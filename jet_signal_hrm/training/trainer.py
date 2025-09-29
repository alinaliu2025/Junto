"""
Training pipeline for Jerome Powell HRM with profit optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import wandb
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.hrm_trading import HRMTrading
from ..utils.config import Config


class ProfitOptimizedLoss(nn.Module):
    """
    Custom loss function that combines classification accuracy with profit metrics
    """
    
    def __init__(self, classification_weight: float = 0.5, profit_weight: float = 0.3, 
                 sharpe_weight: float = 0.2, transaction_cost: float = 0.001):
        super().__init__()
        self.classification_weight = classification_weight
        self.profit_weight = profit_weight
        self.sharpe_weight = sharpe_weight
        self.transaction_cost = transaction_cost
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                market_returns: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute profit-optimized loss
        
        Args:
            predictions: Model predictions [batch_size, seq_len, num_classes]
            targets: True labels [batch_size, seq_len]
            market_returns: Market returns for profit calculation [batch_size, seq_len]
            attention_mask: Mask for valid positions [batch_size, seq_len]
            
        Returns:
            Dictionary with loss components
        """
        batch_size, seq_len, num_classes = predictions.shape
        
        # Reshape for loss computation
        pred_flat = predictions.view(-1, num_classes)
        target_flat = targets.view(-1)
        mask_flat = attention_mask.view(-1)
        
        # Classification loss
        ce_losses = self.ce_loss(pred_flat, target_flat)
        ce_loss = (ce_losses * mask_flat.float()).sum() / mask_flat.float().sum()
        
        # Convert predictions to trading signals
        trading_signals = torch.softmax(predictions, dim=-1)
        
        # Calculate profit-based loss
        profit_loss = self._calculate_profit_loss(trading_signals, market_returns, attention_mask)
        
        # Calculate Sharpe ratio loss
        sharpe_loss = self._calculate_sharpe_loss(trading_signals, market_returns, attention_mask)
        
        # Combined loss
        total_loss = (
            self.classification_weight * ce_loss +
            self.profit_weight * profit_loss +
            self.sharpe_weight * sharpe_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': ce_loss,
            'profit_loss': profit_loss,
            'sharpe_loss': sharpe_loss
        }
    
    def _calculate_profit_loss(self, signals: torch.Tensor, returns: torch.Tensor, 
                              mask: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on trading profits"""
        # Convert signals to positions: sell=-1, hold=0, buy=1
        positions = signals[:, :, 2] - signals[:, :, 0]  # buy_prob - sell_prob
        
        # Calculate returns from trading strategy
        strategy_returns = positions * returns
        
        # Apply transaction costs
        position_changes = torch.abs(torch.diff(positions, dim=1, prepend=torch.zeros_like(positions[:, :1])))
        transaction_costs = position_changes * self.transaction_cost
        
        # Net returns after costs
        net_returns = strategy_returns - transaction_costs
        
        # Mask invalid positions
        masked_returns = net_returns * mask.float()
        
        # Profit loss (negative because we want to maximize profit)
        total_return = masked_returns.sum(dim=1) / mask.float().sum(dim=1)
        profit_loss = -total_return.mean()
        
        return profit_loss
    
    def _calculate_sharpe_loss(self, signals: torch.Tensor, returns: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on Sharpe ratio"""
        # Convert signals to positions
        positions = signals[:, :, 2] - signals[:, :, 0]
        
        # Calculate strategy returns
        strategy_returns = positions * returns
        masked_returns = strategy_returns * mask.float()
        
        # Calculate Sharpe ratio for each sequence
        sharpe_ratios = []
        for i in range(strategy_returns.shape[0]):
            seq_returns = masked_returns[i][mask[i]]
            if len(seq_returns) > 1:
                mean_return = seq_returns.mean()
                std_return = seq_returns.std()
                sharpe = mean_return / (std_return + 1e-8)
                sharpe_ratios.append(sharpe)
            else:
                sharpe_ratios.append(torch.tensor(0.0, device=returns.device))
        
        if sharpe_ratios:
            avg_sharpe = torch.stack(sharpe_ratios).mean()
            sharpe_loss = -avg_sharpe  # Negative because we want to maximize Sharpe
        else:
            sharpe_loss = torch.tensor(0.0, device=returns.device)
        
        return sharpe_loss


class HRMTrainer:
    """
    Main trainer class for Jerome Powell HRM
    """
    
    def __init__(self, config: Config, model: HRMTrading, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = ProfitOptimizedLoss(
            classification_weight=0.5,
            profit_weight=0.3,
            sharpe_weight=0.2,
            transaction_cost=config.trading.transaction_cost
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.model.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_profit = float('-inf')
        self.training_history = []
        
        # Early stopping
        self.patience = config.model.early_stopping_patience
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'classification_loss': [],
            'profit_loss': [],
            'sharpe_loss': [],
            'accuracy': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            pitch_features = batch['pitch_features'].to(self.device)
            word_features = batch['word_features'].to(self.device)
            market_features = batch['market_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Initialize hidden states
            self.model.init_hidden(pitch_features.shape[0], self.device)
            
            # Forward pass
            sentiment_scores, trading_signals = self.model(
                pitch_features, word_features, market_features
            )
            
            # Calculate market returns from market features (assuming returns are in first column)
            market_returns = market_features[:, :, 0]  # Assuming returns are first feature
            
            # Compute loss
            loss_dict = self.criterion(trading_signals, labels, market_returns, attention_mask)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss_dict['total_loss'].item())
            epoch_metrics['classification_loss'].append(loss_dict['classification_loss'].item())
            epoch_metrics['profit_loss'].append(loss_dict['profit_loss'].item())
            epoch_metrics['sharpe_loss'].append(loss_dict['sharpe_loss'].item())
            
            # Calculate accuracy
            predictions = torch.argmax(trading_signals, dim=-1)
            correct = (predictions == labels) & attention_mask
            accuracy = correct.float().sum() / attention_mask.float().sum()
            epoch_metrics['accuracy'].append(accuracy.item())
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss_dict['total_loss'].item():.4f}")
        
        # Calculate epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        avg_metrics['total_loss'] = np.mean(epoch_losses)
        
        return avg_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = {
            'classification_loss': [],
            'profit_loss': [],
            'sharpe_loss': [],
            'accuracy': [],
            'total_profit': [],
            'sharpe_ratio': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                pitch_features = batch['pitch_features'].to(self.device)
                word_features = batch['word_features'].to(self.device)
                market_features = batch['market_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Initialize hidden states
                self.model.init_hidden(pitch_features.shape[0], self.device)
                
                # Forward pass
                sentiment_scores, trading_signals = self.model(
                    pitch_features, word_features, market_features
                )
                
                # Calculate market returns
                market_returns = market_features[:, :, 0]
                
                # Compute loss
                loss_dict = self.criterion(trading_signals, labels, market_returns, attention_mask)
                
                # Track metrics
                epoch_losses.append(loss_dict['total_loss'].item())
                epoch_metrics['classification_loss'].append(loss_dict['classification_loss'].item())
                epoch_metrics['profit_loss'].append(loss_dict['profit_loss'].item())
                epoch_metrics['sharpe_loss'].append(loss_dict['sharpe_loss'].item())
                
                # Calculate accuracy
                predictions = torch.argmax(trading_signals, dim=-1)
                correct = (predictions == labels) & attention_mask
                accuracy = correct.float().sum() / attention_mask.float().sum()
                epoch_metrics['accuracy'].append(accuracy.item())
                
                # Calculate trading performance
                profit, sharpe = self._calculate_trading_performance(
                    trading_signals, market_returns, attention_mask
                )
                epoch_metrics['total_profit'].append(profit)
                epoch_metrics['sharpe_ratio'].append(sharpe)
        
        # Calculate epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        avg_metrics['total_loss'] = np.mean(epoch_losses)
        
        return avg_metrics
    
    def _calculate_trading_performance(self, signals: torch.Tensor, returns: torch.Tensor,
                                     mask: torch.Tensor) -> Tuple[float, float]:
        """Calculate trading performance metrics"""
        # Convert signals to positions
        positions = signals[:, :, 2] - signals[:, :, 0]  # buy_prob - sell_prob
        
        # Calculate strategy returns
        strategy_returns = positions * returns
        masked_returns = strategy_returns * mask.float()
        
        # Calculate total profit
        total_profit = masked_returns.sum().item()
        
        # Calculate Sharpe ratio
        flat_returns = masked_returns[mask].cpu().numpy()
        if len(flat_returns) > 1:
            sharpe_ratio = np.mean(flat_returns) / (np.std(flat_returns) + 1e-8)
        else:
            sharpe_ratio = 0.0
        
        return total_profit, sharpe_ratio
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              save_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training for {self.config.model.max_epochs} epochs...")
        
        for epoch in range(self.config.model.max_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{self.config.model.max_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_log)
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Profit: {val_metrics['total_profit']:.4f}, "
                  f"Val Sharpe: {val_metrics['sharpe_ratio']:.4f}")
            
            # Save best model based on profit
            if val_metrics['total_profit'] > self.best_profit:
                self.best_profit = val_metrics['total_profit']
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(save_path / "best_model.pt", is_best=True)
                print(f"New best model saved! Profit: {self.best_profit:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_path / f"checkpoint_epoch_{epoch + 1}.pt")
        
        print("\nTraining completed!")
        return self.training_history
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_profit': self.best_profit,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            print(f"Best checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_profit = checkpoint['best_profit']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.training_history:
            print("No training history to plot")
            return
        
        epochs = [log['epoch'] for log in self.training_history]
        train_losses = [log['train']['total_loss'] for log in self.training_history]
        val_losses = [log['val']['total_loss'] for log in self.training_history]
        val_profits = [log['val']['total_profit'] for log in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Profit plot
        ax2.plot(epochs, val_profits, label='Val Profit', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Profit')
        ax2.set_title('Validation Profit')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()