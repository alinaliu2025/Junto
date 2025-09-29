"""
Dataset class for Jerome Powell HRM training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from .audio_processor import AudioProcessor
from .speech_processor import BeginningWordsProcessor
from .market_data import MarketDataProcessor


class PowellSpeechDataset(Dataset):
    """
    Dataset for training HRM on Jerome Powell speeches and JPM stock reactions
    """
    
    def __init__(self, data_config: Dict, split: str = "train"):
        """
        Initialize dataset
        
        Args:
            data_config: Configuration dictionary with data paths and parameters
            split: Dataset split ("train", "val", "test")
        """
        self.data_config = data_config
        self.split = split
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=data_config.get('sample_rate', 16000),
            segment_length=data_config.get('segment_length', 30),
            overlap_ratio=data_config.get('overlap_ratio', 0.5)
        )
        
        self.speech_processor = BeginningWordsProcessor(
            embedding_model=data_config.get('embedding_model', 'bert-base-uncased')
        )
        
        self.market_processor = MarketDataProcessor()
        
        # Load data
        self.samples = self._load_samples()
        
        # Normalization statistics
        self.audio_stats = None
        self.market_stats = None
        
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata from JSON files"""
        data_path = Path(self.data_config['processed_data_path'])
        split_file = data_path / f"{self.split}_samples.json"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing model inputs and targets
        """
        sample = self.samples[idx]
        
        # Load audio features
        audio_features = self._load_audio_features(sample['audio_path'])
        
        # Load speech/word features
        word_features = self._load_word_features(sample['audio_path'])
        
        # Load market features
        market_features, labels = self._load_market_features(sample)
        
        # Ensure all sequences have the same length
        min_length = min(
            audio_features.shape[0],
            word_features.shape[0],
            market_features.shape[0]
        )
        
        return {
            'pitch_features': audio_features[:min_length],
            'word_features': word_features[:min_length],
            'market_features': market_features[:min_length],
            'labels': labels[:min_length],
            'sample_id': sample['id'],
            'fomc_datetime': sample['fomc_datetime']
        }
    
    def _load_audio_features(self, audio_path: str) -> torch.Tensor:
        """Load and process audio features"""
        # Check if processed features exist
        processed_path = Path(audio_path).with_suffix('.audio_features.pt')
        
        if processed_path.exists():
            return torch.load(processed_path)
        
        # Process audio file
        features, _ = self.audio_processor.process_speech_file(audio_path)
        
        # Save processed features
        torch.save(features, processed_path)
        
        return features
    
    def _load_word_features(self, audio_path: str) -> torch.Tensor:
        """Load and process word embedding features"""
        # Check if processed features exist
        processed_path = Path(audio_path).with_suffix('.word_features.pt')
        
        if processed_path.exists():
            return torch.load(processed_path)
        
        # Process speech segments
        audio_segments = self._get_audio_segments(audio_path)
        features = self.speech_processor.batch_process_segments(audio_segments)
        
        # Save processed features
        torch.save(features, processed_path)
        
        return features
    
    def _get_audio_segments(self, audio_path: str) -> List[str]:
        """Get list of audio segment paths"""
        # This would be implemented based on how audio is segmented
        # For now, return the original path (assuming it's already segmented)
        return [audio_path]
    
    def _load_market_features(self, sample: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load market features and labels"""
        fomc_datetime = datetime.fromisoformat(sample['fomc_datetime'])
        
        # Get JPM data around FOMC
        jpm_data = self.market_processor.jpm_loader.get_fomc_reaction_data(
            fomc_datetime,
            hours_before=self.data_config.get('hours_before', 2),
            hours_after=self.data_config.get('hours_after', 6)
        )
        
        if jmp_data.empty:
            # Return dummy data if no market data available
            dummy_features = torch.zeros(1, 32)  # Assuming 32 market features
            dummy_labels = torch.zeros(1, dtype=torch.long)
            return dummy_features, dummy_labels
        
        # Process market data
        processed_data = self.market_processor.calculate_technical_indicators(jmp_data)
        
        # Extract features
        features = self.market_processor.extract_price_features(processed_data)
        
        # Create labels
        labels = self.market_processor.create_reaction_labels(processed_data, fomc_datetime)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def normalize_features(self, stats: Optional[Dict] = None) -> Dict:
        """Normalize dataset features"""
        if stats is None:
            # Compute normalization statistics from training data
            all_audio_features = []
            all_market_features = []
            
            for i in range(len(self)):
                sample = self[i]
                all_audio_features.append(sample['pitch_features'])
                all_market_features.append(sample['market_features'])
            
            # Compute statistics
            audio_features_cat = torch.cat(all_audio_features, dim=0)
            market_features_cat = torch.cat(all_market_features, dim=0)
            
            self.audio_stats = {
                'mean': audio_features_cat.mean(dim=0),
                'std': torch.clamp(audio_features_cat.std(dim=0), min=1e-8)
            }
            
            self.market_stats = {
                'mean': market_features_cat.mean(dim=0),
                'std': torch.clamp(market_features_cat.std(dim=0), min=1e-8)
            }
            
            stats = {
                'audio_stats': self.audio_stats,
                'market_stats': self.market_stats
            }
        else:
            self.audio_stats = stats['audio_stats']
            self.market_stats = stats['market_stats']
        
        return stats
    
    def get_normalized_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get normalized sample"""
        sample = self[idx]
        
        if self.audio_stats is not None:
            sample['pitch_features'] = (
                sample['pitch_features'] - self.audio_stats['mean']
            ) / self.audio_stats['std']
        
        if self.market_stats is not None:
            sample['market_features'] = (
                sample['market_features'] - self.market_stats['mean']
            ) / self.market_stats['std']
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences
    """
    # Find maximum sequence length
    max_len = max(sample['pitch_features'].shape[0] for sample in batch)
    
    batch_size = len(batch)
    pitch_dim = batch[0]['pitch_features'].shape[1]
    word_dim = batch[0]['word_features'].shape[1]
    market_dim = batch[0]['market_features'].shape[1]
    
    # Initialize padded tensors
    pitch_features = torch.zeros(batch_size, max_len, pitch_dim)
    word_features = torch.zeros(batch_size, max_len, word_dim)
    market_features = torch.zeros(batch_size, max_len, market_dim)
    labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    sample_ids = []
    fomc_datetimes = []
    
    for i, sample in enumerate(batch):
        seq_len = sample['pitch_features'].shape[0]
        
        pitch_features[i, :seq_len] = sample['pitch_features']
        word_features[i, :seq_len] = sample['word_features']
        market_features[i, :seq_len] = sample['market_features']
        labels[i, :seq_len] = sample['labels']
        attention_mask[i, :seq_len] = True
        
        sample_ids.append(sample['sample_id'])
        fomc_datetimes.append(sample['fomc_datetime'])
    
    return {
        'pitch_features': pitch_features,
        'word_features': word_features,
        'market_features': market_features,
        'labels': labels,
        'attention_mask': attention_mask,
        'sample_ids': sample_ids,
        'fomc_datetimes': fomc_datetimes
    }


def create_data_loaders(data_config: Dict, batch_size: int = 16, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        data_config: Data configuration dictionary
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = PowellSpeechDataset(data_config, split="train")
    val_dataset = PowellSpeechDataset(data_config, split="val")
    test_dataset = PowellSpeechDataset(data_config, split="test")
    
    # Normalize features using training set statistics
    stats = train_dataset.normalize_features()
    val_dataset.normalize_features(stats)
    test_dataset.normalize_features(stats)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader