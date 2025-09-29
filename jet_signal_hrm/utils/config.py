"""Configuration management for Jerome Powell HRM"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """HRM model configuration"""
    # Audio processing
    pitch_features: int = 64
    word_embed_dim: int = 256
    audio_sample_rate: int = 16000
    
    # Model architecture
    high_hidden_size: int = 512
    low_hidden_size: int = 256
    attention_heads: int = 8
    sparsity_level: float = 0.5
    
    # Trading parameters
    jpm_features: int = 32
    sentiment_classes: int = 3  # hawkish, dovish, neutral
    signal_outputs: int = 3     # buy, sell, hold
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Quantization
    enable_quantization: bool = True
    quantization_dtype: str = "qint8"


@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.05   # 5% take profit
    
    # Signal thresholds
    buy_threshold: float = 0.7
    sell_threshold: float = 0.7
    confidence_threshold: float = 0.6
    
    # Backtesting
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005        # 0.05% slippage


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Paths
    audio_data_path: str = "data/powell_speeches"
    jpm_data_path: str = "data/jpm_prices"
    processed_data_path: str = "data/processed"
    
    # Audio processing
    audio_segment_length: int = 30  # seconds
    overlap_ratio: float = 0.5
    
    # Market data
    price_history_days: int = 252   # 1 year
    intraday_frequency: str = "1min"
    
    # Training split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.trading = TradingConfig()
        self.data = DataConfig()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if 'model' in config_dict:
            self.model = ModelConfig(**config_dict['model'])
        if 'trading' in config_dict:
            self.trading = TradingConfig(**config_dict['trading'])
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {
            'model': asdict(self.model),
            'trading': asdict(self.trading),
            'data': asdict(self.data)
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': asdict(self.model),
            'trading': asdict(self.trading),
            'data': asdict(self.data)
        }