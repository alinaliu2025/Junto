"""
Market data processing for JPMorgan Chase stock analysis and correlation with Fed speeches
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class JPMDataLoader:
    """
    Load and process JPMorgan Chase (JPM) stock data with FOMC meeting alignment
    """
    
    def __init__(self, symbol: str = "JPM"):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        
    def fetch_historical_data(self, start_date: str, end_date: str, 
                            interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical JPM stock data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = self.ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol} between {start_date} and {end_date}")
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def get_intraday_data(self, date: str, extended_hours: bool = True) -> pd.DataFrame:
        """
        Get intraday data for a specific date (FOMC meeting day)
        
        Args:
            date: Date in YYYY-MM-DD format
            extended_hours: Include pre/post market data
            
        Returns:
            DataFrame with minute-level data
        """
        try:
            # Get data for the specific day with 1-minute intervals
            start_date = pd.to_datetime(date)
            end_date = start_date + timedelta(days=1)
            
            data = self.ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1m",
                prepost=extended_hours
            )
            
            if not data.empty:
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            return data
            
        except Exception as e:
            print(f"Error fetching intraday data for {date}: {e}")
            return pd.DataFrame()
    
    def get_fomc_reaction_data(self, fomc_datetime: datetime, 
                             hours_before: int = 2, hours_after: int = 6) -> pd.DataFrame:
        """
        Get JPM stock data around FOMC meeting times
        
        Args:
            fomc_datetime: FOMC meeting/announcement datetime
            hours_before: Hours of data before the meeting
            hours_after: Hours of data after the meeting
            
        Returns:
            DataFrame with stock data around FOMC meeting
        """
        # Calculate time window
        start_time = fomc_datetime - timedelta(hours=hours_before)
        end_time = fomc_datetime + timedelta(hours=hours_after)
        
        # Get intraday data for the date range
        start_date = start_time.date()
        end_date = end_time.date()
        
        # If spans multiple days, get data for all days
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_data = self.get_intraday_data(current_date.strftime('%Y-%m-%d'))
            if not daily_data.empty:
                all_data.append(daily_data)
            current_date += timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False)
        
        # Filter to exact time window
        mask = (combined_data.index >= start_time) & (combined_data.index <= end_time)
        filtered_data = combined_data[mask].copy()
        
        # Add time relative to FOMC announcement
        filtered_data['minutes_from_fomc'] = (
            filtered_data.index - fomc_datetime
        ).total_seconds() / 60
        
        return filtered_data


class MarketDataProcessor:
    """
    Process market data to extract features for HRM training
    """
    
    def __init__(self):
        self.jpm_loader = JPMDataLoader()
        
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for JPM stock
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Volatility measures
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price position indicators
        df['price_position_5'] = (df['close'] - df['low'].rolling(5).min()) / (
            df['high'].rolling(5).max() - df['low'].rolling(5).min()
        )
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
        bb_std_dev = df['close'].rolling(window=bb_window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def extract_price_features(self, data: pd.DataFrame, 
                             feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract numerical features from price data for HRM input
        
        Args:
            data: DataFrame with processed market data
            feature_columns: Specific columns to use as features
            
        Returns:
            Feature array [num_samples, num_features]
        """
        if feature_columns is None:
            feature_columns = [
                'returns', 'log_returns', 'volatility_5', 'volatility_20',
                'momentum_5', 'momentum_10', 'volume_ratio', 'price_position_5',
                'bb_position', 'rsi'
            ]
        
        # Select available columns
        available_columns = [col for col in feature_columns if col in data.columns]
        
        if not available_columns:
            raise ValueError("No valid feature columns found in data")
        
        # Extract features and handle NaN values
        features = data[available_columns].values
        
        # Forward fill and backward fill NaN values
        features_df = pd.DataFrame(features, columns=available_columns)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features_df.values.astype(np.float32)
    
    def create_reaction_labels(self, data: pd.DataFrame, fomc_time: datetime,
                             reaction_window_minutes: int = 60) -> np.ndarray:
        """
        Create buy/sell/hold labels based on JPM price reaction to FOMC
        
        Args:
            data: DataFrame with JPM price data around FOMC
            fomc_time: FOMC announcement time
            reaction_window_minutes: Minutes after FOMC to measure reaction
            
        Returns:
            Label array [num_samples] with 0=sell, 1=hold, 2=buy
        """
        labels = np.ones(len(data))  # Default to hold (1)
        
        if 'minutes_from_fomc' not in data.columns:
            return labels
        
        # Find price at FOMC time (or closest)
        fomc_mask = np.abs(data['minutes_from_fomc']) <= 5  # Within 5 minutes
        if not fomc_mask.any():
            return labels
        
        fomc_price = data[fomc_mask]['close'].iloc[0]
        
        # Find price after reaction window
        reaction_mask = (data['minutes_from_fomc'] >= reaction_window_minutes - 5) & \
                       (data['minutes_from_fomc'] <= reaction_window_minutes + 5)
        
        if not reaction_mask.any():
            return labels
        
        reaction_price = data[reaction_mask]['close'].iloc[0]
        
        # Calculate price change
        price_change = (reaction_price - fomc_price) / fomc_price
        
        # Create labels based on thresholds
        buy_threshold = 0.01   # 1% increase
        sell_threshold = -0.01  # 1% decrease
        
        for i, row in data.iterrows():
            minutes_from_fomc = row['minutes_from_fomc']
            
            if minutes_from_fomc < 0:
                # Before FOMC: predict based on eventual reaction
                if price_change > buy_threshold:
                    labels[i] = 2  # Buy signal
                elif price_change < sell_threshold:
                    labels[i] = 0  # Sell signal
                else:
                    labels[i] = 1  # Hold signal
            else:
                # After FOMC: labels based on immediate price action
                current_price = row['close']
                immediate_change = (current_price - fomc_price) / fomc_price
                
                if immediate_change > buy_threshold:
                    labels[i] = 2  # Buy
                elif immediate_change < sell_threshold:
                    labels[i] = 0  # Sell
                else:
                    labels[i] = 1  # Hold
        
        return labels.astype(np.int64)
    
    def prepare_training_data(self, fomc_events: List[Dict], 
                            hours_before: int = 2, hours_after: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from multiple FOMC events
        
        Args:
            fomc_events: List of dicts with 'datetime' and other FOMC info
            hours_before: Hours of data before each FOMC
            hours_after: Hours of data after each FOMC
            
        Returns:
            features: Feature array [total_samples, num_features]
            labels: Label array [total_samples]
        """
        all_features = []
        all_labels = []
        
        for event in fomc_events:
            fomc_datetime = event['datetime']
            
            # Get JPM data around this FOMC event
            jpm_data = self.jpm_loader.get_fomc_reaction_data(
                fomc_datetime, hours_before, hours_after
            )
            
            if jmp_data.empty:
                continue
            
            # Process technical indicators
            processed_data = self.calculate_technical_indicators(jmp_data)
            
            # Extract features
            features = self.extract_price_features(processed_data)
            
            # Create labels
            labels = self.create_reaction_labels(processed_data, fomc_datetime)
            
            all_features.append(features)
            all_labels.append(labels)
        
        if not all_features:
            return np.array([]), np.array([])
        
        # Combine all events
        combined_features = np.vstack(all_features)
        combined_labels = np.concatenate(all_labels)
        
        return combined_features, combined_labels
    
    def normalize_features(self, features: np.ndarray, 
                         stats: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Normalize features using z-score normalization
        
        Args:
            features: Feature array [num_samples, num_features]
            stats: Optional pre-computed normalization statistics
            
        Returns:
            normalized_features: Normalized feature array
            stats: Normalization statistics (mean, std)
        """
        if stats is None:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            stats = {'mean': mean, 'std': std}
        else:
            mean = stats['mean']
            std = stats['std']
        
        normalized_features = (features - mean) / std
        return normalized_features, stats