"""
Audio processing pipeline for Jerome Powell speech analysis
"""

import librosa
import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import scipy.signal
from scipy.stats import zscore


class PitchExtractor:
    """
    Extract pitch features from audio for emotional and stress analysis.
    Focuses on fundamental frequency, variance, and prosodic patterns.
    """
    
    def __init__(self, sample_rate: int = 16000, hop_length: int = 512, 
                 fmin: float = 50.0, fmax: float = 400.0):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin  # Minimum frequency for pitch detection
        self.fmax = fmax  # Maximum frequency for pitch detection
        
    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive pitch features from audio signal
        
        Args:
            audio: Audio signal array
            
        Returns:
            Dictionary containing various pitch features
        """
        # Extract fundamental frequency using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            frame_length=2048
        )
        
        # Handle NaN values in f0
        f0_clean = np.nan_to_num(f0, nan=0.0)
        
        # Basic pitch statistics
        pitch_mean = np.mean(f0_clean[f0_clean > 0]) if np.any(f0_clean > 0) else 0.0
        pitch_std = np.std(f0_clean[f0_clean > 0]) if np.any(f0_clean > 0) else 0.0
        pitch_range = np.ptp(f0_clean[f0_clean > 0]) if np.any(f0_clean > 0) else 0.0
        
        # Pitch contour features
        pitch_slope = self._calculate_pitch_slope(f0_clean)
        pitch_jitter = self._calculate_jitter(f0_clean)
        
        # Prosodic features
        voiced_ratio = np.mean(voiced_flag)
        pitch_velocity = np.diff(f0_clean)
        pitch_acceleration = np.diff(pitch_velocity)
        
        # Emotional indicators
        pitch_variance_norm = pitch_std / pitch_mean if pitch_mean > 0 else 0.0
        high_pitch_ratio = np.mean(f0_clean > pitch_mean + pitch_std) if pitch_mean > 0 else 0.0
        
        return {
            'f0': f0_clean,
            'voiced_flag': voiced_flag.astype(np.float32),
            'voiced_probs': voiced_probs,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'pitch_slope': pitch_slope,
            'pitch_jitter': pitch_jitter,
            'voiced_ratio': voiced_ratio,
            'pitch_variance_norm': pitch_variance_norm,
            'high_pitch_ratio': high_pitch_ratio,
            'pitch_velocity_mean': np.mean(np.abs(pitch_velocity)),
            'pitch_acceleration_mean': np.mean(np.abs(pitch_acceleration))
        }
    
    def _calculate_pitch_slope(self, f0: np.ndarray) -> float:
        """Calculate overall pitch slope (rising/falling tendency)"""
        valid_indices = np.where(f0 > 0)[0]
        if len(valid_indices) < 2:
            return 0.0
        
        # Linear regression to find slope
        x = valid_indices
        y = f0[valid_indices]
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
        return slope
    
    def _calculate_jitter(self, f0: np.ndarray, window_size: int = 5) -> float:
        """Calculate pitch jitter (local pitch variation)"""
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) < window_size:
            return 0.0
        
        jitter_values = []
        for i in range(len(valid_f0) - window_size + 1):
            window = valid_f0[i:i + window_size]
            local_jitter = np.std(window) / np.mean(window) if np.mean(window) > 0 else 0.0
            jitter_values.append(local_jitter)
        
        return np.mean(jitter_values) if jitter_values else 0.0
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features for voice quality analysis"""
        # Compute STFT
        stft = librosa.stft(audio, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff (energy distribution)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth (spread of energy)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate (voice quality indicator)
        zcr = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        # MFCCs for voice characteristics
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossing_rate': zcr,
            'mfccs': mfccs
        }


class AudioProcessor:
    """
    Main audio processing pipeline for Jerome Powell speech analysis.
    Handles audio loading, segmentation, and feature extraction.
    """
    
    def __init__(self, sample_rate: int = 16000, segment_length: int = 30, 
                 overlap_ratio: float = 0.5):
        self.sample_rate = sample_rate
        self.segment_length = segment_length  # seconds
        self.overlap_ratio = overlap_ratio
        self.pitch_extractor = PitchExtractor(sample_rate=sample_rate)
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio: Audio signal array
            duration: Duration in seconds
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(audio) / self.sample_rate
        return audio, duration
    
    def segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Segment audio into overlapping windows for processing
        
        Args:
            audio: Audio signal array
            
        Returns:
            List of audio segments
        """
        segment_samples = int(self.segment_length * self.sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap_ratio))
        
        segments = []
        start = 0
        
        while start + segment_samples <= len(audio):
            segment = audio[start:start + segment_samples]
            segments.append(segment)
            start += hop_samples
        
        # Add final segment if there's remaining audio
        if start < len(audio):
            final_segment = audio[start:]
            # Pad to segment length if needed
            if len(final_segment) < segment_samples:
                padding = segment_samples - len(final_segment)
                final_segment = np.pad(final_segment, (0, padding), mode='constant')
            segments.append(final_segment)
        
        return segments
    
    def extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive audio features for HRM input
        
        Args:
            audio: Audio signal array
            
        Returns:
            Feature vector combining pitch and spectral features
        """
        # Extract pitch features
        pitch_features = self.pitch_extractor.extract_pitch_features(audio)
        
        # Extract spectral features
        spectral_features = self.pitch_extractor.extract_spectral_features(audio)
        
        # Combine features into a single vector
        feature_vector = []
        
        # Add scalar pitch features
        scalar_pitch_features = [
            'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_slope',
            'pitch_jitter', 'voiced_ratio', 'pitch_variance_norm',
            'high_pitch_ratio', 'pitch_velocity_mean', 'pitch_acceleration_mean'
        ]
        
        for feature_name in scalar_pitch_features:
            feature_vector.append(pitch_features[feature_name])
        
        # Add statistical summaries of time-series features
        time_series_features = {
            'f0': pitch_features['f0'],
            'voiced_probs': pitch_features['voiced_probs'],
            'spectral_centroid': spectral_features['spectral_centroid'],
            'spectral_rolloff': spectral_features['spectral_rolloff'],
            'spectral_bandwidth': spectral_features['spectral_bandwidth'],
            'zero_crossing_rate': spectral_features['zero_crossing_rate']
        }
        
        for feature_name, feature_data in time_series_features.items():
            if len(feature_data) > 0:
                feature_vector.extend([
                    np.mean(feature_data),
                    np.std(feature_data),
                    np.min(feature_data),
                    np.max(feature_data),
                    np.median(feature_data)
                ])
            else:
                feature_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add MFCC statistics
        mfccs = spectral_features['mfccs']
        for i in range(mfccs.shape[0]):
            mfcc_coeff = mfccs[i]
            feature_vector.extend([
                np.mean(mfcc_coeff),
                np.std(mfcc_coeff)
            ])
        
        return np.array(feature_vector, dtype=np.float32)
    
    def process_speech_file(self, audio_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process complete speech file and extract features
        
        Args:
            audio_path: Path to Jerome Powell speech audio file
            
        Returns:
            features: Tensor of extracted features [num_segments, feature_dim]
            metadata: Dictionary with processing metadata
        """
        # Load audio
        audio, duration = self.load_audio(audio_path)
        
        # Segment audio
        segments = self.segment_audio(audio)
        
        # Extract features from each segment
        segment_features = []
        for segment in segments:
            features = self.extract_audio_features(segment)
            segment_features.append(features)
        
        # Stack features
        feature_tensor = torch.tensor(np.stack(segment_features), dtype=torch.float32)
        
        # Metadata
        metadata = {
            'audio_path': audio_path,
            'duration': duration,
            'num_segments': len(segments),
            'segment_length': self.segment_length,
            'overlap_ratio': self.overlap_ratio,
            'feature_dim': feature_tensor.shape[-1],
            'sample_rate': self.sample_rate
        }
        
        return feature_tensor, metadata
    
    def normalize_features(self, features: torch.Tensor, 
                          stats: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Normalize features using z-score normalization
        
        Args:
            features: Feature tensor [batch_size, seq_len, feature_dim]
            stats: Optional pre-computed statistics for normalization
            
        Returns:
            normalized_features: Normalized feature tensor
            stats: Normalization statistics (mean, std)
        """
        if stats is None:
            # Compute statistics across batch and sequence dimensions
            mean = features.mean(dim=(0, 1), keepdim=True)
            std = features.std(dim=(0, 1), keepdim=True)
            std = torch.clamp(std, min=1e-8)  # Avoid division by zero
            stats = {'mean': mean, 'std': std}
        else:
            mean = stats['mean']
            std = stats['std']
        
        normalized_features = (features - mean) / std
        return normalized_features, stats