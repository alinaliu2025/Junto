"""
Hierarchical Reasoning Model for Jet Signal Analysis
Multi-level reasoning over corporate flight patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FlightEvent:
    """Represents a flight event at different hierarchy levels"""
    level: int  # 0=single leg, 1=multi-leg trip, 2=company-wide pattern
    timestamp: float
    company_ticker: str
    features: torch.Tensor
    metadata: Dict

class HRMJetModel(nn.Module):
    """
    Hierarchical Reasoning Model for Corporate Jet Analysis
    
    Architecture:
    - Level 0: Single flight leg analysis (unusual routes, timing)
    - Level 1: Multi-leg trip reasoning (executive travel patterns)  
    - Level 2: Company-wide analysis (fleet coordination, event correlation)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.d_model = config.get('d_model', 256)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        self.max_reasoning_steps = config.get('max_reasoning_steps', 10)
        
        # Feature dimensions
        self.flight_features = config.get('flight_features', 32)  # lat, lon, time, aircraft_type, etc.
        self.context_features = config.get('context_features', 64)  # market, news, calendar
        self.company_features = config.get('company_features', 16)  # sector, size, volatility
        
        # Input embeddings
        self.flight_embedding = nn.Linear(self.flight_features, self.d_model)
        self.context_embedding = nn.Linear(self.context_features, self.d_model)
        self.company_embedding = nn.Linear(self.company_features, self.d_model)
        self.level_embedding = nn.Embedding(3, self.d_model)  # 3 hierarchy levels
        
        # Hierarchical reasoning layers
        self.level0_processor = FlightLegProcessor(self.d_model, self.n_heads)
        self.level1_processor = TripProcessor(self.d_model, self.n_heads)
        self.level2_processor = CompanyPatternProcessor(self.d_model, self.n_heads)
        
        # Cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            self.d_model, self.n_heads, batch_first=True
        )
        
        # Reasoning controller (decides when to halt)
        self.reasoning_controller = ReasoningController(self.d_model)
        
        # Output heads
        self.conviction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 3)  # buy, hold, sell
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, flight_events: List[FlightEvent]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning
        
        Args:
            flight_events: List of flight events at different levels
            
        Returns:
            Dictionary with conviction, confidence, and risk scores
        """
        batch_size = len(flight_events)
        device = next(self.parameters()).device
        
        # Group events by level
        level0_events = [e for e in flight_events if e.level == 0]
        level1_events = [e for e in flight_events if e.level == 1]
        level2_events = [e for e in flight_events if e.level == 2]
        
        # Process each level
        level0_repr = self._process_level0(level0_events) if level0_events else None
        level1_repr = self._process_level1(level1_events) if level1_events else None
        level2_repr = self._process_level2(level2_events) if level2_events else None
        
        # Combine representations across levels
        combined_repr = self._combine_levels(level0_repr, level1_repr, level2_repr)
        
        # Iterative reasoning with halting
        final_repr = self._iterative_reasoning(combined_repr)
        
        # Generate outputs
        conviction_logits = self.conviction_head(final_repr)
        confidence_score = self.confidence_head(final_repr)
        risk_score = self.risk_head(final_repr)
        
        return {
            'conviction_logits': conviction_logits,
            'conviction_probs': F.softmax(conviction_logits, dim=-1),
            'confidence': confidence_score,
            'risk': risk_score,
            'reasoning_steps': getattr(self, '_last_reasoning_steps', 1)
        }
    
    def _process_level0(self, events: List[FlightEvent]) -> torch.Tensor:
        """Process Level 0: Individual flight legs"""
        if not events:
            return torch.zeros(1, self.d_model, device=next(self.parameters()).device)
            
        # Stack features and embed
        features = torch.stack([e.features for e in events])
        embedded = self.flight_embedding(features)
        
        # Add level embedding
        level_emb = self.level_embedding(torch.zeros(len(events), dtype=torch.long, device=embedded.device))
        embedded = embedded + level_emb
        
        # Process through Level 0 processor
        processed = self.level0_processor(embedded)
        
        # Aggregate (mean pooling for now)
        return processed.mean(dim=0, keepdim=True)
    
    def _process_level1(self, events: List[FlightEvent]) -> torch.Tensor:
        """Process Level 1: Multi-leg trips"""
        if not events:
            return torch.zeros(1, self.d_model, device=next(self.parameters()).device)
            
        features = torch.stack([e.features for e in events])
        embedded = self.flight_embedding(features)
        
        level_emb = self.level_embedding(torch.ones(len(events), dtype=torch.long, device=embedded.device))
        embedded = embedded + level_emb
        
        processed = self.level1_processor(embedded)
        return processed.mean(dim=0, keepdim=True)
    
    def _process_level2(self, events: List[FlightEvent]) -> torch.Tensor:
        """Process Level 2: Company-wide patterns"""
        if not events:
            return torch.zeros(1, self.d_model, device=next(self.parameters()).device)
            
        features = torch.stack([e.features for e in events])
        embedded = self.flight_embedding(features)
        
        level_emb = self.level_embedding(torch.full((len(events),), 2, dtype=torch.long, device=embedded.device))
        embedded = embedded + level_emb
        
        processed = self.level2_processor(embedded)
        return processed.mean(dim=0, keepdim=True)
    
    def _combine_levels(self, level0: Optional[torch.Tensor], 
                       level1: Optional[torch.Tensor], 
                       level2: Optional[torch.Tensor]) -> torch.Tensor:
        """Combine representations from different levels"""
        representations = []
        
        if level0 is not None:
            representations.append(level0)
        if level1 is not None:
            representations.append(level1)
        if level2 is not None:
            representations.append(level2)
            
        if not representations:
            return torch.zeros(1, self.d_model, device=next(self.parameters()).device)
        
        # Stack and apply cross-level attention
        stacked = torch.cat(representations, dim=0).unsqueeze(0)  # [1, num_levels, d_model]
        
        attended, _ = self.cross_level_attention(stacked, stacked, stacked)
        
        # Aggregate across levels
        return attended.mean(dim=1)  # [1, d_model]
    
    def _iterative_reasoning(self, initial_repr: torch.Tensor) -> torch.Tensor:
        """Iterative reasoning with adaptive halting"""
        current_repr = initial_repr
        
        for step in range(self.max_reasoning_steps):
            # Check if we should halt reasoning
            halt_prob = self.reasoning_controller(current_repr)
            
            if halt_prob > 0.5:  # Halt threshold
                self._last_reasoning_steps = step + 1
                break
                
            # Continue reasoning (simplified - would have more complex reasoning loop)
            current_repr = current_repr + 0.1 * torch.randn_like(current_repr)
        else:
            self._last_reasoning_steps = self.max_reasoning_steps
            
        return current_repr

class FlightLegProcessor(nn.Module):
    """Processes individual flight legs (Level 0)"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class TripProcessor(nn.Module):
    """Processes multi-leg trips (Level 1)"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class CompanyPatternProcessor(nn.Module):
    """Processes company-wide patterns (Level 2)"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class ReasoningController(nn.Module):
    """Controls when to halt iterative reasoning"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.halt_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.halt_predictor(x)