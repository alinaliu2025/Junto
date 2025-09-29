"""
Trading signal generation from HRM jet analysis
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Represents a trading signal generated from jet analysis"""
    ticker: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    conviction: float  # -1 to 1
    confidence: float  # 0 to 1
    risk_score: float  # 0 to 1
    reasoning: str
    timestamp: datetime
    expected_horizon: str  # '1D', '1W', '1M'
    position_size: float  # Recommended position size (0-1)
    stop_loss: Optional[float]
    take_profit: Optional[float]
    metadata: Dict

class SignalGenerator:
    """Generates trading signals from HRM jet model predictions"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        
        # Signal thresholds
        self.buy_threshold = config.get('buy_threshold', 0.6)
        self.sell_threshold = config.get('sell_threshold', -0.6)
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_risk = config.get('max_risk', 0.3)
        
        # Position sizing parameters
        self.base_position_size = config.get('base_position_size', 0.02)  # 2% of portfolio
        self.max_position_size = config.get('max_position_size', 0.10)   # 10% max
        self.risk_scaling = config.get('risk_scaling', True)
        
        # Risk management
        self.default_stop_loss = config.get('default_stop_loss', 0.02)   # 2%
        self.default_take_profit = config.get('default_take_profit', 0.05) # 5%
        
    def generate_signals(self, flight_events: List, market_data: Dict) -> List[TradingSignal]:
        """
        Generate trading signals from flight events
        
        Args:
            flight_events: List of processed flight events
            market_data: Current market data and context
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Group events by company
        company_events = self._group_events_by_company(flight_events)
        
        for ticker, events in company_events.items():
            try:
                signal = self._generate_company_signal(ticker, events, market_data)
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {ticker}: {e}")
                continue
        
        # Filter and rank signals
        signals = self._filter_signals(signals)
        signals = self._rank_signals(signals)
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    def _group_events_by_company(self, flight_events: List) -> Dict[str, List]:
        """Group flight events by company ticker"""
        company_events = {}
        
        for event in flight_events:
            ticker = event.company_ticker
            if ticker not in company_events:
                company_events[ticker] = []
            company_events[ticker].append(event)
            
        return company_events
    
    def _generate_company_signal(self, ticker: str, events: List, 
                               market_data: Dict) -> Optional[TradingSignal]:
        """Generate trading signal for a specific company"""
        
        # Run HRM model prediction
        with torch.no_grad():
            model_output = self.model(events)
            
        conviction_probs = model_output['conviction_probs'].squeeze()
        confidence = model_output['confidence'].squeeze().item()
        risk_score = model_output['risk'].squeeze().item()
        reasoning_steps = model_output['reasoning_steps']
        
        # Convert probabilities to conviction score (-1 to 1)
        # [sell_prob, hold_prob, buy_prob] -> conviction
        sell_prob, hold_prob, buy_prob = conviction_probs.tolist()
        conviction = buy_prob - sell_prob
        
        # Determine signal direction
        if conviction >= self.buy_threshold and confidence >= self.min_confidence:
            signal_type = 'BUY'
        elif conviction <= self.sell_threshold and confidence >= self.min_confidence:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        # Skip low-confidence or high-risk signals
        if confidence < self.min_confidence or risk_score > self.max_risk:
            return None
        
        # Calculate position sizing
        position_size = self._calculate_position_size(conviction, confidence, risk_score)
        
        # Set stop loss and take profit
        stop_loss, take_profit = self._calculate_risk_levels(
            signal_type, conviction, risk_score, market_data.get(ticker, {})
        )
        
        # Generate reasoning explanation
        reasoning = self._generate_reasoning(
            ticker, events, conviction, confidence, reasoning_steps
        )
        
        # Determine expected time horizon
        horizon = self._determine_horizon(events, conviction)
        
        return TradingSignal(
            ticker=ticker,
            signal=signal_type,
            conviction=conviction,
            confidence=confidence,
            risk_score=risk_score,
            reasoning=reasoning,
            timestamp=datetime.utcnow(),
            expected_horizon=horizon,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'num_events': len(events),
                'reasoning_steps': reasoning_steps,
                'model_version': getattr(self.model, 'version', '1.0'),
                'signal_strength': abs(conviction) * confidence
            }
        )
    
    def _calculate_position_size(self, conviction: float, confidence: float, 
                               risk_score: float) -> float:
        """Calculate recommended position size"""
        
        # Base size scaled by conviction and confidence
        size = self.base_position_size * abs(conviction) * confidence
        
        # Scale down by risk
        if self.risk_scaling:
            size *= (1 - risk_score)
        
        # Apply limits
        size = max(0.001, min(size, self.max_position_size))
        
        return round(size, 4)
    
    def _calculate_risk_levels(self, signal_type: str, conviction: float, 
                             risk_score: float, market_data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        
        if signal_type == 'HOLD':
            return None, None
        
        # Adjust risk levels based on conviction and risk score
        stop_loss_pct = self.default_stop_loss * (1 + risk_score)
        take_profit_pct = self.default_take_profit * abs(conviction)
        
        # Get current price for absolute levels
        current_price = market_data.get('price')
        if not current_price:
            return stop_loss_pct, take_profit_pct  # Return as percentages
        
        if signal_type == 'BUY':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # SELL
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        
        return round(stop_loss, 2), round(take_profit, 2)
    
    def _generate_reasoning(self, ticker: str, events: List, conviction: float, 
                          confidence: float, reasoning_steps: int) -> str:
        """Generate human-readable reasoning for the signal"""
        
        # Analyze event patterns
        level_counts = {0: 0, 1: 0, 2: 0}
        for event in events:
            level_counts[event.level] += 1
        
        # Build reasoning string
        reasoning_parts = []
        
        if level_counts[2] > 0:
            reasoning_parts.append(f"Company-wide flight pattern analysis ({level_counts[2]} patterns)")
        
        if level_counts[1] > 0:
            reasoning_parts.append(f"Executive travel analysis ({level_counts[1]} trips)")
            
        if level_counts[0] > 0:
            reasoning_parts.append(f"Individual flight anomalies ({level_counts[0]} flights)")
        
        reasoning = f"{ticker}: " + "; ".join(reasoning_parts)
        reasoning += f". Model confidence: {confidence:.1%}, reasoning depth: {reasoning_steps} steps"
        
        # Add conviction interpretation
        if abs(conviction) > 0.8:
            strength = "Strong"
        elif abs(conviction) > 0.6:
            strength = "Moderate"
        else:
            strength = "Weak"
            
        direction = "bullish" if conviction > 0 else "bearish"
        reasoning += f". {strength} {direction} signal detected."
        
        return reasoning
    
    def _determine_horizon(self, events: List, conviction: float) -> str:
        """Determine expected time horizon for signal"""
        
        # Analyze event timing and types
        has_level2 = any(e.level == 2 for e in events)
        has_level1 = any(e.level == 1 for e in events)
        
        # Company-wide patterns suggest longer horizons
        if has_level2 and abs(conviction) > 0.7:
            return '1M'  # 1 month
        elif has_level1 and abs(conviction) > 0.6:
            return '1W'  # 1 week
        else:
            return '1D'  # 1 day
    
    def _filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on quality criteria"""
        
        filtered = []
        for signal in signals:
            # Skip HOLD signals
            if signal.signal == 'HOLD':
                continue
                
            # Quality filters
            if (signal.confidence >= self.min_confidence and 
                signal.risk_score <= self.max_risk and
                abs(signal.conviction) >= min(self.buy_threshold, abs(self.sell_threshold))):
                filtered.append(signal)
        
        return filtered
    
    def _rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Rank signals by quality score"""
        
        def signal_quality(signal: TradingSignal) -> float:
            return abs(signal.conviction) * signal.confidence * (1 - signal.risk_score)
        
        return sorted(signals, key=signal_quality, reverse=True)
    
    def format_signals_for_display(self, signals: List[TradingSignal]) -> pd.DataFrame:
        """Format signals for display/logging"""
        
        if not signals:
            return pd.DataFrame()
        
        data = []
        for signal in signals:
            data.append({
                'Ticker': signal.ticker,
                'Signal': signal.signal,
                'Conviction': f"{signal.conviction:.2f}",
                'Confidence': f"{signal.confidence:.1%}",
                'Risk': f"{signal.risk_score:.1%}",
                'Position Size': f"{signal.position_size:.1%}",
                'Horizon': signal.expected_horizon,
                'Reasoning': signal.reasoning[:100] + "..." if len(signal.reasoning) > 100 else signal.reasoning
            })
        
        return pd.DataFrame(data)
    
    def export_signals_for_execution(self, signals: List[TradingSignal]) -> List[Dict]:
        """Export signals in format suitable for trading execution"""
        
        execution_orders = []
        
        for signal in signals:
            if signal.signal == 'HOLD':
                continue
                
            order = {
                'symbol': signal.ticker,
                'side': 'buy' if signal.signal == 'BUY' else 'sell',
                'quantity': signal.position_size,  # As percentage of portfolio
                'order_type': 'market',  # Could be 'limit' with price
                'time_in_force': 'day',
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'metadata': {
                    'signal_id': f"{signal.ticker}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    'conviction': signal.conviction,
                    'confidence': signal.confidence,
                    'risk_score': signal.risk_score,
                    'reasoning': signal.reasoning,
                    'expected_horizon': signal.expected_horizon
                }
            }
            
            execution_orders.append(order)
        
        return execution_orders