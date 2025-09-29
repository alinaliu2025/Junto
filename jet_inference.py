"""
Real-time inference for HRM Jet Signal Trading System
"""

import torch
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta
import logging
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from jet_signal_hrm.models.hrm_jet import HRMJetModel, FlightEvent
from jet_signal_hrm.data.flight_data import FlightDataCollector
from jet_signal_hrm.data.company_mapper import CompanyMapper
from jet_signal_hrm.trading.signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JetSignalInference:
    """
    Real-time inference system for jet signal trading
    """
    
    def __init__(self, model_path: str, config_path: str = None, device: str = "auto"):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._create_default_config()
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize components
        self.flight_collector = FlightDataCollector()
        self.company_mapper = CompanyMapper()
        self.signal_generator = SignalGenerator(self.model, self.config.get('trading', {}))
        
        logger.info(f"Jet Signal HRM loaded on {self.device}")
    
    def _create_default_config(self) -> dict:
        """Create default configuration"""
        return {
            "model": {
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "max_reasoning_steps": 10,
                "flight_features": 32,
                "context_features": 64,
                "company_features": 16
            },
            "trading": {
                "buy_threshold": 0.6,
                "sell_threshold": -0.6,
                "min_confidence": 0.7,
                "max_risk": 0.3,
                "base_position_size": 0.02,
                "max_position_size": 0.10
            },
            "data": {
                "flight_update_interval": 300,  # 5 minutes
                "min_flights_per_company": 3
            }
        }
    
    def _load_model(self, model_path: str) -> HRMJetModel:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model_config = checkpoint.get('config', {}).get('model', self.config['model'])
        model = HRMJetModel(model_config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def analyze_current_flights(self, bbox: tuple = None) -> dict:
        """
        Analyze current flight patterns and generate trading signals
        
        Args:
            bbox: Geographic bounding box (lat_min, lon_min, lat_max, lon_max)
            
        Returns:
            Dictionary with analysis results and trading signals
        """
        logger.info("Collecting current flight data...")
        
        # Collect live flight data
        flights_df = self.flight_collector.get_live_flights(bbox)
        
        if flights_df.empty:
            logger.warning("No flight data collected")
            return {"error": "No flight data available"}
        
        logger.info(f"Collected {len(flights_df)} live flights")
        
        # Map flights to companies
        company_flights = self._map_flights_to_companies(flights_df)
        
        if not company_flights:
            logger.warning("No flights mapped to tracked companies")
            return {"error": "No company-mapped flights found"}
        
        logger.info(f"Mapped flights for {len(company_flights)} companies")
        
        # Create flight events for analysis
        flight_events = self._create_flight_events(company_flights)
        
        # Generate trading signals
        signals = self.signal_generator.generate_signals(flight_events, {})
        
        # Format results
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_flights": len(flights_df),
            "company_flights": len(company_flights),
            "signals_generated": len(signals),
            "signals": [self._signal_to_dict(signal) for signal in signals],
            "flight_summary": self._create_flight_summary(company_flights)
        }
        
        return results
    
    def _map_flights_to_companies(self, flights_df: pd.DataFrame) -> dict:
        """Map flights to companies"""
        company_flights = {}
        
        for _, flight in flights_df.iterrows():
            # Try to map using ICAO24
            company_info = self.company_mapper.get_company_from_aircraft(flight['icao24'])
            
            if company_info:
                ticker = company_info['ticker']
                if ticker not in company_flights:
                    company_flights[ticker] = []
                
                company_flights[ticker].append({
                    'flight': flight,
                    'company_info': company_info
                })
        
        # Filter companies with minimum flight count
        min_flights = self.config['data']['min_flights_per_company']
        filtered_companies = {
            ticker: flights for ticker, flights in company_flights.items()
            if len(flights) >= min_flights
        }
        
        return filtered_companies
    
    def _create_flight_events(self, company_flights: dict) -> list:
        """Create flight events for HRM analysis"""
        events = []
        
        for ticker, flights in company_flights.items():
            for flight_data in flights:
                flight = flight_data['flight']
                
                # Create features (simplified - in production would be more sophisticated)
                features = self._extract_flight_features(flight)
                
                # Create flight event
                event = FlightEvent(
                    level=0,  # Single flight leg for now
                    timestamp=flight['timestamp'].timestamp(),
                    company_ticker=ticker,
                    features=features,
                    metadata={
                        'icao24': flight['icao24'],
                        'callsign': flight.get('callsign', ''),
                        'latitude': flight['latitude'],
                        'longitude': flight['longitude'],
                        'altitude': flight.get('baro_altitude', 0),
                        'velocity': flight.get('velocity', 0),
                        'company_info': flight_data['company_info']
                    }
                )
                
                events.append(event)
        
        return events
    
    def _extract_flight_features(self, flight: pd.Series) -> torch.Tensor:
        """Extract features from flight data"""
        # Normalize coordinates and other features
        features = [
            flight['latitude'] / 90.0,  # Normalize latitude
            flight['longitude'] / 180.0,  # Normalize longitude
            (flight.get('baro_altitude', 0) or 0) / 40000.0,  # Normalize altitude
            (flight.get('velocity', 0) or 0) / 500.0,  # Normalize velocity
            (flight.get('true_track', 0) or 0) / 360.0,  # Normalize heading
        ]
        
        # Add time features
        now = datetime.utcnow()
        hour_of_day = now.hour / 24.0
        day_of_week = now.weekday() / 7.0
        features.extend([hour_of_day, day_of_week])
        
        # Pad to expected feature size
        while len(features) < 32:
            features.append(0.0)
        
        return torch.tensor(features[:32], dtype=torch.float32)
    
    def _signal_to_dict(self, signal) -> dict:
        """Convert signal object to dictionary"""
        return {
            "ticker": signal.ticker,
            "signal": signal.signal,
            "conviction": round(signal.conviction, 3),
            "confidence": round(signal.confidence, 3),
            "risk_score": round(signal.risk_score, 3),
            "position_size": round(signal.position_size, 4),
            "expected_horizon": signal.expected_horizon,
            "reasoning": signal.reasoning,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit
        }
    
    def _create_flight_summary(self, company_flights: dict) -> dict:
        """Create summary of flight activity"""
        summary = {}
        
        for ticker, flights in company_flights.items():
            flight_count = len(flights)
            avg_altitude = np.mean([
                f['flight'].get('baro_altitude', 0) or 0 
                for f in flights
            ])
            
            summary[ticker] = {
                "flight_count": flight_count,
                "avg_altitude": round(avg_altitude, 0),
                "company_name": flights[0]['company_info']['company_name']
            }
        
        return summary
    
    def run_continuous_monitoring(self, interval_minutes: int = 5, bbox: tuple = None):
        """
        Run continuous flight monitoring and signal generation
        
        Args:
            interval_minutes: Update interval in minutes
            bbox: Geographic bounding box for flight collection
        """
        logger.info(f"Starting continuous monitoring (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                # Analyze current flights
                results = self.analyze_current_flights(bbox)
                
                # Print results
                self.print_analysis(results)
                
                # Save results with timestamp
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_file = f"jet_signals_{timestamp}.json"
                
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Results saved to {output_file}")
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def print_analysis(self, results: dict):
        """Print formatted analysis results"""
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return
        
        print("\n" + "="*80)
        print("🛩️  HRM JET SIGNAL TRADING SYSTEM")
        print("="*80)
        
        print(f"📊 Analysis Time: {results['timestamp']}")
        print(f"✈️  Total Flights: {results['total_flights']}")
        print(f"🏢 Company Flights: {results['company_flights']}")
        print(f"📈 Signals Generated: {results['signals_generated']}")
        
        # Flight summary
        if results.get('flight_summary'):
            print(f"\n🛩️  FLIGHT ACTIVITY SUMMARY")
            print("-" * 50)
            for ticker, info in results['flight_summary'].items():
                print(f"{ticker:6} | {info['flight_count']:2} flights | {info['company_name']}")
        
        # Trading signals
        if results['signals']:
            print(f"\n💰 TRADING SIGNALS")
            print("-" * 80)
            
            for signal in results['signals']:
                emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
                
                print(f"{emoji} {signal['ticker']:6} | {signal['signal']:4} | "
                      f"Conv: {signal['conviction']:+.2f} | Conf: {signal['confidence']:.1%} | "
                      f"Risk: {signal['risk_score']:.1%} | Size: {signal['position_size']:.1%}")
                
                print(f"   Reasoning: {signal['reasoning'][:100]}...")
                print(f"   Horizon: {signal['expected_horizon']} | "
                      f"Stop: {signal['stop_loss']} | Target: {signal['take_profit']}")
                print()
        else:
            print(f"\n⚪ No trading signals generated")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Run HRM Jet Signal inference")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=5,
                       help="Update interval in minutes for continuous mode")
    parser.add_argument("--bbox", type=str, default=None,
                       help="Geographic bounding box: lat_min,lon_min,lat_max,lon_max")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Parse bounding box
    bbox = None
    if args.bbox:
        try:
            bbox = tuple(map(float, args.bbox.split(',')))
            if len(bbox) != 4:
                raise ValueError("Bounding box must have 4 values")
        except ValueError as e:
            print(f"Error parsing bounding box: {e}")
            return
    
    try:
        # Initialize inference system
        jet_inference = JetSignalInference(args.model, args.config, args.device)
        
        if args.continuous:
            # Run continuous monitoring
            jet_inference.run_continuous_monitoring(args.interval, bbox)
        else:
            # Single analysis
            results = jet_inference.analyze_current_flights(bbox)
            
            # Print results
            jet_inference.print_analysis(results)
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()