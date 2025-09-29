"""
Main training script for HRM Jet Signal Trading System
"""

import torch
import argparse
import json
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from jet_signal_hrm.models.hrm_jet import HRMJetModel, FlightEvent
from jet_signal_hrm.data.flight_data import FlightDataCollector
from jet_signal_hrm.data.company_mapper import CompanyMapper
from jet_signal_hrm.trading.signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train HRM Jet Signal Trading System")
    parser.add_argument("--config", type=str, default="jet_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data/flight_data",
                       help="Path to flight data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints/jet_hrm",
                       help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--setup_data", action="store_true",
                       help="Setup initial data collection and company mapping")
    
    args = parser.parse_args()
    
    # Load or create configuration
    config = load_or_create_config(args.config)
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Setup data collection if requested
    if args.setup_data:
        setup_data_collection(config)
        return
    
    # Initialize components
    flight_collector = FlightDataCollector()
    company_mapper = CompanyMapper()
    
    # Create model
    model = HRMJetModel(config['model'])
    model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create signal generator
    signal_generator = SignalGenerator(model, config['trading'])
    
    # For now, create a simple training loop
    # In production, this would be more sophisticated
    train_model(model, flight_collector, company_mapper, signal_generator, config, device, args.save_dir)

def load_or_create_config(config_path: str) -> dict:
    """Load configuration or create default"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = create_default_config()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created default configuration at {config_path}")
    
    return config

def create_default_config() -> dict:
    """Create default configuration for jet signal system"""
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
            "max_position_size": 0.10,
            "default_stop_loss": 0.02,
            "default_take_profit": 0.05
        },
        "data": {
            "flight_update_interval": 300,  # 5 minutes
            "historical_days": 30,
            "min_flights_per_company": 5
        },
        "training": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
            "profit_weight": 0.7,
            "accuracy_weight": 0.3
        }
    }

def setup_data_collection(config: dict):
    """Setup initial data collection and company mapping"""
    logger.info("Setting up data collection...")
    
    # Initialize components
    flight_collector = FlightDataCollector()
    company_mapper = CompanyMapper()
    
    # Example company mappings (in production, would be more comprehensive)
    example_mappings = {
        "AAPL": "APPLE,CUPERTINO",
        "MSFT": "MICROSOFT,REDMOND",
        "GOOGL": "GOOGLE,ALPHABET,MOUNTAIN VIEW",
        "TSLA": "TESLA,SPACEX,MUSK",
        "AMZN": "AMAZON,BEZOS",
        "META": "META,FACEBOOK,MENLO PARK",
        "NVDA": "NVIDIA,SANTA CLARA",
        "JPM": "JPMORGAN,CHASE",
        "BAC": "BANK OF AMERICA",
        "WMT": "WALMART,BENTONVILLE"
    }
    
    # Auto-map companies
    mapped_count = company_mapper.auto_map_companies(example_mappings)
    logger.info(f"Auto-mapped {mapped_count} aircraft to companies")
    
    # Collect initial flight data
    logger.info("Collecting initial flight data...")
    flights_df = flight_collector.get_live_flights()
    logger.info(f"Collected {len(flights_df)} live flights")
    
    logger.info("Data collection setup complete!")

def train_model(model, flight_collector, company_mapper, signal_generator, 
               config, device, save_dir):
    """Train the HRM jet model"""
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop (simplified for demo)
    logger.info("Starting training...")
    
    for epoch in range(config['training']['epochs']):
        try:
            # Collect fresh flight data
            flights_df = flight_collector.get_live_flights()
            
            if flights_df.empty:
                logger.warning(f"No flight data collected for epoch {epoch}")
                continue
            
            # Convert to flight events (simplified)
            flight_events = create_mock_flight_events(flights_df, company_mapper)
            
            if not flight_events:
                logger.warning(f"No company-mapped flights for epoch {epoch}")
                continue
            
            # Forward pass
            model.train()
            optimizer.zero_grad()
            
            # Group events by company for batch processing
            company_events = {}
            for event in flight_events:
                if event.company_ticker not in company_events:
                    company_events[event.company_ticker] = []
                company_events[event.company_ticker].append(event)
            
            total_loss = 0
            batch_count = 0
            
            for ticker, events in company_events.items():
                try:
                    # Forward pass
                    outputs = model(events)
                    
                    # Create dummy targets (in production, would use real market data)
                    target_conviction = torch.randn(1, 3).softmax(dim=-1)
                    target_confidence = torch.rand(1, 1)
                    
                    # Calculate loss
                    conviction_loss = torch.nn.functional.cross_entropy(
                        outputs['conviction_logits'], target_conviction.argmax(dim=-1)
                    )
                    confidence_loss = torch.nn.functional.mse_loss(
                        outputs['confidence'], target_confidence
                    )
                    
                    loss = conviction_loss + confidence_loss
                    loss.backward()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                    continue
            
            if batch_count > 0:
                optimizer.step()
                avg_loss = total_loss / batch_count
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Companies = {batch_count}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config
                }, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {e}")
            continue
    
    # Save final model
    final_path = Path(save_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    logger.info(f"Training complete! Final model saved to {final_path}")

def create_mock_flight_events(flights_df, company_mapper):
    """Create mock flight events for training (simplified)"""
    events = []
    
    for _, flight in flights_df.iterrows():
        # Try to map to company
        company_info = company_mapper.get_company_from_aircraft(flight['icao24'])
        
        if not company_info:
            continue
        
        # Create mock features (in production, would be real flight features)
        features = torch.randn(32)  # Mock flight features
        
        # Create flight event
        event = FlightEvent(
            level=0,  # Single flight leg
            timestamp=flight['timestamp'].timestamp(),
            company_ticker=company_info['ticker'],
            features=features,
            metadata={
                'icao24': flight['icao24'],
                'callsign': flight.get('callsign', ''),
                'latitude': flight['latitude'],
                'longitude': flight['longitude']
            }
        )
        
        events.append(event)
    
    return events

if __name__ == "__main__":
    main()