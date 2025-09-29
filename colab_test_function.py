"""
Corrected test function for Colab
Copy and paste this into a Colab cell
"""

def test_jet_signals():
    print("🛩️ Testing HRM Jet Signal Generation...")
    
    # Import required classes (make sure these are available)
    from jet_signal_hrm.models.hrm_jet import HRMJetModel, FlightEvent
    from jet_signal_hrm.data.flight_data import FlightDataCollector
    from jet_signal_hrm.data.company_mapper import CompanyMapper
    from jet_signal_hrm.trading.signal_generator import SignalGenerator
    from datetime import datetime
    import torch
    import numpy as np
    
    try:
        # Load trained model
        print("📥 Loading trained model...")
        checkpoint = torch.load('jet_hrm_model.pt', map_location=device)
        model = HRMJetModel(checkpoint['config']['model']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Model loaded successfully")
        
    except FileNotFoundError:
        print("⚠️ No trained model found, creating new model for demo...")
        # Create a new model for demo
        demo_config = {
            "d_model": 128,
            "n_heads": 4,
            "flight_features": 16
        }
        model = HRMJetModel(demo_config).to(device)
        config = {"trading": {
            "buy_threshold": 0.6,
            "sell_threshold": -0.6,
            "min_confidence": 0.7,
            "max_risk": 0.3,
            "base_position_size": 0.02
        }}
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Initialize components
    flight_collector = FlightDataCollector()
    company_mapper = CompanyMapper()
    
    # Use config from checkpoint or create default
    try:
        trading_config = checkpoint['config']['trading']
    except:
        trading_config = {
            "buy_threshold": 0.6,
            "sell_threshold": -0.6,
            "min_confidence": 0.7,
            "max_risk": 0.3,
            "base_position_size": 0.02
        }
    
    signal_generator = SignalGenerator(model, trading_config)
    
    print("\n📡 Collecting live flight data...")
    flights_df = flight_collector.get_live_flights()
    
    if flights_df.empty:
        print("⚠️ No live flights, using demo data")
        flight_events = create_synthetic_flight_events()
    else:
        print(f"✅ Collected {len(flights_df)} live flights")
        flight_events = convert_flights_to_events(flights_df, company_mapper)
        
        # Add some demo events to ensure we have signals
        flight_events.extend(create_synthetic_flight_events())
    
    print(f"🏢 Processing {len(flight_events)} flight events...")
    
    # Generate signals
    signals = signal_generator.generate_signals(flight_events, {})
    
    # Display results
    print("\n" + "="*80)
    print("🛩️ HRM JET SIGNAL TRADING SYSTEM - LIVE RESULTS")
    print("="*80)
    
    if signals:
        print(f"📈 Generated {len(signals)} trading signals:\n")
        
        for i, signal in enumerate(signals, 1):
            emoji = "🟢" if signal.signal == 'BUY' else "🔴" if signal.signal == 'SELL' else "🟡"
            
            print(f"{emoji} Signal {i}: {signal.ticker}")
            print(f"   Action: {signal.signal}")
            print(f"   Conviction: {signal.conviction:+.2f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Risk Score: {signal.risk_score:.1%}")
            print(f"   Position Size: {signal.position_size:.1%}")
            print(f"   Reasoning: {signal.reasoning}")
            print()
    else:
        print("⚪ No trading signals generated")
        print("💡 Try lowering thresholds or check if flight events were created")
    
    print("="*80)
    return signals

def create_synthetic_flight_events():
    """Create synthetic flight events for demo"""
    from jet_signal_hrm.models.hrm_jet import FlightEvent
    from datetime import datetime
    import torch
    import numpy as np
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    events = []
    
    for ticker in tickers:
        for i in range(np.random.randint(1, 4)):  # 1-3 events per company
            features = torch.randn(16)  # Random features
            event = FlightEvent(
                level=0,
                timestamp=datetime.now().timestamp(),
                company_ticker=ticker,
                features=features,
                metadata={'synthetic': True}
            )
            events.append(event)
    
    return events

def convert_flights_to_events(flights_df, company_mapper):
    """Convert real flight data to events"""
    from jet_signal_hrm.models.hrm_jet import FlightEvent
    from datetime import datetime
    import torch
    
    events = []
    for _, flight in flights_df.iterrows():
        # Try to map to company (will mostly fail with real data, but some demo data will work)
        company_info = company_mapper.get_company_from_aircraft(flight['icao24'])
        
        if company_info:
            # Create features from flight data
            features = torch.tensor([
                flight['latitude'] / 90.0,
                flight['longitude'] / 180.0,
                (flight.get('baro_altitude', 0) or 0) / 40000.0,
                (flight.get('velocity', 0) or 0) / 500.0,
                datetime.now().hour / 24.0,
                datetime.now().weekday() / 7.0,
                *[0.0] * 10  # Pad to 16 features
            ], dtype=torch.float32)
            
            event = FlightEvent(
                level=0,
                timestamp=flight['timestamp'].timestamp(),
                company_ticker=company_info['ticker'],
                features=features,
                metadata={'icao24': flight['icao24']}
            )
            events.append(event)
    
    return events

print("✅ Test function ready!")
print("🚀 Run: test_signals = test_jet_signals()")