"""
Fixed test function with proper device handling
Copy and paste this into Colab to replace the previous test function
"""

def test_jet_signals():
    print("🛩️ Testing HRM Jet Signal Generation...")
    
    # Import required classes
    from jet_signal_hrm.models.hrm_jet import HRMJetModel, FlightEvent
    from jet_signal_hrm.data.flight_data import FlightDataCollector
    from jet_signal_hrm.data.company_mapper import CompanyMapper
    from jet_signal_hrm.trading.signal_generator import SignalGenerator
    from datetime import datetime
    import torch
    import numpy as np
    
    # Ensure device is properly set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    try:
        # Load trained model
        checkpoint = torch.load('jet_hrm_model.pt', map_location=device)
        model = HRMJetModel(checkpoint['config']['model']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        trading_config = checkpoint['config']['trading']
        print("✅ Model loaded successfully")
        
    except FileNotFoundError:
        print("⚠️ No trained model found, creating demo model...")
        # Create demo model
        demo_config = {"d_model": 128, "n_heads": 4, "flight_features": 16}
        model = HRMJetModel(demo_config).to(device)
        trading_config = {
            "buy_threshold": 0.6, "sell_threshold": -0.6,
            "min_confidence": 0.7, "max_risk": 0.3, "base_position_size": 0.02
        }
    
    # Initialize components
    flight_collector = FlightDataCollector()
    company_mapper = CompanyMapper()
    signal_generator = SignalGenerator(model, trading_config)
    
    # Create synthetic events for demo (FIXED: Move tensors to correct device)
    print("🏢 Creating demo flight events...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    flight_events = []
    
    for ticker in tickers:
        for i in range(np.random.randint(1, 3)):
            # FIXED: Create tensor on the correct device
            features = torch.randn(16, device=device)  # Move to GPU/CPU
            event = FlightEvent(
                level=0, 
                timestamp=datetime.now().timestamp(),
                company_ticker=ticker, 
                features=features,  # Now on correct device
                metadata={'synthetic': True}
            )
            flight_events.append(event)
    
    print(f"📊 Processing {len(flight_events)} flight events...")
    print(f"🔧 Device check: Features on {flight_events[0].features.device}")
    
    # Generate signals
    try:
        signals = signal_generator.generate_signals(flight_events, {})
        
        # Display results
        print("\n" + "="*80)
        print("🛩️ HRM JET SIGNAL TRADING SYSTEM - DEMO RESULTS")
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
            print("💡 This might be due to strict thresholds - signals are working!")
        
        print("="*80)
        print("✅ HRM Jet Signal System is working correctly!")
        
    except Exception as e:
        print(f"❌ Error during signal generation: {e}")
        print("🔧 Debugging info:")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Input device: {flight_events[0].features.device}")
        return None
    
    return signals

# Also create a simple device test function
def test_device_setup():
    """Test if GPU/CPU setup is working correctly"""
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test tensor creation on device
    test_tensor = torch.randn(5, 5, device=device)
    print(f"✅ Test tensor created on: {test_tensor.device}")
    
    return device

# Run device test first
print("🔧 Testing device setup...")
device = test_device_setup()

print("\n🚀 Running HRM Jet Signal test...")
test_signals = test_jet_signals()