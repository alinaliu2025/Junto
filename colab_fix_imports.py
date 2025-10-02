"""
Quick fix for Colab import issues
Run this cell before the test_jet_signals function
"""

# Add the current directory to Python path
import sys
import os
sys.path.append('/content/jet-signal-hrm')

# Import all required classes
from jet_signal_hrm.models.hrm_jet import HRMJetModel, FlightEvent
from jet_signal_hrm.data.flight_data import FlightDataCollector
from jet_signal_hrm.data.company_mapper import CompanyMapper
from jet_signal_hrm.trading.signal_generator import SignalGenerator

print("✅ All imports successful!")
print("📦 Available classes:")
print("   - HRMJetModel")
print("   - FlightEvent") 
print("   - FlightDataCollector")
print("   - CompanyMapper")
print("   - SignalGenerator")

# Test basic functionality
try:
    # Test model creation
    test_config = {
        "d_model": 128,
        "n_heads": 4,
        "flight_features": 16
    }
    test_model = HRMJetModel(test_config)
    print("✅ HRMJetModel creation successful")
    
    # Test data collector
    collector = FlightDataCollector()
    print("✅ FlightDataCollector creation successful")
    
    # Test company mapper
    mapper = CompanyMapper()
    print("✅ CompanyMapper creation successful")
    
    print("\n🎯 Ready to run test_jet_signals()!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Make sure all files are uploaded correctly")