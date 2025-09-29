"""
Google Colab setup script for HRM Jet Signal Trading System
Run this in a Colab cell to install everything properly
"""

# Colab-specific installation
def setup_colab_environment():
    import subprocess
    import sys
    
    print("🚀 Setting up HRM Jet Signal Trading System in Google Colab...")
    
    # Install packages one by one to handle errors gracefully
    packages = [
        "torch",
        "pandas", 
        "numpy",
        "matplotlib",
        "requests",
        "tqdm",
        "scikit-learn",
        "yfinance",
        "plotly"
    ]
    
    for package in packages:
        try:
            print(f"📥 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"✅ {package} installed")
        except Exception as e:
            print(f"⚠️ {package} failed: {e}")
    
    # Test built-in packages
    print("\n🔍 Testing built-in packages...")
    
    try:
        import sqlite3
        print(f"✅ sqlite3 available (version {sqlite3.sqlite_version})")
    except ImportError as e:
        print(f"❌ sqlite3 error: {e}")
    
    try:
        import json
        print("✅ json available")
    except ImportError as e:
        print(f"❌ json error: {e}")
    
    # Test core imports
    print("\n🧪 Testing core imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ No CUDA (CPU only)")
    except ImportError as e:
        print(f"❌ PyTorch error: {e}")
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas error: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy error: {e}")
    
    print("\n✅ Colab environment setup complete!")
    return True

# Run setup
if __name__ == "__main__":
    setup_colab_environment()