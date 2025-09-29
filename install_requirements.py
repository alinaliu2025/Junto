#!/usr/bin/env python3
"""
Installation script for HRM Jet Signal Trading System
Handles platform-specific requirements and dependencies
"""

import subprocess
import sys
import importlib
import sqlite3

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    print("🚀 Installing HRM Jet Signal Trading System Requirements...")
    print("=" * 60)
    
    # Core requirements (minimal for basic functionality)
    core_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"), 
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("requests", "requests"),
        ("tqdm", "tqdm"),
        ("scikit-learn", "sklearn")
    ]
    
    # Optional packages (for enhanced functionality)
    optional_packages = [
        ("yfinance", "yfinance"),
        ("plotly", "plotly"),
        ("geopy", "geopy"),
        ("python-dateutil", "dateutil"),
        ("pytz", "pytz")
    ]
    
    print("📦 Installing core packages...")
    failed_core = []
    
    for package, import_name in core_packages:
        if check_package(import_name):
            print(f"✅ {package} - already installed")
        else:
            print(f"📥 Installing {package}...")
            if install_package(package):
                print(f"✅ {package} - installed successfully")
            else:
                print(f"❌ {package} - installation failed")
                failed_core.append(package)
    
    print(f"\n📦 Installing optional packages...")
    failed_optional = []
    
    for package, import_name in optional_packages:
        if check_package(import_name):
            print(f"✅ {package} - already installed")
        else:
            print(f"📥 Installing {package}...")
            if install_package(package):
                print(f"✅ {package} - installed successfully")
            else:
                print(f"⚠️  {package} - installation failed (optional)")
                failed_optional.append(package)
    
    # Check built-in packages
    print(f"\n🔍 Checking built-in packages...")
    
    try:
        import sqlite3
        print(f"✅ sqlite3 - built-in (version {sqlite3.sqlite_version})")
    except ImportError:
        print("❌ sqlite3 - not available (this shouldn't happen)")
    
    try:
        import json
        print("✅ json - built-in")
    except ImportError:
        print("❌ json - not available (this shouldn't happen)")
    
    try:
        import datetime
        print("✅ datetime - built-in")
    except ImportError:
        print("❌ datetime - not available (this shouldn't happen)")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Installation Summary:")
    
    if not failed_core:
        print("✅ All core packages installed successfully!")
    else:
        print(f"❌ Failed core packages: {', '.join(failed_core)}")
        print("⚠️  System may not work properly without these packages")
    
    if failed_optional:
        print(f"⚠️  Failed optional packages: {', '.join(failed_optional)}")
        print("ℹ️  System will work but with reduced functionality")
    
    print("\n🎯 Next steps:")
    if not failed_core:
        print("1. Run: python train_jet_hrm.py --setup_data")
        print("2. Then: python train_jet_hrm.py --config jet_config.json")
        print("3. Finally: python jet_inference.py --model checkpoints/jet_hrm/final_model.pt")
    else:
        print("1. Fix failed core package installations")
        print("2. Re-run this script")
    
    print("\n🛩️ Ready to start jet signal trading!")

if __name__ == "__main__":
    main()