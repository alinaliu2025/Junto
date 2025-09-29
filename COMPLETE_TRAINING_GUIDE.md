# Complete HRM Jet Signal Trading System Setup Guide

**Transform corporate aviation data into profitable trading signals with AI**

## 🎯 Overview

This guide will walk you through setting up a complete Hierarchical Reasoning Model (HRM) system that analyzes corporate jet movements to generate trading signals. The system monitors real-time flight data, maps aircraft to public companies, and uses multi-level reasoning to detect patterns that precede market-moving events.

## 📋 Prerequisites

### System Requirements
- **Python 3.8+** with pip
- **16GB+ RAM** (for processing flight data graphs)
- **Internet connection** for real-time ADS-B data
- **Optional**: GPU for faster model training

### API Access (Free Tiers Available)
- **OpenSky Network**: Free ADS-B flight data
- **Yahoo Finance**: Free stock price data
- **Optional**: FlightAware, Alpha Vantage for enhanced data

## 🚀 Step 1: Environment Setup (15 minutes)

### 1.1 Clone and Setup Project
```bash
# Clone the repository
git clone <your-repo-url>
cd jet-signal-hrm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1.2 Verify Installation
```bash
# Test basic imports
python -c "import torch; import pandas; import requests; print('✅ All dependencies installed')"

# Test OpenSky API connection
python -c "
from jet_signal_hrm.data.flight_data import FlightDataCollector
collector = FlightDataCollector()
flights = collector.get_live_flights()
print(f'✅ Collected {len(flights)} live flights')
"
```

## 🛩️ Step 2: Data Collection Setup (30 minutes)

### 2.1 Initialize Company-Aircraft Database
```bash
# Setup initial data collection
python train_jet_hrm.py --setup_data

# This will:
# - Create SQLite databases for aircraft registry
# - Setup company-to-aircraft mappings
# - Collect initial flight data sample
```

### 2.2 Import FAA Aircraft Registry (Optional but Recommended)
```bash
# Download FAA aircraft database
wget https://registry.faa.gov/database/ReleasableAircraft.zip
unzip ReleasableAircraft.zip

# Import into system (this may take 10-15 minutes)
python -c "
from jet_signal_hrm.data.company_mapper import CompanyMapper
mapper = CompanyMapper()
count = mapper.bulk_import_faa_data('MASTER.txt')
print(f'Imported {count} aircraft registrations')
"
```

### 2.3 Configure Company Mappings
Create a file `company_mappings.json` with your target companies:

```json
{
  "AAPL": "APPLE,CUPERTINO,COOK",
  "MSFT": "MICROSOFT,REDMOND,NADELLA", 
  "GOOGL": "GOOGLE,ALPHABET,MOUNTAIN VIEW,PICHAI",
  "TSLA": "TESLA,SPACEX,MUSK,PALO ALTO",
  "AMZN": "AMAZON,BEZOS,SEATTLE",
  "META": "META,FACEBOOK,MENLO PARK,ZUCKERBERG",
  "NVDA": "NVIDIA,SANTA CLARA,HUANG",
  "JPM": "JPMORGAN,CHASE,DIMON",
  "BAC": "BANK OF AMERICA,CHARLOTTE",
  "WMT": "WALMART,BENTONVILLE,MCMILLON"
}
```

Then import the mappings:
```bash
python -c "
import json
from jet_signal_hrm.data.company_mapper import CompanyMapper

with open('company_mappings.json', 'r') as f:
    mappings = json.load(f)

mapper = CompanyMapper()
count = mapper.auto_map_companies(mappings)
print(f'Auto-mapped {count} aircraft to companies')
"
```

## 🤖 Step 3: Model Training (2-4 hours)

### 3.1 Configure Training Parameters
Create `jet_config.json`:

```json
{
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
  "training": {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 100,
    "patience": 10
  }
}
```

### 3.2 Start Training
```bash
# Start model training
python train_jet_hrm.py \
  --config jet_config.json \
  --save_dir checkpoints/jet_hrm \
  --device auto

# Training will:
# - Collect live flight data every epoch
# - Map flights to companies
# - Train HRM on flight patterns
# - Save checkpoints every 10 epochs
```

### 3.3 Monitor Training Progress
The training script will output:
```
Epoch 0: Loss = 0.8234, Companies = 5
Epoch 10: Loss = 0.6891, Companies = 7
Saved checkpoint: checkpoints/jet_hrm/checkpoint_epoch_10.pt
Epoch 20: Loss = 0.5432, Companies = 8
...
Training complete! Final model saved to checkpoints/jet_hrm/final_model.pt
```

## 📈 Step 4: Real-Time Signal Generation (15 minutes)

### 4.1 Test Single Analysis
```bash
# Run single flight analysis
python jet_inference.py \
  --model checkpoints/jet_hrm/final_model.pt \
  --config jet_config.json

# Expected output:
# 🛩️ HRM JET SIGNAL TRADING SYSTEM
# =====================================
# 📊 Analysis Time: 2024-01-15T14:30:00
# ✈️  Total Flights: 1,247
# 🏢 Company Flights: 23
# 📈 Signals Generated: 3
# 
# 💰 TRADING SIGNALS
# ------------------
# 🟢 AAPL   | BUY  | Conv: +0.73 | Conf: 85% | Risk: 12% | Size: 1.8%
#    Reasoning: Executive travel analysis (2 trips); unusual destination patterns...
#    Horizon: 1W | Stop: $185.50 | Target: $195.20
```

### 4.2 Start Continuous Monitoring
```bash
# Run continuous monitoring (updates every 5 minutes)
python jet_inference.py \
  --model checkpoints/jet_hrm/final_model.pt \
  --config jet_config.json \
  --continuous \
  --interval 5

# This will:
# - Monitor flights every 5 minutes
# - Generate trading signals automatically
# - Save results with timestamps
# - Print formatted analysis to console
```

### 4.3 Geographic Filtering (Optional)
```bash
# Monitor only US flights (approximate bounding box)
python jet_inference.py \
  --model checkpoints/jet_hrm/final_model.pt \
  --continuous \
  --bbox "25,-125,50,-65"  # lat_min,lon_min,lat_max,lon_max
```

## 🔧 Step 5: Advanced Configuration

### 5.1 Adjust Signal Sensitivity
Edit your config file to tune signal generation:

```json
{
  "trading": {
    "buy_threshold": 0.5,      // Lower = more buy signals
    "sell_threshold": -0.5,    // Higher = more sell signals  
    "min_confidence": 0.6,     // Lower = more signals overall
    "max_risk": 0.4,          // Higher = accept riskier signals
    "base_position_size": 0.03 // Larger position sizes
  }
}
```

### 5.2 Add More Companies
```bash
# Add new company mappings
python -c "
from jet_signal_hrm.data.company_mapper import CompanyMapper
mapper = CompanyMapper()

# Add individual mapping
mapper.map_aircraft_to_company('N123AB', 'NFLX', 'Netflix Inc', 0.9, 'manual')

# Or bulk import more companies
new_mappings = {
    'NFLX': 'NETFLIX,LOS GATOS',
    'CRM': 'SALESFORCE,SAN FRANCISCO',
    'UBER': 'UBER,SAN FRANCISCO'
}
count = mapper.auto_map_companies(new_mappings)
print(f'Added {count} new mappings')
"
```

### 5.3 Historical Data Collection
```bash
# Collect historical flight data for better training
python -c "
from jet_signal_hrm.data.flight_data import FlightDataCollector
from datetime import datetime, timedelta
import time

collector = FlightDataCollector()

# Collect data for past 7 days (be mindful of API limits)
for i in range(7):
    date = datetime.now() - timedelta(days=i)
    print(f'Collecting data for {date.strftime(\"%Y-%m-%d\")}')
    
    # In practice, you'd use historical API endpoints
    # This is a simplified example
    flights = collector.get_live_flights()
    print(f'Collected {len(flights)} flights')
    
    time.sleep(60)  # Respect API limits
"
```

## 📊 Step 6: Performance Monitoring

### 6.1 Signal Quality Analysis
```bash
# Analyze signal performance over time
python -c "
import json
import glob
from datetime import datetime

# Load all signal files
signal_files = glob.glob('jet_signals_*.json')
all_signals = []

for file in signal_files:
    with open(file, 'r') as f:
        data = json.load(f)
        if 'signals' in data:
            all_signals.extend(data['signals'])

print(f'Total signals generated: {len(all_signals)}')

# Analyze by signal type
buy_signals = [s for s in all_signals if s['signal'] == 'BUY']
sell_signals = [s for s in all_signals if s['signal'] == 'SELL']

print(f'Buy signals: {len(buy_signals)}')
print(f'Sell signals: {len(sell_signals)}')

# Average confidence by signal type
if buy_signals:
    avg_buy_conf = sum(s['confidence'] for s in buy_signals) / len(buy_signals)
    print(f'Average buy confidence: {avg_buy_conf:.1%}')

if sell_signals:
    avg_sell_conf = sum(s['confidence'] for s in sell_signals) / len(sell_signals)
    print(f'Average sell confidence: {avg_sell_conf:.1%}')
"
```

### 6.2 Company Coverage Analysis
```bash
# Check which companies are generating signals
python -c "
from jet_signal_hrm.data.company_mapper import CompanyMapper
import sqlite3

mapper = CompanyMapper()
conn = sqlite3.connect(mapper.db_path)

# Get companies with aircraft mappings
cursor = conn.execute('''
    SELECT ticker_symbol, COUNT(*) as aircraft_count,
           AVG(confidence_score) as avg_confidence
    FROM company_mappings 
    GROUP BY ticker_symbol
    ORDER BY aircraft_count DESC
''')

print('Company Coverage:')
print('Ticker | Aircraft | Avg Confidence')
print('-' * 35)

for row in cursor.fetchall():
    ticker, count, confidence = row
    print(f'{ticker:6} | {count:8} | {confidence:.2f}')

conn.close()
"
```

## 🚨 Troubleshooting

### Common Issues and Solutions

**1. No flight data collected**
```bash
# Check internet connection and API access
python -c "
import requests
response = requests.get('https://opensky-network.org/api/states/all', timeout=10)
print(f'API Status: {response.status_code}')
print(f'Response length: {len(response.text)}')
"
```

**2. No company mappings found**
```bash
# Verify database has mappings
python -c "
from jet_signal_hrm.data.company_mapper import CompanyMapper
mapper = CompanyMapper()
aircraft = mapper.get_company_aircraft('AAPL')
print(f'AAPL aircraft: {len(aircraft)}')
if aircraft:
    print('Sample aircraft:', aircraft[0])
"
```

**3. Model training fails**
```bash
# Check CUDA availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
print(f'PyTorch version: {torch.__version__}')
"
```

**4. Low signal generation**
```bash
# Lower thresholds for more signals
python jet_inference.py \
  --model checkpoints/jet_hrm/final_model.pt \
  --config <(echo '{
    "trading": {
      "buy_threshold": 0.3,
      "sell_threshold": -0.3,
      "min_confidence": 0.5,
      "max_risk": 0.5
    }
  }')
```

## 🎯 Next Steps

### Production Deployment
1. **Database Upgrade**: Move from SQLite to PostgreSQL for better performance
2. **Real-time Pipeline**: Setup continuous data ingestion with Apache Kafka
3. **API Integration**: Connect to broker APIs for automated trading
4. **Monitoring**: Add Grafana dashboards for system monitoring
5. **Backtesting**: Implement historical performance analysis

### Model Improvements
1. **Feature Engineering**: Add weather, airport congestion, fuel prices
2. **Multi-timeframe**: Analyze patterns across different time horizons
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Alternative Data**: Integrate satellite imagery, social media sentiment

### Risk Management
1. **Position Sizing**: Implement Kelly Criterion for optimal sizing
2. **Portfolio Limits**: Set maximum exposure per company/sector
3. **Drawdown Controls**: Automatic system shutdown on large losses
4. **Compliance**: Ensure regulatory compliance for algorithmic trading

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review log files in the project directory
3. Ensure all dependencies are correctly installed
4. Verify API access and internet connectivity

## ⚠️ Important Disclaimers

- **Not Financial Advice**: This system is for educational/research purposes
- **Risk Warning**: Trading involves substantial risk of loss
- **Regulatory Compliance**: Check local laws regarding algorithmic trading
- **Data Usage**: Respect API terms of service and rate limits
- **Privacy**: Be mindful of aviation privacy concerns

---

**🎉 Congratulations!** You now have a complete HRM Jet Signal Trading System running. The system will continuously monitor corporate aviation activity and generate trading signals based on unusual flight patterns. Remember to start with paper trading to validate performance before risking real capital.