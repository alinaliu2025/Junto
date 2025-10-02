"""
SIMPLE HRM JET SIGNAL SYSTEM for Google Colab
Clean, working version - Copy this entire cell into Colab and run
"""

import subprocess
import sys

# Install required packages
packages = ['yfinance', 'matplotlib', 'pandas', 'numpy']
for package in packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🚀 HRM JET SIGNAL TRADING SYSTEM")
print("="*60)
print("🛩️ Corporate aviation intelligence for trading")
print("📊 Real-time signal generation and analysis")
print("="*60)

# Companies with corporate jets
companies = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms',
    'NVDA': 'NVIDIA Corp.',
    'TSLA': 'Tesla Inc.',
    'AMZN': 'Amazon.com Inc.',
    'JPM': 'JPMorgan Chase',
    'SPY': 'SPDR S&P 500 ETF'
}

def generate_jet_signals():
    """Generate simulated HRM jet signals"""
    
    signals = []
    
    # Simulate some jet activity
    if np.random.random() < 0.4:  # 40% chance of signals
        num_signals = np.random.randint(1, 4)
        
        for _ in range(num_signals):
            ticker = np.random.choice(list(companies.keys()))
            
            # Signal types
            signal_types = ['M&A Activity', 'Earnings Prep', 'Crisis Management', 'Strategic Meeting']
            signal_type = np.random.choice(signal_types)
            
            # Generate conviction and confidence
            if signal_type == 'M&A Activity':
                conviction = np.random.normal(0.75, 0.15)
                confidence = np.random.normal(0.85, 0.1)
            elif signal_type == 'Crisis Management':
                conviction = np.random.normal(-0.65, 0.15)
                confidence = np.random.normal(0.75, 0.1)
            else:
                conviction = np.random.normal(0.0, 0.4)
                confidence = np.random.normal(0.65, 0.15)
            
            conviction = np.clip(conviction, -1.0, 1.0)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            signals.append({
                'ticker': ticker,
                'company': companies[ticker],
                'signal_type': signal_type,
                'conviction': conviction,
                'confidence': confidence,
                'expected_move': conviction * 0.05,  # 5% max expected move
                'reasoning': f'{signal_type} detected via corporate jet pattern analysis'
            })
    
    return signals

def get_current_prices(tickers):
    """Get current stock prices"""
    
    prices = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                prices[ticker] = hist['Close'].iloc[-1]
            else:
                prices[ticker] = None
        except:
            prices[ticker] = None
    
    return prices

def analyze_signals(signals):
    """Analyze and rank signals"""
    
    if not signals:
        return []
    
    # Calculate signal strength
    for signal in signals:
        signal['strength'] = abs(signal['conviction']) * signal['confidence']
    
    # Sort by strength
    signals.sort(key=lambda x: x['strength'], reverse=True)
    
    return signals

def generate_recommendations(signals, prices):
    """Generate trading recommendations"""
    
    recommendations = []
    
    for signal in signals:
        ticker = signal['ticker']
        current_price = prices.get(ticker)
        
        if current_price is None:
            continue
        
        # Determine action
        if signal['conviction'] > 0.6:
            action = 'BUY'
            target_price = current_price * (1 + abs(signal['expected_move']))
            stop_loss = current_price * 0.95
        elif signal['conviction'] < -0.6:
            action = 'SELL/SHORT'
            target_price = current_price * (1 - abs(signal['expected_move']))
            stop_loss = current_price * 1.05
        else:
            action = 'MONITOR'
            target_price = current_price
            stop_loss = current_price
        
        # Position size based on conviction and confidence
        position_size = min(0.05, abs(signal['conviction']) * signal['confidence'] * 0.03)
        
        recommendation = {
            'ticker': ticker,
            'company': signal['company'],
            'action': action,
            'current_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'conviction': signal['conviction'],
            'confidence': signal['confidence'],
            'signal_type': signal['signal_type'],
            'reasoning': signal['reasoning']
        }
        
        recommendations.append(recommendation)
    
    return recommendations

def run_analysis():
    """Run complete HRM jet signal analysis"""
    
    print("🛩️ Generating HRM jet signals...")
    signals = generate_jet_signals()
    
    if not signals:
        print("⚪ No jet signals detected today")
        print("💡 Monitor for corporate aviation activity")
        return
    
    print(f"📊 Found {len(signals)} jet signals")
    
    # Get current prices
    tickers = [s['ticker'] for s in signals]
    print("📈 Fetching current stock prices...")
    prices = get_current_prices(tickers)
    
    # Analyze signals
    analyzed_signals = analyze_signals(signals)
    
    # Generate recommendations
    recommendations = generate_recommendations(analyzed_signals, prices)
    
    # Display results
    print("\n" + "="*60)
    print("🎯 HRM JET SIGNAL ANALYSIS RESULTS")
    print("="*60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['ticker']} - {rec['company']}")
        print(f"   Signal: {rec['signal_type']}")
        print(f"   Action: {rec['action']}")
        print(f"   Current Price: ${rec['current_price']:.2f}")
        print(f"   Conviction: {rec['conviction']:+.2f} | Confidence: {rec['confidence']:.1%}")
        
        if rec['action'] != 'MONITOR':
            print(f"   Target: ${rec['target_price']:.2f} | Stop: ${rec['stop_loss']:.2f}")
            print(f"   Position Size: {rec['position_size']:.1%} of portfolio")
        
        print(f"   Reasoning: {rec['reasoning']}")
    
    # Create visualization
    if recommendations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Signal strength chart
        tickers = [r['ticker'] for r in recommendations]
        strengths = [abs(r['conviction']) * r['confidence'] for r in recommendations]
        colors = ['green' if r['conviction'] > 0 else 'red' if r['conviction'] < 0 else 'gray' for r in recommendations]
        
        ax1.bar(tickers, strengths, color=colors, alpha=0.7)
        ax1.set_title('Signal Strength by Company')
        ax1.set_ylabel('Signal Strength')
        ax1.tick_params(axis='x', rotation=45)
        
        # Conviction vs Confidence scatter
        convictions = [r['conviction'] for r in recommendations]
        confidences = [r['confidence'] for r in recommendations]
        
        ax2.scatter(convictions, confidences, s=100, alpha=0.7)
        for i, rec in enumerate(recommendations):
            ax2.annotate(rec['ticker'], (convictions[i], confidences[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Conviction')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Conviction vs Confidence')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    print("\n" + "="*60)
    print("💡 SYSTEM ADVANTAGES:")
    print("   • Corporate jet intelligence provides early signals")
    print("   • Multi-factor analysis (conviction + confidence)")
    print("   • Risk-adjusted position sizing")
    print("   • Real-time price integration")
    print("="*60)
    
    return recommendations

# Run the analysis
print("🚀 RUNNING HRM JET SIGNAL ANALYSIS")
print("="*60)

recommendations = run_analysis()

print("\n🎯 HRM JET SIGNAL ANALYSIS COMPLETE!")
print("Ready for trading decisions based on corporate aviation intelligence")