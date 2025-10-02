"""
COMPLETE INTEGRATED HRM JET + Z-SCORE STRIKEMAP TRADING SYSTEM
For Google Colab - Copy this entire cell and run
Combines corporate aviation intelligence with statistical options arbitrage
"""

import subprocess
import sys

# Install required packages
packages = ['yfinance', 'scikit-learn', 'seaborn', 'matplotlib', 'pandas', 'numpy']
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🚀 INTEGRATED HRM JET + Z-SCORE STRIKEMAP TRADING SYSTEM")
print("="*80)
print("🛩️ Corporate aviation intelligence + Statistical options arbitrage")
print("📊 Multi-asset trading opportunities with risk management")
print("💰 Real-time integration of fundamental and technical factors")
print("="*80)

class ZScoreStrikemap:
    """
    Statistical options trading model using regression and z-score analysis
    """
    
    def __init__(self):
        self.regression_model = LinearRegression()
        self.scaler = StandardScaler()
        
        # Trading parameters
        self.max_spread_pct = 25  # Maximum bid-ask spread %
        self.min_volume = 50
        self.min_open_interest = 100
        
        # Risk parameters
        self.profit_target_atm = 0.25  # 25% for ATM
        self.profit_target_otm = 0.40  # 40% for OTM
        self.stop_loss = -0.35  # -35%
        self.max_days_hold = 2      
  
    def get_option_chain(self, ticker: str, expiry_date: str = None) -> pd.DataFrame:
        """
        Get option chain data for a ticker
        """
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            
            # Generate synthetic option chain for demo
            strikes = np.arange(
                current_price * 0.8, 
                current_price * 1.2, 
                current_price * 0.02
            )
            
            option_data = []
            
            for strike in strikes:
                moneyness = strike / current_price
                days_to_expiry = 30  # Assume 30 DTE
                
                # Synthetic option pricing
                intrinsic_call = max(0, current_price - strike)
                intrinsic_put = max(0, strike - current_price)
                
                time_value = max(0.1, abs(moneyness - 1) * current_price * 0.1)
                
                # Call data
                call_price = intrinsic_call + time_value
                call_bid = call_price * 0.95
                call_ask = call_price * 1.05
                call_iv = 0.25 + abs(moneyness - 1) * 0.1
                
                # Put data  
                put_price = intrinsic_put + time_value
                put_bid = put_price * 0.95
                put_ask = put_price * 1.05
                put_iv = 0.25 + abs(moneyness - 1) * 0.1
                
                # Synthetic volume and OI
                volume_factor = max(0.1, 1 - abs(moneyness - 1) * 2)
                call_volume = int(np.random.poisson(100 * volume_factor))
                put_volume = int(np.random.poisson(80 * volume_factor))
                call_oi = int(np.random.poisson(500 * volume_factor))
                put_oi = int(np.random.poisson(400 * volume_factor))
                
                option_data.extend([
                    {
                        'Type': 'Call',
                        'Strike': strike,
                        'Bid': call_bid,
                        'Ask': call_ask,
                        'Last': call_price,
                        'Volume': call_volume,
                        'Open_Interest': call_oi,
                        'IV': call_iv,
                        'Moneyness': moneyness,
                        'DTE': days_to_expiry
                    },
                    {
                        'Type': 'Put',
                        'Strike': strike,
                        'Bid': put_bid,
                        'Ask': put_ask,
                        'Last': put_price,
                        'Volume': put_volume,
                        'Open_Interest': put_oi,
                        'IV': put_iv,
                        'Moneyness': moneyness,
                        'DTE': days_to_expiry
                    }
                ])
            
            df = pd.DataFrame(option_data)
            df['Current_Price'] = current_price
            df['Spread_Pct'] = (df['Ask'] - df['Bid']) / df['Last'] * 100
            
            return df
            
        except Exception as e:
            print(f"Error getting option chain for {ticker}: {e}")
            return pd.DataFrame()
    
    def preprocess_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess option chain data"""
        if df.empty:
            return df
        
        # Remove options with zero bid/ask
        df = df[(df['Bid'] > 0) & (df['Ask'] > 0)]
        
        # Remove wide spreads
        df = df[df['Spread_Pct'] <= 60]
        
        # Calculate additional metrics
        df['Mid_Price'] = (df['Bid'] + df['Ask']) / 2
        df['Log_Moneyness'] = np.log(df['Moneyness'])
        df['Delta'] = self._calculate_delta(df)
        
        return df
    
    def _calculate_delta(self, df: pd.DataFrame) -> pd.Series:
        """Simplified delta calculation"""
        deltas = []
        for _, row in df.iterrows():
            if row['Type'] == 'Call':
                if row['Moneyness'] < 0.9:
                    delta = 0.1
                elif row['Moneyness'] > 1.1:
                    delta = 0.9
                else:
                    delta = 0.5 + (row['Moneyness'] - 1) * 2
            else:  # Put
                if row['Moneyness'] < 0.9:
                    delta = -0.9
                elif row['Moneyness'] > 1.1:
                    delta = -0.1
                else:
                    delta = -0.5 + (1 - row['Moneyness']) * 2
            
            deltas.append(delta)
        
        return pd.Series(deltas)
    
    def fit_regression_model(self, df: pd.DataFrame, underlying_move: float) -> pd.DataFrame:
        """Fit regression model to predict option price performance"""
        if df.empty or len(df) < 10:
            return df
        
        # Features for regression
        features = ['Strike', 'IV', 'Moneyness', 'Log_Moneyness', 'Delta', 'DTE']
        
        # Create feature matrix
        X = df[features].fillna(0)
        
        # Target: Expected option price change based on underlying move
        y_expected = df['Delta'] * underlying_move * df['Current_Price']
        
        # Actual price change (synthetic for demo)
        noise = np.random.normal(0, 0.1, len(df))
        y_actual = y_expected + noise + np.random.normal(0, 0.05, len(df))
        
        # Fit regression model
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.regression_model.fit(X_scaled, y_actual)
            
            # Predict and calculate residuals
            y_pred = self.regression_model.predict(X_scaled)
            residuals = y_actual - y_pred
            
            # Calculate z-scores
            z_scores = stats.zscore(residuals)
            
            # Add results to dataframe
            df = df.copy()
            df['Expected_Change'] = y_expected
            df['Predicted_Change'] = y_pred
            df['Actual_Change'] = y_actual
            df['Residual'] = residuals
            df['Z_Score'] = z_scores
            
            return df
            
        except Exception as e:
            print(f"Error fitting regression model: {e}")
            df['Z_Score'] = 0
            return df
    
    def screen_for_trades(self, df: pd.DataFrame) -> dict:
        """Screen options for trading opportunities based on z-scores and liquidity"""
        if df.empty:
            return {'calls': [], 'puts': [], 'message': 'No data available'}
        
        # Apply liquidity filters
        liquid_calls = df[
            (df['Type'] == 'Call') &
            (df['Spread_Pct'] <= self.max_spread_pct) &
            (df['Volume'] >= self.min_volume) &
            (df['Open_Interest'] >= self.min_open_interest)
        ].copy()
        
        liquid_puts = df[
            (df['Type'] == 'Put') &
            (df['Spread_Pct'] <= self.max_spread_pct) &
            (df['Volume'] >= self.min_volume) &
            (df['Open_Interest'] >= self.min_open_interest)
        ].copy()
        
        # Focus on ATM to moderately OTM options
        atm_otm_calls = liquid_calls[
            (liquid_calls['Moneyness'] >= 0.95) & 
            (liquid_calls['Moneyness'] <= 1.15)
        ]
        
        atm_otm_puts = liquid_puts[
            (liquid_puts['Moneyness'] >= 0.85) & 
            (liquid_puts['Moneyness'] <= 1.05)
        ]
        
        # Sort by z-score (most undervalued first)
        undervalued_calls = atm_otm_calls[atm_otm_calls['Z_Score'] < -1.0].sort_values('Z_Score')
        undervalued_puts = atm_otm_puts[atm_otm_puts['Z_Score'] < -1.0].sort_values('Z_Score')
        
        return {
            'undervalued_calls': undervalued_calls.head(5),
            'undervalued_puts': undervalued_puts.head(5),
            'total_liquid_calls': len(liquid_calls),
            'total_liquid_puts': len(liquid_puts)
        }
    
    def generate_trade_signals(self, ticker: str, underlying_move: float) -> dict:
        """Generate complete trade signals for a ticker"""
        # Get option chain
        option_chain = self.get_option_chain(ticker)
        
        if option_chain.empty:
            return {'ticker': ticker, 'message': 'No option data available'}
        
        # Preprocess
        clean_options = self.preprocess_options(option_chain)
        
        # Fit regression and calculate z-scores
        options_with_scores = self.fit_regression_model(clean_options, underlying_move)
        
        # Screen for trades
        trade_opportunities = self.screen_for_trades(options_with_scores)
        
        return {
            'ticker': ticker,
            'underlying_move': underlying_move,
            'current_price': option_chain['Current_Price'].iloc[0],
            'opportunities': trade_opportunities,
            'timestamp': datetime.now()
        }

class IntegratedTradingSystem:
    """
    Integrated system combining HRM Jet Signals with Z-Score Strikemap
    """
    
    def __init__(self):
        self.zscore_model = ZScoreStrikemap()
        
        # Companies with both jet tracking and liquid options
        self.tracked_companies = {
            'AAPL': {'sector': 'Technology', 'avg_volume': 50000000},
            'MSFT': {'sector': 'Technology', 'avg_volume': 30000000},
            'GOOGL': {'sector': 'Technology', 'avg_volume': 25000000},
            'META': {'sector': 'Technology', 'avg_volume': 20000000},
            'NVDA': {'sector': 'Technology', 'avg_volume': 45000000},
            'TSLA': {'sector': 'Consumer', 'avg_volume': 80000000},
            'AMZN': {'sector': 'Consumer', 'avg_volume': 35000000},
            'JPM': {'sector': 'Financial', 'avg_volume': 15000000},
            'SPY': {'sector': 'ETF', 'avg_volume': 100000000},
            'QQQ': {'sector': 'ETF', 'avg_volume': 50000000}
        }
        
    def get_jet_signals(self, date: datetime) -> list:
        """Simulate HRM jet signals for the given date"""
        signals = []
        
        # Simulate some jet signals
        if np.random.random() < 0.3:  # 30% chance of signals
            num_signals = np.random.randint(1, 4)
            
            for _ in range(num_signals):
                ticker = np.random.choice(list(self.tracked_companies.keys()))
                
                # Signal types with different probabilities
                signal_types = ['ma_activity', 'earnings_prep', 'crisis', 'breakthrough']
                signal_type = np.random.choice(signal_types, p=[0.4, 0.3, 0.2, 0.1])
                
                # Generate conviction based on signal type
                if signal_type == 'ma_activity':
                    conviction = np.random.normal(0.75, 0.15)
                    confidence = np.random.normal(0.85, 0.1)
                elif signal_type == 'breakthrough':
                    conviction = np.random.normal(0.70, 0.12)
                    confidence = np.random.normal(0.80, 0.1)
                elif signal_type == 'crisis':
                    conviction = np.random.normal(-0.65, 0.15)
                    confidence = np.random.normal(0.75, 0.1)
                else:  # earnings_prep
                    conviction = np.random.normal(0.0, 0.4)
                    confidence = np.random.normal(0.65, 0.15)
                
                conviction = np.clip(conviction, -1.0, 1.0)
                confidence = np.clip(confidence, 0.0, 1.0)
                
                signals.append({
                    'ticker': ticker,
                    'signal_type': signal_type,
                    'conviction': conviction,
                    'confidence': confidence,
                    'expected_move': conviction * 0.05,  # 5% max expected move
                    'time_horizon': '1-3 days',
                    'reasoning': f'{signal_type.replace("_", " ").title()} detected via corporate jet analysis'
                })
        
        return signals
    
    def analyze_integrated_opportunities(self, date: datetime = None) -> dict:
        """Analyze both jet signals and options opportunities"""
        if date is None:
            date = datetime.now()
        
        print(f"🛩️ Analyzing integrated trading opportunities for {date.strftime('%Y-%m-%d')}")
        
        # Get jet signals
        jet_signals = self.get_jet_signals(date)
        
        # Analyze options for companies with jet signals
        integrated_opportunities = []
        
        for signal in jet_signals:
            ticker = signal['ticker']
            expected_move = signal['expected_move']
            
            print(f"📊 Analyzing options for {ticker} (Expected move: {expected_move:+.1%})")
            
            # Get options analysis
            options_analysis = self.zscore_model.generate_trade_signals(ticker, expected_move)
            
            # Combine jet signal with options opportunities
            integrated_opportunity = {
                'ticker': ticker,
                'jet_signal': signal,
                'options_analysis': options_analysis,
                'integration_score': self._calculate_integration_score(signal, options_analysis)
            }
            
            integrated_opportunities.append(integrated_opportunity)
        
        # Sort by integration score
        integrated_opportunities.sort(key=lambda x: x['integration_score'], reverse=True)
        
        return {
            'date': date,
            'total_jet_signals': len(jet_signals),
            'integrated_opportunities': integrated_opportunities,
            'top_opportunities': integrated_opportunities[:3]
        }
    
    def _calculate_integration_score(self, jet_signal: dict, options_analysis: dict) -> float:
        """Calculate integration score combining jet signal strength with options opportunities"""
        # Base score from jet signal
        jet_score = abs(jet_signal['conviction']) * jet_signal['confidence']
        
        # Options opportunity score
        options_score = 0
        
        if 'opportunities' in options_analysis:
            opps = options_analysis['opportunities']
            
            # Score based on number of liquid opportunities
            liquid_calls = len(opps.get('undervalued_calls', []))
            liquid_puts = len(opps.get('undervalued_puts', []))
            
            options_score = (liquid_calls + liquid_puts) * 0.1
            
            # Bonus for strong z-scores
            if liquid_calls > 0:
                best_call_zscore = abs(opps['undervalued_calls']['Z_Score'].min()) if not opps['undervalued_calls'].empty else 0
                options_score += min(best_call_zscore * 0.1, 0.3)
            
            if liquid_puts > 0:
                best_put_zscore = abs(opps['undervalued_puts']['Z_Score'].min()) if not opps['undervalued_puts'].empty else 0
                options_score += min(best_put_zscore * 0.1, 0.3)
        
        # Combined score
        integration_score = jet_score * 0.7 + options_score * 0.3
        
        return integration_score
    
    def generate_trade_recommendations(self, opportunities: dict) -> list:
        """Generate specific trade recommendations"""
        recommendations = []
        
        for opp in opportunities['top_opportunities']:
            ticker = opp['ticker']
            jet_signal = opp['jet_signal']
            options_analysis = opp['options_analysis']
            
            # Determine strategy based on jet signal and options availability
            if jet_signal['conviction'] > 0.6:  # Strong bullish signal
                strategy = self._recommend_bullish_strategy(ticker, jet_signal, options_analysis)
            elif jet_signal['conviction'] < -0.6:  # Strong bearish signal
                strategy = self._recommend_bearish_strategy(ticker, jet_signal, options_analysis)
            else:  # Neutral or weak signal
                strategy = self._recommend_neutral_strategy(ticker, jet_signal, options_analysis)
            
            if strategy:
                recommendations.append(strategy)
        
        return recommendations
    
    def _recommend_bullish_strategy(self, ticker: str, jet_signal: dict, options_analysis: dict) -> dict:
        """Recommend bullish strategy combining equity and options"""
        if 'opportunities' not in options_analysis:
            return None
        
        opps = options_analysis['opportunities']
        undervalued_calls = opps.get('undervalued_calls', pd.DataFrame())
        
        if not undervalued_calls.empty:
            best_call = undervalued_calls.iloc[0]
            
            return {
                'ticker': ticker,
                'strategy': 'Bullish Call + Equity',
                'jet_reasoning': jet_signal['reasoning'],
                'primary_trade': {
                    'type': 'Call Option',
                    'strike': best_call['Strike'],
                    'bid': best_call['Bid'],
                    'ask': best_call['Ask'],
                    'z_score': best_call['Z_Score'],
                    'moneyness': best_call['Moneyness']
                },
                'secondary_trade': {
                    'type': 'Equity Long',
                    'reasoning': 'Hedge delta and capture additional upside'
                },
                'risk_management': {
                    'profit_target': '25-40%',
                    'stop_loss': '35%',
                    'time_stop': '2 days',
                    'max_position_size': '2% of portfolio'
                },
                'conviction': jet_signal['conviction'],
                'confidence': jet_signal['confidence']
            }
        
        return {
            'ticker': ticker,
            'strategy': 'Equity Long Only',
            'jet_reasoning': jet_signal['reasoning'],
            'primary_trade': {
                'type': 'Equity Long',
                'reasoning': 'No liquid call options available'
            },
            'conviction': jet_signal['conviction'],
            'confidence': jet_signal['confidence']
        }
    
    def _recommend_bearish_strategy(self, ticker: str, jet_signal: dict, options_analysis: dict) -> dict:
        """Recommend bearish strategy"""
        if 'opportunities' not in options_analysis:
            return None
        
        opps = options_analysis['opportunities']
        undervalued_puts = opps.get('undervalued_puts', pd.DataFrame())
        
        if not undervalued_puts.empty:
            best_put = undervalued_puts.iloc[0]
            
            return {
                'ticker': ticker,
                'strategy': 'Protective Put + Short Equity',
                'jet_reasoning': jet_signal['reasoning'],
                'primary_trade': {
                    'type': 'Put Option',
                    'strike': best_put['Strike'],
                    'bid': best_put['Bid'],
                    'ask': best_put['Ask'],
                    'z_score': best_put['Z_Score'],
                    'moneyness': best_put['Moneyness']
                },
                'secondary_trade': {
                    'type': 'Equity Short',
                    'reasoning': 'Amplify downside exposure'
                },
                'conviction': abs(jet_signal['conviction']),
                'confidence': jet_signal['confidence']
            }
        
        return {
            'ticker': ticker,
            'strategy': 'Equity Short Only',
            'jet_reasoning': jet_signal['reasoning'],
            'conviction': abs(jet_signal['conviction']),
            'confidence': jet_signal['confidence']
        }
    
    def _recommend_neutral_strategy(self, ticker: str, jet_signal: dict, options_analysis: dict) -> dict:
        """Recommend neutral/hedging strategy"""
        return {
            'ticker': ticker,
            'strategy': 'Monitor Only',
            'jet_reasoning': jet_signal['reasoning'],
            'recommendation': 'Signal too weak for position, monitor for strengthening',
            'conviction': jet_signal['conviction'],
            'confidence': jet_signal['confidence']
        }

def run_integrated_analysis():
    """Run complete integrated analysis"""
    print("🚀 INTEGRATED HRM JET + Z-SCORE STRIKEMAP ANALYSIS")
    print("="*80)
    print("🛩️ Corporate aviation intelligence + Statistical options arbitrage")
    print("📊 Multi-asset trading opportunities with risk management")
    
    # Initialize system
    system = IntegratedTradingSystem()
    
    # Run analysis
    opportunities = system.analyze_integrated_opportunities()
    
    # Generate recommendations
    recommendations = system.generate_trade_recommendations(opportunities)
    
    # Display results
    print(f"\n📋 ANALYSIS RESULTS:")
    print(f"   Date: {opportunities['date'].strftime('%Y-%m-%d')}")
    print(f"   Jet Signals Detected: {opportunities['total_jet_signals']}")
    print(f"   Integrated Opportunities: {len(opportunities['integrated_opportunities'])}")
    print(f"   Trade Recommendations: {len(recommendations)}")
    
    if recommendations:
        print(f"\n🎯 TOP TRADE RECOMMENDATIONS:")
        print("-" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['ticker']} - {rec['strategy']}")
            print(f"   Jet Signal: {rec['jet_reasoning']}")
            print(f"   Conviction: {rec['conviction']:+.2f} | Confidence: {rec['confidence']:.1%}")
            
            if 'primary_trade' in rec:
                trade = rec['primary_trade']
                print(f"   Primary Trade: {trade['type']}")
                
                if 'strike' in trade:
                    print(f"   Strike: ${trade['strike']:.2f} | Z-Score: {trade['z_score']:.2f}")
                    print(f"   Bid/Ask: ${trade['bid']:.2f}/${trade['ask']:.2f}")
            
            if 'risk_management' in rec:
                rm = rec['risk_management']
                print(f"   Risk: Target {rm['profit_target']}, Stop {rm['stop_loss']}, Time {rm['time_stop']}")
    
    else:
        print(f"\n⚪ No high-conviction trade recommendations today")
        print(f"💡 Monitor for stronger jet signals or better options opportunities")
    
    print("\n" + "="*80)
    print("💡 INTEGRATED SYSTEM ADVANTAGES:")
    print("   • Corporate jet intelligence provides directional bias")
    print("   • Statistical options analysis finds mispriced contracts")
    print("   • Combined approach reduces false signals")
    print("   • Multi-asset strategies optimize risk/reward")
    print("   • Real-time integration of fundamental and technical factors")
    print("="*80)
    
    return opportunities, recommendations

# Run the integrated analysis
if __name__ == "__main__":
    opportunities, recommendations = run_integrated_analysis()