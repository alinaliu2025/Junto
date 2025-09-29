"""
Historical flight data collection script for HRM Jet Signal System
"""

import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

from jet_signal_hrm.data.flight_data import FlightDataCollector
from jet_signal_hrm.data.company_mapper import CompanyMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_historical_data(days: int = 7, output_dir: str = "data/historical"):
    """
    Collect historical flight data for training
    
    Args:
        days: Number of days to collect data for
        output_dir: Directory to save collected data
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize collectors
    flight_collector = FlightDataCollector()
    company_mapper = CompanyMapper()
    
    logger.info(f"Starting historical data collection for {days} days")
    
    all_flights = []
    company_flight_counts = {}
    
    for day in range(days):
        try:
            date = datetime.now() - timedelta(days=day)
            logger.info(f"Collecting data for {date.strftime('%Y-%m-%d')}")
            
            # Collect live flights (in practice, would use historical API)
            flights_df = flight_collector.get_live_flights()
            
            if flights_df.empty:
                logger.warning(f"No flights collected for {date.strftime('%Y-%m-%d')}")
                continue
            
            # Add date column
            flights_df['collection_date'] = date.strftime('%Y-%m-%d')
            
            # Map to companies and count
            mapped_count = 0
            for _, flight in flights_df.iterrows():
                company_info = company_mapper.get_company_from_aircraft(flight['icao24'])
                if company_info:
                    ticker = company_info['ticker']
                    company_flight_counts[ticker] = company_flight_counts.get(ticker, 0) + 1
                    mapped_count += 1
            
            all_flights.append(flights_df)
            
            logger.info(f"Collected {len(flights_df)} flights, {mapped_count} mapped to companies")
            
            # Save daily data
            daily_file = Path(output_dir) / f"flights_{date.strftime('%Y%m%d')}.csv"
            flights_df.to_csv(daily_file, index=False)
            
            # Respect API limits
            time.sleep(60)  # Wait 1 minute between requests
            
        except Exception as e:
            logger.error(f"Error collecting data for day {day}: {e}")
            continue
    
    # Combine all data
    if all_flights:
        combined_df = pd.concat(all_flights, ignore_index=True)
        
        # Save combined dataset
        combined_file = Path(output_dir) / "combined_flights.csv"
        combined_df.to_csv(combined_file, index=False)
        
        # Save summary statistics
        summary = {
            "collection_period": f"{days} days",
            "total_flights": len(combined_df),
            "unique_aircraft": combined_df['icao24'].nunique(),
            "company_flight_counts": company_flight_counts,
            "collection_dates": sorted(combined_df['collection_date'].unique().tolist())
        }
        
        summary_file = Path(output_dir) / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data collection complete!")
        logger.info(f"Total flights: {len(combined_df)}")
        logger.info(f"Unique aircraft: {combined_df['icao24'].nunique()}")
        logger.info(f"Companies with flights: {len(company_flight_counts)}")
        logger.info(f"Data saved to: {output_dir}")
        
        return combined_df, summary
    
    else:
        logger.error("No flight data collected")
        return None, None

def analyze_flight_patterns(data_dir: str = "data/historical"):
    """Analyze collected flight data for patterns"""
    
    combined_file = Path(data_dir) / "combined_flights.csv"
    if not combined_file.exists():
        logger.error(f"No combined flight data found at {combined_file}")
        return
    
    # Load data
    df = pd.read_csv(combined_file)
    logger.info(f"Analyzing {len(df)} flights")
    
    # Initialize company mapper
    company_mapper = CompanyMapper()
    
    # Analyze patterns by company
    company_patterns = {}
    
    for _, flight in df.iterrows():
        company_info = company_mapper.get_company_from_aircraft(flight['icao24'])
        if not company_info:
            continue
            
        ticker = company_info['ticker']
        if ticker not in company_patterns:
            company_patterns[ticker] = {
                'flights': [],
                'unique_aircraft': set(),
                'flight_dates': [],
                'avg_altitude': 0,
                'avg_velocity': 0
            }
        
        company_patterns[ticker]['flights'].append(flight.to_dict())
        company_patterns[ticker]['unique_aircraft'].add(flight['icao24'])
        company_patterns[ticker]['flight_dates'].append(flight['collection_date'])
    
    # Calculate statistics
    pattern_summary = {}
    for ticker, data in company_patterns.items():
        flights = data['flights']
        
        altitudes = [f.get('baro_altitude', 0) or 0 for f in flights]
        velocities = [f.get('velocity', 0) or 0 for f in flights]
        
        pattern_summary[ticker] = {
            'total_flights': len(flights),
            'unique_aircraft': len(data['unique_aircraft']),
            'flight_days': len(set(data['flight_dates'])),
            'avg_altitude': sum(altitudes) / len(altitudes) if altitudes else 0,
            'avg_velocity': sum(velocities) / len(velocities) if velocities else 0,
            'flights_per_day': len(flights) / len(set(data['flight_dates'])) if data['flight_dates'] else 0
        }
    
    # Save analysis
    analysis_file = Path(data_dir) / "pattern_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(pattern_summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FLIGHT PATTERN ANALYSIS")
    print("="*60)
    
    print(f"{'Ticker':<8} {'Flights':<8} {'Aircraft':<8} {'Days':<6} {'Flights/Day':<12}")
    print("-" * 60)
    
    for ticker, stats in sorted(pattern_summary.items(), 
                               key=lambda x: x[1]['total_flights'], 
                               reverse=True):
        print(f"{ticker:<8} {stats['total_flights']:<8} {stats['unique_aircraft']:<8} "
              f"{stats['flight_days']:<6} {stats['flights_per_day']:<12.1f}")
    
    logger.info(f"Pattern analysis saved to: {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description="Collect historical flight data")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days to collect data for")
    parser.add_argument("--output_dir", type=str, default="data/historical",
                       help="Directory to save collected data")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing collected data")
    parser.add_argument("--collect", action="store_true",
                       help="Collect new flight data")
    
    args = parser.parse_args()
    
    if args.collect:
        collect_historical_data(args.days, args.output_dir)
    
    if args.analyze:
        analyze_flight_patterns(args.output_dir)
    
    if not args.collect and not args.analyze:
        # Default: collect then analyze
        collect_historical_data(args.days, args.output_dir)
        analyze_flight_patterns(args.output_dir)

if __name__ == "__main__":
    main()