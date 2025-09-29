"""
Real-time ADS-B flight data collection and processing
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FlightDataCollector:
    """Collects and processes real-time ADS-B flight data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.opensky_url = "https://opensky-network.org/api"
        self.session = requests.Session()
        
    def get_live_flights(self, bbox: Optional[Tuple[float, float, float, float]] = None) -> pd.DataFrame:
        """
        Get live flight data from OpenSky Network
        
        Args:
            bbox: Bounding box (lat_min, lon_min, lat_max, lon_max) for geographic filtering
            
        Returns:
            DataFrame with columns: icao24, callsign, origin_country, time_position,
                                  last_contact, longitude, latitude, baro_altitude, 
                                  on_ground, velocity, true_track, vertical_rate
        """
        try:
            url = f"{self.opensky_url}/states/all"
            params = {}
            
            if bbox:
                params.update({
                    'lamin': bbox[0], 'lomin': bbox[1],
                    'lamax': bbox[2], 'lomax': bbox[3]
                })
                
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data or 'states' not in data:
                return pd.DataFrame()
                
            # Convert to DataFrame
            columns = [
                'icao24', 'callsign', 'origin_country', 'time_position',
                'last_contact', 'longitude', 'latitude', 'baro_altitude',
                'on_ground', 'velocity', 'true_track', 'vertical_rate',
                'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'
            ]
            
            df = pd.DataFrame(data['states'], columns=columns)
            df['timestamp'] = datetime.utcnow()
            
            # Clean and filter data
            df = df.dropna(subset=['longitude', 'latitude'])
            df = df[df['on_ground'] == False]  # Only airborne aircraft
            
            logger.info(f"Collected {len(df)} live flights")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting flight data: {e}")
            return pd.DataFrame()
    
    def get_flight_history(self, icao24: str, begin: int, end: int) -> pd.DataFrame:
        """
        Get historical flight data for specific aircraft
        
        Args:
            icao24: Aircraft ICAO 24-bit address
            begin: Unix timestamp for start time
            end: Unix timestamp for end time
            
        Returns:
            DataFrame with flight track data
        """
        try:
            url = f"{self.opensky_url}/tracks/all"
            params = {
                'icao24': icao24,
                'time': begin
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data or 'path' not in data:
                return pd.DataFrame()
                
            # Convert path data to DataFrame
            path_data = []
            for point in data['path']:
                path_data.append({
                    'icao24': icao24,
                    'timestamp': point[0],
                    'latitude': point[1],
                    'longitude': point[2],
                    'baro_altitude': point[3],
                    'true_track': point[4]
                })
                
            df = pd.DataFrame(path_data)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting flight history for {icao24}: {e}")
            return pd.DataFrame()
    
    def detect_flight_legs(self, track_df: pd.DataFrame, min_ground_time: int = 1800) -> List[Dict]:
        """
        Detect individual flight legs from continuous track data
        
        Args:
            track_df: DataFrame with flight track data
            min_ground_time: Minimum ground time in seconds to separate legs
            
        Returns:
            List of flight leg dictionaries
        """
        if track_df.empty:
            return []
            
        legs = []
        track_df = track_df.sort_values('timestamp')
        
        # Simple leg detection based on altitude changes
        # In production, would use more sophisticated ground detection
        current_leg = None
        
        for _, row in track_df.iterrows():
            altitude = row.get('baro_altitude', 0)
            
            if altitude and altitude > 1000:  # Airborne
                if current_leg is None:
                    current_leg = {
                        'icao24': row['icao24'],
                        'start_time': row['timestamp'],
                        'start_lat': row['latitude'],
                        'start_lon': row['longitude'],
                        'waypoints': [(row['latitude'], row['longitude'], row['timestamp'])]
                    }
                else:
                    current_leg['waypoints'].append((row['latitude'], row['longitude'], row['timestamp']))
                    
            else:  # On ground or low altitude
                if current_leg is not None:
                    current_leg.update({
                        'end_time': row['timestamp'],
                        'end_lat': row['latitude'],
                        'end_lon': row['longitude'],
                        'duration': row['timestamp'] - current_leg['start_time']
                    })
                    legs.append(current_leg)
                    current_leg = None
        
        # Close final leg if still airborne
        if current_leg is not None:
            last_point = track_df.iloc[-1]
            current_leg.update({
                'end_time': last_point['timestamp'],
                'end_lat': last_point['latitude'],
                'end_lon': last_point['longitude'],
                'duration': last_point['timestamp'] - current_leg['start_time']
            })
            legs.append(current_leg)
            
        logger.info(f"Detected {len(legs)} flight legs for {track_df.iloc[0]['icao24']}")
        return legs
    
    def get_airport_from_coordinates(self, lat: float, lon: float, radius: float = 0.1) -> Optional[str]:
        """
        Get nearest airport code from coordinates
        
        Args:
            lat: Latitude
            lon: Longitude  
            radius: Search radius in degrees
            
        Returns:
            Airport ICAO code or None
        """
        # In production, would use airport database lookup
        # For now, return placeholder
        return f"AIRPORT_{lat:.2f}_{lon:.2f}"