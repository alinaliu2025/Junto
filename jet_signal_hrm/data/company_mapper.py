"""
Maps aircraft tail numbers to public companies and executives
"""

import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)

class CompanyMapper:
    """Maps aircraft registrations to public companies"""
    
    def __init__(self, db_path: str = "company_aircraft.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for company-aircraft mappings"""
        conn = sqlite3.connect(self.db_path)
        
        # Aircraft registration table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aircraft_registry (
                tail_number TEXT PRIMARY KEY,
                icao24 TEXT UNIQUE,
                owner_name TEXT,
                owner_address TEXT,
                aircraft_type TEXT,
                manufacturer TEXT,
                model TEXT,
                registration_date DATE,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Public company mappings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS company_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tail_number TEXT,
                ticker_symbol TEXT,
                company_name TEXT,
                confidence_score REAL,
                mapping_source TEXT,
                verified BOOLEAN DEFAULT FALSE,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tail_number) REFERENCES aircraft_registry (tail_number)
            )
        """)
        
        # Executive travel patterns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS executive_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker_symbol TEXT,
                executive_name TEXT,
                executive_title TEXT,
                typical_routes TEXT,  -- JSON array of common routes
                travel_frequency INTEGER,  -- flights per month
                last_analysis_date DATE
            )
        """)
        
        conn.commit()
        conn.close()
        
    def add_aircraft_registration(self, tail_number: str, icao24: str, 
                                owner_info: Dict) -> bool:
        """Add aircraft registration data"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO aircraft_registry 
                (tail_number, icao24, owner_name, owner_address, aircraft_type, 
                 manufacturer, model, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tail_number, icao24, owner_info.get('owner_name'),
                owner_info.get('owner_address'), owner_info.get('aircraft_type'),
                owner_info.get('manufacturer'), owner_info.get('model'),
                owner_info.get('registration_date')
            ))
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding aircraft registration {tail_number}: {e}")
            return False
    
    def map_aircraft_to_company(self, tail_number: str, ticker: str, 
                              company_name: str, confidence: float,
                              source: str = "manual") -> bool:
        """Map aircraft to public company"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO company_mappings 
                (tail_number, ticker_symbol, company_name, confidence_score, mapping_source)
                VALUES (?, ?, ?, ?, ?)
            """, (tail_number, ticker, company_name, confidence, source))
            conn.commit()
            conn.close()
            
            logger.info(f"Mapped {tail_number} to {ticker} ({company_name}) with {confidence:.2f} confidence")
            return True
            
        except Exception as e:
            logger.error(f"Error mapping {tail_number} to {ticker}: {e}")
            return False
    
    def get_company_from_aircraft(self, identifier: str) -> Optional[Dict]:
        """
        Get company information from tail number or ICAO24
        
        Args:
            identifier: Tail number (N123AB) or ICAO24 hex code
            
        Returns:
            Dictionary with company info or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Determine if identifier is tail number or ICAO24
            if re.match(r'^[A-F0-9]{6}$', identifier.upper()):
                # ICAO24 format
                query = """
                    SELECT cm.ticker_symbol, cm.company_name, cm.confidence_score,
                           ar.tail_number, ar.owner_name, ar.aircraft_type
                    FROM company_mappings cm
                    JOIN aircraft_registry ar ON cm.tail_number = ar.tail_number
                    WHERE ar.icao24 = ?
                    ORDER BY cm.confidence_score DESC
                    LIMIT 1
                """
                cursor = conn.execute(query, (identifier.upper(),))
            else:
                # Tail number format
                query = """
                    SELECT cm.ticker_symbol, cm.company_name, cm.confidence_score,
                           ar.tail_number, ar.owner_name, ar.aircraft_type
                    FROM company_mappings cm
                    JOIN aircraft_registry ar ON cm.tail_number = ar.tail_number
                    WHERE ar.tail_number = ?
                    ORDER BY cm.confidence_score DESC
                    LIMIT 1
                """
                cursor = conn.execute(query, (identifier.upper(),))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'ticker': result[0],
                    'company_name': result[1],
                    'confidence': result[2],
                    'tail_number': result[3],
                    'owner_name': result[4],
                    'aircraft_type': result[5]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting company for {identifier}: {e}")
            return None
    
    def get_company_aircraft(self, ticker: str) -> List[Dict]:
        """Get all aircraft for a company"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT ar.tail_number, ar.icao24, ar.aircraft_type, 
                       ar.manufacturer, ar.model, cm.confidence_score
                FROM aircraft_registry ar
                JOIN company_mappings cm ON ar.tail_number = cm.tail_number
                WHERE cm.ticker_symbol = ?
                ORDER BY cm.confidence_score DESC
            """
            
            cursor = conn.execute(query, (ticker.upper(),))
            results = cursor.fetchall()
            conn.close()
            
            aircraft_list = []
            for row in results:
                aircraft_list.append({
                    'tail_number': row[0],
                    'icao24': row[1],
                    'aircraft_type': row[2],
                    'manufacturer': row[3],
                    'model': row[4],
                    'confidence': row[5]
                })
                
            return aircraft_list
            
        except Exception as e:
            logger.error(f"Error getting aircraft for {ticker}: {e}")
            return []
    
    def bulk_import_faa_data(self, faa_csv_path: str) -> int:
        """
        Import FAA aircraft registration database
        
        Args:
            faa_csv_path: Path to FAA MASTER.txt file
            
        Returns:
            Number of records imported
        """
        try:
            # FAA MASTER.txt format (fixed width)
            # This is a simplified version - actual FAA format is more complex
            df = pd.read_csv(faa_csv_path, dtype=str)
            
            imported = 0
            conn = sqlite3.connect(self.db_path)
            
            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO aircraft_registry 
                        (tail_number, owner_name, owner_address, aircraft_type, 
                         manufacturer, model)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('N-NUMBER', '').strip(),
                        row.get('NAME', '').strip(),
                        f"{row.get('STREET', '')} {row.get('CITY', '')} {row.get('STATE', '')}".strip(),
                        row.get('TYPE AIRCRAFT', '').strip(),
                        row.get('MFR MDL CODE', '').strip(),
                        row.get('MODEL', '').strip()
                    ))
                    imported += 1
                    
                except Exception as e:
                    logger.warning(f"Error importing row: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Imported {imported} aircraft registrations from FAA data")
            return imported
            
        except Exception as e:
            logger.error(f"Error importing FAA data: {e}")
            return 0
    
    def auto_map_companies(self, company_keywords: Dict[str, str]) -> int:
        """
        Automatically map aircraft to companies based on owner name keywords
        
        Args:
            company_keywords: Dict mapping ticker symbols to owner name keywords
            
        Returns:
            Number of mappings created
        """
        try:
            conn = sqlite3.connect(self.db_path)
            mapped = 0
            
            for ticker, keywords in company_keywords.items():
                # Get company name (simplified - would use proper company database)
                company_name = ticker  # Placeholder
                
                # Find aircraft with matching owner names
                for keyword in keywords.split(','):
                    keyword = keyword.strip().upper()
                    
                    query = """
                        SELECT tail_number, owner_name 
                        FROM aircraft_registry 
                        WHERE UPPER(owner_name) LIKE ?
                    """
                    
                    cursor = conn.execute(query, (f'%{keyword}%',))
                    matches = cursor.fetchall()
                    
                    for tail_number, owner_name in matches:
                        # Calculate confidence based on keyword match strength
                        confidence = 0.8 if keyword in owner_name.upper() else 0.6
                        
                        try:
                            conn.execute("""
                                INSERT OR IGNORE INTO company_mappings 
                                (tail_number, ticker_symbol, company_name, 
                                 confidence_score, mapping_source)
                                VALUES (?, ?, ?, ?, ?)
                            """, (tail_number, ticker, company_name, confidence, "auto_keyword"))
                            mapped += 1
                            
                        except Exception as e:
                            logger.warning(f"Error auto-mapping {tail_number}: {e}")
            
            conn.commit()
            conn.close()
            
            logger.info(f"Auto-mapped {mapped} aircraft to companies")
            return mapped
            
        except Exception as e:
            logger.error(f"Error in auto-mapping: {e}")
            return 0