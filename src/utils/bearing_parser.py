"""
Utilities for parsing and converting bearing formats.
Handles multiple formats: N 45°30'15" E, S22°10'W, 123°45'30", 123.75°, N45-30-15E
"""
import re
import math
from typing import Optional, Tuple, Union


class BearingParser:
    """Parse various bearing formats and convert to azimuth degrees"""
    
    # General directional terms and their corresponding bearings
    DIRECTIONAL_TERMS = {
        'north': 'N 0°0\'0" E',
        'northerly': 'N 0°0\'0" E',
        'northeast': 'N 45°0\'0" E',
        'northeasterly': 'N 45°0\'0" E',
        'east': 'N 90°0\'0" E',
        'easterly': 'N 90°0\'0" E',
        'southeast': 'S 45°0\'0" E',
        'southeasterly': 'S 45°0\'0" E',
        'south': 'S 0°0\'0" E',
        'southerly': 'S 0°0\'0" E',
        'southwest': 'S 45°0\'0" W',
        'southwesterly': 'S 45°0\'0" W',
        'west': 'N 90°0\'0" W',
        'westerly': 'N 90°0\'0" W',
        'northwest': 'N 45°0\'0" W',
        'northwesterly': 'N 45°0\'0" W',
        # Also add abbreviated forms
        'n': 'N 0°0\'0" E',
        'ne': 'N 45°0\'0" E',
        'e': 'N 90°0\'0" E',
        'se': 'S 45°0\'0" E',
        's': 'S 0°0\'0" E',
        'sw': 'S 45°0\'0" W',
        'w': 'N 90°0\'0" W',
        'nw': 'N 45°0\'0" W',
    }
    
    # Regex patterns for different bearing formats
    PATTERNS = {
        'quadrant_dms': re.compile(
            r'([NS])\s*(\d{1,3})°?\s*(\d{1,2})\'?\s*(\d{1,2}(?:\.\d+)?)\"?\s*([EW])',
            re.IGNORECASE
        ),
        'quadrant_dm': re.compile(
            r'([NS])\s*(\d{1,3})°?\s*(\d{1,2}(?:\.\d+)?)\'?\s*([EW])',
            re.IGNORECASE
        ),
        'quadrant_d': re.compile(
            r'([NS])\s*(\d{1,3}(?:\.\d+)?)°?\s*([EW])',
            re.IGNORECASE
        ),
        'quadrant_dash': re.compile(
            r'([NS])\s*(\d{1,3})-(\d{1,2})-(\d{1,2}(?:\.\d+)?)\s*([EW])',
            re.IGNORECASE
        ),
        'azimuth_dms': re.compile(
            r'(\d{1,3})°?\s*(\d{1,2})\'?\s*(\d{1,2}(?:\.\d+)?)\"?'
        ),
        'azimuth_dm': re.compile(
            r'(\d{1,3})°?\s*(\d{1,2}(?:\.\d+)?)\'?'
        ),
        'azimuth_d': re.compile(
            r'(\d{1,3}(?:\.\d+)?)°?'
        )
    }
    
    @classmethod
    def parse_bearing(cls, bearing_str: str) -> Optional[float]:
        """
        Parse bearing string and return azimuth in degrees (0-360).
        
        Args:
            bearing_str: Bearing string in various formats
            
        Returns:
            Azimuth in degrees (0-360) or None if parsing fails
        """
        if not bearing_str or not isinstance(bearing_str, str):
            return None
            
        bearing_str = bearing_str.strip()
        
        # First check for general directional terms
        azimuth = cls._parse_directional_term(bearing_str)
        if azimuth is not None:
            return azimuth
        
        # Try quadrant formats (most common in deeds)
        azimuth = cls._parse_quadrant_bearing(bearing_str)
        if azimuth is not None:
            return azimuth
            
        # Try azimuth formats
        azimuth = cls._parse_azimuth_bearing(bearing_str)
        if azimuth is not None:
            return azimuth
            
        return None
    
    @classmethod
    def _parse_directional_term(cls, bearing_str: str) -> Optional[float]:
        """Parse general directional terms like NORTHWESTERLY, EASTERLY, etc."""
        # Extract just the directional word from phrases like "RUNNING THENCE NORTHWESTERLY"
        # Look for directional terms in the string
        bearing_str_lower = bearing_str.lower()
        
        for term, bearing in cls.DIRECTIONAL_TERMS.items():
            # Check if the term appears as a word (not part of another word)
            # Use word boundaries or check for the term followed by space or end of string
            if term in bearing_str_lower:
                # Make sure it's a complete word match
                pattern = r'\b' + re.escape(term) + r'(?:ly)?\b'
                if re.search(pattern, bearing_str_lower):
                    # Parse the standard bearing format
                    return cls._parse_quadrant_bearing(bearing)
        
        return None
    
    @classmethod
    def _parse_quadrant_bearing(cls, bearing_str: str) -> Optional[float]:
        """Parse quadrant bearing formats (N 45°30' E, etc.)"""
        
        # Try DMS format: N 45°30'15" E
        match = cls.PATTERNS['quadrant_dms'].search(bearing_str)
        if match:
            ns, deg, min_val, sec, ew = match.groups()
            decimal_deg = cls._dms_to_decimal(float(deg), float(min_val), float(sec))
            return cls._quadrant_to_azimuth(decimal_deg, ns.upper(), ew.upper())
        
        # Try DM format: N 45°30' E
        match = cls.PATTERNS['quadrant_dm'].search(bearing_str)
        if match:
            ns, deg, min_val, ew = match.groups()
            decimal_deg = cls._dms_to_decimal(float(deg), float(min_val), 0)
            return cls._quadrant_to_azimuth(decimal_deg, ns.upper(), ew.upper())
        
        # Try D format: N 45° E
        match = cls.PATTERNS['quadrant_d'].search(bearing_str)
        if match:
            ns, deg, ew = match.groups()
            return cls._quadrant_to_azimuth(float(deg), ns.upper(), ew.upper())
        
        # Try dash format: N45-30-15E
        match = cls.PATTERNS['quadrant_dash'].search(bearing_str)
        if match:
            ns, deg, min_val, sec, ew = match.groups()
            decimal_deg = cls._dms_to_decimal(float(deg), float(min_val), float(sec))
            return cls._quadrant_to_azimuth(decimal_deg, ns.upper(), ew.upper())
        
        return None
    
    @classmethod
    def _parse_azimuth_bearing(cls, bearing_str: str) -> Optional[float]:
        """Parse azimuth bearing formats (123°45'30", 123.75°, etc.)"""
        
        # Try DMS format: 123°45'30"
        match = cls.PATTERNS['azimuth_dms'].search(bearing_str)
        if match:
            deg, min_val, sec = match.groups()
            return cls._dms_to_decimal(float(deg), float(min_val), float(sec))
        
        # Try DM format: 123°45'
        match = cls.PATTERNS['azimuth_dm'].search(bearing_str)
        if match:
            deg, min_val = match.groups()
            return cls._dms_to_decimal(float(deg), float(min_val), 0)
        
        # Try D format: 123.75°
        match = cls.PATTERNS['azimuth_d'].search(bearing_str)
        if match:
            deg = match.group(1)
            azimuth = float(deg)
            return azimuth if 0 <= azimuth <= 360 else None
        
        return None
    
    @staticmethod
    def _dms_to_decimal(degrees: float, minutes: float, seconds: float) -> float:
        """Convert degrees-minutes-seconds to decimal degrees"""
        return degrees + minutes/60.0 + seconds/3600.0
    
    @staticmethod
    def _quadrant_to_azimuth(bearing: float, ns: str, ew: str) -> float:
        """Convert quadrant bearing to azimuth (0-360°)"""
        if ns == 'N' and ew == 'E':
            # NE quadrant: azimuth = bearing
            return bearing
        elif ns == 'S' and ew == 'E':
            # SE quadrant: azimuth = 180 - bearing
            return 180.0 - bearing
        elif ns == 'S' and ew == 'W':
            # SW quadrant: azimuth = 180 + bearing
            return 180.0 + bearing
        elif ns == 'N' and ew == 'W':
            # NW quadrant: azimuth = 360 - bearing
            return 360.0 - bearing
        else:
            raise ValueError(f"Invalid quadrant: {ns}{ew}")
    
    @classmethod
    def azimuth_to_quadrant(cls, azimuth: float) -> str:
        """Convert azimuth to quadrant bearing format"""
        azimuth = azimuth % 360  # Normalize to 0-360
        
        if 0 <= azimuth <= 90:
            # NE quadrant
            bearing = azimuth
            return f"N {cls._decimal_to_dms(bearing)} E"
        elif 90 < azimuth <= 180:
            # SE quadrant
            bearing = 180 - azimuth
            return f"S {cls._decimal_to_dms(bearing)} E"
        elif 180 < azimuth <= 270:
            # SW quadrant
            bearing = azimuth - 180
            return f"S {cls._decimal_to_dms(bearing)} W"
        else:
            # NW quadrant
            bearing = 360 - azimuth
            return f"N {cls._decimal_to_dms(bearing)} W"
    
    @staticmethod
    def _decimal_to_dms(decimal_degrees: float) -> str:
        """Convert decimal degrees to DMS format"""
        degrees = int(decimal_degrees)
        minutes_float = (decimal_degrees - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        
        if seconds < 0.01:  # Less than 1/100 of a second
            if minutes == 0:
                return f"{degrees}°"
            else:
                return f"{degrees}°{minutes:02d}'"
        else:
            return f"{degrees}°{minutes:02d}'{seconds:05.2f}\""


class DistanceParser:
    """Parse distance values with units"""
    
    # Distance unit conversion factors to feet
    UNIT_CONVERSIONS = {
        'ft': 1.0,
        'feet': 1.0,
        'foot': 1.0,
        "'": 1.0,
        'm': 3.28084,
        'meter': 3.28084,
        'meters': 3.28084,
        'ch': 66.0,
        'chain': 66.0,
        'chains': 66.0,
        'rd': 16.5,
        'rod': 16.5,
        'rods': 16.5
    }
    
    DISTANCE_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*([a-zA-Z\']+)?',
        re.IGNORECASE
    )
    
    @classmethod
    def parse_distance(cls, distance_str: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse distance string and return (value, unit).
        
        Args:
            distance_str: Distance string like "120.5 ft", "36.58 m", "120'"
            
        Returns:
            Tuple of (distance_value, unit) or (None, None) if parsing fails
        """
        if not distance_str or not isinstance(distance_str, str):
            return None, None
            
        distance_str = distance_str.strip()
        match = cls.DISTANCE_PATTERN.search(distance_str)
        
        if match:
            value_str, unit_str = match.groups()
            value = float(value_str)
            
            # Normalize unit
            if unit_str:
                unit_str = unit_str.lower().strip()
                # Map common variations to standard units
                if unit_str in cls.UNIT_CONVERSIONS:
                    if unit_str in ['feet', 'foot', "'"]:
                        unit = 'ft'
                    elif unit_str in ['meter', 'meters']:
                        unit = 'm'
                    elif unit_str in ['chain', 'chains']:
                        unit = 'ch'
                    elif unit_str in ['rod', 'rods']:
                        unit = 'rd'
                    else:
                        unit = unit_str
                    return value, unit
            else:
                # No unit specified, assume feet (common in US deeds)
                return value, 'ft'
        
        return None, None
    
    @classmethod
    def convert_to_feet(cls, value: float, unit: str) -> float:
        """Convert distance value to feet"""
        unit = unit.lower()
        if unit in cls.UNIT_CONVERSIONS:
            return value * cls.UNIT_CONVERSIONS[unit]
        return value  # Assume feet if unknown unit
