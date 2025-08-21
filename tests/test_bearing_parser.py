"""
Tests for bearing parser utilities.
"""
import pytest
from src.utils.bearing_parser import BearingParser, DistanceParser


class TestBearingParser:
    """Test bearing parsing functionality"""
    
    def test_quadrant_dms_format(self):
        """Test degrees-minutes-seconds quadrant format"""
        parser = BearingParser()
        
        # Test all quadrants
        assert parser.parse_bearing("N 45°30'15\" E") == pytest.approx(45.504167, rel=1e-5)
        assert parser.parse_bearing("S 30°15'30\" E") == pytest.approx(149.7417, rel=1e-5)
        assert parser.parse_bearing("S 60°45'00\" W") == pytest.approx(240.75, rel=1e-5)
        assert parser.parse_bearing("N 15°30'45\" W") == pytest.approx(344.4875, rel=1e-5)
    
    def test_quadrant_dm_format(self):
        """Test degrees-minutes quadrant format"""
        parser = BearingParser()
        
        assert parser.parse_bearing("N 45°30' E") == pytest.approx(45.5, rel=1e-5)
        assert parser.parse_bearing("S 30°15' W") == pytest.approx(210.25, rel=1e-5)
    
    def test_quadrant_d_format(self):
        """Test degrees-only quadrant format"""
        parser = BearingParser()
        
        assert parser.parse_bearing("N 45° E") == pytest.approx(45.0, rel=1e-5)
        assert parser.parse_bearing("S 90° W") == pytest.approx(270.0, rel=1e-5)
    
    def test_dash_format(self):
        """Test dash-separated format"""
        parser = BearingParser()
        
        assert parser.parse_bearing("N45-30-15E") == pytest.approx(45.504167, rel=1e-5)
        assert parser.parse_bearing("S30-15-30W") == pytest.approx(210.258333, rel=1e-5)
    
    def test_azimuth_formats(self):
        """Test azimuth formats"""
        parser = BearingParser()
        
        assert parser.parse_bearing("45.5°") == pytest.approx(45.5, rel=1e-5)
        assert parser.parse_bearing("123°30'15\"") == pytest.approx(123.504167, rel=1e-5)
        assert parser.parse_bearing("270") == pytest.approx(270.0, rel=1e-5)
    
    def test_invalid_bearings(self):
        """Test invalid bearing inputs"""
        parser = BearingParser()
        
        assert parser.parse_bearing("") is None
        assert parser.parse_bearing(None) is None
        assert parser.parse_bearing("invalid") is None
        assert parser.parse_bearing("N 400° E") is None  # Invalid degrees
    
    def test_azimuth_to_quadrant_conversion(self):
        """Test conversion from azimuth back to quadrant"""
        parser = BearingParser()
        
        # Test each quadrant
        assert "N" in parser.azimuth_to_quadrant(45.0) and "E" in parser.azimuth_to_quadrant(45.0)
        assert "S" in parser.azimuth_to_quadrant(135.0) and "E" in parser.azimuth_to_quadrant(135.0)
        assert "S" in parser.azimuth_to_quadrant(225.0) and "W" in parser.azimuth_to_quadrant(225.0)
        assert "N" in parser.azimuth_to_quadrant(315.0) and "W" in parser.azimuth_to_quadrant(315.0)
    
    def test_round_trip_conversion(self):
        """Test that parsing and converting back gives similar results"""
        parser = BearingParser()
        
        original = "N 45°30'15\" E"
        azimuth = parser.parse_bearing(original)
        quadrant = parser.azimuth_to_quadrant(azimuth)
        azimuth2 = parser.parse_bearing(quadrant)
        
        assert abs(azimuth - azimuth2) < 0.01  # Should be very close


class TestDistanceParser:
    """Test distance parsing functionality"""
    
    def test_basic_distance_parsing(self):
        """Test basic distance parsing"""
        parser = DistanceParser()
        
        # Test with units
        value, unit = parser.parse_distance("120.5 ft")
        assert value == 120.5
        assert unit == "ft"
        
        value, unit = parser.parse_distance("36.58 m")
        assert value == 36.58
        assert unit == "m"
    
    def test_distance_without_units(self):
        """Test distance parsing without explicit units"""
        parser = DistanceParser()
        
        value, unit = parser.parse_distance("120.5")
        assert value == 120.5
        assert unit == "ft"  # Default to feet
    
    def test_various_unit_formats(self):
        """Test various unit format variations"""
        parser = DistanceParser()
        
        # Feet variations
        for unit_str in ["ft", "feet", "foot", "'"]:
            value, unit = parser.parse_distance(f"120 {unit_str}")
            assert value == 120
            assert unit == "ft"
        
        # Meter variations  
        for unit_str in ["m", "meter", "meters"]:
            value, unit = parser.parse_distance(f"36.5 {unit_str}")
            assert value == 36.5
            assert unit == "m"
    
    def test_unit_conversions(self):
        """Test unit conversion factors"""
        parser = DistanceParser()
        
        # Test conversion to feet
        assert parser.convert_to_feet(1.0, "m") == pytest.approx(3.28084, rel=1e-5)
        assert parser.convert_to_feet(1.0, "ch") == pytest.approx(66.0, rel=1e-5)
        assert parser.convert_to_feet(1.0, "rd") == pytest.approx(16.5, rel=1e-5)
        assert parser.convert_to_feet(1.0, "ft") == pytest.approx(1.0, rel=1e-5)
    
    def test_invalid_distance_inputs(self):
        """Test invalid distance inputs"""
        parser = DistanceParser()
        
        assert parser.parse_distance("") == (None, None)
        assert parser.parse_distance(None) == (None, None)
        assert parser.parse_distance("invalid") == (None, None)


if __name__ == "__main__":
    pytest.main([__file__])
