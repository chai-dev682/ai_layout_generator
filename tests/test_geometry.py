"""
Tests for geometry calculation functionality.
"""
import pytest
import math
from src.geometry.calculator import GeometryCalculator
from src.models.deed_models import SurveyCall, ProjectSettings


class TestGeometryCalculator:
    """Test geometry calculation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.settings = ProjectSettings(pob_x=0.0, pob_y=0.0)
        self.calculator = GeometryCalculator(self.settings)
    
    def test_simple_square_calculation(self):
        """Test calculation of a simple square"""
        calls = [
            SurveyCall(sequence=1, type="line", raw_text="N 90° E 100.00 ft",
                      bearing="N 90° E", azimuth_deg=90.0, distance=100.0, distance_unit="ft", confidence=1.0),
            SurveyCall(sequence=2, type="line", raw_text="S 0° E 100.00 ft", 
                      bearing="S 0° E", azimuth_deg=180.0, distance=100.0, distance_unit="ft", confidence=1.0),
            SurveyCall(sequence=3, type="line", raw_text="S 90° W 100.00 ft",
                      bearing="S 90° W", azimuth_deg=270.0, distance=100.0, distance_unit="ft", confidence=1.0),
            SurveyCall(sequence=4, type="line", raw_text="N 0° W 100.00 ft",
                      bearing="N 0° W", azimuth_deg=0.0, distance=100.0, distance_unit="ft", confidence=1.0)
        ]
        
        geometry = self.calculator.calculate_polygon(calls)
        
        # Check vertices
        assert len(geometry.vertices) == 5  # POB + 4 corners
        
        # Check that we get back to near the starting point
        assert geometry.closure_error < 0.01  # Very small closure error
        
        # Check perimeter
        assert geometry.perimeter == pytest.approx(400.0, rel=1e-5)
        
        # Check area (should be 10,000 sq ft)
        assert geometry.area == pytest.approx(10000.0, rel=1e-3)
    
    def test_line_endpoint_calculation(self):
        """Test calculation of line endpoints"""
        # Test cardinal directions
        end_x, end_y = self.calculator._calculate_line_endpoint(0, 0, 0, 100)  # North
        assert end_x == pytest.approx(0.0, abs=1e-10)
        assert end_y == pytest.approx(100.0, rel=1e-5)
        
        end_x, end_y = self.calculator._calculate_line_endpoint(0, 0, 90, 100)  # East
        assert end_x == pytest.approx(100.0, rel=1e-5)
        assert end_y == pytest.approx(0.0, abs=1e-10)
        
        end_x, end_y = self.calculator._calculate_line_endpoint(0, 0, 180, 100)  # South
        assert end_x == pytest.approx(0.0, abs=1e-10)
        assert end_y == pytest.approx(-100.0, rel=1e-5)
        
        end_x, end_y = self.calculator._calculate_line_endpoint(0, 0, 270, 100)  # West
        assert end_x == pytest.approx(-100.0, rel=1e-5)
        assert end_y == pytest.approx(0.0, abs=1e-10)
    
    def test_diagonal_line_calculation(self):
        """Test calculation of diagonal lines"""
        # 45-degree line
        end_x, end_y = self.calculator._calculate_line_endpoint(0, 0, 45, 100)
        expected_component = 100 / math.sqrt(2)
        assert end_x == pytest.approx(expected_component, rel=1e-5)
        assert end_y == pytest.approx(expected_component, rel=1e-5)
    
    def test_curve_calculation_basic(self):
        """Test basic curve calculation"""
        # Simple curve with known parameters
        call = SurveyCall(
            sequence=1, type="curve", raw_text="curve right R=100, arc=50",
            radius=100.0, arc_length=50.0, chord_bearing="N 45° E", 
            chord_length=49.9, curve_direction="R", confidence=1.0
        )
        
        points = self.calculator._calculate_curve_points(0, 0, call, num_segments=8)
        
        # Should generate multiple points along the curve
        assert len(points) == 8
        
        # All points should be roughly the same distance from the calculated center
        # (This is a basic sanity check)
        for point in points:
            assert point.x is not None
            assert point.y is not None
    
    def test_closure_error_calculation(self):
        """Test closure error calculation"""
        from src.models.deed_models import GeometryPoint
        
        # Perfect square (should have zero closure error)
        vertices = [
            GeometryPoint(x=0, y=0, description="POB"),
            GeometryPoint(x=100, y=0, description="Corner 1"),
            GeometryPoint(x=100, y=100, description="Corner 2"),
            GeometryPoint(x=0, y=100, description="Corner 3"),
            GeometryPoint(x=0, y=0, description="Back to POB")  # Perfect closure
        ]
        
        closure_error = self.calculator._calculate_closure_error(vertices)
        assert closure_error == pytest.approx(0.0, abs=1e-10)
        
        # Square with small error
        vertices[-1] = GeometryPoint(x=1, y=1, description="Near POB")  # 1 ft off
        closure_error = self.calculator._calculate_closure_error(vertices)
        assert closure_error == pytest.approx(math.sqrt(2), rel=1e-5)  # sqrt(1^2 + 1^2)
    
    def test_area_calculation(self):
        """Test area calculation using shoelace formula"""
        from src.models.deed_models import GeometryPoint
        
        # Simple rectangle: 100x50
        vertices = [
            GeometryPoint(x=0, y=0, description="POB"),
            GeometryPoint(x=100, y=0, description="Corner 1"),
            GeometryPoint(x=100, y=50, description="Corner 2"),
            GeometryPoint(x=0, y=50, description="Corner 3")
        ]
        
        area = self.calculator._calculate_area(vertices)
        assert area == pytest.approx(5000.0, rel=1e-5)  # 100 * 50 = 5000
        
        # Triangle: base=100, height=50
        vertices = [
            GeometryPoint(x=0, y=0, description="POB"),
            GeometryPoint(x=100, y=0, description="Base end"),
            GeometryPoint(x=50, y=50, description="Apex")
        ]
        
        area = self.calculator._calculate_area(vertices)
        assert area == pytest.approx(2500.0, rel=1e-5)  # 0.5 * 100 * 50 = 2500
    
    def test_bounding_box_calculation(self):
        """Test bounding box calculation"""
        from src.models.deed_models import GeometryPoint
        
        vertices = [
            GeometryPoint(x=10, y=20, description="Point 1"),
            GeometryPoint(x=50, y=5, description="Point 2"),
            GeometryPoint(x=30, y=40, description="Point 3"),
            GeometryPoint(x=5, y=15, description="Point 4")
        ]
        
        min_x, min_y, max_x, max_y = self.calculator.get_bounding_box(vertices)
        
        assert min_x == 5
        assert min_y == 5
        assert max_x == 50
        assert max_y == 40
    
    def test_missing_data_handling(self):
        """Test handling of calls with missing data"""
        calls = [
            SurveyCall(sequence=1, type="line", raw_text="incomplete call",
                      bearing=None, azimuth_deg=None, distance=None, confidence=0.3)
        ]
        
        # Should not crash, but may produce warnings or default behavior
        with pytest.raises(ValueError):  # Should raise error for incomplete data
            geometry = self.calculator.calculate_polygon(calls)


if __name__ == "__main__":
    pytest.main([__file__])
