"""
Geometry calculations for survey lines and curves.
Converts bearings/distances to coordinates and generates polygon geometry.
"""
import math
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import transform
import numpy as np

from ..models.deed_models import SurveyCall, GeometryPoint, PolygonGeometry, ProjectSettings


class GeometryCalculator:
    """Calculate coordinates and geometry from survey calls"""
    
    def __init__(self, settings: ProjectSettings):
        """
        Initialize calculator with project settings.
        
        Args:
            settings: Project settings including POB and units
        """
        self.settings = settings
        self.pob_x = settings.pob_x
        self.pob_y = settings.pob_y
    
    def calculate_polygon(self, calls: List[SurveyCall]) -> PolygonGeometry:
        """
        Calculate polygon geometry from survey calls.
        
        Args:
            calls: List of survey calls in sequence
            
        Returns:
            PolygonGeometry with vertices and closure information
        """
        vertices = self._calculate_vertices(calls)
        
        if len(vertices) < 3:
            raise ValueError("Need at least 3 vertices to form a polygon")
        
        # Calculate closure
        closure_error = self._calculate_closure_error(vertices)
        perimeter = self._calculate_perimeter(calls)
        closure_percentage = (closure_error / perimeter * 100) if perimeter > 0 else 0
        
        # Calculate area if polygon is reasonably closed
        area = None
        if closure_error < self.settings.closure_tolerance:
            area = self._calculate_area(vertices)
        
        return PolygonGeometry(
            vertices=vertices,
            closure_error=closure_error,
            closure_percentage=closure_percentage,
            perimeter=perimeter,
            area=area
        )
    
    def _calculate_vertices(self, calls: List[SurveyCall]) -> List[GeometryPoint]:
        """Calculate all vertices from survey calls"""
        vertices = [GeometryPoint(x=self.pob_x, y=self.pob_y, description="Point of Beginning")]
        
        current_x, current_y = self.pob_x, self.pob_y
        
        for call in calls:
            try:
                if call.type in ["line", "tie_line"]:
                    # Check if we have required data before calculating
                    if call.azimuth_deg is None or call.distance is None:
                        print(f"Warning: Skipping call {call.sequence} - missing data - azimuth: {call.azimuth_deg}, distance: {call.distance}")
                        print(f"Raw text: {call.raw_text}")
                        # Skip this call completely - don't add unnecessary vertices
                        continue
                    
                    next_x, next_y = self._calculate_line_endpoint(
                        current_x, current_y, call.azimuth_deg, call.distance
                    )
                    vertices.append(GeometryPoint(
                        x=next_x, 
                        y=next_y, 
                        description=f"Line {call.sequence} endpoint"
                    ))
                    current_x, current_y = next_x, next_y
                    
                elif call.type in ["curve", "tie_curve"]:
                    # For polygon boundary, we only need the curve endpoint, not intermediate points
                    curve_endpoint = self._calculate_curve_endpoint(
                        current_x, current_y, call
                    )
                    if curve_endpoint:
                        vertices.append(curve_endpoint)
                        current_x, current_y = curve_endpoint.x, curve_endpoint.y
                        
            except Exception as e:
                print(f"Error processing call {call.sequence}: {e}")
                print(f"Call data - type: {call.type}, azimuth: {call.azimuth_deg}, distance: {call.distance}")
                # Skip this call - don't add unnecessary vertices for errors
                continue
        
        # Validate vertex count
        expected_vertices = len([call for call in calls if call.azimuth_deg is not None and call.distance is not None]) + 1  # +1 for POB
        actual_vertices = len(vertices)
        
        print(f"Vertex count validation:")
        print(f"  Valid calls: {expected_vertices - 1}")
        print(f"  Expected vertices (including POB): {expected_vertices}")
        print(f"  Actual vertices: {actual_vertices}")
        
        if actual_vertices != expected_vertices:
            print(f"  WARNING: Vertex count mismatch!")
        else:
            print(f"  ✓ Vertex count is correct")
        
        return vertices
    
    def _calculate_line_endpoint(self, start_x: float, start_y: float, 
                               azimuth_deg: Optional[float], distance: Optional[float]) -> Tuple[float, float]:
        """Calculate endpoint of a line segment"""
        if azimuth_deg is None or distance is None:
            raise ValueError("Line requires azimuth and distance")
        
        # Convert azimuth to radians (azimuth is clockwise from North)
        azimuth_rad = math.radians(azimuth_deg)
        
        # Calculate deltas (x = East, y = North)
        dx = distance * math.sin(azimuth_rad)
        dy = distance * math.cos(azimuth_rad)
        
        return start_x + dx, start_y + dy
    
    def _calculate_curve_endpoint(self, start_x: float, start_y: float, 
                                call: SurveyCall) -> Optional[GeometryPoint]:
        """Calculate only the endpoint of a curve for polygon boundary"""
        if not call.radius or not call.chord_length:
            # Fallback to straight line if curve data incomplete
            if call.chord_bearing and call.chord_length:
                from ..utils.bearing_parser import BearingParser
                chord_azimuth = BearingParser().parse_bearing(call.chord_bearing)
                if chord_azimuth is not None:
                    end_x, end_y = self._calculate_line_endpoint(
                        start_x, start_y, chord_azimuth, call.chord_length
                    )
                    return GeometryPoint(x=end_x, y=end_y, description=f"Curve {call.sequence} endpoint (as line)")
            return None
        
        # Use chord to calculate endpoint directly
        if call.chord_bearing and call.chord_length:
            from ..utils.bearing_parser import BearingParser
            chord_azimuth = BearingParser().parse_bearing(call.chord_bearing)
            if chord_azimuth is not None:
                end_x, end_y = self._calculate_line_endpoint(
                    start_x, start_y, chord_azimuth, call.chord_length
                )
                return GeometryPoint(x=end_x, y=end_y, description=f"Curve {call.sequence} endpoint")
        
        return None
    
    def _calculate_curve_points(self, start_x: float, start_y: float, 
                              call: SurveyCall, num_segments: int = 16) -> List[GeometryPoint]:
        """Calculate points along a curve"""
        if not call.radius or not call.chord_length:
            # Fallback to straight line if curve data incomplete
            if call.chord_bearing and call.chord_length:
                from ..utils.bearing_parser import BearingParser
                chord_azimuth = BearingParser.parse_bearing(call.chord_bearing)
                if chord_azimuth is not None:
                    end_x, end_y = self._calculate_line_endpoint(
                        start_x, start_y, chord_azimuth, call.chord_length
                    )
                    return [GeometryPoint(x=end_x, y=end_y, description=f"Curve {call.sequence} (as line)")]
            return []
        
        # Calculate curve geometry
        radius = call.radius
        chord_length = call.chord_length
        
        # Calculate central angle from chord and radius
        if chord_length > 2 * radius:
            # Invalid geometry, use chord as straight line
            if call.chord_bearing:
                from ..utils.bearing_parser import BearingParser
                chord_azimuth = BearingParser.parse_bearing(call.chord_bearing)
                if chord_azimuth is not None:
                    end_x, end_y = self._calculate_line_endpoint(
                        start_x, start_y, chord_azimuth, chord_length
                    )
                    return [GeometryPoint(x=end_x, y=end_y, description=f"Curve {call.sequence} (invalid)")]
            return []
        
        central_angle = 2 * math.asin(chord_length / (2 * radius))
        
        # Determine curve direction and calculate center
        direction_multiplier = -1 if call.curve_direction == "L" else 1
        
        # Get chord bearing
        chord_azimuth = None
        if call.chord_bearing:
            from ..utils.bearing_parser import BearingParser
            chord_azimuth = BearingParser.parse_bearing(call.chord_bearing)
        
        if chord_azimuth is None:
            return []
        
        # Calculate chord midpoint
        chord_azimuth_rad = math.radians(chord_azimuth)
        mid_x = start_x + (chord_length / 2) * math.sin(chord_azimuth_rad)
        mid_y = start_y + (chord_length / 2) * math.cos(chord_azimuth_rad)
        
        # Calculate perpendicular distance to center
        perp_distance = math.sqrt(radius**2 - (chord_length/2)**2)
        
        # Calculate center point (perpendicular to chord)
        perp_azimuth_rad = chord_azimuth_rad + direction_multiplier * math.pi/2
        center_x = mid_x + perp_distance * math.sin(perp_azimuth_rad)
        center_y = mid_y + perp_distance * math.cos(perp_azimuth_rad)
        
        # Calculate start and end angles relative to center
        start_angle = math.atan2(start_y - center_y, start_x - center_x)
        
        # Generate points along the arc
        points = []
        for i in range(1, num_segments + 1):
            t = i / num_segments
            angle = start_angle + direction_multiplier * central_angle * t
            
            point_x = center_x + radius * math.cos(angle)
            point_y = center_y + radius * math.sin(angle)
            
            points.append(GeometryPoint(
                x=point_x, 
                y=point_y, 
                description=f"Curve {call.sequence} point {i}"
            ))
        
        return points
    
    def _calculate_closure_error(self, vertices: List[GeometryPoint]) -> float:
        """Calculate closure error (distance from last point to POB)"""
        if len(vertices) < 2:
            return 0.0
        
        last_vertex = vertices[-1]
        pob = vertices[0]
        
        dx = last_vertex.x - pob.x
        dy = last_vertex.y - pob.y
        
        return math.sqrt(dx**2 + dy**2)
    
    def _calculate_perimeter(self, calls: List[SurveyCall]) -> float:
        """Calculate total perimeter from calls"""
        perimeter = 0.0
        
        for call in calls:
            if call.distance:
                perimeter += call.distance
            elif call.arc_length:  # For curves
                perimeter += call.arc_length
            elif call.chord_length:  # Fallback for curves
                perimeter += call.chord_length
        
        return perimeter
    
    def _calculate_area(self, vertices: List[GeometryPoint]) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(vertices) < 3:
            return 0.0
        
        # Create coordinate arrays
        coords = [(v.x, v.y) for v in vertices]
        
        # Add POB at end to close polygon
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        # Shoelace formula
        n = len(coords)
        area = 0.0
        
        for i in range(n - 1):
            area += coords[i][0] * coords[i + 1][1]
            area -= coords[i + 1][0] * coords[i][1]
        
        return abs(area) / 2.0
    
    def create_shapely_polygon(self, vertices: List[GeometryPoint]) -> Polygon:
        """Create a Shapely polygon from vertices"""
        coords = [(v.x, v.y) for v in vertices]
        
        # Close the polygon if not already closed
        if len(coords) > 2 and coords[0] != coords[-1]:
            coords.append(coords[0])
        
        return Polygon(coords)
    
    def get_bounding_box(self, vertices: List[GeometryPoint]) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y) of vertices"""
        if not vertices:
            return 0, 0, 0, 0
        
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        
        return min(xs), min(ys), max(xs), max(ys)


class CoordinateTransformer:
    """Handle coordinate system transformations"""
    
    def __init__(self, source_crs: str = "EPSG:4326", target_crs: str = "EPSG:3857"):
        """
        Initialize transformer.
        
        Args:
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
        """
        self.source_crs = source_crs
        self.target_crs = target_crs
    
    def transform_vertices(self, vertices: List[GeometryPoint]) -> List[GeometryPoint]:
        """Transform vertices between coordinate systems"""
        # This would require pyproj for full implementation
        # For now, return vertices unchanged (local coordinate system)
        return vertices
    
    def to_geographic(self, vertices: List[GeometryPoint]) -> List[GeometryPoint]:
        """Convert to geographic coordinates (lat/lon)"""
        # Placeholder - would implement with pyproj
        return vertices
