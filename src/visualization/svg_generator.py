"""
SVG generation for survey polygon visualization.
Creates scalable vector graphics with measurement labels and call annotations.
"""
import math
from typing import List, Tuple, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from ..models.deed_models import SurveyCall, GeometryPoint, PolygonGeometry


class SVGGenerator:
    """Generate SVG visualizations of survey polygons"""
    
    def __init__(self, width: int = 800, height: int = 600, margin: int = 50):
        """
        Initialize SVG generator.
        
        Args:
            width: SVG canvas width in pixels
            height: SVG canvas height in pixels  
            margin: Margin around the polygon in pixels
        """
        self.width = width
        self.height = height
        self.margin = margin
        self.drawing_width = width - 2 * margin
        self.drawing_height = height - 2 * margin
        
        # Scale control
        self.feet_per_pixel = None  # Auto-calculate if None
        
        # Visualization options
        self.show_measurements = True
        self.show_bearings = True
        self.show_vertices = True
        self.show_grid = False
        self.show_north_arrow = True
        self.show_info_box = True
        self.line_style = "solid"  # solid, dashed, dotted
        self.color_scheme = "default"  # default, dark, high_contrast
    
    def configure(self, **options):
        """Configure visualization options"""
        for key, value in options.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def generate_multi_tract_svg(self, tracts: List, title: str = "Multi-Tract Survey") -> str:
        """
        Generate SVG visualization for multiple tracts.
        
        Args:
            tracts: List of Tract objects
            title: Title for the drawing
            
        Returns:
            SVG string
        """
        # Create root SVG element
        svg = Element('svg', {
            'width': str(self.width),
            'height': str(self.height),
            'viewBox': f'0 0 {self.width} {self.height}',
            'xmlns': 'http://www.w3.org/2000/svg'
        })
        
        # Add styles
        self._add_styles(svg)
        
        # Add title
        title_elem = SubElement(svg, 'title')
        title_elem.text = title
        
        # Collect all vertices for transformation calculation
        all_vertices = []
        for tract in tracts:
            if tract.geometry and tract.geometry.vertices:
                all_vertices.extend(tract.geometry.vertices)
        
        if not all_vertices:
            return '<svg><text x="50" y="50">No geometry data available</text></svg>'
        
        # Calculate coordinate transformation
        transform = self._calculate_transform(all_vertices)
        
        # Draw each tract with different colors
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#7B68EE', '#32CD32', '#FF6347']
        
        for i, tract in enumerate(tracts):
            if not tract.geometry or not tract.geometry.vertices:
                continue
                
            color = colors[i % len(colors)]
            self._draw_tract(svg, tract, transform, color, i)
        
        # Draw optional elements
        if self.show_grid:
            self._draw_grid(svg, transform)
            
        if self.show_north_arrow:
            self._draw_north_arrow(svg)
        
        if self.show_info_box:
            self._draw_multi_tract_info_box(svg, tracts)
        
        # Convert to string
        rough_string = tostring(svg, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def _draw_tract(self, svg: Element, tract, transform: dict, color: str, tract_index: int) -> None:
        """Draw a single tract with custom color"""
        vertices = tract.geometry.vertices
        calls = tract.calls
        
        # Draw polygon with custom color
        if len(vertices) >= 3:
            path_data = []
            first_x, first_y = self._transform_point(vertices[0].x, vertices[0].y, transform)
            path_data.append(f'M {first_x:.2f} {first_y:.2f}')
            
            for vertex in vertices[1:]:
                x, y = self._transform_point(vertex.x, vertex.y, transform)
                path_data.append(f'L {x:.2f} {y:.2f}')
            
            path_data.append('Z')
            
            # Custom polygon style
            SubElement(svg, 'path', {
                'd': ' '.join(path_data),
                'fill': f'{color}30',  # 30% opacity
                'stroke': color,
                'stroke-width': '2'
            })
        
        # Draw vertices with custom color
        if self.show_vertices:
            for i, vertex in enumerate(vertices):
                x, y = self._transform_point(vertex.x, vertex.y, transform)
                
                radius = 5 if i == 0 else 3
                circle = SubElement(svg, 'circle', {
                    'cx': str(x),
                    'cy': str(y),
                    'r': str(radius),
                    'fill': color,
                    'stroke': 'white',
                    'stroke-width': '1'
                })
                
                # Add tooltip
                title = SubElement(circle, 'title')
                if i == 0:
                    title.text = f"{tract.tract_id} - POB\n({vertex.x:.2f}, {vertex.y:.2f})\n{tract.pob_description}"
                else:
                    title.text = f"{tract.tract_id} - Vertex {i}\n({vertex.x:.2f}, {vertex.y:.2f})"
        
        # Draw measurements with tract-specific styling
        if self.show_measurements or self.show_bearings:
            self._draw_tract_measurements(svg, vertices, calls, transform, color, tract.tract_id)
    
    def _draw_tract_measurements(self, svg: Element, vertices: List, calls: List, 
                               transform: dict, color: str, tract_id: str) -> None:
        """Draw measurements for a specific tract"""
        if len(vertices) < 2:
            return
        
        for i, call in enumerate(calls):
            if i >= len(vertices) - 1:
                break
            
            start_vertex = vertices[i]
            end_vertex = vertices[i + 1]
            
            start_x, start_y = self._transform_point(start_vertex.x, start_vertex.y, transform)
            end_x, end_y = self._transform_point(end_vertex.x, end_vertex.y, transform)
            
            # Calculate label position
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                perp_x = -dy / length * 15
                perp_y = dx / length * 15
                
                label_x = mid_x + perp_x
                label_y = mid_y + perp_y
                
                # Distance label
                if self.show_measurements and call.distance:
                    distance_text = f"{call.distance:.1f} {call.distance_unit}"
                    SubElement(svg, 'text', {
                        'x': str(label_x),
                        'y': str(label_y),
                        'class': 'measurement',
                        'fill': color
                    }).text = distance_text
                
                # Tract ID label (small)
                if i == 0:  # Only on first call
                    SubElement(svg, 'text', {
                        'x': str(label_x),
                        'y': str(label_y - 20),
                        'class': 'call-label',
                        'fill': color
                    }).text = tract_id
    
    def _draw_multi_tract_info_box(self, svg: Element, tracts: List) -> None:
        """Draw information box for multiple tracts"""
        box_x = 10
        box_y = 10
        box_width = 250
        box_height = 30 + len(tracts) * 60
        
        # Background box
        SubElement(svg, 'rect', {
            'x': str(box_x),
            'y': str(box_y),
            'width': str(box_width),
            'height': str(box_height),
            'class': 'info-box'
        })
        
        # Title
        SubElement(svg, 'text', {
            'x': str(box_x + 10),
            'y': str(box_y + 20),
            'class': 'title'
        }).text = f'Multi-Tract Survey ({len(tracts)} tracts)'
        
        # Tract information
        y_offset = 40
        for i, tract in enumerate(tracts):
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#7B68EE', '#32CD32', '#FF6347']
            color = colors[i % len(colors)]
            
            # Tract name with color indicator
            SubElement(svg, 'rect', {
                'x': str(box_x + 10),
                'y': str(box_y + y_offset - 8),
                'width': '10',
                'height': '10',
                'fill': color
            })
            
            SubElement(svg, 'text', {
                'x': str(box_x + 25),
                'y': str(box_y + y_offset),
                'class': 'info'
            }).text = tract.tract_id
            
            # Tract stats
            if tract.geometry:
                SubElement(svg, 'text', {
                    'x': str(box_x + 15),
                    'y': str(box_y + y_offset + 15),
                    'class': 'info',
                    'font-size': '9px'
                }).text = f"Calls: {len(tract.calls)} | Closure: {tract.geometry.closure_error:.2f} ft"
                
                if tract.geometry.area:
                    acres = tract.geometry.area / 43560
                    SubElement(svg, 'text', {
                        'x': str(box_x + 15),
                        'y': str(box_y + y_offset + 28),
                        'class': 'info',
                        'font-size': '9px'
                    }).text = f"Area: {acres:.3f} acres"
            
            y_offset += 50
    
    def generate_svg(self, geometry: PolygonGeometry, calls: List[SurveyCall], 
                    title: str = "Survey Polygon") -> str:
        """
        Generate complete SVG visualization.
        
        Args:
            geometry: Polygon geometry with vertices
            calls: Survey calls for annotations
            title: Title for the drawing
            
        Returns:
            SVG string
        """
        # Create root SVG element
        svg = Element('svg', {
            'width': str(self.width),
            'height': str(self.height),
            'viewBox': f'0 0 {self.width} {self.height}',
            'xmlns': 'http://www.w3.org/2000/svg'
        })
        
        # Add styles
        self._add_styles(svg)
        
        # Add title
        title_elem = SubElement(svg, 'title')
        title_elem.text = title
        
        # Calculate coordinate transformation
        transform = self._calculate_transform(geometry.vertices)
        
        # Draw polygon
        self._draw_polygon(svg, geometry.vertices, transform)
        
        # Draw vertices
        self._draw_vertices(svg, geometry.vertices, transform)
        
        # Draw measurements and labels
        self._draw_measurements(svg, geometry.vertices, calls, transform)
        
        # Draw optional elements
        if self.show_grid:
            self._draw_grid(svg, transform)
            
        if self.show_north_arrow:
            self._draw_north_arrow(svg)
        
        if self.show_info_box:
            self._draw_info_box(svg, geometry, calls)
        
        # Convert to string
        rough_string = tostring(svg, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def _add_styles(self, svg: Element) -> None:
        """Add CSS styles to SVG"""
        style = SubElement(svg, 'style')
        
        # Get color scheme
        colors = self._get_color_scheme()
        
        # Build line style
        line_dash = {
            "solid": "none",
            "dashed": "5,5",
            "dotted": "2,2"
        }.get(self.line_style, "none")
        
        style.text = f"""
            .polygon {{ 
                fill: {colors['polygon_fill']}; 
                stroke: {colors['polygon_stroke']}; 
                stroke-width: 2; 
            }}
            .vertex {{ 
                fill: {colors['vertex_fill']}; 
                stroke: {colors['vertex_stroke']}; 
                stroke-width: 1; 
                cursor: pointer;
            }}
            .vertex:hover {{
                fill: {colors['vertex_hover']};
                stroke-width: 2;
            }}
            .pob {{ 
                fill: {colors['pob_fill']}; 
                stroke: {colors['pob_stroke']}; 
                stroke-width: 2; 
                cursor: pointer;
            }}
            .pob:hover {{
                fill: {colors['pob_hover']};
                stroke-width: 3;
            }}
            .line {{ 
                stroke: {colors['line_stroke']}; 
                stroke-width: 1; 
                stroke-dasharray: {line_dash}; 
            }}
            .curve {{ 
                stroke: {colors['curve_stroke']}; 
                stroke-width: 1; 
                fill: none;
            }}
            .measurement {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 10px; 
                fill: {colors['text_primary']}; 
                text-anchor: middle;
                dominant-baseline: middle;
            }}
            .bearing {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 9px; 
                fill: {colors['text_secondary']}; 
                text-anchor: middle;
                dominant-baseline: middle;
            }}
            .title {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 16px; 
                font-weight: bold; 
                fill: {colors['text_primary']}; 
            }}
            .info {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 11px; 
                fill: {colors['text_primary']}; 
            }}
            .north-arrow {{ 
                stroke: {colors['text_primary']}; 
                stroke-width: 2; 
                fill: {colors['text_primary']}; 
            }}
            .grid {{ 
                stroke: {colors['grid']}; 
                stroke-width: 0.5; 
            }}
            .info-box {{
                fill: {colors['info_bg']};
                stroke: {colors['info_border']};
                stroke-width: 1;
                rx: 5;
                ry: 5;
            }}
            .call-label {{
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 8px;
                fill: {colors['text_secondary']};
                text-anchor: middle;
            }}
        """
    
    def _get_color_scheme(self) -> dict:
        """Get colors for the selected color scheme"""
        schemes = {
            "default": {
                "polygon_fill": "rgba(135, 206, 235, 0.3)",
                "polygon_stroke": "#2E86AB",
                "vertex_fill": "#A23B72",
                "vertex_stroke": "white",
                "vertex_hover": "#D63384",
                "pob_fill": "#F18F01",
                "pob_stroke": "white", 
                "pob_hover": "#FD7E14",
                "line_stroke": "#2E86AB",
                "curve_stroke": "#A23B72",
                "text_primary": "#333333",
                "text_secondary": "#666666",
                "grid": "#DDDDDD",
                "info_bg": "rgba(255, 255, 255, 0.95)",
                "info_border": "#CCCCCC"
            },
            "dark": {
                "polygon_fill": "rgba(100, 150, 200, 0.4)",
                "polygon_stroke": "#4A90E2",
                "vertex_fill": "#E74C3C",
                "vertex_stroke": "#2C3E50",
                "vertex_hover": "#C0392B",
                "pob_fill": "#F39C12",
                "pob_stroke": "#2C3E50",
                "pob_hover": "#E67E22",
                "line_stroke": "#4A90E2",
                "curve_stroke": "#E74C3C",
                "text_primary": "#ECF0F1",
                "text_secondary": "#BDC3C7",
                "grid": "#34495E",
                "info_bg": "rgba(44, 62, 80, 0.95)",
                "info_border": "#7F8C8D"
            },
            "high_contrast": {
                "polygon_fill": "rgba(255, 255, 0, 0.2)",
                "polygon_stroke": "#000000",
                "vertex_fill": "#FF0000",
                "vertex_stroke": "#000000",
                "vertex_hover": "#CC0000",
                "pob_fill": "#0000FF",
                "pob_stroke": "#000000",
                "pob_hover": "#0000CC",
                "line_stroke": "#000000",
                "curve_stroke": "#FF0000",
                "text_primary": "#000000",
                "text_secondary": "#333333",
                "grid": "#999999",
                "info_bg": "rgba(255, 255, 255, 0.98)",
                "info_border": "#000000"
            }
        }
        return schemes.get(self.color_scheme, schemes["default"])
    
    def _draw_grid(self, svg: Element, transform: dict) -> None:
        """Draw coordinate grid"""
        # Calculate grid spacing
        scale = transform['scale']
        grid_spacing = 50  # pixels
        
        # Convert to data units
        data_spacing = grid_spacing / scale
        
        # Round to nice numbers
        if data_spacing < 1:
            data_spacing = 1
        elif data_spacing < 5:
            data_spacing = 5
        elif data_spacing < 10:
            data_spacing = 10
        elif data_spacing < 50:
            data_spacing = 50
        else:
            data_spacing = 100
        
        # Draw vertical lines
        min_x = transform.get('min_x', 0)
        max_x = transform.get('max_x', 100)
        
        x = int(min_x / data_spacing) * data_spacing
        while x <= max_x:
            svg_x, _ = self._transform_point(x, 0, transform)
            if 0 <= svg_x <= self.width:
                SubElement(svg, 'line', {
                    'x1': str(svg_x), 'y1': '0',
                    'x2': str(svg_x), 'y2': str(self.height),
                    'class': 'grid'
                })
            x += data_spacing
        
        # Draw horizontal lines
        min_y = transform.get('min_y', 0)
        max_y = transform.get('max_y', 100)
        
        y = int(min_y / data_spacing) * data_spacing
        while y <= max_y:
            _, svg_y = self._transform_point(0, y, transform)
            if 0 <= svg_y <= self.height:
                SubElement(svg, 'line', {
                    'x1': '0', 'y1': str(svg_y),
                    'x2': str(self.width), 'y2': str(svg_y),
                    'class': 'grid'
                })
            y += data_spacing
    
    def _calculate_transform(self, vertices: List[GeometryPoint]) -> dict:
        """Calculate coordinate transformation to fit polygon in SVG canvas"""
        if not vertices:
            return {'scale': 1.0, 'offset_x': 0, 'offset_y': 0, 'feet_per_pixel': 1.0}
        
        # Find bounding box
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Calculate center point
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Determine scale
        if self.feet_per_pixel is not None:
            # Use custom scale
            pixels_per_foot = 1.0 / self.feet_per_pixel
            scale = pixels_per_foot
        else:
            # Auto-calculate scale to fit in drawing area
            data_width = max_x - min_x
            data_height = max_y - min_y
            
            if data_width == 0 and data_height == 0:
                scale = 1.0
                self.feet_per_pixel = 1.0
            else:
                scale_x = self.drawing_width / data_width if data_width > 0 else float('inf')
                scale_y = self.drawing_height / data_height if data_height > 0 else float('inf')
                scale = min(scale_x, scale_y) * 0.8  # 80% to leave some padding
                self.feet_per_pixel = 1.0 / scale
        
        # Calculate offset to center the polygon
        svg_center_x = self.width / 2
        svg_center_y = self.height / 2
        
        offset_x = svg_center_x - center_x * scale
        offset_y = svg_center_y + center_y * scale  # Flip Y axis
        
        return {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'feet_per_pixel': self.feet_per_pixel or 1.0
        }
    
    def _transform_point(self, x: float, y: float, transform: dict) -> Tuple[float, float]:
        """Transform a point from data coordinates to SVG coordinates"""
        svg_x = x * transform['scale'] + transform['offset_x']
        svg_y = -y * transform['scale'] + transform['offset_y']  # Flip Y axis
        return svg_x, svg_y
    
    def _draw_polygon(self, svg: Element, vertices: List[GeometryPoint], transform: dict) -> None:
        """Draw the main polygon"""
        if len(vertices) < 3:
            return
        
        # Create path
        path_data = []
        
        # Move to first point
        first_x, first_y = self._transform_point(vertices[0].x, vertices[0].y, transform)
        path_data.append(f'M {first_x:.2f} {first_y:.2f}')
        
        # Line to subsequent points
        for vertex in vertices[1:]:
            x, y = self._transform_point(vertex.x, vertex.y, transform)
            path_data.append(f'L {x:.2f} {y:.2f}')
        
        # Close path
        path_data.append('Z')
        
        # Add path element
        SubElement(svg, 'path', {
            'd': ' '.join(path_data),
            'class': 'polygon'
        })
    
    def _draw_vertices(self, svg: Element, vertices: List[GeometryPoint], transform: dict) -> None:
        """Draw vertex points"""
        if not self.show_vertices:
            return
            
        for i, vertex in enumerate(vertices):
            x, y = self._transform_point(vertex.x, vertex.y, transform)
            
            # Different style for POB
            css_class = 'pob' if i == 0 else 'vertex'
            radius = 5 if i == 0 else 3
            
            # Add title for hover tooltip
            circle = SubElement(svg, 'circle', {
                'cx': str(x),
                'cy': str(y),
                'r': str(radius),
                'class': css_class
            })
            
            # Add tooltip
            title = SubElement(circle, 'title')
            if i == 0:
                title.text = f"Point of Beginning\n({vertex.x:.2f}, {vertex.y:.2f})"
            else:
                title.text = f"Vertex {i}\n({vertex.x:.2f}, {vertex.y:.2f})\n{vertex.description or ''}"
    
    def _draw_measurements(self, svg: Element, vertices: List[GeometryPoint], 
                         calls: List[SurveyCall], transform: dict) -> None:
        """Draw measurement labels and bearing annotations"""
        if len(vertices) < 2:
            return
        
        for i, call in enumerate(calls):
            if i >= len(vertices) - 1:
                break
            
            # Get line segment points
            start_vertex = vertices[i]
            end_vertex = vertices[i + 1]
            
            start_x, start_y = self._transform_point(start_vertex.x, start_vertex.y, transform)
            end_x, end_y = self._transform_point(end_vertex.x, end_vertex.y, transform)
            
            # Draw construction line (dashed)
            line_class = 'curve' if call.type in ['curve', 'tie_curve'] else 'line'
            SubElement(svg, 'line', {
                'x1': str(start_x),
                'y1': str(start_y),
                'x2': str(end_x),
                'y2': str(end_y),
                'class': line_class
            })
            
            # Calculate label position (midpoint offset)
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # Calculate perpendicular offset for label
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Perpendicular vector
                perp_x = -dy / length * 15  # 15 pixel offset
                perp_y = dx / length * 15
                
                label_x = mid_x + perp_x
                label_y = mid_y + perp_y
                
                # Distance label
                if self.show_measurements and call.distance:
                    # Handle both enum and string distance units
                    unit_str = str(call.distance_unit) if call.distance_unit else "ft"
                    if hasattr(call.distance_unit, 'value'):
                        unit_str = call.distance_unit.value
                    distance_text = f"{call.distance:.1f} {unit_str}"
                    SubElement(svg, 'text', {
                        'x': str(label_x),
                        'y': str(label_y),
                        'class': 'measurement'
                    }).text = distance_text
                
                # Bearing label (slightly offset)
                if self.show_bearings and call.bearing:
                    bearing_y = label_y + (12 if self.show_measurements else 0)
                    SubElement(svg, 'text', {
                        'x': str(label_x),
                        'y': str(bearing_y),
                        'class': 'bearing'
                    }).text = call.bearing
                
                # Call sequence number
                if self.show_vertices:
                    seq_y = label_y - (12 if self.show_measurements or self.show_bearings else 0)
                    SubElement(svg, 'text', {
                        'x': str(label_x),
                        'y': str(seq_y),
                        'class': 'call-label'
                    }).text = f"#{call.sequence}"
    
    def _draw_north_arrow(self, svg: Element) -> None:
        """Draw north arrow in top right corner"""
        arrow_x = self.width - 60
        arrow_y = 40
        arrow_size = 20
        
        # Arrow group
        g = SubElement(svg, 'g', {'class': 'north-arrow'})
        
        # Arrow shaft
        SubElement(g, 'line', {
            'x1': str(arrow_x),
            'y1': str(arrow_y + arrow_size),
            'x2': str(arrow_x),
            'y2': str(arrow_y)
        })
        
        # Arrow head
        SubElement(g, 'polygon', {
            'points': f'{arrow_x},{arrow_y} {arrow_x-5},{arrow_y+10} {arrow_x+5},{arrow_y+10}'
        })
        
        # North label
        SubElement(g, 'text', {
            'x': str(arrow_x),
            'y': str(arrow_y + arrow_size + 15),
            'class': 'bearing',
            'text-anchor': 'middle'
        }).text = 'N'
    
    def _draw_info_box(self, svg: Element, geometry: PolygonGeometry, calls: List[SurveyCall]) -> None:
        """Draw information box with polygon statistics"""
        box_x = 10
        box_y = 10
        box_width = 200
        box_height = 120
        
        # Background box
        SubElement(svg, 'rect', {
            'x': str(box_x),
            'y': str(box_y),
            'width': str(box_width),
            'height': str(box_height),
            'class': 'info-box'
        })
        
        # Title
        SubElement(svg, 'text', {
            'x': str(box_x + 10),
            'y': str(box_y + 20),
            'class': 'title'
        }).text = 'Survey Information'
        
        # Statistics
        info_lines = [
            f"Calls: {len(calls)}",
            f"Perimeter: {geometry.perimeter:.1f} ft",
            f"Closure Error: {geometry.closure_error:.2f} ft",
            f"Closure %: {geometry.closure_percentage:.2f}%"
        ]
        
        # Add scale information
        if hasattr(self, 'feet_per_pixel') and self.feet_per_pixel:
            info_lines.append(f"Scale: {self.feet_per_pixel:.2f} ft/px")
        
        if geometry.area:
            acres = geometry.area / 43560
            info_lines.append(f"Area: {acres:.3f} acres")
        
        for i, line in enumerate(info_lines):
            SubElement(svg, 'text', {
                'x': str(box_x + 10),
                'y': str(box_y + 40 + i * 14),
                'class': 'info'
            }).text = line
    
    def generate_simple_svg(self, vertices: List[GeometryPoint]) -> str:
        """Generate a simple SVG with just the polygon (for quick preview)"""
        if not vertices:
            return '<svg></svg>'
        
        svg = Element('svg', {
            'width': str(self.width),
            'height': str(self.height),
            'viewBox': f'0 0 {self.width} {self.height}',
            'xmlns': 'http://www.w3.org/2000/svg'
        })
        
        # Simple styles
        style = SubElement(svg, 'style')
        style.text = ".polygon { fill: rgba(135, 206, 235, 0.3); stroke: #2E86AB; stroke-width: 2; }"
        
        # Calculate transform and draw polygon
        transform = self._calculate_transform(vertices)
        self._draw_polygon(svg, vertices, transform)
        
        rough_string = tostring(svg, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


def create_sample_svg():
    """Create a sample SVG for testing"""
    # Sample vertices for a rectangle
    vertices = [
        GeometryPoint(x=0, y=0, description="POB"),
        GeometryPoint(x=100, y=0, description="Corner 1"),
        GeometryPoint(x=100, y=50, description="Corner 2"),
        GeometryPoint(x=0, y=50, description="Corner 3")
    ]
    
    # Sample calls
    calls = [
        SurveyCall(sequence=1, type="line", raw_text="N 90° E 100.00 ft", 
                  bearing="N 90° E", azimuth_deg=90, distance=100, distance_unit="ft", confidence=1.0),
        SurveyCall(sequence=2, type="line", raw_text="S 0° E 50.00 ft",
                  bearing="S 0° E", azimuth_deg=180, distance=50, distance_unit="ft", confidence=1.0),
        SurveyCall(sequence=3, type="line", raw_text="S 90° W 100.00 ft",
                  bearing="S 90° W", azimuth_deg=270, distance=100, distance_unit="ft", confidence=1.0),
        SurveyCall(sequence=4, type="line", raw_text="N 0° W 50.00 ft",
                  bearing="N 0° W", azimuth_deg=0, distance=50, distance_unit="ft", confidence=1.0)
    ]
    
    # Create geometry
    from ..geometry.calculator import GeometryCalculator
    from ..models.deed_models import ProjectSettings
    
    settings = ProjectSettings()
    calculator = GeometryCalculator(settings)
    geometry = calculator.calculate_polygon(calls)
    
    # Generate SVG
    generator = SVGGenerator()
    return generator.generate_svg(geometry, calls, "Sample Rectangle")


if __name__ == "__main__":
    print(create_sample_svg())
