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
        
        # Add styles and interactive features
        self._add_styles(svg)
        self._add_interactive_controls(svg)
        
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
        
        # Draw POB information for each tract
        self._draw_tract_pobs(svg, tracts, transform)
        
        # Draw boundary coordinates for all tracts
        self._draw_multi_tract_boundary_coordinates(svg, tracts, transform)
        
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
        
        # Draw actual survey calls for this tract
        self._draw_tract_survey_calls(svg, vertices, calls, transform, color, tract.tract_id)
        
        # Draw measurements with tract-specific styling
        if self.show_measurements or self.show_bearings:
            self._draw_tract_measurements(svg, vertices, calls, transform, color, tract.tract_id)
    
    def _draw_tract_survey_calls(self, svg: Element, vertices: List, calls: List, 
                               transform: dict, color: str, tract_id: str) -> None:
        """Draw actual survey calls (boundary lines/curves) for a specific tract"""
        if len(vertices) < 2:
            return
        
        for i, call in enumerate(calls):
            if i >= len(vertices) - 1:
                break
            
            # Get start and end points for this survey call
            start_vertex = vertices[i]
            end_vertex = vertices[i + 1]
            
            start_x, start_y = self._transform_point(start_vertex.x, start_vertex.y, transform)
            end_x, end_y = self._transform_point(end_vertex.x, end_vertex.y, transform)
            
            if call.type in ['curve', 'tie_curve']:
                # Draw curve boundary for this tract
                self._draw_tract_curve_boundary(svg, start_x, start_y, end_x, end_y, call, color, tract_id)
            else:
                # Draw line boundary for this tract
                self._draw_tract_line_boundary(svg, start_x, start_y, end_x, end_y, call, color, tract_id)
    
    def _draw_tract_line_boundary(self, svg: Element, start_x: float, start_y: float, 
                                 end_x: float, end_y: float, call: SurveyCall, color: str, tract_id: str) -> None:
        """Draw a straight line boundary for a specific tract"""
        SubElement(svg, 'line', {
            'x1': f'{start_x:.2f}',
            'y1': f'{start_y:.2f}',
            'x2': f'{end_x:.2f}',
            'y2': f'{end_y:.2f}',
            'stroke': color,
            'stroke-width': '2',
            'data-tract-id': tract_id,
            'data-call-sequence': str(call.sequence),
            'data-call-type': call.type
        })
    
    def _draw_tract_curve_boundary(self, svg: Element, start_x: float, start_y: float, 
                                  end_x: float, end_y: float, call: SurveyCall, color: str, tract_id: str) -> None:
        """Draw a curved boundary for a specific tract"""
        if not call.radius or not call.chord_length:
            # Fallback to straight line
            self._draw_tract_line_boundary(svg, start_x, start_y, end_x, end_y, call, color, tract_id)
            return
        
        # Calculate curve parameters
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        dx = end_x - start_x
        dy = end_y - start_y
        chord_length = math.sqrt(dx**2 + dy**2)
        
        if chord_length > 0 and call.radius > chord_length / 2:
            # Calculate curve control point
            perp_x = -dy / chord_length
            perp_y = dx / chord_length
            
            curve_height = call.radius - math.sqrt(call.radius**2 - (chord_length/2)**2)
            direction = 1 if call.curve_direction == "R" else -1
            
            control_x = mid_x + perp_x * curve_height * direction
            control_y = mid_y + perp_y * curve_height * direction
            
            # Draw quadratic curve
            path_data = f'M {start_x:.2f} {start_y:.2f} Q {control_x:.2f} {control_y:.2f} {end_x:.2f} {end_y:.2f}'
            SubElement(svg, 'path', {
                'd': path_data,
                'fill': 'none',
                'stroke': color,
                'stroke-width': '2',
                'data-tract-id': tract_id,
                'data-call-sequence': str(call.sequence),
                'data-call-type': call.type
            })
        else:
            # Invalid curve, draw as line
            self._draw_tract_line_boundary(svg, start_x, start_y, end_x, end_y, call, color, tract_id)
    
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
    
    def _add_interactive_controls(self, svg: Element) -> None:
        """Add interactive zoom and pan controls to SVG"""
        # Add JavaScript for zoom and pan functionality
        script = SubElement(svg, 'script')
        script.text = '''
        let currentZoom = 1;
        let currentPanX = 0;
        let currentPanY = 0;
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        
        function initializeInteractivity() {
            const svg = document.querySelector('svg');
            const mainGroup = document.getElementById('main-group');
            
            if (!mainGroup) {
                // Create main group if it doesn't exist
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('id', 'main-group');
                
                // Move all children except script to the group
                const children = Array.from(svg.children);
                children.forEach(child => {
                    if (child.tagName !== 'script' && child.tagName !== 'style') {
                        g.appendChild(child);
                    }
                });
                svg.appendChild(g);
            }
            
            // Zoom functionality
            svg.addEventListener('wheel', function(e) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                currentZoom *= delta;
                currentZoom = Math.max(0.1, Math.min(10, currentZoom));
                updateTransform();
            });
            
            // Pan functionality
            svg.addEventListener('mousedown', function(e) {
                isDragging = true;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                svg.style.cursor = 'grabbing';
            });
            
            svg.addEventListener('mousemove', function(e) {
                if (isDragging) {
                    const deltaX = e.clientX - lastMouseX;
                    const deltaY = e.clientY - lastMouseY;
                    currentPanX += deltaX / currentZoom;
                    currentPanY += deltaY / currentZoom;
                    lastMouseX = e.clientX;
                    lastMouseY = e.clientY;
                    updateTransform();
                }
            });
            
            svg.addEventListener('mouseup', function() {
                isDragging = false;
                svg.style.cursor = 'grab';
            });
            
            svg.addEventListener('mouseleave', function() {
                isDragging = false;
                svg.style.cursor = 'default';
            });
            
            function updateTransform() {
                const mainGroup = document.getElementById('main-group');
                if (mainGroup) {
                    const transform = `translate(${currentPanX}, ${currentPanY}) scale(${currentZoom})`;
                    mainGroup.setAttribute('transform', transform);
                }
            }
            
            svg.style.cursor = 'grab';
        }
        
        function resetView() {
            currentZoom = 1;
            currentPanX = 0;
            currentPanY = 0;
            const mainGroup = document.getElementById('main-group');
            if (mainGroup) {
                mainGroup.setAttribute('transform', 'translate(0, 0) scale(1)');
            }
        }
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeInteractivity);
        } else {
            initializeInteractivity();
        }
        '''
    
    def _draw_tract_pobs(self, svg: Element, tracts: List, transform: dict) -> None:
        """Draw POB information for each tract"""
        pob_info_y = self.height - 150
        
        # Background for POB info
        SubElement(svg, 'rect', {
            'x': '10',
            'y': str(pob_info_y),
            'width': '320',
            'height': str(len(tracts) * 30 + 25),
            'class': 'info-box'
        })
        
        # Title
        SubElement(svg, 'text', {
            'x': '20',
            'y': str(pob_info_y + 18),
            'class': 'title',
            'font-size': '14px'
        }).text = 'Points of Beginning (POB)'
        
        # POB details for each tract
        for i, tract in enumerate(tracts):
            y_pos = pob_info_y + 40 + (i * 30)
            
            # Tract color indicator
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#7B68EE', '#32CD32', '#FF6347']
            color = colors[i % len(colors)]
            
            SubElement(svg, 'rect', {
                'x': '20',
                'y': str(y_pos - 10),
                'width': '10',
                'height': '10',
                'fill': color
            })
            
            # POB coordinates and description
            pob_text = f"{tract.tract_id}: ({tract.pob_x:.2f}, {tract.pob_y:.2f})"
            SubElement(svg, 'text', {
                'x': '35',
                'y': str(y_pos),
                'class': 'info',
                'font-size': '11px',
                'font-weight': 'bold'
            }).text = pob_text
            
            # POB description (truncated)
            desc_text = tract.pob_description[:40] + "..." if len(tract.pob_description) > 40 else tract.pob_description
            SubElement(svg, 'text', {
                'x': '35',
                'y': str(y_pos + 12),
                'class': 'info',
                'font-size': '9px',
                'fill': '#666666'
            }).text = desc_text
            
            # Add POB marker on the actual plot
            pob_svg_x, pob_svg_y = self._transform_point(tract.pob_x, tract.pob_y, transform)
            
            # POB marker circle (larger and more prominent)
            SubElement(svg, 'circle', {
                'cx': str(pob_svg_x),
                'cy': str(pob_svg_y),
                'r': '8',
                'fill': color,
                'stroke': 'white',
                'stroke-width': '3',
                'class': 'pob-marker'
            })
            
            # POB label with background
            label_bg = SubElement(svg, 'rect', {
                'x': str(pob_svg_x + 12),
                'y': str(pob_svg_y - 15),
                'width': f'{len(tract.tract_id) * 6 + 10}',
                'height': '16',
                'fill': 'white',
                'stroke': color,
                'stroke-width': '1',
                'rx': '3'
            })
            
            SubElement(svg, 'text', {
                'x': str(pob_svg_x + 17),
                'y': str(pob_svg_y - 5),
                'class': 'pob-label',
                'font-size': '10px',
                'font-weight': 'bold',
                'fill': color
            }).text = f"POB-{tract.tract_id}"
    
    def _draw_single_tract_pob(self, svg: Element, geometry: PolygonGeometry, transform: dict) -> None:
        """Draw POB information for single tract"""
        if not geometry.vertices:
            return
        
        # Get POB (first vertex)
        pob_vertex = geometry.vertices[0]
        pob_svg_x, pob_svg_y = self._transform_point(pob_vertex.x, pob_vertex.y, transform)
        
        # POB info box
        pob_info_y = self.height - 80
        SubElement(svg, 'rect', {
            'x': '10',
            'y': str(pob_info_y),
            'width': '280',
            'height': '60',
            'class': 'info-box'
        })
        
        # POB title
        SubElement(svg, 'text', {
            'x': '20',
            'y': str(pob_info_y + 18),
            'class': 'title',
            'font-size': '14px'
        }).text = 'Point of Beginning (POB)'
        
        # POB coordinates
        SubElement(svg, 'text', {
            'x': '20',
            'y': str(pob_info_y + 35),
            'class': 'info',
            'font-size': '12px',
            'font-weight': 'bold'
        }).text = f"Coordinates: ({pob_vertex.x:.2f}, {pob_vertex.y:.2f})"
        
        # POB description
        SubElement(svg, 'text', {
            'x': '20',
            'y': str(pob_info_y + 50),
            'class': 'info',
            'font-size': '10px',
            'fill': '#666666'
        }).text = pob_vertex.description or "Point of Beginning"
        
        # Enhanced POB marker
        SubElement(svg, 'circle', {
            'cx': str(pob_svg_x),
            'cy': str(pob_svg_y),
            'r': '10',
            'fill': '#F18F01',
            'stroke': 'white',
            'stroke-width': '4',
            'class': 'pob-marker-single'
        })
        
        # POB label with background
        SubElement(svg, 'rect', {
            'x': str(pob_svg_x + 15),
            'y': str(pob_svg_y - 18),
            'width': '35',
            'height': '18',
            'fill': 'white',
            'stroke': '#F18F01',
            'stroke-width': '2',
            'rx': '4'
        })
        
        SubElement(svg, 'text', {
            'x': str(pob_svg_x + 20),
            'y': str(pob_svg_y - 6),
            'class': 'pob-label',
            'font-size': '11px',
            'font-weight': 'bold',
            'fill': '#F18F01'
        }).text = 'POB'
    
    def _draw_boundary_coordinates(self, svg: Element, geometry: PolygonGeometry, transform: dict) -> None:
        """Draw coordinate labels for each boundary point in single tract view"""
        if not geometry or not geometry.vertices:
            return
        
        for i, vertex in enumerate(geometry.vertices):
            svg_x, svg_y = self._transform_point(vertex.x, vertex.y, transform)
            
            # Skip POB as it's already labeled
            if i == 0:
                continue
            
            # Small circle marker for boundary point
            SubElement(svg, 'circle', {
                'cx': str(svg_x),
                'cy': str(svg_y),
                'r': '3',
                'fill': '#007bff',
                'stroke': 'white',
                'stroke-width': '1',
                'class': 'boundary-point'
            })
            
            # Coordinate label background
            coord_text = f"({vertex.x:.1f}, {vertex.y:.1f})"
            text_width = len(coord_text) * 6 + 8
            
            # Position label to avoid overlap
            label_x = svg_x + 8
            label_y = svg_y - 8
            
            # Adjust if near edges
            if label_x + text_width > self.width - 10:
                label_x = svg_x - text_width - 8
            if label_y < 20:
                label_y = svg_y + 20
            
            SubElement(svg, 'rect', {
                'x': str(label_x),
                'y': str(label_y - 12),
                'width': str(text_width),
                'height': '16',
                'fill': 'rgba(255, 255, 255, 0.9)',
                'stroke': '#007bff',
                'stroke-width': '0.5',
                'rx': '2'
            })
            
            # Coordinate text
            SubElement(svg, 'text', {
                'x': str(label_x + 4),
                'y': str(label_y - 2),
                'font-size': '8px',
                'font-family': 'monospace',
                'fill': '#007bff',
                'font-weight': 'bold'
            }).text = coord_text
    
    def _draw_multi_tract_boundary_coordinates(self, svg: Element, tracts: List, transform: dict) -> None:
        """Draw coordinate labels for boundary points in multi-tract view"""
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#7B68EE', '#32CD32', '#FF6347']
        
        for tract_index, tract in enumerate(tracts):
            if not tract.geometry or not tract.geometry.vertices:
                continue
            
            color = colors[tract_index % len(colors)]
            
            for i, vertex in enumerate(tract.geometry.vertices):
                svg_x, svg_y = self._transform_point(vertex.x, vertex.y, transform)
                
                # Skip POB as it's already labeled with larger markers
                if i == 0:
                    continue
                
                # Small circle marker for boundary point
                SubElement(svg, 'circle', {
                    'cx': str(svg_x),
                    'cy': str(svg_y),
                    'r': '2.5',
                    'fill': color,
                    'stroke': 'white',
                    'stroke-width': '1',
                    'class': f'boundary-point-{tract_index}'
                })
                
                # Coordinate label (smaller for multi-tract to avoid clutter)
                coord_text = f"({vertex.x:.0f},{vertex.y:.0f})"
                text_width = len(coord_text) * 5 + 6
                
                # Position label to avoid overlap
                label_x = svg_x + 6
                label_y = svg_y - 6
                
                # Adjust if near edges
                if label_x + text_width > self.width - 10:
                    label_x = svg_x - text_width - 6
                if label_y < 15:
                    label_y = svg_y + 15
                
                SubElement(svg, 'rect', {
                    'x': str(label_x),
                    'y': str(label_y - 10),
                    'width': str(text_width),
                    'height': '12',
                    'fill': 'rgba(255, 255, 255, 0.9)',
                    'stroke': color,
                    'stroke-width': '0.5',
                    'rx': '1'
                })
                
                # Coordinate text
                SubElement(svg, 'text', {
                    'x': str(label_x + 3),
                    'y': str(label_y - 2),
                    'font-size': '7px',
                    'font-family': 'monospace',
                    'fill': color,
                    'font-weight': 'bold'
                }).text = coord_text
    
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
        
        # Add styles and interactive features
        self._add_styles(svg)
        self._add_interactive_controls(svg)
        
        # Add title
        title_elem = SubElement(svg, 'title')
        title_elem.text = title
        
        # Calculate coordinate transformation
        transform = self._calculate_transform(geometry.vertices)
        
        # Draw polygon
        self._draw_polygon(svg, geometry.vertices, transform)
        
        # Draw vertices
        self._draw_vertices(svg, geometry.vertices, transform)
        
        # Draw actual survey calls (boundary lines/curves)
        self._draw_survey_calls(svg, geometry.vertices, calls, transform)
        
        # Draw measurements and labels
        self._draw_measurements(svg, geometry.vertices, calls, transform)
        
        # Draw optional elements
        if self.show_grid:
            self._draw_grid(svg, transform)
            
        if self.show_north_arrow:
            self._draw_north_arrow(svg)
        
        if self.show_info_box:
            self._draw_info_box(svg, geometry, calls)
        
        # Draw POB information for single tract
        self._draw_single_tract_pob(svg, geometry, transform)
        
        # Draw boundary point coordinates
        self._draw_boundary_coordinates(svg, geometry, transform)
        
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
    
    def _draw_survey_calls(self, svg: Element, vertices: List[GeometryPoint], 
                          calls: List[SurveyCall], transform: dict) -> None:
        """Draw actual survey calls (lines and curves) that form property boundaries"""
        if len(vertices) < 2:
            return
        
        for i, call in enumerate(calls):
            if i >= len(vertices) - 1:
                break
            
            # Get start and end points for this survey call
            start_vertex = vertices[i]
            end_vertex = vertices[i + 1]
            
            start_x, start_y = self._transform_point(start_vertex.x, start_vertex.y, transform)
            end_x, end_y = self._transform_point(end_vertex.x, end_vertex.y, transform)
            
            if call.type in ['curve', 'tie_curve']:
                # Draw curve as an arc (approximation with path)
                self._draw_curve_boundary(svg, start_x, start_y, end_x, end_y, call)
            else:
                # Draw straight line boundary
                self._draw_line_boundary(svg, start_x, start_y, end_x, end_y, call)
    
    def _draw_line_boundary(self, svg: Element, start_x: float, start_y: float, 
                           end_x: float, end_y: float, call: SurveyCall) -> None:
        """Draw a straight line boundary"""
        SubElement(svg, 'line', {
            'x1': f'{start_x:.2f}',
            'y1': f'{start_y:.2f}',
            'x2': f'{end_x:.2f}',
            'y2': f'{end_y:.2f}',
            'class': 'boundary-line',
            'stroke': '#2E86AB',
            'stroke-width': '2',
            'data-call-sequence': str(call.sequence),
            'data-call-type': call.type
        })
    
    def _draw_curve_boundary(self, svg: Element, start_x: float, start_y: float, 
                           end_x: float, end_y: float, call: SurveyCall) -> None:
        """Draw a curved boundary using SVG arc"""
        if not call.radius or not call.chord_length:
            # Fallback to straight line if curve data is incomplete
            self._draw_line_boundary(svg, start_x, start_y, end_x, end_y, call)
            return
        
        # Calculate arc parameters for SVG path
        # For now, approximate with a quadratic curve
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Offset the midpoint based on curve direction and radius
        dx = end_x - start_x
        dy = end_y - start_y
        chord_length = math.sqrt(dx**2 + dy**2)
        
        if chord_length > 0:
            # Calculate perpendicular offset for curve
            perp_x = -dy / chord_length
            perp_y = dx / chord_length
            
            # Approximate curve height from radius and chord
            if call.radius > chord_length / 2:
                curve_height = call.radius - math.sqrt(call.radius**2 - (chord_length/2)**2)
                direction = 1 if call.curve_direction == "R" else -1
                
                control_x = mid_x + perp_x * curve_height * direction
                control_y = mid_y + perp_y * curve_height * direction
                
                # Draw quadratic curve
                path_data = f'M {start_x:.2f} {start_y:.2f} Q {control_x:.2f} {control_y:.2f} {end_x:.2f} {end_y:.2f}'
                SubElement(svg, 'path', {
                    'd': path_data,
                    'fill': 'none',
                    'class': 'boundary-curve',
                    'stroke': '#A23B72',
                    'stroke-width': '2',
                    'data-call-sequence': str(call.sequence),
                    'data-call-type': call.type
                })
            else:
                # Invalid curve, draw as line
                self._draw_line_boundary(svg, start_x, start_y, end_x, end_y, call)
        else:
            # Zero-length chord, draw as line
            self._draw_line_boundary(svg, start_x, start_y, end_x, end_y, call)
    
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
            
            # Note: Actual boundary lines/curves are drawn by _draw_survey_calls method
            # This method only handles measurement labels
            
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
