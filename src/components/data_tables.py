"""
Enhanced data table components for better survey data visualization.
"""
import streamlit as st
import pandas as pd
from typing import List, Optional
from ..models.deed_models import SurveyCall, PolygonGeometry


def display_enhanced_calls_table(calls: List[SurveyCall], title: str = "📋 Survey Calls"):
    """Display enhanced survey calls table with better formatting"""
    st.subheader(title)
    
    if not calls:
        st.info("No survey calls to display")
        return
    
    # Prepare data with better formatting
    table_data = []
    for call in calls:
        # Format confidence with emoji indicators
        if call.confidence >= 0.8:
            conf_display = f"🟢 {call.confidence:.2f}"
            conf_color = "#d4edda"
        elif call.confidence >= 0.6:
            conf_display = f"🟡 {call.confidence:.2f}"
            conf_color = "#fff3cd"
        else:
            conf_display = f"🔴 {call.confidence:.2f}"
            conf_color = "#f8d7da"
        
        # Format distance with proper units
        if call.distance and call.distance_unit:
            unit_str = str(call.distance_unit) if call.distance_unit else "ft"
            if hasattr(call.distance_unit, 'value'):
                unit_str = call.distance_unit.value
            distance_display = f"{call.distance:.1f} {unit_str}"
        else:
            distance_display = "—"
        
        # Format bearing
        bearing_display = call.bearing if call.bearing else "—"
        
        # Format type with icon
        type_icons = {
            "line": "📏",
            "curve": "🌀", 
            "tie_line": "🔗",
            "tie_curve": "🔗🌀"
        }
        type_display = f"{type_icons.get(call.type, '📐')} {call.type.upper()}"
        
        table_data.append({
            "#": call.sequence,
            "Type": type_display,
            "Bearing": bearing_display,
            "Distance": distance_display,
            "Confidence": conf_display,
            "Raw Text": call.raw_text[:40] + "..." if len(call.raw_text) > 40 else call.raw_text
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display with custom styling
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Bearing": st.column_config.TextColumn("Bearing", width="medium"),
            "Distance": st.column_config.TextColumn("Distance", width="small"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            "Raw Text": st.column_config.TextColumn("Raw Text", width="large")
        }
    )


def display_geometry_summary(geometry: PolygonGeometry, tract_name: str = "Survey"):
    """Display geometry summary with key metrics"""
    st.subheader(f"📐 {tract_name} Geometry")
    
    if not geometry:
        st.info("No geometry data available")
        return
    
    # Key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Perimeter",
            f"{geometry.perimeter:.1f} ft",
            help="Total boundary length"
        )
        
        if geometry.area:
            acres = geometry.area / 43560
            st.metric(
                "Area", 
                f"{acres:.3f} acres",
                delta=f"{geometry.area:.0f} sq ft",
                help="Property area in acres and square feet"
            )
    
    with col2:
        # Closure quality with color coding
        closure_status = "Excellent" if geometry.closure_error <= 0.1 else "Good" if geometry.closure_error <= 0.5 else "Fair" if geometry.closure_error <= 1.0 else "Poor"
        
        st.metric(
            "Closure Error",
            f"{geometry.closure_error:.2f} ft",
            delta=closure_status,
            delta_color="normal" if closure_status in ["Excellent", "Good"] else "inverse"
        )
        
        st.metric(
            "Closure %",
            f"{geometry.closure_percentage:.2f}%",
            help="Closure error as percentage of perimeter"
        )
    
    with col3:
        st.metric(
            "Vertices",
            len(geometry.vertices),
            help="Number of polygon vertices"
        )
        
        # Shape complexity
        vertices_count = len(geometry.vertices)
        if vertices_count <= 4:
            complexity = "Simple"
        elif vertices_count <= 8:
            complexity = "Moderate"
        else:
            complexity = "Complex"
        
        st.metric(
            "Complexity",
            complexity,
            help="Shape complexity based on vertex count"
        )


def display_vertices_table(geometry: PolygonGeometry, show_all: bool = False):
    """Display vertices table with coordinates"""
    if not geometry or not geometry.vertices:
        st.info("No vertex data available")
        return
    
    vertices = geometry.vertices
    
    if show_all:
        st.subheader(f"📍 All Vertices ({len(vertices)} points)")
        display_vertices = vertices
    else:
        st.subheader("📍 Key Vertices")
        # Show POB, midpoint, and end point for large polygons
        if len(vertices) <= 6:
            display_vertices = vertices
        else:
            display_vertices = [
                vertices[0],  # POB
                vertices[len(vertices)//3],  # 1/3 point
                vertices[2*len(vertices)//3],  # 2/3 point
                vertices[-1] if len(vertices) > 1 else vertices[0]  # End point
            ]
    
    # Prepare vertex data
    vertices_data = []
    for i, vertex in enumerate(display_vertices):
        # Find original index
        original_index = vertices.index(vertex) if vertex in vertices else i
        
        point_name = "POB" if original_index == 0 else f"V{original_index}"
        
        # Calculate distance from POB
        if original_index > 0:
            pob = vertices[0]
            distance_from_pob = math.sqrt((vertex.x - pob.x)**2 + (vertex.y - pob.y)**2)
            dist_display = f"{distance_from_pob:.1f} ft"
        else:
            dist_display = "0.0 ft"
        
        vertices_data.append({
            "Point": point_name,
            "X Coordinate": f"{vertex.x:.2f}",
            "Y Coordinate": f"{vertex.y:.2f}",
            "Distance from POB": dist_display,
            "Description": vertex.description or "—"
        })
    
    # Display table
    df = pd.DataFrame(vertices_data)
    st.dataframe(df, hide_index=True)
    
    # Show all vertices toggle
    if not show_all and len(vertices) > 6:
        if st.button(f"Show All {len(vertices)} Vertices"):
            display_vertices_table(geometry, show_all=True)


def display_tract_comparison(tracts: List):
    """Display comparison table for multiple tracts"""
    if len(tracts) <= 1:
        return
    
    st.subheader("📊 Tract Comparison")
    
    comparison_data = []
    for tract in tracts:
        if tract.geometry:
            area_acres = tract.geometry.area / 43560 if tract.geometry.area else 0
            closure_status = "Good" if tract.geometry.closure_error <= 0.5 else "Fair" if tract.geometry.closure_error <= 1.0 else "Poor"
            
            comparison_data.append({
                "Tract": tract.tract_id,
                "Calls": len(tract.calls),
                "Perimeter": f"{tract.geometry.perimeter:.1f} ft",
                "Area": f"{area_acres:.3f} ac",
                "Closure": f"{tract.geometry.closure_error:.2f} ft",
                "Quality": closure_status,
                "Confidence": f"{tract.total_confidence:.2f}"
            })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, hide_index=True)


# Import math for distance calculations
import math
