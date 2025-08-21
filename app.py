"""
Main Streamlit application for the Deed Parser system.
"""
import streamlit as st
import pandas as pd
import json
import os
from typing import List, Optional
import traceback

# Import our modules
from src.models.deed_models import SurveyCall, ProjectSettings, DeedParseResult, DistanceUnit, BearingConvention, Tract
from src.parsers.openai_parser import OpenAIDeedParser
from src.geometry.calculator import GeometryCalculator
from src.visualization.svg_generator import SVGGenerator
from src.utils.bearing_parser import BearingParser
from src.components.data_tables import display_enhanced_calls_table, display_geometry_summary, display_vertices_table, display_tract_comparison


def get_closure_status_color(closure_error: float) -> tuple[str, str]:
    """Get closure status and color based on error"""
    if closure_error <= 0.1:
        return "Excellent", "green"
    elif closure_error <= 0.5:
        return "Good", "green"
    elif closure_error <= 1.0:
        return "Fair", "orange"
    else:
        return "Poor", "red"


def generate_summary_report(geometry, calls: List[SurveyCall]) -> str:
    """Generate a text summary report"""
    import datetime
    
    report = []
    report.append("PROPERTY BOUNDARY SURVEY SUMMARY")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Geometry summary
    report.append("GEOMETRY SUMMARY:")
    report.append(f"  Total Calls: {len(calls)}")
    report.append(f"  Perimeter: {geometry.perimeter:.2f} ft")
    report.append(f"  Closure Error: {geometry.closure_error:.3f} ft ({geometry.closure_percentage:.3f}%)")
    
    if geometry.area:
        acres = geometry.area / 43560
        report.append(f"  Area: {geometry.area:.0f} sq ft ({acres:.3f} acres)")
    
    closure_status, _ = get_closure_status_color(geometry.closure_error)
    report.append(f"  Closure Quality: {closure_status}")
    report.append("")
    
    # Call details
    report.append("SURVEY CALLS:")
    report.append("-" * 30)
    
    for call in calls:
        report.append(f"Call {call.sequence}: {call.type.upper()}")
        report.append(f"  Raw Text: {call.raw_text}")
        
        if call.bearing:
            report.append(f"  Bearing: {call.bearing}")
        if call.azimuth_deg is not None:
            report.append(f"  Azimuth: {call.azimuth_deg:.2f}°")
        if call.distance:
            report.append(f"  Distance: {call.distance:.2f} {call.distance_unit}")
        
        # Curve details
        if call.type in ["curve", "tie_curve"]:
            if call.radius:
                report.append(f"  Radius: {call.radius:.2f} ft")
            if call.arc_length:
                report.append(f"  Arc Length: {call.arc_length:.2f} ft")
            if call.chord_length:
                report.append(f"  Chord Length: {call.chord_length:.2f} ft")
        
        report.append(f"  Confidence: {call.confidence:.2f}")
        if call.notes:
            report.append(f"  Notes: {call.notes}")
        report.append("")
    
    # Quality assessment
    report.append("QUALITY ASSESSMENT:")
    report.append("-" * 20)
    
    low_confidence_calls = [c for c in calls if c.confidence < 0.7]
    if low_confidence_calls:
        report.append(f"Low confidence calls: {len(low_confidence_calls)}")
        for call in low_confidence_calls:
            report.append(f"  - Call {call.sequence}: {call.confidence:.2f}")
    else:
        report.append("All calls have acceptable confidence levels")
    
    report.append("")
    report.append("End of Report")
    
    return "\n".join(report)


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Deed Parser System",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🗺️ Deed Parser System")
    st.markdown("Extract survey calls from legal descriptions and visualize property boundaries")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Input & Parsing", "📊 Review & Edit", "🗺️ Visualization"])
    
    with tab1:
        text_input_tab()
    
    with tab2:
        review_edit_tab()
    
    with tab3:
        visualization_tab()


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'settings' not in st.session_state:
        st.session_state.settings = ProjectSettings()
    
    if 'calls' not in st.session_state:
        st.session_state.calls = []
    
    if 'geometry' not in st.session_state:
        st.session_state.geometry = None
    
    if 'original_text' not in st.session_state:
        st.session_state.original_text = ""
    
    if 'parse_result' not in st.session_state:
        st.session_state.parse_result = None
    
    # Multi-tract support
    if 'tracts' not in st.session_state:
        st.session_state.tracts = []
    
    if 'selected_tract' not in st.session_state:
        st.session_state.selected_tract = 0
    
    if 'has_multiple_tracts' not in st.session_state:
        st.session_state.has_multiple_tracts = False


def setup_sidebar():
    """Setup sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # OpenAI Settings
    st.sidebar.subheader("OpenAI Settings")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.settings.openai_api_key or "",
        help="Enter your OpenAI API key. This is stored securely in your session."
    )
    
    # Model selection
    model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    selected_model = st.sidebar.selectbox(
        "OpenAI Model",
        model_options,
        index=model_options.index(st.session_state.settings.openai_model)
        if st.session_state.settings.openai_model in model_options else 0
    )
    
    # Project Settings
    st.sidebar.subheader("Project Settings")
    
    # Units
    units = st.sidebar.selectbox(
        "Distance Units",
        ["ft", "m"],
        index=0 if str(st.session_state.settings.units) == "ft" else 1
    )
    
    # Bearing convention
    bearing_convention = st.sidebar.selectbox(
        "Bearing Convention",
        ["quadrant", "azimuth"],
        index=0 if str(st.session_state.settings.bearing_convention) == "quadrant" else 1
    )
    
    # Point of Beginning
    st.sidebar.subheader("Point of Beginning")
    pob_x = st.sidebar.number_input("POB X Coordinate", value=st.session_state.settings.pob_x)
    pob_y = st.sidebar.number_input("POB Y Coordinate", value=st.session_state.settings.pob_y)
    pob_desc = st.sidebar.text_input("POB Description", value=st.session_state.settings.pob_description)
    
    # Quality thresholds
    st.sidebar.subheader("Quality Thresholds")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0, st.session_state.settings.confidence_threshold,
        help="Minimum confidence for auto-accepting parsed calls"
    )
    
    closure_tolerance = st.sidebar.number_input(
        "Closure Tolerance (ft)",
        value=st.session_state.settings.closure_tolerance,
        help="Maximum acceptable closure error"
    )
    
    # Update settings
    st.session_state.settings.openai_api_key = api_key
    st.session_state.settings.openai_model = selected_model
    st.session_state.settings.units = DistanceUnit.FEET if units == "ft" else DistanceUnit.METERS
    st.session_state.settings.bearing_convention = BearingConvention.QUADRANT if bearing_convention == "quadrant" else BearingConvention.AZIMUTH
    st.session_state.settings.pob_x = pob_x
    st.session_state.settings.pob_y = pob_y
    st.session_state.settings.pob_description = pob_desc
    st.session_state.settings.confidence_threshold = confidence_threshold
    st.session_state.settings.closure_tolerance = closure_tolerance


def text_input_tab():
    """Text input and parsing tab"""
    st.header("📝 Deed Text Input")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload PDF or text file",
        type=['pdf', 'txt'],
        help="Upload a PDF deed or text file containing the legal description"
    )
    
    # Text input
    deed_text = st.text_area(
        "Paste deed/legal description text:",
        height=300,
        value=st.session_state.original_text,
        placeholder="Paste your deed or legal description here..."
    )
    
    # Sample text buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Single Tract Sample"):
            st.session_state.use_multi_tract_sample = False
            sample_text = get_sample_deed_text()
            deed_text = sample_text
            st.session_state.original_text = sample_text
            st.rerun()
    
    with col2:
        if st.button("Load Multi-Tract Sample"):
            st.session_state.use_multi_tract_sample = True
            sample_text = get_sample_deed_text()
            deed_text = sample_text
            st.session_state.original_text = sample_text
            st.rerun()
    
    # Parse button
    col1, col2 = st.columns([1, 4])
    with col1:
        parse_button = st.button("🔍 Parse Deed", type="primary", disabled=not deed_text or not st.session_state.settings.openai_api_key)
    
    with col2:
        if not st.session_state.settings.openai_api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
        elif not deed_text:
            st.info("💡 Enter deed text above to parse")
    
    # Handle parsing
    if parse_button and deed_text and st.session_state.settings.openai_api_key:
        st.session_state.original_text = deed_text
        
        with st.spinner("🤖 Parsing deed with AI..."):
            try:
                # Initialize parser
                parser = OpenAIDeedParser(
                    api_key=st.session_state.settings.openai_api_key,
                    model=st.session_state.settings.openai_model
                )
                
                # Parse the text (now returns DeedParseResult)
                parse_result = parser.parse_deed_text(deed_text, st.session_state.settings)
                st.session_state.parse_result = parse_result
                
                # Update session state
                st.session_state.tracts = parse_result.tracts
                st.session_state.has_multiple_tracts = parse_result.has_multiple_tracts
                st.session_state.calls = parse_result.calls  # Legacy support
                st.session_state.geometry = parse_result.geometry  # Legacy support
                
                # Show success message
                total_calls = sum(len(tract.calls) for tract in parse_result.tracts)
                if parse_result.has_multiple_tracts:
                    st.success(f"✅ Successfully parsed {len(parse_result.tracts)} tracts with {total_calls} total survey calls!")
                    
                    # Show tract summary
                    for i, tract in enumerate(parse_result.tracts):
                        st.info(f"📋 **{tract.tract_id}**: {len(tract.calls)} calls | "
                               f"Confidence: {tract.total_confidence:.2f} | "
                               f"Closure error: {tract.geometry.closure_error:.3f} ft")
                else:
                    st.success(f"✅ Successfully parsed {total_calls} survey calls!")
                    
                    # Show quick summary
                    primary_tract = parse_result.primary_tract
                    if primary_tract:
                        st.info(f"📊 Average confidence: {primary_tract.total_confidence:.2f} | "
                               f"Closure error: {primary_tract.geometry.closure_error:.3f} ft")
                
            except Exception as e:
                st.error(f"❌ Parsing failed: {str(e)}")
                st.expander("Error Details").code(traceback.format_exc())
    
    # Show raw LLM output toggle
    if st.session_state.calls:
        if st.checkbox("Show Raw Parsing Details"):
            st.subheader("Parsed Calls Summary")
            for call in st.session_state.calls:
                with st.expander(f"Call {call.sequence}: {call.type} (confidence: {call.confidence:.2f})"):
                    st.code(call.raw_text)
                    col1, col2 = st.columns(2)
                    with col1:
                        if call.bearing:
                            st.write(f"**Bearing:** {call.bearing}")
                        if call.distance:
                            st.write(f"**Distance:** {call.distance} {call.distance_unit}")
                    with col2:
                        if call.azimuth_deg is not None:
                            st.write(f"**Azimuth:** {call.azimuth_deg:.2f}°")
                        if call.notes:
                            st.write(f"**Notes:** {call.notes}")


def review_edit_tab():
    """Review and edit calls tab with editable table"""
    st.header("📊 Review & Edit Survey Calls")
    
    if not st.session_state.tracts and not st.session_state.calls:
        st.info("💡 Parse deed text in the first tab to see editable calls here")
        return
    
    # Handle multiple tracts
    if st.session_state.has_multiple_tracts:
        # Tract selector
        tract_options = [f"{tract.tract_id} ({len(tract.calls)} calls)" for tract in st.session_state.tracts]
        selected_tract_idx = st.selectbox(
            "📋 Select Tract to Edit:",
            range(len(st.session_state.tracts)),
            format_func=lambda x: tract_options[x],
            index=st.session_state.selected_tract
        )
        st.session_state.selected_tract = selected_tract_idx
        
        current_tract = st.session_state.tracts[selected_tract_idx]
        current_calls = current_tract.calls
        
        # Show tract info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tract", current_tract.tract_id)
        with col2:
            st.metric("POB", f"({current_tract.pob_x:.1f}, {current_tract.pob_y:.1f})")
        with col3:
            st.metric("Calls", len(current_calls))
        
        st.text_area("Tract Description:", value=current_tract.description, height=60, disabled=True)
        st.text_area("POB Description:", value=current_tract.pob_description, height=40, disabled=True)
        
    else:
        current_calls = st.session_state.calls
        current_tract = st.session_state.tracts[0] if st.session_state.tracts else None
    
    # Convert calls to DataFrame for editing
    df_data = []
    for call in current_calls:
        df_data.append({
            'Sequence': call.sequence,
            'Type': call.type,
            'Raw Text': call.raw_text[:50] + "..." if len(call.raw_text) > 50 else call.raw_text,
            'Bearing': call.bearing or "",
            'Azimuth (°)': call.azimuth_deg or 0.0,
            'Distance': call.distance or 0.0,
            'Unit': call.distance_unit or "ft",
            'Radius': call.radius or 0.0,
            'Arc Length': call.arc_length or 0.0,
            'Chord Length': call.chord_length or 0.0,
            'Curve Dir': call.curve_direction or "",
            'Confidence': call.confidence,
            'Notes': call.notes or ""
        })
    
    df = pd.DataFrame(df_data)
    
    # Color coding for confidence
    def color_confidence(val):
        if val >= 0.8:
            return 'background-color: #d4edda'  # Green
        elif val >= 0.6:
            return 'background-color: #fff3cd'  # Yellow
        else:
            return 'background-color: #f8d7da'  # Red
    
    # Display editable table
    st.subheader("📋 Survey Calls Table")
    
    # Configure columns
    column_config = {
        'Sequence': st.column_config.NumberColumn('Seq', min_value=1, max_value=100, step=1),
        'Type': st.column_config.SelectboxColumn('Type', options=['line', 'curve', 'tie_line', 'tie_curve']),
        'Bearing': st.column_config.TextColumn('Bearing', help='e.g., N 45°30\' E'),
        'Azimuth (°)': st.column_config.NumberColumn('Azimuth (°)', min_value=0, max_value=360, format="%.2f"),
        'Distance': st.column_config.NumberColumn('Distance', min_value=0, format="%.2f"),
        'Unit': st.column_config.SelectboxColumn('Unit', options=['ft', 'm', 'ch', 'rd']),
        'Confidence': st.column_config.ProgressColumn('Confidence', min_value=0, max_value=1),
    }
    
    # Editable data editor
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic"  # Allow adding/deleting rows
    )
    
    # Update button
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("💾 Update Calls", type="primary"):
            update_calls_from_dataframe(edited_df)
            st.success("✅ Calls updated!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Recalculate Geometry"):
            recalculate_geometry()
            st.success("✅ Geometry recalculated!")
            st.rerun()
    
    # Show validation warnings
    if st.session_state.calls:
        show_validation_warnings()


def visualization_tab():
    """Visualization tab with SVG display"""
    st.header("🗺️ Property Boundary Visualization")
    
    if not st.session_state.tracts and not st.session_state.geometry:
        st.info("💡 Parse and review deed calls to see visualization here")
        return
    
    # Handle multiple tracts
    if st.session_state.has_multiple_tracts:
        # Tract selector for visualization
        col1, col2 = st.columns([3, 1])
        with col1:
            tract_options = [f"{tract.tract_id} - {tract.description[:50]}..." for tract in st.session_state.tracts]
            selected_viz_tract = st.selectbox(
                "🗺️ Select Tract to Visualize:",
                range(len(st.session_state.tracts)),
                format_func=lambda x: tract_options[x],
                index=st.session_state.selected_tract
            )
        with col2:
            show_all_tracts = st.checkbox("Show All Tracts", value=False, help="Overlay all tracts in one view")
        
        if show_all_tracts:
            current_tract = None  # Will show all tracts
            current_geometry = None
            current_calls = []
            for tract in st.session_state.tracts:
                current_calls.extend(tract.calls)
        else:
            current_tract = st.session_state.tracts[selected_viz_tract]
            current_geometry = current_tract.geometry
            current_calls = current_tract.calls
            
    else:
        current_tract = st.session_state.tracts[0] if st.session_state.tracts else None
        current_geometry = st.session_state.geometry
        current_calls = st.session_state.calls
        show_all_tracts = False
    
    # Visualization controls in sidebar
    with st.sidebar:
        st.subheader("🎨 Visualization Options")
        
        # Interactive mode toggle
        use_interactive = st.checkbox("🚀 Interactive Mode", value=True, 
                                    help="Enable interactive SVG with zoom and scale controls")
        
        if use_interactive:
            # Scale controls for interactive mode
            st.subheader("📏 Scale Controls")
            
            # Auto-calculate optimal scale
            if current_geometry and current_geometry.vertices:
                # Calculate rough optimal scale
                xs = [v.x for v in current_geometry.vertices]
                ys = [v.y for v in current_geometry.vertices]
                width_ft = max(xs) - min(xs) if xs else 100
                height_ft = max(ys) - min(ys) if ys else 100
                
                # Rough calculation for display area
                display_width = 600  # pixels
                display_height = 400  # pixels
                
                optimal_scale = max(width_ft / display_width, height_ft / display_height) if width_ft > 0 and height_ft > 0 else 1.0
                optimal_scale = max(0.1, min(20.0, optimal_scale))
            else:
                optimal_scale = 1.0
            
            feet_per_pixel = st.slider(
                "Scale (feet per pixel)", 
                min_value=0.1, 
                max_value=20.0, 
                value=optimal_scale,
                step=0.1,
                help="Adjust zoom level: lower values = more detail"
            )
            
            # Quick scale buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Fine\n(0.5)", help="High detail view"):
                    feet_per_pixel = 0.5
            with col2:
                if st.button("Normal\n(2.0)", help="Standard view"):
                    feet_per_pixel = 2.0
            with col3:
                if st.button("Coarse\n(5.0)", help="Overview"):
                    feet_per_pixel = 5.0
            with col4:
                if st.button("Auto\nFit", help="Optimal fit"):
                    feet_per_pixel = optimal_scale
            
            # Interactive options
            show_grid = st.checkbox("Show Coordinate Grid", value=True)
            show_scale_bar = st.checkbox("Show Scale Bar", value=True)
            
        else:
            # Basic SVG options
            show_grid = st.checkbox("Show Grid", value=False)
            show_scale_bar = False
            feet_per_pixel = 1.0
        
        # Common display options
        show_measurements = st.checkbox("Show Measurements", value=True)
        show_bearings = st.checkbox("Show Bearings", value=True)
        show_vertices = st.checkbox("Show Vertices", value=True)
        show_north_arrow = st.checkbox("Show North Arrow", value=True)
        show_info_box = st.checkbox("Show Info Box", value=True)
        
        # Style options
        color_scheme = st.selectbox(
            "Color Scheme",
            ["default", "dark", "high_contrast"],
            help="Choose visualization color scheme"
        )
        
        if not use_interactive:
            line_style = st.selectbox(
                "Line Style", 
                ["solid", "dashed", "dotted"],
                help="Style for survey lines"
            )
            
            # Size options for basic SVG
            svg_width = st.slider("Width", 400, 1200, 800, 50)
            svg_height = st.slider("Height", 300, 900, 600, 50)
        else:
            line_style = "solid"
            svg_width = 800
            svg_height = 600
    
    # Geometry statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Perimeter", f"{st.session_state.geometry.perimeter:.2f} ft")
    
    with col2:
        closure_error = st.session_state.geometry.closure_error
        closure_status, closure_color = get_closure_status_color(closure_error)
        st.metric(
            "Closure Error", 
            f"{closure_error:.3f} ft",
            delta=closure_status,
            delta_color="normal" if closure_status == "Excellent" else "inverse"
        )
    
    with col3:
        st.metric("Closure %", f"{st.session_state.geometry.closure_percentage:.3f}%")
    
    with col4:
        if st.session_state.geometry.area:
            acres = st.session_state.geometry.area / 43560
            st.metric("Area", f"{acres:.3f} acres")
    
    # SVG Generation and Display
    try:
        # Use enhanced SVG generator with scale control
        generator = SVGGenerator(width=svg_width, height=svg_height)
        
        # Set custom scale if in interactive mode
        if use_interactive:
            generator.feet_per_pixel = feet_per_pixel
        
        # Configure visualization options
        generator.configure(
            show_measurements=show_measurements,
            show_bearings=show_bearings,
            show_vertices=show_vertices,
            show_grid=show_grid,
            show_north_arrow=show_north_arrow,
            show_info_box=show_info_box,
            color_scheme=color_scheme,
            line_style=line_style
        )
        
        # Generate appropriate SVG based on tract selection
        if show_all_tracts and st.session_state.has_multiple_tracts:
            svg_content = generator.generate_multi_tract_svg(
                st.session_state.tracts,
                title="Multi-Tract Property Survey"
            )
        elif current_geometry:
            svg_content = generator.generate_svg(
                current_geometry,
                current_calls,
                title=f"Property Boundary Survey - {current_tract.tract_id if current_tract else 'Single Tract'}"
            )
        else:
            svg_content = '<svg><text x="50" y="50">No geometry data available</text></svg>'
        
        # Display SVG with enhanced styling
        if use_interactive:
            # Enhanced display with scale information
            st.components.v1.html(
                f'''
                <div style="text-align: center; background: {"#2C3E50" if color_scheme == "dark" else "#FFFFFF"}; padding: 20px; border-radius: 10px; border: 2px solid #007bff;">
                    <div style="margin-bottom: 10px; font-weight: bold; color: #007bff;">
                        📏 Scale: {feet_per_pixel:.2f} ft/px | Resolution: {1/feet_per_pixel:.1f} px/ft
                    </div>
                    {svg_content}
                </div>
                ''',
                height=svg_height + 150
            )
            
            # Interactive scale controls below SVG
            st.subheader("🔍 Scale Controls")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("🔍+ Fine Detail", help="0.5 ft/px - High detail"):
                    st.session_state.scale_override = 0.5
                    st.rerun()
            
            with col2:
                if st.button("📐 Normal", help="2.0 ft/px - Standard view"):
                    st.session_state.scale_override = 2.0
                    st.rerun()
            
            with col3:
                if st.button("🔍- Overview", help="5.0 ft/px - Wide view"):
                    st.session_state.scale_override = 5.0
                    st.rerun()
            
            with col4:
                if st.button("🎯 Auto Fit", help="Optimal scale"):
                    st.session_state.scale_override = None
                    st.rerun()
            
            with col5:
                if st.button("📊 Show Data", help="Toggle data tables"):
                    st.session_state.show_data_tables = not st.session_state.get('show_data_tables', False)
                    st.rerun()
            
            # Show data tables if requested
            if st.session_state.get('show_data_tables', False):
                show_enhanced_data_tables(current_geometry, current_calls, current_tract)
        
        else:
            # Basic display
            display_height = svg_height + 100
            st.components.v1.html(
                f'''
                <div style="text-align: center; background: {"#2C3E50" if color_scheme == "dark" else "#FFFFFF"}; padding: 20px; border-radius: 10px;">
                    {svg_content}
                </div>
                ''',
                height=display_height
            )
        
        # Export options
        st.subheader("📤 Export Options")
        
        # Generate timestamp for filenames
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                "📄 Download SVG",
                data=svg_content,
                file_name=f"survey_boundary_{timestamp}.svg",
                mime="image/svg+xml",
                help="Download as scalable vector graphics"
            )
        
        with col2:
            # Export calls as CSV
            df = pd.DataFrame([call.dict() for call in st.session_state.calls])
            csv = df.to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                data=csv,
                file_name=f"survey_calls_{timestamp}.csv",
                mime="text/csv",
                help="Download survey calls as CSV"
            )
        
        with col3:
            # Export as JSON
            export_data = {
                'settings': st.session_state.settings.dict(),
                'calls': [call.dict() for call in st.session_state.calls],
                'geometry': st.session_state.geometry.dict() if st.session_state.geometry else None,
                'export_timestamp': timestamp
            }
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                "📋 Download JSON",
                data=json_str,
                file_name=f"deed_parse_result_{timestamp}.json",
                mime="application/json",
                help="Download complete project data"
            )
        
        with col4:
            # Export summary report
            summary_report = generate_summary_report(st.session_state.geometry, st.session_state.calls)
            st.download_button(
                "📑 Summary Report",
                data=summary_report,
                file_name=f"survey_summary_{timestamp}.txt",
                mime="text/plain",
                help="Download text summary report"
            )
        
        # Quick preview section
        with st.expander("🔍 Quick Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Call Statistics")
                line_calls = len([c for c in st.session_state.calls if c.type in ["line", "tie_line"]])
                curve_calls = len([c for c in st.session_state.calls if c.type in ["curve", "tie_curve"]])
                avg_confidence = sum(c.confidence for c in st.session_state.calls) / len(st.session_state.calls)
                
                st.write(f"📏 **Line calls:** {line_calls}")
                st.write(f"🌀 **Curve calls:** {curve_calls}")
                st.write(f"🎯 **Average confidence:** {avg_confidence:.2f}")
                
            with col2:
                st.subheader("Geometry Analysis")
                if st.session_state.geometry.area:
                    area_acres = st.session_state.geometry.area / 43560
                    if area_acres < 0.1:
                        size_desc = "Small lot"
                    elif area_acres < 1.0:
                        size_desc = "Residential lot"
                    elif area_acres < 10:
                        size_desc = "Large property"
                    else:
                        size_desc = "Commercial/Agricultural"
                    
                    st.write(f"🏠 **Property type:** {size_desc}")
                
                closure_status, _ = get_closure_status_color(st.session_state.geometry.closure_error)
                st.write(f"✅ **Closure quality:** {closure_status}")
                
                # Shape analysis
                vertices_count = len(st.session_state.geometry.vertices)
                if vertices_count <= 4:
                    shape_desc = "Simple polygon"
                elif vertices_count <= 8:
                    shape_desc = "Complex polygon"
                else:
                    shape_desc = "Very complex polygon"
                st.write(f"📐 **Shape complexity:** {shape_desc}")
        
    except Exception as e:
        st.error(f"❌ Visualization failed: {str(e)}")
        st.expander("Error Details").code(traceback.format_exc())


def update_calls_from_dataframe(df: pd.DataFrame):
    """Update session state calls from edited DataFrame"""
    updated_calls = []
    
    for _, row in df.iterrows():
        # Find existing call or create new one
        existing_call = None
        for call in st.session_state.calls:
            if call.sequence == row['Sequence']:
                existing_call = call
                break
        
        if existing_call:
            # Update existing call
            existing_call.type = row['Type']
            existing_call.bearing = row['Bearing'] if row['Bearing'] else None
            existing_call.distance = row['Distance'] if row['Distance'] > 0 else None
            existing_call.distance_unit = row['Unit']
            existing_call.confidence = row['Confidence']
            existing_call.notes = row['Notes'] if row['Notes'] else None
            
            # Recalculate azimuth from bearing
            if existing_call.bearing:
                azimuth = BearingParser.parse_bearing(existing_call.bearing)
                existing_call.azimuth_deg = azimuth
            
            updated_calls.append(existing_call)
        else:
            # Create new call
            new_call = SurveyCall(
                sequence=int(row['Sequence']),
                type=row['Type'],
                raw_text=row['Raw Text'],
                bearing=row['Bearing'] if row['Bearing'] else None,
                distance=row['Distance'] if row['Distance'] > 0 else None,
                distance_unit=row['Unit'],
                confidence=row['Confidence']
            )
            
            # Calculate azimuth
            if new_call.bearing:
                azimuth = BearingParser.parse_bearing(new_call.bearing)
                new_call.azimuth_deg = azimuth
            
            updated_calls.append(new_call)
    
    st.session_state.calls = sorted(updated_calls, key=lambda x: x.sequence)


def recalculate_geometry():
    """Recalculate geometry from current calls"""
    if st.session_state.calls:
        calculator = GeometryCalculator(st.session_state.settings)
        st.session_state.geometry = calculator.calculate_polygon(st.session_state.calls)


def show_validation_warnings():
    """Show validation warnings for calls"""
    warnings = []
    
    for call in st.session_state.calls:
        if call.confidence < st.session_state.settings.confidence_threshold:
            warnings.append(f"Call {call.sequence}: Low confidence ({call.confidence:.2f})")
        
        if not call.bearing and call.type in ['line', 'tie_line']:
            warnings.append(f"Call {call.sequence}: Missing bearing")
        
        if not call.distance and call.type in ['line', 'tie_line']:
            warnings.append(f"Call {call.sequence}: Missing distance")
    
    if warnings:
        st.warning("⚠️ **Validation Warnings:**\n" + "\n".join(f"• {w}" for w in warnings))


def show_enhanced_data_tables(geometry, calls: List[SurveyCall], tract):
    """Show enhanced data tables for interactive mode"""
    st.subheader("📊 Enhanced Survey Data")
    
    # Use the enhanced components
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Enhanced calls table
        display_enhanced_calls_table(calls, "📋 Survey Calls")
        
        # Show tract comparison if multiple tracts
        if st.session_state.has_multiple_tracts:
            display_tract_comparison(st.session_state.tracts)
    
    with col2:
        # Geometry summary
        tract_name = tract.tract_id if tract else "Survey"
        display_geometry_summary(geometry, tract_name)
        
        # Vertices table
        display_vertices_table(geometry)


def get_sample_deed_text() -> str:
    """Return sample deed text for testing"""
    # Check if user wants multi-tract sample
    if st.session_state.get('use_multi_tract_sample', False):
        return """
        TRACT 1: Being a tract of land situated in Harris County, Texas, described as follows:
        
        BEGINNING at a concrete monument at the intersection of Main Street and Oak Avenue;
        THENCE N 0°00'00" E along Oak Avenue, 150.00 feet to an iron rod;
        THENCE N 90°00'00" E, 200.00 feet to an iron rod;
        THENCE S 0°00'00" E, 150.00 feet to a point on Main Street;
        THENCE S 90°00'00" W along Main Street, 200.00 feet to the POINT OF BEGINNING;
        CONTAINING 0.688 acres.
        
        TRACT 2: Being another tract of land adjacent to Tract 1, described as follows:
        
        BEGINNING at the northeast corner of Tract 1 described above;
        THENCE N 0°00'00" E, 100.00 feet to an iron rod;
        THENCE with a curve to the right having a radius of 150.00 feet, an arc length of 78.54 feet,
        chord bearing N 45°00'00" E, chord length 76.54 feet to an iron rod;
        THENCE S 45°00'00" E, 141.42 feet to the southeast corner of Tract 1;
        THENCE S 90°00'00" W along the north line of Tract 1, 200.00 feet to the POINT OF BEGINNING;
        CONTAINING 0.459 acres.
        
        PARCEL A: Being a small triangular parcel, described as follows:
        
        BEGINNING at a point 50 feet north of the northwest corner of Tract 1;
        THENCE N 45°00'00" E, 70.71 feet to an iron rod;
        THENCE S 45°00'00" E, 70.71 feet to the northwest corner of Tract 1;
        THENCE S 0°00'00" E along Oak Avenue, 50.00 feet to the POINT OF BEGINNING;
        CONTAINING 0.057 acres.
        """
    else:
        return """
        Beginning at a concrete monument found at the intersection of the north right-of-way line 
        of State Highway 123 and the east right-of-way line of County Road 456;
        
        THENCE N 15°30'45" E along said east right-of-way line, a distance of 125.50 feet to a 
        point for corner;
        
        THENCE with a curve to the right having a radius of 285.00 feet, an arc length of 42.15 feet,
        chord bearing N 19°45'30" E, chord length 42.08 feet to a point for corner;
        
        THENCE S 75°15'00" E, a distance of 200.00 feet to an iron rod found for corner;
        
        THENCE S 15°30'45" W, a distance of 150.25 feet to a point on the north right-of-way 
        line of said State Highway 123;
        
        THENCE N 75°15'00" W along said north right-of-way line, a distance of 225.00 feet to 
        the POINT OF BEGINNING, containing 0.75 acres, more or less.
        """


if __name__ == "__main__":
    main()
