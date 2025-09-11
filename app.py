"""
SmartLandPlanner - AI-Powered Subdivision Layout System
Main Streamlit application for comprehensive subdivision planning.
"""
import streamlit as st
import pandas as pd
import json
import os
import logging
from typing import List, Optional
import traceback
import tempfile
from io import BytesIO
import base64

# Additional imports for PNG export and site planning
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Import platform configuration for cross-platform compatibility
from src.utils.platform_config import platform_config, configure_for_deployment
from google import genai
from google.genai import types
import matplotlib.pyplot as plt

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deed_parser.log')
    ]
)

logger = logging.getLogger(__name__)

# Configure platform-specific settings for deployment
try:
    deployment_ready = configure_for_deployment()
    if not deployment_ready:
        logger.warning("Some platform dependencies may not be properly configured")
except Exception as e:
    logger.error(f"Failed to configure platform dependencies: {e}")
logger.propagate = False  # Prevent duplicate logging

# Import our modules
from src.models.deed_models import SurveyCall, ProjectSettings, DeedParseResult, Tract
from src.parsers.openai_parser import OpenAIDeedParser
from src.parsers.pdf_extractor import PDFExtractor, PDFExtractionMethod
from src.parsers.pdf_boundary_parser import PDFBoundaryParser
from src.geometry.calculator import GeometryCalculator
from src.utils.cost_calculator import CostCalculator, format_cost_display
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
        page_title="SmartLandPlanner - AI Subdivision Planning",
        page_icon="🏘️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏘️ SmartLandPlanner")
    st.markdown("**AI-Powered Subdivision Planning System** - Transform property boundaries into comprehensive subdivision layouts")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content tabs - Complete subdivision planning workflow
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Property Boundary Extraction", "📝 Deed Text Processing", "📊 Review & Edit Data", "🗺️ Boundary Visualization", "🏘️ AI Site Planning"])
    
    with tab1:
        pdf_input_tab()  # Property Boundary Extraction
    
    with tab2:
        text_input_tab()  # Deed Text Processing
    
    with tab3:
        review_edit_tab()  # Review & Edit Data
    
    with tab4:
        visualization_tab()  # Boundary Visualization
    
    with tab5:
        site_planning_tab()  # AI Site Planning


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
    
    # PDF processing state
    if 'pdf_extractor' not in st.session_state:
        st.session_state.pdf_extractor = None
    
    if 'pdf_boundary_parser' not in st.session_state:
        st.session_state.pdf_boundary_parser = None
    
    if 'pdf_extraction_result' not in st.session_state:
        st.session_state.pdf_extraction_result = None
    
    if 'boundary_extraction_result' not in st.session_state:
        st.session_state.boundary_extraction_result = None
    
    if 'processing_step' not in st.session_state:
        st.session_state.processing_step = "upload"  # upload, extract, boundary, format, review
    
    # Site planning state
    if 'site_planning_step' not in st.session_state:
        st.session_state.site_planning_step = 0  # 0-3 for the 4-step workflow
    
    if 'boundary_png_path' not in st.session_state:
        st.session_state.boundary_png_path = None
    
    if 'pixel_per_foot' not in st.session_state:
        st.session_state.pixel_per_foot = 2.0
    
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None
    
    if 'site_boundary_image' not in st.session_state:
        st.session_state.site_boundary_image = None
    
    if 'road_network_image' not in st.session_state:
        st.session_state.road_network_image = None
    
    if 'final_site_plan' not in st.session_state:
        st.session_state.final_site_plan = None
    
    if 'site_analysis_results' not in st.session_state:
        st.session_state.site_analysis_results = None
    
    if 'selected_road_type' not in st.session_state:
        st.session_state.selected_road_type = 'grid'


def setup_sidebar():
    """Setup sidebar with configuration options"""
    st.sidebar.header("⚙️ SmartLandPlanner Configuration")
    
    # OpenAI Settings
    st.sidebar.subheader("OpenAI Settings")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.settings.openai_api_key or "",
        help="Enter your OpenAI API key. This is stored securely in your session."
    )
    
    # PDF Processing API Keys
    st.sidebar.subheader("PDF Processing APIs")
    
    llamaindex_key = st.sidebar.text_input(
        "LlamaIndex API Key",
        type="password",
        value=st.session_state.get('llamaindex_api_key', ""),
        help="For premium OCR with LlamaExtract"
    )
    
    # Gemini API Key for Site Planning
    st.sidebar.subheader("AI Site Planning")
    gemini_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.get('gemini_api_key', ""),
        help="For AI-powered site planning and road network generation"
    )
    
    # Google Document AI settings
    with st.sidebar.expander("🔧 Google Document AI Setup", expanded=False):
        google_project_id = st.text_input(
            "Google Cloud Project ID",
            value=st.session_state.get('google_project_id', ""),
            help="Your Google Cloud Project ID"
        )
        
        google_processor_id = st.text_input(
            "Document AI Processor ID", 
            value=st.session_state.get('google_processor_id', ""),
            help="Document AI processor ID for PDF processing"
        )
        
        google_location = st.selectbox(
            "Google Cloud Location",
            ["us", "eu", "asia"],
            index=0,
            help="Google Cloud region for Document AI"
        )
        
        st.markdown("**Setup Instructions:**")
        st.markdown("1. Create Google Cloud Project")
        st.markdown("2. Enable Document AI API")
        st.markdown("3. Create Document AI Processor")
        st.markdown("4. Set up authentication (service account)")
    
    # Store API keys in session state
    if llamaindex_key:
        st.session_state.llamaindex_api_key = llamaindex_key
    if gemini_key:
        st.session_state.gemini_api_key = gemini_key
        # Initialize Gemini client if key is provided
        if not st.session_state.gemini_client:
            try:
                st.session_state.gemini_client = genai.Client(api_key=gemini_key)
            except Exception as e:
                st.sidebar.error(f"Failed to initialize Gemini client: {str(e)}")
    if google_project_id:
        st.session_state.google_project_id = google_project_id
    if google_processor_id:
        st.session_state.google_processor_id = google_processor_id
    st.session_state.google_location = google_location
    
    # Model selection with categories
    st.sidebar.markdown("### 🤖 OpenAI Model Selection")
    
    model_categories = {
        "Standard GPT Models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "New GPT Models": ["gpt-4.1", "gpt-4.1-mini", "gpt-5"],
        "O1-Series Reasoning": ["o1-preview", "o1-mini"],
        "New O-Series Reasoning": ["o3", "o3-pro", "o4-mini"]
    }
    
    # Flatten for selectbox
    all_models = []
    for category, models in model_categories.items():
        all_models.extend(models)
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        all_models,
        index=all_models.index(st.session_state.settings.openai_model)
        if st.session_state.settings.openai_model in all_models else 0,
        help="Reasoning models (o1-preview, o1-mini, o3, o3-pro, o4-mini) use advanced reasoning but don't support function calling"
    )
    
    # Show model info
    reasoning_models = ["o1-preview", "o1-mini", "o3", "o3-pro", "o4-mini"]
    new_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "o3", "o3-pro", "o4-mini"]
    no_temperature_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "o1-preview", "o1-mini", "o3", "o3-pro", "o4-mini"]
    
    if selected_model in reasoning_models:
        st.sidebar.info("🧠 Reasoning model selected - Uses advanced reasoning capabilities")
        st.sidebar.warning("⚠️ Limitations: No system messages, no function calling, no custom temperature")
        if selected_model in ["o3", "o3-pro", "o4-mini"]:
            st.sidebar.success("✨ New reasoning model - Enhanced capabilities!")
    else:
        st.sidebar.info("⚡ Standard model selected - Uses function calling for structured output")
        if selected_model in no_temperature_models and selected_model not in reasoning_models:
            st.sidebar.warning("⚠️ Note: Uses default temperature (no custom temperature support)")
        if selected_model in new_models:
            st.sidebar.success("🚀 New GPT model - Latest capabilities!")
    
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
    st.session_state.settings.units = units
    st.session_state.settings.bearing_convention = bearing_convention
    st.session_state.settings.pob_x = pob_x
    st.session_state.settings.pob_y = pob_y
    st.session_state.settings.pob_description = pob_desc
    st.session_state.settings.confidence_threshold = confidence_threshold
    st.session_state.settings.closure_tolerance = closure_tolerance
    
    # Cost Analysis Section
    if st.session_state.get('pdf_extraction_result') or st.session_state.get('boundary_extraction_result'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Cost Analysis")
        
        # Initialize cost calculator
        if 'cost_calculator' not in st.session_state:
            st.session_state.cost_calculator = CostCalculator()
        
        calculator = st.session_state.cost_calculator
        
        # Show current session costs
        total_session_cost = 0.0
        
        if st.session_state.get('pdf_extraction_result'):
            extraction_cost = st.session_state.pdf_extraction_result.cost_estimate
            total_session_cost += extraction_cost
            st.sidebar.metric("PDF Extraction", format_cost_display(extraction_cost))
        
        if st.session_state.get('boundary_extraction_result'):
            boundary_cost = st.session_state.boundary_extraction_result.cost_estimate
            total_session_cost += boundary_cost
            st.sidebar.metric("AI Processing", format_cost_display(boundary_cost))
        
        st.sidebar.metric("Session Total", format_cost_display(total_session_cost), 
                         help="Total cost for current PDF processing session")
        
        # Cost optimization tips
        with st.sidebar.expander("💡 Cost Tips", expanded=False):
            if total_session_cost == 0:
                st.write("✅ You're using free methods!")
            else:
                st.write("💡 **Optimization Tips:**")
                st.write("• Use PDFMiner+Tesseract for simple PDFs")
                st.write("• Try GPT-4o-mini for 85% cost savings")
                st.write("• Process multiple PDFs in batches")
                
                # Monthly estimation
                if st.checkbox("📊 Monthly Cost Estimate"):
                    pdfs_per_month = st.number_input("PDFs per month:", min_value=1, max_value=1000, value=10)
                    
                    if st.session_state.get('pdf_extraction_result'):
                        pages = st.session_state.pdf_extraction_result.page_count
                        method = st.session_state.pdf_extraction_result.method.value
                        
                        monthly_estimate = calculator.estimate_monthly_cost(
                            pdfs_per_month=pdfs_per_month,
                            avg_pages_per_pdf=pages,
                            extraction_method=method,
                            openai_model=st.session_state.settings.openai_model
                        )
                        
                        st.write(f"**Monthly:** {format_cost_display(monthly_estimate['monthly_cost'])}")
                        st.write(f"**Yearly:** {format_cost_display(monthly_estimate['yearly_cost'])}")


def pdf_input_tab():
    """Property boundary extraction from PDFs and images"""
    st.header("📄 Property Boundary Extraction")
    st.markdown("**Extract property boundaries from PDFs and site plan images** using AI-powered analysis")
    
    # Progress indicator
    steps = ["Upload", "Extract", "Boundary", "Format", "Review", "Visualize"]
    current_step = st.session_state.processing_step
    
    # Create progress bar
    step_index = steps.index(current_step.title()) if current_step.title() in steps else 0
    progress = (step_index) / (len(steps) - 1)
    st.progress(progress)
    
    # Step indicator
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i <= step_index:
                st.markdown(f"✅ **{step}**")
            else:
                st.markdown(f"⏳ {step}")
    
    st.markdown("---")
    
    # Step 1: PDF Upload
    if current_step == "upload":
        pdf_upload_step()
    
    # Step 2: Text Extraction 
    elif current_step == "extract":
        pdf_extraction_step()
    
    # Step 3: Boundary Information Extraction
    elif current_step == "boundary":
        boundary_extraction_step()
    
    # Step 4: CAD Table Formatting
    elif current_step == "format":
        cad_formatting_step()
    
    # Step 5: Review (redirect to review tab)
    elif current_step == "review":
        st.info("📊 Please switch to the 'Review & Edit' tab to review and edit the extracted survey calls.")
        if st.button("🔙 Back to Format Step"):
            st.session_state.processing_step = "format"
            st.rerun()
    
    # Step 6: Visualization (redirect to visualization tab)
    elif current_step == "visualize":
        st.info("🗺️ Please switch to the 'Visualization' tab to see the property boundary visualization.")
        if st.button("🔙 Back to Review Step"):
            st.session_state.processing_step = "review"
            st.rerun()


def pdf_upload_step():
    """Step 1: PDF Upload"""
    st.subheader("📤 Step 1: Upload Site Survey PDF")
    
    st.markdown("""
    **Upload your site survey PDF document.** This should contain:
    - Property boundary descriptions
    - Line and curve tables
    - Legal descriptions
    - Survey notes and measurements
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=['pdf'],
        help="Upload a PDF site survey document"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.session_state.pdf_file_path = tmp_file_path
        st.session_state.pdf_filename = uploaded_file.name
        
        # Show file info
        file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
        st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size:.1f} MB)")
        
        # Initialize PDF extractor
        try:
            extractor = PDFExtractor(
                llamaindex_api_key=st.session_state.get('llamaindex_api_key'),
                google_project_id=st.session_state.get('google_project_id'),
                google_processor_id=st.session_state.get('google_processor_id'),
                google_location=st.session_state.get('google_location', 'us')
            )
            st.session_state.pdf_extractor = extractor
            
            # Show available methods
            available_methods = extractor.get_available_methods()
            st.info(f"📋 Available extraction methods: {len(available_methods)}")
            
            for method in available_methods:
                method_info = extractor.get_method_info(method)
                with st.expander(f"ℹ️ {method_info['name']}", expanded=False):
                    st.write(f"**Cost:** {method_info['cost']}")
                    st.write(f"**Best for:** {method_info['best_for']}")
                    st.write(f"**Pros:** {', '.join(method_info['pros'])}")
                    st.write(f"**Cons:** {', '.join(method_info['cons'])}")
            
            # Proceed button
            if st.button("➡️ Proceed to Text Extraction", type="primary"):
                st.session_state.processing_step = "extract"
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error initializing PDF extractor: {str(e)}")
    
    else:
        st.info("👆 Please upload a PDF file to begin processing")


def pdf_extraction_step():
    """Step 2: PDF Text Extraction"""
    st.subheader("🔍 Step 2: Extract Text from PDF")
    
    if not hasattr(st.session_state, 'pdf_file_path'):
        st.error("❌ No PDF file found. Please go back to Step 1.")
        return
    
    extractor = st.session_state.pdf_extractor
    if not extractor:
        st.error("❌ PDF extractor not initialized. Please go back to Step 1.")
        return
    
    # Show file info
    st.info(f"📄 Processing: {st.session_state.pdf_filename}")
    
    # Method selection
    st.markdown("### Choose Extraction Method")
    
    available_methods = extractor.get_available_methods()
    method_names = [extractor.get_method_info(method)['name'] for method in available_methods]
    method_costs = [extractor.estimate_cost(st.session_state.pdf_file_path, method) for method in available_methods]
    
    # Create method selection with cost info
    method_options = []
    for i, method in enumerate(available_methods):
        name = method_names[i]
        cost = method_costs[i]
        cost_str = "Free" if cost == 0 else f"${cost:.4f}"
        method_options.append(f"{name} ({cost_str})")
    
    selected_idx = st.radio(
        "Select extraction method:",
        range(len(method_options)),
        format_func=lambda x: method_options[x],
        help="Choose based on PDF complexity and cost preference"
    )
    
    selected_method = available_methods[selected_idx]
    
    # Show detailed method info
    method_info = extractor.get_method_info(selected_method)
    st.markdown(f"**Selected:** {method_info['name']}")
    st.markdown(f"**Best for:** {method_info['best_for']}")
    st.markdown(f"**Estimated cost:** {method_costs[selected_idx]:.4f} USD" if method_costs[selected_idx] > 0 else "**Cost:** Free")
    
    # Extract button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        extract_button = st.button("🚀 Extract Text", type="primary")
    
    with col2:
        if st.button("🔙 Back to Upload"):
            st.session_state.processing_step = "upload"
            st.rerun()
    
    # Perform extraction
    if extract_button:
        with st.spinner(f"🔄 Extracting text using {method_info['name']}..."):
            try:
                result = extractor.extract_text(st.session_state.pdf_file_path, selected_method)
                st.session_state.pdf_extraction_result = result
                
                if result.success:
                    st.success(f"✅ Text extraction completed!")
                    st.info(f"📊 Extracted {len(result.extracted_text)} characters in {result.processing_time:.2f} seconds")
                    st.info(f"💰 Actual cost: ${result.cost_estimate:.4f}")
                    
                    # Show warnings if any
                    if result.warnings:
                        for warning in result.warnings:
                            st.warning(f"⚠️ {warning}")
                    
                    # Show extracted text preview
                    with st.expander("📝 Extracted Text Preview", expanded=False):
                        st.text_area(
                            "Extracted text (first 2000 characters):",
                            result.extracted_text[:2000] + "..." if len(result.extracted_text) > 2000 else result.extracted_text,
                            height=300,
                            disabled=True
                        )
                    
                    # Proceed button
                    if st.button("➡️ Proceed to Boundary Extraction", type="primary"):
                        st.session_state.processing_step = "boundary"
                        st.rerun()
                        
                else:
                    st.error("❌ Text extraction failed!")
                    if result.warnings:
                        for warning in result.warnings:
                            st.error(f"❌ {warning}")
                    
            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
    
    # Show previous result if available
    elif st.session_state.pdf_extraction_result:
        result = st.session_state.pdf_extraction_result
        if result.success:
            st.success(f"✅ Text already extracted ({len(result.extracted_text)} characters)")
            
            with st.expander("📝 Extracted Text Preview", expanded=False):
                st.text_area(
                    "Extracted text:",
                    result.extracted_text[:2000] + "..." if len(result.extracted_text) > 2000 else result.extracted_text,
                    height=300,
                    disabled=True
                )
            
            if st.button("➡️ Proceed to Boundary Extraction", type="primary"):
                st.session_state.processing_step = "boundary"
                st.rerun()


def boundary_extraction_step():
    """Step 3: Site Boundary Information Extraction"""
    st.subheader("🎯 Step 3: Extract Site Boundary Information")
    
    if not st.session_state.pdf_extraction_result or not st.session_state.pdf_extraction_result.success:
        st.error("❌ No extracted text found. Please complete Step 2 first.")
        return
    
    # Initialize boundary parser if not done
    if not st.session_state.pdf_boundary_parser and st.session_state.settings.openai_api_key:
        st.session_state.pdf_boundary_parser = PDFBoundaryParser(
            openai_api_key=st.session_state.settings.openai_api_key,
            model=st.session_state.settings.openai_model
        )
    
    if not st.session_state.pdf_boundary_parser:
        st.error("❌ Please enter your OpenAI API key in the sidebar to proceed.")
        return
    
    st.markdown("**Extract site boundary data from the PDF text using AI analysis.**")
    
    extraction_result = st.session_state.pdf_extraction_result
    st.info(f"📄 Analyzing {len(extraction_result.extracted_text)} characters of extracted text")
    
    # Extract button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        extract_button = st.button("🤖 Extract Boundary Info", type="primary")
    
    with col2:
        if st.button("🔙 Back to Text Extraction"):
            st.session_state.processing_step = "extract"
            st.rerun()
    
    # Perform boundary extraction
    if extract_button:
        with st.spinner("🔄 Analyzing PDF text for boundary information..."):
            try:
                boundary_result = st.session_state.pdf_boundary_parser.extract_boundary_information(
                    extraction_result.extracted_text,
                    st.session_state.settings
                )
                st.session_state.boundary_extraction_result = boundary_result
                
                if boundary_result.has_boundary_data:
                    st.success(f"✅ Boundary information extracted!")
                    st.info(f"⏱️ Processing time: {boundary_result.processing_time:.2f} seconds")
                    st.info(f"💰 Estimated cost: ${boundary_result.cost_estimate:.4f}")
                    
                    # Show what was found
                    st.markdown("### 📋 Found Boundary Data:")
                    
                    for data_type, data in boundary_result.extracted_data.items():
                        if data.strip():
                            confidence = boundary_result.confidence_scores.get(data_type, 0.0)
                            confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                            
                            with st.expander(f"{confidence_color} {data_type.value.replace('_', ' ').title()} (confidence: {confidence:.2f})", expanded=False):
                                st.text_area(
                                    f"{data_type.value} content:",
                                    data[:1000] + "..." if len(data) > 1000 else data,
                                    height=200,
                                    disabled=True,
                                    key=f"boundary_{data_type.value}"
                                )
                    
                    # Show warnings
                    if boundary_result.warnings:
                        st.markdown("### ⚠️ Warnings:")
                        for warning in boundary_result.warnings:
                            st.warning(warning)
                    
                    # Proceed button
                    if st.button("➡️ Proceed to CAD Formatting", type="primary"):
                        st.session_state.processing_step = "format"
                        st.rerun()
                        
                else:
                    st.error("❌ No boundary information found in the PDF text!")
                    st.markdown("**Possible reasons:**")
                    st.markdown("- PDF doesn't contain site survey data")
                    st.markdown("- Text extraction quality was poor")
                    st.markdown("- Boundary information is in unexpected format")
                    
                    if boundary_result.warnings:
                        for warning in boundary_result.warnings:
                            st.error(f"❌ {warning}")
                    
            except Exception as e:
                st.error(f"❌ Boundary extraction failed: {str(e)}")
    
    # Show previous result if available
    elif st.session_state.boundary_extraction_result:
        result = st.session_state.boundary_extraction_result
        if result.has_boundary_data:
            st.success("✅ Boundary information already extracted")
            
            # Show summary
            found_types = [dt for dt, data in result.extracted_data.items() if data.strip()]
            st.info(f"📊 Found {len(found_types)} boundary data types")
            
            if st.button("➡️ Proceed to CAD Formatting", type="primary"):
                st.session_state.processing_step = "format"
                st.rerun()


def cad_formatting_step():
    """Step 4: Format to CAD-ready line/curve table"""
    st.subheader("📐 Step 4: Format to CAD-Ready Table")
    
    if not st.session_state.boundary_extraction_result or not st.session_state.boundary_extraction_result.has_boundary_data:
        st.error("❌ No boundary data found. Please complete Step 3 first.")
        return
    
    boundary_result = st.session_state.boundary_extraction_result
    
    st.markdown("**Convert ALL extracted boundary information to structured survey calls for CAD drawing.**")
    
    # Show all extracted boundary data
    found_data = {data_type: data for data_type, data in boundary_result.extracted_data.items() if data.strip()}
    
    st.info(f"📄 Processing {len(found_data)} boundary data types with total {sum(len(data) for data in found_data.values())} characters")
    
    # Show preview of all data types
    with st.expander("📝 All Extracted Boundary Data", expanded=False):
        for data_type, data in found_data.items():
            confidence = boundary_result.confidence_scores.get(data_type, 0.0)
            confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
            
            st.markdown(f"### {confidence_color} {data_type.value.replace('_', ' ').title()} (confidence: {confidence:.2f})")
            st.text_area(
                f"{data_type.value} content:",
                data[:500] + "..." if len(data) > 500 else data,
                height=150,
                disabled=True,
                key=f"cad_preview_{data_type.value}"
            )
    
    # Format button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        format_button = st.button("🔄 Format to CAD Table", type="primary")
    
    with col2:
        if st.button("🔙 Back to Boundary Extraction"):
            st.session_state.processing_step = "boundary"
            st.rerun()
    
    # Perform CAD formatting
    if format_button:
        with st.spinner("🔄 Converting to CAD-ready format using AI..."):
            try:
                deed_result = st.session_state.pdf_boundary_parser.format_to_cad_table(
                    boundary_result,
                    st.session_state.settings
                )
                
                if deed_result.calls:
                    st.success(f"✅ Converted to {len(deed_result.calls)} survey calls!")
                    
                    # Update session state with results
                    st.session_state.tracts = deed_result.tracts
                    st.session_state.has_multiple_tracts = deed_result.has_multiple_tracts
                    st.session_state.calls = deed_result.calls
                    st.session_state.geometry = deed_result.geometry
                    st.session_state.parse_result = deed_result
                    # Get the combined boundary text for original_text
                    found_data = {data_type: data for data_type, data in boundary_result.extracted_data.items() if data.strip()}
                    combined_text = "\n\n".join([f"=== {dt.value.upper()} ===\n{data}" for dt, data in found_data.items()])
                    st.session_state.original_text = combined_text
                    
                    # Show results summary
                    if deed_result.has_multiple_tracts:
                        st.info(f"📋 Found {len(deed_result.tracts)} tracts")
                        for tract in deed_result.tracts:
                            st.write(f"- **{tract.tract_id}**: {len(tract.calls)} calls")
                    else:
                        st.info(f"📋 Single tract with {len(deed_result.calls)} calls")
                    
                    # Show calls table preview
                    st.markdown("### 📊 Survey Calls Preview:")
                    
                    preview_data = []
                    for call in deed_result.calls[:10]:  # Show first 10 calls
                        preview_data.append({
                            'Seq': call.sequence,
                            'Type': call.type,
                            'Bearing': call.bearing or "N/A",
                            'Distance': f"{call.distance:.2f}" if call.distance else "N/A",
                            'Confidence': f"{call.confidence:.2f}"
                        })
                    
                    if preview_data:
                        import pandas as pd
                        df = pd.DataFrame(preview_data)
                        st.dataframe(df, use_container_width=True)
                        
                        if len(deed_result.calls) > 10:
                            st.info(f"Showing first 10 of {len(deed_result.calls)} calls")
                    
                    # Show geometry info if available
                    if deed_result.geometry:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Perimeter", f"{deed_result.geometry.perimeter:.2f} ft")
                        with col2:
                            st.metric("Closure Error", f"{deed_result.geometry.closure_error:.3f} ft")
                        with col3:
                            if deed_result.geometry.area:
                                acres = deed_result.geometry.area / 43560
                                st.metric("Area", f"{acres:.3f} acres")
                    
                    # Proceed button
                    if st.button("➡️ Proceed to Review & Edit", type="primary"):
                        st.session_state.processing_step = "review"
                        st.rerun()
                        
                else:
                    st.error("❌ Failed to convert boundary data to survey calls!")
                    st.markdown("**This may happen if:**")
                    st.markdown("- Boundary data format is not recognized")
                    st.markdown("- Text lacks sufficient survey information")
                    st.markdown("- AI parsing encountered issues")
                    
            except Exception as e:
                st.error(f"❌ CAD formatting failed: {str(e)}")
    
    # Show existing results if available
    elif st.session_state.calls:
        st.success(f"✅ Already formatted to {len(st.session_state.calls)} survey calls")
        
        if st.button("➡️ Proceed to Review & Edit", type="primary"):
            st.session_state.processing_step = "review"
            st.rerun()


def text_input_tab():
    """Deed text processing and parsing"""
    st.header("📝 Deed Text Processing")
    st.markdown("**Parse legal deed descriptions** to extract survey calls and property boundaries")
    
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
        
        # Create a log capture for this session
        log_capture = []
        
        class StreamlitLogHandler(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        # Add the handler to capture logs
        streamlit_handler = StreamlitLogHandler()
        streamlit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Get the root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.addHandler(streamlit_handler)
        
        with st.spinner("🤖 Parsing deed with AI..."):
            try:
                logger.info(f"=== Starting deed parsing session ===")
                logger.info(f"User selected model: {st.session_state.settings.openai_model}")
                logger.info(f"Deed text length: {len(deed_text)} characters")
                
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
                
                # Remove the log handler after parsing
                root_logger.removeHandler(streamlit_handler)
                
                # Display parsing logs in an expander
                if log_capture:
                    with st.expander("📋 Parsing Process Log", expanded=False):
                        st.markdown("### Processing Timeline")
                        for log_entry in log_capture:
                            if "ERROR" in log_entry:
                                st.error(log_entry)
                            elif "WARNING" in log_entry:
                                st.warning(log_entry)
                            elif "INFO" in log_entry:
                                st.info(log_entry)
                            else:
                                st.text(log_entry)
                    
                    # Show quick summary
                    primary_tract = parse_result.primary_tract
                    if primary_tract:
                        st.info(f"📊 Average confidence: {primary_tract.total_confidence:.2f} | "
                               f"Closure error: {primary_tract.geometry.closure_error:.3f} ft")
                
            except Exception as e:
                # Remove the log handler in case of error
                try:
                    root_logger.removeHandler(streamlit_handler)
                except:
                    pass
                
                logger.error(f"Parsing failed: {str(e)}")
                st.error(f"❌ Parsing failed: {str(e)}")
                
                # Show logs even on error
                if log_capture:
                    with st.expander("📋 Parsing Process Log (Error)", expanded=True):
                        for log_entry in log_capture:
                            if "ERROR" in log_entry:
                                st.error(log_entry)
                            elif "WARNING" in log_entry:
                                st.warning(log_entry)
                            else:
                                st.text(log_entry)
                
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
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
    """Review and edit survey data"""
    st.header("📊 Review & Edit Data")
    st.markdown("**Review and refine extracted survey data** before proceeding to visualization and site planning")
    
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
        
        st.text_area("Tract Description:", value=current_tract.description, height=70, disabled=True)
        st.text_area("POB Description:", value=current_tract.pob_description, height=70, disabled=True)
        
        # POB Coordinate Editing
        st.subheader("📍 Edit POB Coordinates")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            new_pob_x = st.number_input(
                "POB X Coordinate",
                value=float(current_tract.pob_x),
                step=0.01,
                format="%.2f",
                key=f"pob_x_{selected_tract_idx}",
                help="X coordinate of the Point of Beginning"
            )
        
        with col2:
            new_pob_y = st.number_input(
                "POB Y Coordinate", 
                value=float(current_tract.pob_y),
                step=0.01,
                format="%.2f",
                key=f"pob_y_{selected_tract_idx}",
                help="Y coordinate of the Point of Beginning"
            )
        
        with col3:
            if st.button("🔄 Update POB", key=f"update_pob_{selected_tract_idx}", 
                        help="Update POB coordinates and recalculate geometry"):
                # Update the tract's POB coordinates
                current_tract.pob_x = new_pob_x
                current_tract.pob_y = new_pob_y
                
                # Recalculate geometry with new POB
                from src.geometry.calculator import GeometryCalculator
                tract_settings = ProjectSettings(
                    units=st.session_state.settings.units,
                    bearing_convention=st.session_state.settings.bearing_convention,
                    pob_x=new_pob_x,
                    pob_y=new_pob_y,
                    pob_description=current_tract.pob_description,
                    confidence_threshold=st.session_state.settings.confidence_threshold,
                    closure_tolerance=st.session_state.settings.closure_tolerance
                )
                
                calculator = GeometryCalculator(tract_settings)
                current_tract.geometry = calculator.calculate_polygon(current_tract.calls)
                
                st.success(f"✅ Updated POB for {current_tract.tract_id} to ({new_pob_x:.2f}, {new_pob_y:.2f})")
                st.rerun()
        
        # Show coordinate change information
        if abs(new_pob_x - current_tract.pob_x) > 0.01 or abs(new_pob_y - current_tract.pob_y) > 0.01:
            delta_x = new_pob_x - current_tract.pob_x
            delta_y = new_pob_y - current_tract.pob_y
            st.info(f"📏 Coordinate change: ΔX = {delta_x:+.2f} ft, ΔY = {delta_y:+.2f} ft")
        
        # Show POB relationship info if available
        pob_coords = getattr(current_tract, 'pob_coordinates', None)
        if pob_coords and isinstance(pob_coords, dict):
            if pob_coords.get('reference_tract'):
                st.info(f"ℹ️ This POB is referenced to: {pob_coords['reference_tract']}")
                if 'offset_x' in pob_coords and 'offset_y' in pob_coords:
                    st.caption(f"Offset: ({pob_coords['offset_x']:+.2f}, {pob_coords['offset_y']:+.2f}) ft")
        
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
    st.header("🗺️ Boundary Visualization")
    st.markdown("**Visualize property boundaries** and prepare data for subdivision planning")
    
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
            # Add zoom control buttons
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
            with col1:
                if st.button("🔍+ Zoom In", key="zoom_in"):
                    st.rerun()
            with col2:
                if st.button("🔍- Zoom Out", key="zoom_out"):
                    st.rerun()
            with col3:
                if st.button("🎯 Reset View", key="reset_view"):
                    st.rerun()
            with col4:
                if st.button("📐 Fit to View", key="fit_view"):
                    st.rerun()
            with col5:
                st.markdown("**🖱️ Mouse**: Scroll=Zoom, Drag=Pan")
            
            # Enhanced display with interactive controls
            st.components.v1.html(
                f'''
                <div style="text-align: center; background: {"#2C3E50" if color_scheme == "dark" else "#FFFFFF"}; padding: 20px; border-radius: 10px; border: 2px solid #007bff; position: relative;">
                    <div style="margin-bottom: 10px; font-weight: bold; color: #007bff;">
                        📏 Scale: {feet_per_pixel:.2f} ft/px | Resolution: {1/feet_per_pixel:.1f} px/ft
                        <br/>🎮 Interactive: Mouse wheel to zoom, click and drag to pan
                    </div>
                    
                    <!-- Zoom indicator overlay -->
                    <div id="zoom-indicator" style="position: absolute; top: 10px; right: 10px; background: rgba(0,123,255,0.9); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;">
                        Zoom: 100%
                    </div>
                    
                    <div style="border: 2px solid {"#7F8C8D" if color_scheme == "dark" else "#DDDDDD"}; border-radius: 5px; overflow: hidden; background: white; cursor: grab;">
                        {svg_content}
                    </div>
                    
                    <div style="margin-top: 10px; font-size: 12px; color: {"#BDC3C7" if color_scheme == "dark" else "#666666"};">
                        💡 Tip: Use mouse wheel to zoom, click and drag to pan around the survey
                    </div>
                    
                    <script>
                        // Update zoom indicator when zoom changes
                        function updateZoomIndicator() {{
                            if (typeof currentZoom !== 'undefined') {{
                                const indicator = document.getElementById('zoom-indicator');
                                if (indicator) {{
                                    indicator.textContent = `Zoom: ${{Math.round(currentZoom * 100)}}%`;
                                }}
                            }}
                        }}
                        
                        // Override the updateTransform function to include zoom indicator
                        if (typeof updateTransform === 'function') {{
                            const originalUpdate = updateTransform;
                            updateTransform = function() {{
                                originalUpdate();
                                updateZoomIndicator();
                            }};
                        }}
                        
                        // Initialize zoom indicator
                        setTimeout(updateZoomIndicator, 100);
                    </script>
                </div>
                ''',
                height=svg_height + 200
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
        
            # Show boundary points table
            st.subheader("📍 Boundary Points")
            show_boundary_points_table(current_geometry, current_tract, show_all_tracts and st.session_state.has_multiple_tracts)
        
        else:
            # Basic display with interactive features
            # Add zoom control buttons for basic mode too
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
            with col1:
                if st.button("🔍+ Zoom In", key="basic_zoom_in"):
                    st.rerun()
            with col2:
                if st.button("🔍- Zoom Out", key="basic_zoom_out"):
                    st.rerun()
            with col3:
                if st.button("🎯 Reset View", key="basic_reset_view"):
                    st.rerun()
            with col4:
                if st.button("📐 Fit to View", key="basic_fit_view"):
                    st.rerun()
            with col5:
                st.markdown("**🖱️ Mouse**: Scroll=Zoom, Drag=Pan")
            
            display_height = svg_height + 150
            st.components.v1.html(
                f'''
                <div style="text-align: center; background: {"#2C3E50" if color_scheme == "dark" else "#FFFFFF"}; padding: 20px; border-radius: 10px; position: relative;">
                    <div style="margin-bottom: 10px; font-weight: bold; color: {"#ECF0F1" if color_scheme == "dark" else "#007bff"};">
                        🎮 Interactive Survey Viewer - Mouse wheel to zoom, click and drag to pan
                    </div>
                    
                    <!-- Zoom indicator -->
                    <div id="zoom-indicator-basic" style="position: absolute; top: 10px; right: 10px; background: rgba(0,123,255,0.9); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold;">
                        Zoom: 100%
                    </div>
                    
                    <div style="border: 2px solid {"#7F8C8D" if color_scheme == "dark" else "#DDDDDD"}; border-radius: 5px; overflow: hidden; background: white; cursor: grab;">
                        {svg_content}
                    </div>
                    
                    <div style="margin-top: 10px; font-size: 12px; color: {"#BDC3C7" if color_scheme == "dark" else "#666666"};">
                        💡 Tip: Use mouse wheel to zoom, click and drag to pan around the survey
                    </div>
                    
                    <script>
                        // Update zoom indicator for basic mode
                        function updateBasicZoomIndicator() {{
                            if (typeof currentZoom !== 'undefined') {{
                                const indicator = document.getElementById('zoom-indicator-basic');
                                if (indicator) {{
                                    indicator.textContent = `Zoom: ${{Math.round(currentZoom * 100)}}%`;
                                }}
                            }}
                        }}
                        
                        // Initialize zoom indicator for basic mode
                        setTimeout(updateBasicZoomIndicator, 100);
                        
                        // Update indicator on zoom changes
                        if (typeof updateTransform === 'function') {{
                            const originalUpdate = updateTransform;
                            updateTransform = function() {{
                                originalUpdate();
                                updateBasicZoomIndicator();
                            }};
                        }}
                    </script>
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
            df = pd.DataFrame([call.model_dump() for call in st.session_state.calls])
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
                'settings': st.session_state.settings.model_dump(),
                'calls': [call.model_dump() for call in st.session_state.calls],
                'geometry': st.session_state.geometry.model_dump() if st.session_state.geometry else None,
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
        
        # New row for site planning export
        st.markdown("---")
        st.subheader("🏘️ Subdivision Planning Export")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            # Export boundary as PNG for site planning
            if st.button("🖼️ Export Boundary PNG", help="Export boundary polygon as PNG for site planning"):
                with st.spinner("Creating boundary PNG..."):
                    png_path, pixel_per_foot = export_boundary_as_png(st.session_state.geometry, st.session_state.calls)
                    if png_path:
                        st.session_state.boundary_png_path = png_path
                        st.session_state.pixel_per_foot = pixel_per_foot
                        
                        # Read the PNG file for download
                        with open(png_path, 'rb') as f:
                            png_data = f.read()
                        
                        st.download_button(
                            "📥 Download PNG",
                            data=png_data,
                            file_name=f"boundary_polygon_{timestamp}.png",
                            mime="image/png",
                            help="Download boundary polygon as PNG"
                        )
                        
                        st.success(f"✅ Boundary exported! Scale: {pixel_per_foot:.2f} pixels/foot")
        
        with col2:
            # Quick link to site planning tab
            if st.button("🏘️ Go to AI Site Planning", help="Switch to AI Site Planning tab to continue"):
                st.info("💡 Switch to the 'AI Site Planning' tab to generate road networks and analyze subdivision plans")
        
        with col3:
            st.info("💡 **Subdivision Planning Workflow:** Export boundary PNG → Switch to AI Site Planning tab → Generate roads → Configure lots → Analyze development")
        
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


def show_boundary_points_table(geometry, tract=None, is_multi_tract=False):
    """Show boundary points table with coordinates for each tract"""
    
    if is_multi_tract and st.session_state.tracts:
        # Multi-tract boundary points
        for i, tract_obj in enumerate(st.session_state.tracts):
            if tract_obj.geometry and tract_obj.geometry.vertices:
                with st.expander(f"🔍 {tract_obj.tract_id} - Boundary Points ({len(tract_obj.geometry.vertices)} points)", expanded=i==0):
                    
                    # Create boundary points dataframe
                    points_data = []
                    for j, vertex in enumerate(tract_obj.geometry.vertices):
                        points_data.append({
                            'Point #': j + 1,
                            'Point Type': 'POB' if j == 0 else f'Vertex {j}',
                            'X Coordinate': f"{vertex.x:.2f}",
                            'Y Coordinate': f"{vertex.y:.2f}",
                            'Description': vertex.description or f"Boundary point {j+1}"
                        })
                    
                    df = pd.DataFrame(points_data)
                    
                    # Display with color coding
                    colors = ['#2E86AB', '#A23B72', '#F18F01', '#7B68EE', '#32CD32', '#FF6347']
                    tract_color = colors[i % len(colors)]
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {tract_color}; padding-left: 10px; margin-bottom: 10px;">
                        <strong style="color: {tract_color};">{tract_obj.tract_id}</strong> - 
                        POB: ({tract_obj.pob_x:.2f}, {tract_obj.pob_y:.2f})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Point #": st.column_config.NumberColumn("Point #", width="small"),
                            "Point Type": st.column_config.TextColumn("Type", width="medium"),
                            "X Coordinate": st.column_config.TextColumn("X (ft)", width="medium"),
                            "Y Coordinate": st.column_config.TextColumn("Y (ft)", width="medium"),
                            "Description": st.column_config.TextColumn("Description", width="large")
                        }
                    )
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Points", len(tract_obj.geometry.vertices))
                    with col2:
                        if tract_obj.geometry.perimeter:
                            st.metric("Perimeter", f"{tract_obj.geometry.perimeter:.2f} ft")
                    with col3:
                        if tract_obj.geometry.area:
                            acres = tract_obj.geometry.area / 43560
                            st.metric("Area", f"{acres:.3f} acres")
    
    else:
        # Single tract boundary points
        if geometry and geometry.vertices:
            st.markdown("### Boundary Points Coordinates")
            
            # Create boundary points dataframe
            points_data = []
            for i, vertex in enumerate(geometry.vertices):
                points_data.append({
                    'Point #': i + 1,
                    'Point Type': 'POB' if i == 0 else f'Vertex {i}',
                    'X Coordinate': f"{vertex.x:.2f}",
                    'Y Coordinate': f"{vertex.y:.2f}",
                    'Description': vertex.description or f"Boundary point {i+1}"
                })
            
            df = pd.DataFrame(points_data)
            
            # Display POB info
            if tract:
                st.markdown(f"""
                <div style="border-left: 4px solid #F18F01; padding-left: 10px; margin-bottom: 10px;">
                    <strong style="color: #F18F01;">Point of Beginning:</strong> 
                    ({tract.pob_x:.2f}, {tract.pob_y:.2f})
                </div>
                """, unsafe_allow_html=True)
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Point #": st.column_config.NumberColumn("Point #", width="small"),
                    "Point Type": st.column_config.TextColumn("Type", width="medium"),
                    "X Coordinate": st.column_config.TextColumn("X (ft)", width="medium"),
                    "Y Coordinate": st.column_config.TextColumn("Y (ft)", width="medium"),
                    "Description": st.column_config.TextColumn("Description", width="large")
                }
            )
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Points", len(geometry.vertices))
            with col2:
                if geometry.perimeter:
                    st.metric("Perimeter", f"{geometry.perimeter:.2f} ft")
            with col3:
                if geometry.area:
                    acres = geometry.area / 43560
                    st.metric("Area", f"{acres:.3f} acres")
        else:
            st.info("No boundary points available to display.")


def export_boundary_as_png(geometry, calls: List[SurveyCall], width: int = 800, height: int = 600) -> tuple[str, float]:
    """
    Export boundary polygon as PNG image with red lines and calculate pixel/foot ratio.
    
    Returns:
        tuple: (png_file_path, pixel_per_foot_ratio)
    """
    if not geometry or not geometry.vertices:
        return None, 1.0
    
    # Create PIL image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Calculate transformation
    vertices = geometry.vertices
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Calculate center and scale
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    data_width = max_x - min_x
    data_height = max_y - min_y
    
    # Leave 10% margin
    margin = 0.1
    usable_width = width * (1 - 2 * margin)
    usable_height = height * (1 - 2 * margin)
    
    if data_width == 0 and data_height == 0:
        scale = 1.0
    else:
        scale_x = usable_width / data_width if data_width > 0 else float('inf')
        scale_y = usable_height / data_height if data_height > 0 else float('inf')
        scale = min(scale_x, scale_y)
    
    # Calculate pixel per foot ratio
    pixel_per_foot = scale
    
    def transform_point(x, y):
        """Transform survey coordinates to image coordinates"""
        img_x = (x - center_x) * scale + width / 2
        img_y = height / 2 - (y - center_y) * scale  # Flip Y axis
        return int(img_x), int(img_y)
    
    # Draw polygon boundary with red lines
    polygon_points = []
    for vertex in vertices:
        x, y = transform_point(vertex.x, vertex.y)
        polygon_points.append((x, y))
    
    # Draw lines between consecutive vertices
    for i in range(len(polygon_points)):
        start = polygon_points[i]
        end = polygon_points[(i + 1) % len(polygon_points)]  # Connect last to first
        draw.line([start, end], fill='red', width=3)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='.') as tmp_file:
        img.save(tmp_file.name)
        return tmp_file.name, pixel_per_foot


# Site Planning Functions (integrated from site_plan_app.py)
def extract_site_boundary_ai(site_plan_image):
    """Extract site boundary polygon from site plan image using Gemini AI"""
    if not st.session_state.gemini_client:
        st.error("❌ Gemini API key required for AI site boundary extraction")
        return None
    
    print("[PROGRESS] Starting site boundary extraction...")
    with st.spinner("🔄 Extracting site boundary polygon..."):
        prompt = """
        Extract ONLY site boundary polygon from provided site plan image (lines: red, no vertices).
        """
        
        try:
            print("[PROGRESS] Sending request to Gemini API for boundary extraction...")
            response = st.session_state.gemini_client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[[site_plan_image, prompt, 
                "Remove ALL elements Except ONLY the Outermost border of site boundary polygon (red lines)"], 
                "There MUST NOT be any other elements except ONLY red lines. Confirm it and REMOVE other ALL elements Except only red lines!", 
                "Remove ALL elements Inside of site boundary polygon", 
                "Remove ALL elements Outside of site boundary polygon",
                "There must be nothing except red lines. Confirm it and fix it!", 
                "There must be NOthing inside and outside of red lines. Confirm it and fix it!"],
            )
            print("[PROGRESS] Gemini API response received")
            
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    final_image = Image.open(BytesIO(part.inline_data.data))
                    print("[SUCCESS] Site boundary extracted")
                    return final_image
            
            print("[WARNING] No image received from API")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to extract boundary: {str(e)}")
            st.error(f"Error extracting boundary: {str(e)}")
            return None


def generate_road_network_ai(boundary_image, road_type):
    """Generate road network inside site boundary using Gemini AI"""
    if not st.session_state.gemini_client:
        st.error("❌ Gemini API key required for AI road network generation")
        return None
    
    print(f"[PROGRESS] Starting {road_type} road network generation...")
    with st.spinner(f"🔄 Generating {road_type} road network..."):
        prompt = (
            "Provided image is extracted site boundary polygon (red lines)",
            "I am going to prepare in the area within the red lines (site boundary)",
            f"Generate {road_type}-type road network INSIDE of site boundary polygon ONLY while considering lotting",
        )
        
        try:
            print(f"[PROGRESS] Sending request to Gemini API for {road_type} road network...")
            response = st.session_state.gemini_client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[boundary_image, prompt,
                       "POB(Point Of Beginning) of road network is on the LEFT of the site boundary. Redesign road network!!!", 
                       "POB(Point Of Beginning) of road network is on the LEFT of the site boundary. Confirm it and if not, Redesign road network!!!",
                       "When considering lotting, there can be at most one house in each lot.", 
                       "When considering lotting, every lots with one house must have access into the road. You can add cul-de-sacs if only necessary. Confirm it and if not, Redesign road network while not add additional roads into the lots!!!", 
                       "Every roads must be connected. Confirm it and if not, Redesign road network!!!",
                       "add green spaces to make site plan more engineer-like", 
                       "Update to more engineer-like road network"],
            )
            print("[PROGRESS] Gemini API response received for road network")
            
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    print(f"[SUCCESS] {road_type.capitalize()} road network generated")
                    return image
            
            print("[WARNING] No road network image received from API")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to generate road network: {str(e)}")
            st.error(f"Error generating road network: {str(e)}")
            return None


def finalize_site_plan_ai(road_network_image):
    """Finalize site plan with lotting using Gemini AI"""
    if not st.session_state.gemini_client:
        st.error("❌ Gemini API key required for AI site plan finalization")
        return None
    
    print("[PROGRESS] Starting finalization with lotting...")
    with st.spinner("🔄 Finalizing site plan with lotting..."):
        try:
            print("[PROGRESS] Sending request to Gemini API for lotting...")
            response = st.session_state.gemini_client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[road_network_image, 
                "Segment provided image to road network, lots, green spaces", 
                "Fill out black color to road network, GRAY color inside of Lots, green color to green spaces.", 
                "Remove ALL elements inside of lots and Highlight ALL lots with RED borders", 
                "Remove ALL other elements except road network, lots, green spaces, open spaces, so that extract boundaries.", 
                "Remove all elements except boundaries, especially texts"],
            )
            print("[PROGRESS] Gemini API response received for lotting")
            
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    print("[SUCCESS] Site plan finalized")
                    return image
            
            print("[WARNING] No finalized image received from API")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to finalize site plan: {str(e)}")
            st.error(f"Error finalizing site plan: {str(e)}")
            return None


def make_json_serializable(obj):
    """Convert numpy data types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def highlight_site_plan_edges(site_plan_image):
    """Highlight edges of site plan image with red borders using Gemini AI"""
    if not st.session_state.gemini_client:
        st.error("❌ Gemini API key required for edge highlighting")
        return None
    
    print("[PROGRESS] Starting edge highlighting...")
    
    prompt = """
    I am going to detect edges by filtering red color range.
    
    highlight every edges of provided site plan image with RED borders.
    """
    
    try:
        print("[PROGRESS] Sending request to Gemini AI for edge highlighting...")
        response = st.session_state.gemini_client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[site_plan_image, prompt, "highlight EVERY edges of provided site plan image with RED borders.", "Only the road should be black. Change the color of any other elements that can be detected as black to white. (especailly, houses inside lots)"],
        )
        print("[PROGRESS] Gemini AI response received for edge highlighting")
        
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                highlighted_image = Image.open(BytesIO(part.inline_data.data))
                print("[SUCCESS] Site plan edges highlighted")
                return highlighted_image
        
        print("[WARNING] No highlighted image received from API")
        return None
        
    except Exception as e:
        print(f"[ERROR] Failed to highlight edges: {str(e)}")
        st.error(f"Error highlighting edges: {str(e)}")
        return None


def analyze_site_plan_comprehensive(final_image_path, pixel_per_foot, progress_container=None):
    """
    Comprehensive site plan analysis with progress visualization
    """
    print("[PROGRESS] Starting comprehensive site plan analysis...")
    
    try:
        # Read the original image
        original_img = cv2.imread(final_image_path)
        if original_img is None:
            st.error("❌ Could not read finalized site plan image")
            return None
        
        # Preprocess: Edge highlighting preprocessing
        if progress_container:
            with progress_container:
                st.info("🔄 Preprocess: Highlighting edges for better detection...")
        
        # Convert to PIL Image for Gemini AI
        original_pil = Image.open(final_image_path)
        
        # Highlight edges using Gemini AI
        highlighted_pil = highlight_site_plan_edges(original_pil)
        
        if highlighted_pil:
            # Save highlighted image
            highlighted_path = "highlighted_site_plan.png"
            highlighted_pil.save(highlighted_path)
            
            # Convert back to OpenCV format
            img = cv2.imread(highlighted_path)
            
            if progress_container:
                with progress_container:
                    st.image(highlighted_pil, caption="Preprocess: Edge Highlighting - Red Borders Added", use_column_width=True)
                    st.success("✅ Edges highlighted successfully!")
        else:
            # Fallback to original image if highlighting fails
            img = original_img
            if progress_container:
                with progress_container:
                    st.warning("⚠️ Edge highlighting failed, using original image")
        
        # Step 1: Binary conversion and red line detection
        if progress_container:
            with progress_container:
                st.info("🔄 Step 1: Converting to binary and detecting boundaries...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect green spaces early for buildable lot analysis
        hsv_for_green = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])   # Lower bound for green
        upper_green = np.array([85, 255, 255]) # Upper bound for green
        green_mask = cv2.inRange(hsv_for_green, lower_green, upper_green)
        
        # Detect red boundaries using same method as test_analyse.py
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define WIDER range for red color in HSV to capture more variations
        # Lower red range (0-20 degrees in hue)
        lower_red1 = np.array([0, 30, 30])    # Reduced saturation and value thresholds
        upper_red1 = np.array([20, 255, 255]) # Extended hue range to include orange-reds
        
        # Upper red range (160-180 degrees in hue)
        lower_red2 = np.array([160, 30, 30])  # Start from 160 to catch more purple-reds
        upper_red2 = np.array([180, 255, 255])
        
        # Additional range for very light reds/pinks
        lower_red3 = np.array([0, 20, 100])   # Very low saturation for pinks
        upper_red3 = np.array([10, 150, 255]) # Medium saturation, high value
        
        # Create masks for all red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
        
        # Combine all masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.bitwise_or(red_mask, mask3)
        
        # Apply morphological operations to enhance edges (from test_analyse.py)
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        
        # Close small gaps in lines
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        # Dilate to strengthen edges
        red_mask = cv2.dilate(red_mask, kernel_small, iterations=1)
        # Apply median filter to reduce noise
        red_mask = cv2.medianBlur(red_mask, 3)
        # Final closing to ensure continuity
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Apply edge detection and add back to enhance boundaries
        edges = cv2.Canny(red_mask, 50, 150)
        red_mask = cv2.bitwise_or(red_mask, edges)
        
        # Create binary visualization
        binary_viz = np.zeros_like(img)
        binary_viz[red_mask > 0] = [0, 0, 255]  # Red lines
        binary_viz[red_mask == 0] = [255, 255, 255]  # White background
        
        # Save and show binary image
        binary_path = "analysis_binary_viz.png"
        cv2.imwrite(binary_path, binary_viz)
        
        if progress_container:
            with progress_container:
                st.image(binary_viz, caption="Step 1: Binary Conversion - Red Line Detection", use_column_width=True)
        
        # Step 2: Detect lots and roads
        if progress_container:
            with progress_container:
                st.info("🔄 Step 2: Analyzing lots and roads...")
        
        # Use same lot detection method as test_analyse.py
        # Apply morphological closing to ensure closed boundaries
        kernel = np.ones((3, 3), np.uint8)
        closed_img = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Invert image (regions become white, boundaries become black)
        inverted = cv2.bitwise_not(closed_img)
        
        # Use connected components with stats (same as test_analyse.py)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
        
        print(f"Found {num_labels} components (including background)")
        
        # Process each component (skip label 0 which is background)
        min_area_threshold = 500  # Minimum pixels to be considered a valid lot
        
        # Calculate lot data using exact method from test_analyse.py
        lot_data = []
        
        for label in range(1, num_labels):
            # Extract statistics for this component (exact method from test_analyse.py)
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]  # Exact pixel count!
            
            # Filter out small noise
            if area < min_area_threshold:
                continue
            
            # Get centroid
            cx, cy = centroids[label]
            
            # Extract mask for this component
            component_mask = (labels == label).astype(np.uint8) * 255
            
            # Calculate perimeter using the component mask
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perimeter = cv2.arcLength(contours[0], True) if contours else 0
            
            # Calculate additional metrics (from test_analyse.py)
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate area in square feet
            area_sq_ft = area / (pixel_per_foot ** 2)
            perimeter_ft = perimeter / pixel_per_foot
            
            # Calculate width and depth in feet (from test_analyse.py)
            if w <= h:
                width_ft = w / pixel_per_foot
                depth_ft = h / pixel_per_foot
                orientation = 'North-South' if h > w * 1.2 else 'Square'
            else:
                width_ft = h / pixel_per_foot
                depth_ft = w / pixel_per_foot
                orientation = 'East-West' if w > h * 1.2 else 'Square'
            
            # Determine if buildable (updated logic: not a green space)
            # Check if this lot area overlaps with green spaces
            green_pixels_in_lot = np.sum(green_mask[component_mask > 0])
            lot_total_pixels = np.sum(component_mask > 0)
            green_percentage = (green_pixels_in_lot / lot_total_pixels) * 100 if lot_total_pixels > 0 else 0
            
            # Lot is buildable if it's not primarily a green space (less than 50% green)
            buildable = green_percentage < 50.0 and area_sq_ft >= 500  # Minimum 500 sf and not green space
            
            # Estimate house size based on lot area (from test_analyse.py)
            est_house_sf = area_sq_ft * 0.25 if buildable else 0  # 25% coverage
            
            # Store all data (matching test_analyse.py structure)
            lot_data.append({
                'lot_id': len(lot_data) + 1,
                'label': int(label),
                'area_pixels': int(area),
                'area_sqft': float(area_sq_ft),
                'perimeter_pixels': float(perimeter),
                'perimeter_ft': float(perimeter_ft),
                'centroid_x': float(cx),
                'centroid_y': float(cy),
                'bbox_x': int(x),
                'bbox_y': int(y),
                'bbox_width': int(w),
                'bbox_height': int(h),
                'width_ft': float(width_ft),
                'depth_ft': float(depth_ft),
                'orientation': str(orientation),
                'compactness': float(compactness),
                'aspect_ratio': float(aspect_ratio),
                'buildable': bool(buildable),  # Based on green space analysis
                'est_house_sf': float(est_house_sf),
                'green_percentage': float(green_percentage)  # Add green space percentage
            })
        
        print(f"Detected {len(lot_data)} valid lots (area > {min_area_threshold} pixels)")
        
        # Initialize all variables at the beginning to avoid scope issues
        total_pixels = img.shape[0] * img.shape[1]
        pixels_per_sqft = pixel_per_foot ** 2
        total_area_sqft = total_pixels / pixels_per_sqft
        
        # Find the largest lot (site boundary)
        if lot_data:
            max_area_lot = max(lot_data, key=lambda x: x['area_pixels'])
            site_boundary_label = max_area_lot['label']
            buildable_lots = [lot for lot in lot_data if lot['label'] != site_boundary_label]
            site_area_sf = max_area_lot['area_sqft']
        else:
            buildable_lots = []
            max_area_lot = {'area_sqft': total_area_sqft, 'label': 0}
            site_area_sf = total_area_sqft
        
        # Calculate buildable area metrics
        if buildable_lots:
            total_buildable_area = sum(lot['area_sqft'] for lot in buildable_lots)
            usage_rate = (total_buildable_area / site_area_sf) * 100
            avg_lot_size = total_buildable_area / len(buildable_lots)
        else:
            total_buildable_area = sum(lot['area_sqft'] for lot in lot_data) if lot_data else 0
            usage_rate = 0
            avg_lot_size = total_buildable_area / len(lot_data) if lot_data else 0
        
        # Create professional matplotlib overlay visualization
        original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert red lines to black for cleaner visualization
        hsv_for_viz = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1_viz = np.array([0, 50, 50])
        upper_red1_viz = np.array([10, 255, 255])
        lower_red2_viz = np.array([170, 50, 50])
        upper_red2_viz = np.array([180, 255, 255])
        mask1_viz = cv2.inRange(hsv_for_viz, lower_red1_viz, upper_red1_viz)
        mask2_viz = cv2.inRange(hsv_for_viz, lower_red2_viz, upper_red2_viz)
        red_mask_viz = cv2.bitwise_or(mask1_viz, mask2_viz)
        
        # Change red pixels to black
        modified_img = original_img_rgb.copy()
        modified_img[red_mask_viz > 0] = [0, 0, 0]  # Change red to black
        
        # Create matplotlib figure (matching test_analyse.py)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(18, 12), dpi=100)
        
        # Display the modified site plan image
        ax.imshow(modified_img, alpha=1.0)
        
        # Create semi-transparent overlay for lot areas
        overlay = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.float32)
        
        # Generate distinct colors for buildable lots
        np.random.seed(42)
        cmap = plt.colormaps.get_cmap('Set3')  # Updated matplotlib syntax
        colors_mpl = [cmap(i / max(len(buildable_lots), 1)) for i in range(len(buildable_lots))]
        
        # Create masks for each buildable lot and add semi-transparent color overlay
        for idx, lot in enumerate(buildable_lots):
            label_val = lot['label']
            mask = labels == label_val
            
            # Apply semi-transparent color to this lot area
            color = colors_mpl[idx % len(colors_mpl)]
            overlay[mask] = [color[0], color[1], color[2], 0.3]  # 30% opacity
        
        # Display the overlay
        ax.imshow(overlay, interpolation='nearest')
        
        # Add lot numbers and areas for buildable lots only (matching test_analyse.py)
        for i, lot in enumerate(buildable_lots, 1):
            cx = lot['centroid_x']
            cy = lot['centroid_y']
            area_sf = lot['area_sqft']
            
            # Determine font size based on lot area
            if area_sf > 1000:
                font_size = 11
                text_padding = 0.4
            elif area_sf > 500:
                font_size = 10
                text_padding = 0.35
            else:
                font_size = 9
                text_padding = 0.3
            
            # Create text with lot number and area
            lot_text = f"Lot {i}"
            area_text = f"{area_sf:.0f} sf"
            
            # Add lot number with strong visibility
            ax.text(cx, cy - 10, lot_text,
                    fontsize=font_size, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle=f'round,pad={text_padding}', 
                             facecolor='white', 
                             edgecolor='darkblue', 
                             linewidth=1.5, 
                             alpha=0.95),
                    color='darkblue')
            
            # Add area below lot number
            ax.text(cx, cy + 10, area_text,
                    fontsize=font_size * 0.9,
                    ha='center', va='center',
                    bbox=dict(boxstyle=f'round,pad={text_padding}', 
                             facecolor='yellow', 
                             edgecolor='darkgreen', 
                             linewidth=1.5, 
                             alpha=0.95),
                    color='darkgreen')
        
        # Site metrics already calculated above - no need to recalculate
        
        # Set title with usage rate (matching test_analyse.py)
        title = (f'Subdivision Lot Analysis - Original Site Plan Overlay\n'
                 f'{len(buildable_lots)} Buildable Lots | '
                 f'Site Area: {site_area_sf:,.0f} sf | '
                 f'Buildable Area: {total_buildable_area:,.0f} sf | '
                 f'Usage Rate: {usage_rate:.1f}%')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set axis labels
        ax.set_xlabel(f'Width (pixels) - Scale: {pixel_per_foot:.2f} pixels/foot', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        
        # Add subtle grid
        ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
        
        # Set axis limits to match image dimensions
        ax.set_xlim(0, original_img_rgb.shape[1])
        ax.set_ylim(original_img_rgb.shape[0], 0)
        
        # Create detailed statistics box (matching test_analyse.py)
        if buildable_lots:
            avg_lot = np.mean([lot['area_sqft'] for lot in buildable_lots])
            min_lot = min([lot['area_sqft'] for lot in buildable_lots])
            max_lot = max([lot['area_sqft'] for lot in buildable_lots])
            
            stats_text = (
                f"╔══════════════════════════╗\n"
                f"║   SITE STATISTICS        ║\n"
                f"╠══════════════════════════╣\n"
                f"║ Buildable Lots: {len(buildable_lots):>9}║\n"
                f"║ Site Area:  {site_area_sf:>10,.0f} sf║\n"
                f"║ Buildable:  {total_buildable_area:>10,.0f} sf║\n"
                f"║ Usage Rate:     {usage_rate:>8.1f}%║\n"
                f"╠══════════════════════════╣\n"
                f"║ Avg Lot: {avg_lot:>13,.0f} sf║\n"
                f"║ Min Lot: {min_lot:>13,.0f} sf║\n"
                f"║ Max Lot: {max_lot:>13,.0f} sf║\n"
                f"╚══════════════════════════╝"
            )
            
            # Add statistics box with better styling
            props = dict(boxstyle='round,pad=0.5', 
                        facecolor='lightcyan', 
                        edgecolor='navy', 
                        linewidth=2, 
                        alpha=0.95)
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, 
                   fontsize=10,
                   verticalalignment='top', 
                   fontfamily='monospace', 
                   bbox=props)
        
        # Save matplotlib visualization
        plt.tight_layout()
        lot_viz_matplotlib_path = "analysis_lots_matplotlib.png"
        plt.savefig(lot_viz_matplotlib_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Create simple OpenCV visualization for progress display
        lot_viz = img.copy()
        colors_cv = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for idx, lot in enumerate(lot_data):
            # Color the lot
            mask = (labels == lot['label']).astype(np.uint8)
            color = colors_cv[idx % len(colors_cv)]
            lot_viz[mask > 0] = color
            
            # Add lot number and size
            cx, cy = int(lot['centroid_x']), int(lot['centroid_y'])
            cv2.putText(lot_viz, f"Lot {lot['lot_id']}", (cx-20, cy-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(lot_viz, f"{lot['area_sqft']:.0f} sf", (cx-25, cy+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save OpenCV visualization
        lot_viz_path = "analysis_lots_viz.png"
        cv2.imwrite(lot_viz_path, lot_viz)
        
        if progress_container:
            with progress_container:
                st.image(lot_viz, caption=f"Step 2: Lot Analysis - {len(lot_data)} Lots Detected", use_column_width=True)
        
        # Step 3: Road network analysis (comprehensive method from test_analyse.py)
        if progress_container:
            with progress_container:
                st.info("🔄 Step 3: Analyzing road network...")
        
        # Detect road areas (black regions) - same method as test_analyse.py
        _, road_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Skeletonize to find road centerlines (from test_analyse.py)
        from skimage import morphology
        skeleton = morphology.skeletonize(road_mask // 255)
        
        # Calculate total road length from skeleton
        road_pixels = np.sum(skeleton)
        total_road_length_ft = road_pixels / pixel_per_foot
        
        # Find road widths using distance transform
        dist_transform = cv2.distanceTransform(road_mask, cv2.DIST_L2, 5)
        
        # Get road widths along the skeleton
        road_widths_px = dist_transform[skeleton > 0] * 2  # Multiply by 2 for full width
        if len(road_widths_px) > 0:
            avg_road_width_ft = float(np.mean(road_widths_px) / pixel_per_foot)
            min_road_width_ft = float(np.min(road_widths_px) / pixel_per_foot)
            max_road_width_ft = float(np.max(road_widths_px) / pixel_per_foot)
        else:
            avg_road_width_ft = min_road_width_ft = max_road_width_ft = 0.0
        
        # Classify roads by width (from test_analyse.py)
        main_road_threshold = 30  # Roads wider than 30 ft are main roads
        main_road_pixels = np.sum(road_widths_px > (main_road_threshold * pixel_per_foot)) if len(road_widths_px) > 0 else 0
        residential_road_pixels = np.sum(road_widths_px <= (main_road_threshold * pixel_per_foot)) if len(road_widths_px) > 0 else 0
        
        # Estimate number of road segments using connected components on skeleton
        num_labels_road, labels_road = cv2.connectedComponents(skeleton.astype(np.uint8))
        num_road_segments = num_labels_road - 1  # Subtract background
        
        # Calculate road area coverage
        total_road_area_px = np.sum(road_mask > 0)
        total_road_area_sf = total_road_area_px / (pixel_per_foot ** 2)
        
        # Estimate ROW and pavement widths (from test_analyse.py)
        row_width_ft = avg_road_width_ft * 1.5  # ROW is typically 1.5x pavement width
        pavement_width_ft = avg_road_width_ft
        
        road_stats = {
            'total_road_length_ft': float(total_road_length_ft),
            'total_road_area_sf': float(total_road_area_sf),
            'avg_road_width_ft': float(avg_road_width_ft),
            'min_road_width_ft': float(min_road_width_ft),
            'max_road_width_ft': float(max_road_width_ft),
            'row_width_ft': float(row_width_ft),
            'pavement_width_ft': float(pavement_width_ft),
            'num_road_segments': int(num_road_segments),
            'main_road_coverage': float(main_road_pixels / max(road_pixels, 1)),
            'residential_road_coverage': float(residential_road_pixels / max(road_pixels, 1))
        }
        
        # Create comprehensive road visualization (matching test_analyse.py)
        fig_road, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original with road mask
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].contour(road_mask, colors='red', linewidths=1)
        axes[0].set_title('Original with Road Detection')
        axes[0].axis('off')
        
        # Road skeleton
        axes[1].imshow(skeleton, cmap='gray')
        axes[1].set_title('Road Centerlines (Skeleton)')
        axes[1].axis('off')
        
        # Distance transform (road widths)
        im = axes[2].imshow(dist_transform, cmap='jet')
        axes[2].set_title('Road Width Analysis')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Road Network Analysis\nTotal Length: {total_road_length_ft:.0f} ft | Avg Width: {avg_road_width_ft:.1f} ft')
        plt.tight_layout()
        
        # Save road analysis
        road_viz_path = "analysis_roads_matplotlib.png"
        plt.savefig(road_viz_path, dpi=150, bbox_inches='tight')
        plt.close(fig_road)
        
        if progress_container:
            with progress_container:
                # Show the matplotlib road analysis instead
                if os.path.exists(road_viz_path):
                    st.image(road_viz_path, caption="Step 3: Road Network Analysis", use_column_width=True)
                else:
                    st.info("Road network analysis visualization generated")
        
        # Step 4: Final annotated site plan using matplotlib
        if progress_container:
            with progress_container:
                st.info("🔄 Step 4: Creating final annotated site plan...")
        
        # Convert to grayscale but preserve green spaces (using already detected green_mask)
        final_viz = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final_viz = cv2.cvtColor(final_viz, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
        
        # Restore green areas using the green_mask detected earlier
        final_viz[green_mask > 0] = img[green_mask > 0]  # Keep original green colors
        
        # Create matplotlib figure for final visualization
        fig_final, ax_final = plt.subplots(1, 1, figsize=(16, 12), dpi=150)
        
        # Convert to RGB for matplotlib
        final_viz_rgb = cv2.cvtColor(final_viz, cv2.COLOR_BGR2RGB)
        ax_final.imshow(final_viz_rgb)
        
        # Find site boundary to position summary outside
        if buildable_lots and max_area_lot:
            # Find the rightmost and bottommost points of the site boundary
            site_mask = (labels == max_area_lot['label']).astype(np.uint8)
            contours, _ = cv2.findContours(site_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get bounding rectangle of site boundary
                site_x, site_y, site_w, site_h = cv2.boundingRect(contours[0])
                # Position summary to the right of site boundary
                summary_x = site_x + site_w + 20
                summary_y = site_y + 20
                
                # If too close to right edge, position below site
                if summary_x + 300 > img.shape[1]:
                    summary_x = max(10, site_x)
                    summary_y = site_y + site_h + 20
            else:
                summary_x, summary_y = img.shape[1] - 320, 20
        else:
            # Default position if no site boundary found
            summary_x, summary_y = img.shape[1] - 320, 20
        
        # Ensure summary stays within image bounds
        summary_x = max(10, min(summary_x, img.shape[1] - 310))
        summary_y = max(20, min(summary_y, img.shape[0] - 160))
        
        # Add lot labels and information using matplotlib (positioned inside each lot)
        for lot in lot_data:
            # Get lot mask to ensure text is inside
            lot_mask = (labels == lot['label']).astype(np.uint8)
            
            # Find the actual lot area to position text inside
            lot_contours, _ = cv2.findContours(lot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if lot_contours:
                # Get bounding rectangle of the lot
                lot_x, lot_y, lot_w, lot_h = cv2.boundingRect(lot_contours[0])
                
                # Position text in the center of the lot bounding box
                cx = lot_x + lot_w // 2
                cy = lot_y + lot_h // 2
                
                # Ensure text fits inside the lot
                text_offset_y = min(15, lot_h // 4)
                
                # Lot number using matplotlib (smaller size)
                ax_final.text(cx, cy - text_offset_y, f"Lot {lot['lot_id']}", 
                            fontsize=8, fontweight='bold', ha='center', va='center',
                            color='black', fontfamily='monospace')
                
                # Area using matplotlib (smaller size)
                ax_final.text(cx, cy + text_offset_y, f"{lot['area_sqft']:.0f} sf", 
                            fontsize=7, ha='center', va='center',
                            color='black', fontfamily='monospace')
        
        # Use already calculated metrics for statistics box
        total_buildable_area_display = total_buildable_area
        usage_rate_display = usage_rate
        
        # Site analysis summary using matplotlib (positioned outside site boundary)
        summary_text = (
            f"SITE ANALYSIS SUMMARY\n"
            f"Total Lots: {len(lot_data)}\n"
            f"Total Area: {total_area_sqft:,.0f} sf\n"
            f"Buildable: {total_buildable_area_display:,.0f} sf\n"
            f"Road Area: {total_road_area_sf:,.0f} sf\n"
            f"Usage Rate: {usage_rate_display:.1f}%\n"
            f"Avg Lot: {avg_lot_size:,.0f} sf"
        )
        
        # Add summary text using matplotlib
        ax_final.text(summary_x, summary_y, summary_text, 
                     fontsize=9, fontweight='bold', va='top',
                     color='black', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', 
                              facecolor='white', 
                              edgecolor='black', 
                              alpha=0.9))
        
        # Remove axes and set proper limits
        ax_final.axis('off')
        ax_final.set_xlim(0, final_viz_rgb.shape[1])
        ax_final.set_ylim(final_viz_rgb.shape[0], 0)
        
        # Save final visualization using matplotlib
        final_viz_path = "analysis_final_viz.png"
        plt.tight_layout()
        plt.savefig(final_viz_path, dpi=150, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.close(fig_final)
        
        if progress_container:
            with progress_container:
                st.image(final_viz, caption="Step 4: Final Annotated Site Plan", use_column_width=True)
                st.success("✅ Site plan analysis completed!")
        
        # All metrics already calculated above - using existing variables
        
        # Prepare comprehensive results (ensure all values are JSON serializable)
        analysis_results = {
            'total_area_sqft': float(site_area_sf),
            'road_area_sqft': float(total_road_area_sf),
            'lot_area_sqft': float(total_buildable_area),
            'num_lots': int(len(buildable_lots)),
            'usage_rate': float(usage_rate),
            'avg_lot_size': float(avg_lot_size),
            'pixel_per_foot': float(pixel_per_foot),
            'lot_data': lot_data,
            'buildable_lots': buildable_lots,
            'road_stats': road_stats,
            'visualization_paths': {
                'highlighted': highlighted_path if highlighted_pil else None,
                'binary': binary_path,
                'lots': lot_viz_path,
                'lots_matplotlib': lot_viz_matplotlib_path,
                'roads': road_viz_path,
                'roads_matplotlib': road_viz_path,
                'final': final_viz_path
            }
        }
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None


def site_planning_tab():
    """AI Site Planning tab with 4-step workflow"""
    st.header("🏘️ AI-Powered Subdivision Planning")
    st.markdown("**Transform property boundaries into comprehensive subdivision layouts** with intelligent road networks and lot configurations")
    
    # Workflow steps
    workflow_steps = [
        {"name": "Boundary Input", "icon": "📤"},
        {"name": "AI Road Generation", "icon": "🛣️"},
        {"name": "Lot Configuration", "icon": "✅"},
        {"name": "Development Analysis", "icon": "📊"}
    ]
    
    # Progress indicator
    current_step = st.session_state.site_planning_step
    progress = current_step / (len(workflow_steps) - 1) if len(workflow_steps) > 1 else 0
    st.progress(progress)
    
    # Step indicator
    cols = st.columns(len(workflow_steps))
    for i, step in enumerate(workflow_steps):
        with cols[i]:
            if i < current_step:
                st.markdown(f"✅ **{step['name']}**")
            elif i == current_step:
                st.markdown(f"{step['icon']} **{step['name']}** ⏳")
            else:
                st.markdown(f"⭕ {step['name']}")
    
    st.markdown("---")
    
    # Preprocess: Input Preparation
    if current_step == 0:
        site_planning_input_step()
    
    # Step 1: Road Network Generation
    elif current_step == 1:
        site_planning_road_generation_step()
    
    # Step 2: Site Plan Finalization
    elif current_step == 2:
        site_planning_finalization_step()
    
    # Step 3: Site Analysis
    elif current_step == 3:
        site_planning_analysis_step()


def site_planning_input_step():
    """Preprocess: Input Preparation - Choose input source"""
    st.subheader("📤 Step 1: Input Preparation")
    
    # Check if we have parsed geometry from deed/PDF processing
    has_geometry = st.session_state.geometry is not None and st.session_state.geometry.vertices
    
    if has_geometry:
        st.success("✅ Boundary polygon available from deed/PDF parsing!")
        
        # Show geometry info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vertices", len(st.session_state.geometry.vertices))
        with col2:
            st.metric("Perimeter", f"{st.session_state.geometry.perimeter:.1f} ft")
        with col3:
            if st.session_state.geometry.area:
                acres = st.session_state.geometry.area / 43560
                st.metric("Area", f"{acres:.2f} acres")
        
        # Export boundary as PNG
        if st.button("🖼️ Export Boundary as PNG", type="primary"):
            with st.spinner("Creating boundary PNG..."):
                png_path, pixel_per_foot = export_boundary_as_png(st.session_state.geometry, st.session_state.calls)
                if png_path:
                    st.session_state.boundary_png_path = png_path
                    st.session_state.pixel_per_foot = pixel_per_foot
                    st.session_state.site_boundary_image = Image.open(png_path)
                    
                    st.success(f"✅ Boundary exported! Scale: {pixel_per_foot:.2f} pixels/foot")
                    
                    # Show the exported image
                    st.image(st.session_state.site_boundary_image, caption="Exported Boundary Polygon (Red Lines)", use_column_width=True)
                    
                    # Proceed to next step
                    st.session_state.site_planning_step = 1
                    st.rerun()
    else:
        st.info("💡 No boundary polygon found from deed/PDF processing")
        
        # Alternative: Upload site image
        st.markdown("### Upload Site Survey Image")
        uploaded_file = st.file_uploader(
            "Choose a site survey image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload a site plan or survey image for processing",
            key="site_image_uploader"
        )
        
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.session_state.uploaded_site_image = uploaded_image
            st.image(uploaded_image, caption="Uploaded Site Image", use_column_width=True)
            
            # Manual pixel/foot input
            pixel_per_foot = st.number_input(
                "Pixels per foot (scale)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Enter the scale conversion rate (pixels per foot)"
            )
            st.session_state.pixel_per_foot = pixel_per_foot
            
            # AI Boundary Extraction Section
            st.markdown("### AI Boundary Extraction")
            
            # Show existing boundary if available
            if st.session_state.site_boundary_image:
                st.success("✅ Site boundary already extracted!")
                st.image(st.session_state.site_boundary_image, caption="Extracted Site Boundary", use_column_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Re-extract Boundary", help="Try extracting boundary again"):
                        boundary_image = extract_site_boundary_ai(uploaded_image)
                        if boundary_image:
                            st.session_state.site_boundary_image = boundary_image
                            st.success("✅ Site boundary re-extracted!")
                            st.rerun()
                
                with col2:
                    if st.button("✅ Accept & Continue", type="primary"):
                        st.session_state.site_planning_step = 1
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Clear & Start Over"):
                        st.session_state.site_boundary_image = None
                        st.rerun()
            else:
                # First time extraction
                if st.button("🤖 Extract Boundary with AI", type="primary"):
                    boundary_image = extract_site_boundary_ai(uploaded_image)
                    if boundary_image:
                        st.session_state.site_boundary_image = boundary_image
                        st.success("✅ Site boundary extracted!")
                        st.rerun()


def site_planning_road_generation_step():
    """Step 1: Road Network Generation"""
    st.subheader("🛣️ Step 2: Road Network Generation")
    
    if not st.session_state.site_boundary_image:
        st.error("❌ No site boundary image found. Please complete Step 1.")
        if st.button("🔙 Back to Input"):
            st.session_state.site_planning_step = 0
            st.rerun()
        return
    
    # Show current boundary
    st.image(st.session_state.site_boundary_image, caption="Site Boundary", width=400)
    
    # Road type selection
    st.markdown("### Select Road Network Type")
    road_type = st.selectbox(
        "Choose road network style:",
        options=['grid', 'organic', 'cul-de-sac', 'hybrid'],
        index=['grid', 'organic', 'cul-de-sac', 'hybrid'].index(st.session_state.selected_road_type),
        help="Select the type of road network to generate"
    )
    st.session_state.selected_road_type = road_type
    
    st.info(f"Selected: **{road_type.capitalize()}** road network")
    
    # Road Network Generation Section
    st.markdown("### Road Network Generation")
    
    # Show existing road network if available
    if st.session_state.road_network_image:
        st.success("✅ Road network already generated!")
        st.image(st.session_state.road_network_image, caption=f"Generated {road_type.capitalize()} Road Network", use_column_width=True)
        
        # Review and action buttons
        st.markdown("### Review Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Regenerate Roads", help="Generate new road network with current settings"):
                road_network = generate_road_network_ai(st.session_state.site_boundary_image, road_type)
                if road_network:
                    st.session_state.road_network_image = road_network
                    st.success("✅ Road network regenerated!")
                    st.rerun()
        
        with col2:
            if st.button("🎨 Try Different Style", help="Change road type and regenerate"):
                # Clear current network to force regeneration with new style
                if road_type != st.session_state.get('last_road_type', road_type):
                    road_network = generate_road_network_ai(st.session_state.site_boundary_image, road_type)
                    if road_network:
                        st.session_state.road_network_image = road_network
                        st.session_state.last_road_type = road_type
                        st.success(f"✅ {road_type.capitalize()} road network generated!")
                        st.rerun()
                else:
                    st.info("💡 Select a different road type first, then click this button")
        
        with col3:
            if st.button("✅ Accept & Continue", type="primary"):
                st.session_state.site_planning_step = 2
                st.rerun()
        
        with col4:
            if st.button("🔙 Back to Input"):
                st.session_state.site_planning_step = 0
                st.rerun()
    
    else:
        # First time generation
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Generate Roads", type="primary"):
                road_network = generate_road_network_ai(st.session_state.site_boundary_image, road_type)
                if road_network:
                    st.session_state.road_network_image = road_network
                    st.session_state.last_road_type = road_type
                    st.success("✅ Road network generated!")
                    st.rerun()
        
        with col2:
            if st.button("🔙 Back to Input"):
                st.session_state.site_planning_step = 0
                st.rerun()


def site_planning_finalization_step():
    """Step 2: Site Plan Finalization"""
    st.subheader("✅ Step 3: Site Plan Finalization")
    
    if not st.session_state.road_network_image:
        st.error("❌ No road network image found. Please complete Step 2.")
        if st.button("🔙 Back to Road Generation"):
            st.session_state.site_planning_step = 1
            st.rerun()
        return
    
    # Show current road network
    st.image(st.session_state.road_network_image, caption="Road Network", width=400)
    
    # Site Plan Finalization Section
    st.markdown("### Site Plan Finalization")
    
    # Show existing final plan if available
    if st.session_state.final_site_plan:
        st.success("✅ Site plan already finalized!")
        st.image(st.session_state.final_site_plan, caption="Finalized Site Plan with Lots", use_column_width=True)
        
        # Review and action buttons
        st.markdown("### Review Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Re-finalize Plan", help="Generate new finalization with current road network"):
                final_plan = finalize_site_plan_ai(st.session_state.road_network_image)
                if final_plan:
                    st.session_state.final_site_plan = final_plan
                    
                    # Save to temporary file for analysis
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='.') as tmp_file:
                        final_plan.save(tmp_file.name)
                        st.session_state.final_site_plan_path = tmp_file.name
                    
                    st.success("✅ Site plan re-finalized!")
                    st.rerun()
        
        with col2:
            if st.button("🔙 Modify Roads", help="Go back to modify road network"):
                st.session_state.site_planning_step = 1
                st.rerun()
        
        with col3:
            if st.button("✅ Accept & Continue", type="primary"):
                st.session_state.site_planning_step = 3
                st.rerun()
        
        with col4:
            if st.button("🗑️ Clear & Restart", help="Clear finalization and start over"):
                st.session_state.final_site_plan = None
                if hasattr(st.session_state, 'final_site_plan_path'):
                    try:
                        os.unlink(st.session_state.final_site_plan_path)
                    except:
                        pass
                    del st.session_state.final_site_plan_path
                st.rerun()
    
    else:
        # First time finalization
        st.markdown("**Ready to finalize the site plan with lot divisions and green spaces.**")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎯 Finalize with Lotting", type="primary"):
                final_plan = finalize_site_plan_ai(st.session_state.road_network_image)
                if final_plan:
                    st.session_state.final_site_plan = final_plan
                    
                    # Save to temporary file for analysis
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='.') as tmp_file:
                        final_plan.save(tmp_file.name)
                        st.session_state.final_site_plan_path = tmp_file.name
                    
                    st.success("✅ Site plan finalized!")
                    st.rerun()
        
        with col2:
            if st.button("🔙 Back to Road Generation"):
                st.session_state.site_planning_step = 1
                st.rerun()


def site_planning_analysis_step():
    """Step 3: Site Analysis"""
    st.subheader("📊 Step 4: Site Plan Analysis")
    
    if not st.session_state.final_site_plan:
        st.error("❌ No finalized site plan found. Please complete Step 3.")
        if st.button("🔙 Back to Finalization"):
            st.session_state.site_planning_step = 2
            st.rerun()
        return
    
    # Show final plan
    st.image(st.session_state.final_site_plan, caption="Finalized Site Plan", width=400)
    
    # Scale adjustment
    st.markdown("### Analysis Settings")
    pixel_per_foot = st.number_input(
        "Pixels per foot (for area calculations)",
        min_value=0.1,
        max_value=10.0,
        value=st.session_state.pixel_per_foot,
        step=0.1,
        help="Adjust scale if needed for accurate area calculations"
    )
    st.session_state.pixel_per_foot = pixel_per_foot
    
    # Site Plan Analysis Section
    st.markdown("### Site Plan Analysis")
    
    # Show existing analysis if available
    if st.session_state.site_analysis_results:
        results = st.session_state.site_analysis_results
        
        st.success("✅ Site analysis completed!")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Area", f"{results['total_area_sqft']:,.0f} sf")
        with col2:
            st.metric("Number of Lots", f"{results['num_lots']}")
        with col3:
            st.metric("Average Lot Size", f"{results['avg_lot_size']:,.0f} sf")
        with col4:
            st.metric("Land Usage Rate", f"{results['usage_rate']:.1f}%")
        
        # Additional metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lot Area", f"{results['lot_area_sqft']:,.0f} sf")
            acres_lots = results['lot_area_sqft'] / 43560
            st.caption(f"({acres_lots:.2f} acres)")
        
        with col2:
            st.metric("Road Area", f"{results['road_area_sqft']:,.0f} sf")
            acres_roads = results['road_area_sqft'] / 43560
            st.caption(f"({acres_roads:.2f} acres)")
        
        # Comprehensive Lot Table
        if 'lot_data' in results and results['lot_data']:
            st.markdown("---")
            st.subheader("📋 Detailed Lot Analysis Table")
            
            # Create comprehensive lot DataFrame
            import pandas as pd
            
            lot_table_data = []
            # Use buildable_lots if available, otherwise use all lot_data
            display_lots = results.get('buildable_lots', results['lot_data'])
            
            for lot in display_lots:
                # Use exact metrics from test_analyse.py structure
                area_sqft = lot['area_sqft']
                area_acres = area_sqft / 43560
                
                # Determine lot type based on size (same as test_analyse.py)
                if area_sqft < 2000:
                    lot_type = "Small"
                elif area_sqft < 5000:
                    lot_type = "Standard"
                elif area_sqft < 10000:
                    lot_type = "Large"
                else:
                    lot_type = "Estate"
                
                # Use actual calculated dimensions from test_analyse.py method
                width_ft = lot.get('width_ft', 0)
                depth_ft = lot.get('depth_ft', 0)
                est_house_sf = lot.get('est_house_sf', area_sqft * 0.25)
                orientation = lot.get('orientation', 'Unknown')
                buildable = lot.get('buildable', False)
                compactness = lot.get('compactness', 0)
                green_pct = lot.get('green_percentage', 0)
                
                # Determine lot type based on green space content
                if green_pct >= 50:
                    lot_type = "Green Space"
                elif area_sqft < 2000:
                    lot_type = "Small"
                elif area_sqft < 5000:
                    lot_type = "Standard"
                elif area_sqft < 10000:
                    lot_type = "Large"
                else:
                    lot_type = "Estate"
                
                lot_table_data.append({
                    'Lot #': lot['lot_id'],
                    'Area (sf)': f"{area_sqft:,.0f}",
                    'Area (acres)': f"{area_acres:.3f}",
                    'Width (ft)': f"{width_ft:.1f}",
                    'Depth (ft)': f"{depth_ft:.1f}",
                    'Type': lot_type,
                    'Orientation': orientation,
                    'Green %': f"{green_pct:.1f}%",
                    'Buildable': "✅" if buildable else "❌",
                    'Max House (sf)': f"{est_house_sf:,.0f}",
                    'Compactness': f"{compactness:.2f}",
                    'Location (px)': f"({lot['centroid_x']:.0f}, {lot['centroid_y']:.0f})"
                })
            
            lot_df = pd.DataFrame(lot_table_data)
            
            # Display the table with styling (matching test_analyse.py comprehensive data)
            st.dataframe(
                lot_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Lot #": st.column_config.NumberColumn("Lot #", width="small"),
                    "Area (sf)": st.column_config.TextColumn("Area (sf)", width="medium"),
                    "Area (acres)": st.column_config.TextColumn("Area (acres)", width="small"),
                    "Width (ft)": st.column_config.TextColumn("Width (ft)", width="small"),
                    "Depth (ft)": st.column_config.TextColumn("Depth (ft)", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="medium"),
                    "Orientation": st.column_config.TextColumn("Orientation", width="medium"),
                    "Green %": st.column_config.TextColumn("Green %", width="small"),
                    "Buildable": st.column_config.TextColumn("Buildable", width="small"),
                    "Max House (sf)": st.column_config.TextColumn("Max House", width="medium"),
                    "Compactness": st.column_config.TextColumn("Compactness", width="small"),
                    "Location (px)": st.column_config.TextColumn("Location", width="medium")
                }
            )
            
            # Lot summary statistics
            st.markdown("### 📊 Lot Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                buildable_count = len(results.get('buildable_lots', []))
                st.metric("Buildable Lots", f"{buildable_count}")
            
            with col2:
                all_lots = results['lot_data']
                green_spaces = sum(1 for lot in all_lots if lot.get('green_percentage', 0) >= 50)
                st.metric("Green Spaces", f"{green_spaces}")
            
            with col3:
                large_lots = sum(1 for lot in all_lots if lot['area_sqft'] >= 10000 and lot.get('green_percentage', 0) < 50)
                st.metric("Estate Lots", f"{large_lots}")
            
            with col4:
                buildable_lots_data = results.get('buildable_lots', all_lots)
                total_building_capacity = sum(lot.get('est_house_sf', 0) for lot in buildable_lots_data)
                st.metric("Total Building Capacity", f"{total_building_capacity:,.0f} sf")
        
        # Analysis visualizations if available
        if 'visualization_paths' in results:
            st.markdown("---")
            st.subheader("🔍 Analysis Visualizations")
            
            # Show all analysis steps including matplotlib visualizations (matching test_analyse.py)
            viz_tabs = st.tabs(["Edge Highlighting", "Binary Detection", "Lot Analysis (Matplotlib)", "Road Analysis (Matplotlib)", "Final Plan"])
            
            with viz_tabs[0]:
                if results['visualization_paths']['highlighted'] and os.path.exists(results['visualization_paths']['highlighted']):
                    st.image(results['visualization_paths']['highlighted'], 
                           caption="Preprocess: Edge Highlighting - Red Borders Added", 
                           use_container_width=True)
                else:
                    st.info("Edge highlighting was not performed or failed")
            
            with viz_tabs[1]:
                if os.path.exists(results['visualization_paths']['binary']):
                    st.image(results['visualization_paths']['binary'], 
                           caption="Step 1: Binary Conversion - Red Line Detection", 
                           use_container_width=True)
            
            with viz_tabs[2]:
                # Show matplotlib lot analysis (same as test_analyse.py)
                if os.path.exists(results['visualization_paths']['lots_matplotlib']):
                    st.image(results['visualization_paths']['lots_matplotlib'], 
                           caption="Step 2: Professional Lot Analysis (Matplotlib Overlay)", 
                           use_container_width=True)
                elif os.path.exists(results['visualization_paths']['lots']):
                    st.image(results['visualization_paths']['lots'], 
                           caption="Step 2: Lot Detection and Sizing", 
                           use_container_width=True)
            
            with viz_tabs[3]:
                # Show matplotlib road analysis (same as test_analyse.py)
                if os.path.exists(results['visualization_paths']['roads_matplotlib']):
                    st.image(results['visualization_paths']['roads_matplotlib'], 
                           caption="Step 3: Comprehensive Road Network Analysis (Matplotlib)", 
                           use_container_width=True)
                elif os.path.exists(results['visualization_paths']['roads']):
                    st.image(results['visualization_paths']['roads'], 
                           caption="Step 3: Road Network Analysis", 
                           use_container_width=True)
            
            with viz_tabs[4]:
                if os.path.exists(results['visualization_paths']['final']):
                    st.image(results['visualization_paths']['final'], 
                           caption="Step 4: Final Annotated Site Plan", 
                           use_container_width=True)
        
        # Road Network Statistics (matching test_analyse.py)
        if 'road_stats' in results and results['road_stats']:
            st.markdown("---")
            st.subheader("🛣️ Comprehensive Road Network Analysis")
            
            road_stats = results['road_stats']
            
            # Road metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Road Length", f"{road_stats['total_road_length_ft']:,.0f} ft")
            with col2:
                st.metric("Average Width", f"{road_stats['avg_road_width_ft']:.1f} ft")
            with col3:
                st.metric("Road Segments", f"{road_stats['num_road_segments']}")
            with col4:
                st.metric("ROW Width", f"{road_stats['row_width_ft']:.1f} ft")
            
            # Additional road metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Width", f"{road_stats['min_road_width_ft']:.1f} ft")
            with col2:
                st.metric("Max Width", f"{road_stats['max_road_width_ft']:.1f} ft")
            with col3:
                main_pct = road_stats['main_road_coverage'] * 100
                st.metric("Main Roads", f"{main_pct:.1f}%")
            with col4:
                res_pct = road_stats['residential_road_coverage'] * 100
                st.metric("Residential Roads", f"{res_pct:.1f}%")
        
        # Review and action buttons
        st.markdown("### Review Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Re-analyze", help="Run analysis again with current settings"):
                if hasattr(st.session_state, 'final_site_plan_path'):
                    # Create progress container
                    progress_container = st.empty()
                    
                    analysis_results = analyze_site_plan_comprehensive(
                        st.session_state.final_site_plan_path, 
                        pixel_per_foot,
                        progress_container
                    )
                    if analysis_results:
                        st.session_state.site_analysis_results = analysis_results
                        progress_container.empty()  # Clear progress
                        st.success("✅ Analysis re-run completed!")
                        st.rerun()
        
        with col2:
            if st.button("🔧 Adjust Scale", help="Change scale and re-analyze"):
                st.info("💡 Adjust the 'Pixels per foot' value above, then click 'Re-analyze'")
        
        with col3:
            if st.button("🔙 Modify Plan", help="Go back to modify the site plan"):
                st.session_state.site_planning_step = 2
                st.rerun()
        
        with col4:
            if st.button("✅ Analysis Complete", type="primary", help="Finish workflow"):
                st.balloons()
                st.success("🎉 Site planning workflow completed successfully!")
        
        # Export options
        st.markdown("---")
        st.subheader("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download final site plan
            if st.session_state.final_site_plan:
                buf = BytesIO()
                st.session_state.final_site_plan.save(buf, format='PNG')
                byte_data = buf.getvalue()
                
                st.download_button(
                    label="🖼️ Download Site Plan",
                    data=byte_data,
                    file_name="final_site_plan.png",
                    mime="image/png"
                )
        
        with col2:
            # Export analysis as JSON (ensure JSON serializable)
            serializable_results = make_json_serializable(results)
            json_data = json.dumps(serializable_results, indent=2)
            st.download_button(
                label="📄 Download Analysis",
                data=json_data,
                file_name="site_analysis.json",
                mime="application/json"
            )
        
        with col3:
            # Reset workflow
            if st.button("🔄 Start New Project"):
                # Reset site planning state
                for key in list(st.session_state.keys()):
                    if key.startswith(('site_', 'boundary_png', 'road_network', 'final_site')):
                        del st.session_state[key]
                st.session_state.site_planning_step = 0
                st.rerun()
    
    else:
        # First time analysis
        st.markdown("**Ready to analyze the finalized site plan for lot metrics and efficiency.**")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Analyze Site Plan", type="primary"):
                if hasattr(st.session_state, 'final_site_plan_path'):
                    # Create progress container for live updates
                    progress_container = st.empty()
                    
                    analysis_results = analyze_site_plan_comprehensive(
                        st.session_state.final_site_plan_path, 
                        pixel_per_foot,
                        progress_container
                    )
                    if analysis_results:
                        st.session_state.site_analysis_results = analysis_results
                        progress_container.empty()  # Clear progress display
                        st.success("✅ Analysis completed!")
                        st.rerun()
        
        with col2:
            if st.button("🔙 Back to Finalization"):
                st.session_state.site_planning_step = 2
                st.rerun()


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
