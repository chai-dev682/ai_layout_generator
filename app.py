"""
Main Streamlit application for the Deed Parser system.
"""
import streamlit as st
import pandas as pd
import json
import os
import logging
from typing import List, Optional
import traceback

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
    tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF Input & Processing", "📝 Text Input & Parsing", "📊 Review & Edit", "🗺️ Visualization"])
    
    with tab1:
        pdf_input_tab()
    
    with tab2:
        text_input_tab()
    
    with tab3:
        review_edit_tab()
    
    with tab4:
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
    
    # PDF Processing API Keys
    st.sidebar.subheader("PDF Processing APIs")
    
    llamaindex_key = st.sidebar.text_input(
        "LlamaIndex API Key",
        type="password",
        value=st.session_state.get('llamaindex_api_key', ""),
        help="For premium OCR with LlamaExtract"
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
    """PDF input and processing tab with 6-step workflow"""
    st.header("📄 PDF Site Survey Processing")
    
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
