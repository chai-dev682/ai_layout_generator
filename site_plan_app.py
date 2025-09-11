import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
import time
from datetime import datetime
import cv2
import numpy as np

# Import platform configuration for cross-platform compatibility
from src.utils.platform_config import platform_config, configure_for_deployment
import csv
import matplotlib.pyplot as plt

# Configure platform-specific settings
try:
    deployment_ready = configure_for_deployment()
    if not deployment_ready:
        print("Warning: Some platform dependencies may not be properly configured")
except Exception as e:
    print(f"Failed to configure platform dependencies: {e}")
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage import morphology
import json

# Page configuration
st.set_page_config(
    page_title="Subdivision Planning Tool",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Center the image container */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #0056b3;
    }
    
    /* Title styling */
    h1 {
        text-align: center;
        color: #333;
        margin-bottom: 1rem;
    }
    
    h2, h3 {
        color: #333;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini client
def get_client():
    """Get Gemini client with current API key"""
    api_key = st.session_state.get('gemini_api_key', '')
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        print("[INIT] Gemini client initialized successfully")
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini client: {str(e)}")
        return None

# Initialize session state
if 'workflow_step' not in st.session_state:
    st.session_state.workflow_step = 0
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'boundary_image' not in st.session_state:
    st.session_state.boundary_image = None
if 'road_network_image' not in st.session_state:
    st.session_state.road_network_image = None
if 'final_image' not in st.session_state:
    st.session_state.final_image = None
if 'road_type' not in st.session_state:
    st.session_state.road_type = 'grid'
if 'current_display_image' not in st.session_state:
    st.session_state.current_display_image = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'pixel_per_foot' not in st.session_state:
    st.session_state.pixel_per_foot = 2.0

# Workflow steps definition
workflow_steps = [
    {"name": "Upload Site Survey", "icon": "📤"},
    {"name": "Extract Site Boundary", "icon": "🗺️"},
    {"name": "Select Road Type", "icon": "🛣️"},
    {"name": "Generate Road Network", "icon": "🏗️"},
    {"name": "Finalize Site Plan", "icon": "✅"},
    {"name": "Analyze Site Plan", "icon": "📊"}
]

def extract_site_boundary(site_plan_image):
    """Extract site boundary polygon from site plan image"""
    print("[PROGRESS] Starting site boundary extraction...")
    
    # Get client and check if API key is configured
    client = get_client()
    if not client:
        st.error("❌ Please enter your Google API key in the sidebar to proceed.")
        return None
    
    # Get selected model
    model = st.session_state.get('selected_model', 'gemini-2.5-flash-image-preview')
    
    with st.spinner("🔄 Extracting site boundary polygon..."):
        prompt = """
        Extract ONLY site boundary polygon from provided site plan image (lines: red, no vertices).
        """
        
        try:
            print("[PROGRESS] Sending request to Gemini API for boundary extraction...")
            response = client.models.generate_content(
                model=model,
                contents=[[site_plan_image, prompt, 
                "Remove ALL elements Except only site boundary polygon (red lines)"], 
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
                    # Save the boundary image
                    final_image.save("site_boundary.png")
                    print("[SUCCESS] Site boundary extracted and saved as site_boundary.png")
                    return final_image
            
            print("[WARNING] No image received from API")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to extract boundary: {str(e)}")
            st.error(f"Error extracting boundary: {str(e)}")
            return None

def generate_road_network(boundary_image, road_type):
    """Generate road network inside site boundary"""
    print(f"[PROGRESS] Starting {road_type} road network generation...")
    
    # Get client and check if API key is configured
    client = get_client()
    if not client:
        st.error("❌ Please enter your Google API key in the sidebar to proceed.")
        return None
    
    # Get selected model
    model = st.session_state.get('selected_model', 'gemini-2.5-flash-image-preview')
    
    with st.spinner(f"🔄 Generating {road_type} road network..."):
        prompt = (
            "Provided image is extracted site boundary polygon (red lines)",
            f"Generate {road_type}-type road network INSIDE of site boundary polygon ONLY while considering lotting",
        )
        
        try:
            print(f"[PROGRESS] Sending request to Gemini API for {road_type} road network...")
            response = client.models.generate_content(
                model=model,
                contents=[boundary_image, prompt,
                       "POB(Point Of Beginning) of road network is on the LEFT of the site boundary. Redesign road network!!!", 
                       "POB(Point Of Beginning) of road network is on the LEFT of the site boundary. Confirm it and if not, Redesign road network!!!",
                       "When considering lotting, there can be at most one house in each lot.", 
                       "When considering lotting, every lots with one house must have access into the road. You can add cul-de-sacs if only necessary. Confirm it and if not, Redesign road network!!!", 
                       "Every roads must be connected. Confirm it and if not, Redesign road network!!!",
                       "add green spaces to make site plan more engineer-like", 
                       "Update to more engineer-like road network"],
            )
            print("[PROGRESS] Gemini API response received for road network")
            
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    # Save the road network image
                    image.save("road_network.png")
                    print(f"[SUCCESS] {road_type.capitalize()} road network generated and saved as road_network.png")
                    return image
            
            print("[WARNING] No road network image received from API")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to generate road network: {str(e)}")
            st.error(f"Error generating road network: {str(e)}")
            return None

def finalize_with_lotting(road_network_image):
    """Finalize site plan with lotting"""
    print("[PROGRESS] Starting finalization with lotting...")
    
    # Get client and check if API key is configured
    client = get_client()
    if not client:
        st.error("❌ Please enter your Google API key in the sidebar to proceed.")
        return None
    
    # Get selected model
    model = st.session_state.get('selected_model', 'gemini-2.5-flash-image-preview')
    
    with st.spinner("🔄 Finalizing site plan with lotting..."):
        try:
            print("[PROGRESS] Sending request to Gemini API for lotting...")
            response = client.models.generate_content(
                model=model,
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
                    # Save the finalized image
                    image.save("finalized_plan.png")
                    print("[SUCCESS] Site plan finalized and saved as finalized_plan.png")
                    return image
            
            print("[WARNING] No finalized image received from API")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to finalize site plan: {str(e)}")
            st.error(f"Error finalizing site plan: {str(e)}")
            return None

def convert_red_to_binary(input_path, output_path):
    """
    Convert an image to binary: red pixels to white, others to black
    """
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return False
    
    # Convert BGR to HSV for better red detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define WIDER range for red color in HSV
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([20, 255, 255])
    
    lower_red2 = np.array([160, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    
    lower_red3 = np.array([0, 20, 100])
    upper_red3 = np.array([10, 150, 255])
    
    # Create masks for all red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
    
    # Combine all masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.bitwise_or(red_mask, mask3)
    
    # Apply morphological operations to enhance edges
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    
    # Close small gaps and strengthen edges
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    red_mask = cv2.dilate(red_mask, kernel_small, iterations=1)
    red_mask = cv2.medianBlur(red_mask, 3)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # Apply edge detection and add back
    edges = cv2.Canny(red_mask, 50, 150)
    red_mask = cv2.bitwise_or(red_mask, edges)
    
    # Save the binary image
    cv2.imwrite(output_path, red_mask)
    print(f"Binary image saved to {output_path}")
    
    return True

def detect_bounding_boxes(binary_path, pixel_per_foot=1.0):
    """
    Detect irregular bounding boxes using connected components with pixel-accurate area measurement
    """
    # Read binary image
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    if binary_img is None:
        print(f"Error: Could not read binary image from {binary_path}")
        return None, None
    
    # Apply morphological closing to ensure closed boundaries
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Invert image (regions become white, boundaries become black)
    inverted = cv2.bitwise_not(closed_img)
    
    # Use connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    # Process each component (skip label 0 which is background)
    lot_data = []
    min_area_threshold = 500  # Minimum pixels to be considered a valid lot
    
    for label in range(1, num_labels):
        # Extract statistics for this component
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        
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
        
        # Calculate additional metrics
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate area in square feet
        area_sq_ft = area / (pixel_per_foot ** 2)
        perimeter_ft = perimeter / pixel_per_foot
        
        # Store all data
        lot_data.append({
            'Lot_ID': len(lot_data) + 1,
            'Label': label,
            'Area_Pixels': area,
            'Area_SqFt': area_sq_ft,
            'Perimeter_Pixels': perimeter,
            'Perimeter_Ft': perimeter_ft,
            'Centroid_X': cx,
            'Centroid_Y': cy,
            'BBox_X': x,
            'BBox_Y': y,
            'BBox_Width': w,
            'BBox_Height': h,
            'Compactness': compactness,
            'Aspect_Ratio': aspect_ratio
        })
    
    # Extended lot analysis
    for lot in lot_data:
        # Calculate width and depth in feet
        width_px = lot['BBox_Width']
        height_px = lot['BBox_Height']
        
        if width_px <= height_px:
            lot['Width_Ft'] = width_px / pixel_per_foot
            lot['Depth_Ft'] = height_px / pixel_per_foot
            lot['Orientation'] = 'North-South' if height_px > width_px * 1.2 else 'Square'
        else:
            lot['Width_Ft'] = height_px / pixel_per_foot
            lot['Depth_Ft'] = width_px / pixel_per_foot
            lot['Orientation'] = 'East-West' if width_px > height_px * 1.2 else 'Square'
        
        # Check if lot faces road (simplified)
        lot['Faces_Road'] = lot['Area_Pixels'] != max(lot_data, key=lambda x: x['Area_Pixels'])['Area_Pixels']
        
        # Determine if buildable
        min_buildable_area = 2000  # Minimum 2000 sf for buildable lot
        lot['Buildable'] = lot['Area_SqFt'] >= min_buildable_area and lot['Area_SqFt'] < 50000
        
        # Estimate house size based on lot area
        if lot['Buildable']:
            coverage_ratio = 0.25  # 25% coverage
            lot['Est_House_SF'] = lot['Area_SqFt'] * coverage_ratio
        else:
            lot['Est_House_SF'] = 0
    
    return lot_data, labels

def analyze_road_network(original_image_path, pixel_per_foot):
    """
    Analyze the road network from the black areas in the original image
    """
    # Read the original image
    img = cv2.imread(original_image_path)
    if img is None:
        print(f"Error: Could not read image from {original_image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect black areas (roads)
    _, road_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Skeletonize to find road centerlines
    skeleton = morphology.skeletonize(road_mask // 255)
    
    # Calculate total road length from skeleton
    road_pixels = np.sum(skeleton)
    total_road_length_ft = road_pixels / pixel_per_foot
    
    # Find road widths using distance transform
    dist_transform = cv2.distanceTransform(road_mask, cv2.DIST_L2, 5)
    
    # Get road widths along the skeleton
    road_widths_px = dist_transform[skeleton > 0] * 2  # Multiply by 2 for full width
    if len(road_widths_px) > 0:
        avg_road_width_ft = np.mean(road_widths_px) / pixel_per_foot
        min_road_width_ft = np.min(road_widths_px) / pixel_per_foot
        max_road_width_ft = np.max(road_widths_px) / pixel_per_foot
    else:
        avg_road_width_ft = min_road_width_ft = max_road_width_ft = 0
    
    # Classify roads by width
    main_road_threshold = 30  # Roads wider than 30 ft are main roads
    main_road_pixels = np.sum(road_widths_px > (main_road_threshold * pixel_per_foot))
    residential_road_pixels = np.sum(road_widths_px <= (main_road_threshold * pixel_per_foot))
    
    # Estimate number of road segments
    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
    num_road_segments = num_labels - 1
    
    # Calculate road area coverage
    total_road_area_px = np.sum(road_mask > 0)
    total_road_area_sf = total_road_area_px / (pixel_per_foot ** 2)
    
    # Estimate ROW and pavement widths
    row_width_ft = avg_road_width_ft * 1.5
    pavement_width_ft = avg_road_width_ft
    
    road_stats = {
        'Total_Road_Length_Ft': total_road_length_ft,
        'Total_Road_Area_SF': total_road_area_sf,
        'Avg_Road_Width_Ft': avg_road_width_ft,
        'Min_Road_Width_Ft': min_road_width_ft,
        'Max_Road_Width_Ft': max_road_width_ft,
        'ROW_Width_Ft': row_width_ft,
        'Pavement_Width_Ft': pavement_width_ft,
        'Num_Road_Segments': num_road_segments,
        'Main_Road_Coverage': main_road_pixels / max(road_pixels, 1),
        'Residential_Road_Coverage': residential_road_pixels / max(road_pixels, 1)
    }
    
    return road_stats

def create_matplotlib_visualization(original_image_path, lot_data, labels_img, pixel_per_foot, road_stats=None, output_path="lot_matplotlib_viz.png"):
    """
    Create matplotlib visualization overlaying lot numbers on the original site plan image.
    Skip the largest lot (site boundary) and calculate usage rate.
    """
    print("\nCreating matplotlib visualization with lot details...")
    
    # Read the original site plan image
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        print(f"Error: Could not read original image from {original_image_path}")
        return None
    # Convert BGR to RGB for matplotlib
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Find the largest lot (site boundary)
    max_area_lot = max(lot_data, key=lambda x: x['Area_Pixels'])
    site_boundary_label = max_area_lot['Label']
    site_area_sf = max_area_lot['Area_SqFt']
    
    # Filter out the site boundary from regular buildable lots
    buildable_lots = [lot for lot in lot_data if lot['Label'] != site_boundary_label]
    
    # Calculate usage rate
    total_buildable_area = sum(lot['Area_SqFt'] for lot in buildable_lots)
    usage_rate = (total_buildable_area / site_area_sf) * 100
    
    print(f"\nSite Analysis:")
    print(f"  Site Boundary Area: {site_area_sf:,.0f} sf")
    print(f"  Total Buildable Area: {total_buildable_area:,.0f} sf")
    print(f"  Usage Rate: {usage_rate:.1f}%")
    print(f"  Number of Buildable Lots: {len(buildable_lots)}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
    
    # Convert red lines to black in the original image for cleaner visualization
    # Detect red pixels
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Change red pixels to black
    modified_img = original_img_rgb.copy()
    modified_img[red_mask > 0] = [0, 0, 0]  # Change red to black
    
    # Display the modified site plan image
    ax.imshow(modified_img, alpha=1.0)
    
    # Create semi-transparent overlay for lot areas
    overlay = np.zeros((labels_img.shape[0], labels_img.shape[1], 4), dtype=np.float32)
    
    # Generate distinct colors for buildable lots
    np.random.seed(42)
    cmap = plt.cm.get_cmap('Set3')  # Use Set3 for nice pastel colors
    colors = [cmap(i / len(buildable_lots)) for i in range(len(buildable_lots))]
    
    # Create masks for each buildable lot and add semi-transparent color overlay
    for idx, lot in enumerate(buildable_lots):
        label = lot['Label']
        mask = labels_img == label
        
        # Apply semi-transparent color to this lot area
        color = colors[idx % len(colors)]
        overlay[mask] = [color[0], color[1], color[2], 0.3]  # 30% opacity
    
    # Display the overlay
    ax.imshow(overlay, interpolation='nearest')
    
    # Add lot numbers and areas for buildable lots only
    for i, lot in enumerate(buildable_lots, 1):
        cx = lot['Centroid_X']
        cy = lot['Centroid_Y']
        area_sf = lot['Area_SqFt']
        
        # Determine font size based on lot area
        if area_sf > 1000:
            font_size = 10
            text_padding = 0.4
        elif area_sf > 500:
            font_size = 9
            text_padding = 0.35
        else:
            font_size = 8
            text_padding = 0.3
        
        # Create text with lot number and area
        lot_text = f"Lot {i}"
        area_text = f"{area_sf:.0f} sf"
        
        # Add lot number with strong visibility
        ax.text(cx, cy - 8, lot_text,
                fontsize=font_size, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle=f'round,pad={text_padding}', 
                         facecolor='white', 
                         edgecolor='darkblue', 
                         linewidth=1.5, 
                         alpha=0.95),
                color='darkblue')
        
        # Add area below lot number
        ax.text(cx, cy + 8, area_text,
                fontsize=font_size * 0.9,
                ha='center', va='center',
                bbox=dict(boxstyle=f'round,pad={text_padding}', 
                         facecolor='yellow', 
                         edgecolor='darkgreen', 
                         linewidth=1.5, 
                         alpha=0.95),
                color='darkgreen')
    
    # Set title with usage rate
    title = (f'Subdivision Lot Analysis\n'
             f'{len(buildable_lots)} Lots | '
             f'Site: {site_area_sf:,.0f} sf | '
             f'Buildable: {total_buildable_area:,.0f} sf | '
             f'Usage: {usage_rate:.1f}%')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set axis labels
    ax.set_xlabel(f'Width (pixels) - Scale: {pixel_per_foot:.2f} pixels/foot', fontsize=10)
    ax.set_ylabel('Height (pixels)', fontsize=10)
    
    # Add subtle grid
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Set axis limits to match image dimensions
    ax.set_xlim(0, original_img_rgb.shape[1])
    ax.set_ylim(original_img_rgb.shape[0], 0)
    
    # Create statistics box
    if road_stats:
        stats_text = (
            f"Statistics:\n"
            f"Lots: {len(buildable_lots)}\n"
            f"Avg Lot: {np.mean([lot['Area_SqFt'] for lot in buildable_lots]):,.0f} sf\n"
            f"Roads: {road_stats['Total_Road_Length_Ft']:,.0f} ft"
        )
    else:
        stats_text = (
            f"Statistics:\n"
            f"Lots: {len(buildable_lots)}\n"
            f"Avg Lot: {np.mean([lot['Area_SqFt'] for lot in buildable_lots]):,.0f} sf"
        )
    
    # Add statistics box
    props = dict(boxstyle='round,pad=0.5', 
                facecolor='lightcyan', 
                edgecolor='navy', 
                linewidth=2, 
                alpha=0.95)
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes, 
           fontsize=9,
           verticalalignment='top', 
           fontfamily='monospace', 
           bbox=props)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"\nMatplotlib visualization saved to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Return the path to the saved image
    return output_path

def analyze_site_plan(final_image_path, pixel_per_foot):
    """
    Comprehensive site plan analysis
    """
    print("[PROGRESS] Starting comprehensive site plan analysis...")
    
    # Convert to binary
    binary_path = "analysis_binary.png"
    success = convert_red_to_binary(final_image_path, binary_path)
    
    if not success:
        return None
    
    # Detect lots
    lot_data, labels = detect_bounding_boxes(binary_path, pixel_per_foot)
    
    if not lot_data:
        return None
    
    # Analyze road network
    road_stats = analyze_road_network(final_image_path, pixel_per_foot)
    
    # Calculate site efficiency
    max_area_lot = max(lot_data, key=lambda x: x['Area_Pixels'])
    buildable_lots = [lot for lot in lot_data if lot['Label'] != max_area_lot['Label']]
    
    total_buildable_area = sum(lot['Area_SqFt'] for lot in buildable_lots if lot['Buildable'])
    site_area = max_area_lot['Area_SqFt']
    usage_rate = (total_buildable_area / site_area) * 100 if site_area > 0 else 0
    
    # Create matplotlib visualization with lot numbers and sizes
    viz_path = create_matplotlib_visualization(
        final_image_path,
        lot_data,
        labels,
        pixel_per_foot,
        road_stats,
        output_path="lot_analysis_visualization.png"
    )
    
    analysis_results = {
        'lot_data': lot_data,
        'buildable_lots': buildable_lots,
        'road_stats': road_stats,
        'site_area': site_area,
        'total_buildable_area': total_buildable_area,
        'usage_rate': usage_rate,
        'num_buildable_lots': len(buildable_lots),
        'labels': labels,
        'visualization_path': viz_path
    }
    
    return analysis_results

# Sidebar configuration
def setup_sidebar():
    """Setup sidebar with Google API key configuration"""
    st.sidebar.header("⚙️ Configuration")
    
    # Google Gemini API Key
    st.sidebar.subheader("Google Gemini API")
    google_api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        value=st.session_state.get('gemini_api_key', ""),
        help="Enter your Google Gemini API key for AI-powered site planning"
    )
    
    if google_api_key:
        st.session_state.gemini_api_key = google_api_key
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.warning("⚠️ API Key required for processing")
    
    # Additional settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Processing Settings")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash-image-preview", "gemini-1.5-pro-vision-latest"],
        index=0,
        help="Select the Gemini model for image processing"
    )
    st.session_state.selected_model = model

# Main app
print("\n" + "="*50)
print("[START] Subdivision Planning Tool Started")
print("="*50 + "\n")

# Setup sidebar
setup_sidebar()

st.title("🏘️ Smart Subdivision Planning Tool")
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>AI-Powered Site Development Workflow</p>", unsafe_allow_html=True)

# Progress indicator in main area
st.markdown("---")
cols = st.columns(len(workflow_steps))
for idx, (col, step) in enumerate(zip(cols, workflow_steps)):
    with col:
        if idx < st.session_state.workflow_step:
            st.success(f"{step['icon']} {step['name']} ✓")
        elif idx == st.session_state.workflow_step:
            st.info(f"{step['icon']} {step['name']} ⏳")
        else:
            st.markdown(f"⭕ {step['name']}")

# Progress bar
progress = st.session_state.workflow_step / len(workflow_steps)
st.progress(progress)
st.caption(f"Step {st.session_state.workflow_step + 1} of {len(workflow_steps)}")
st.markdown("---")

# Create layout
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Image display container
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    if st.session_state.current_display_image:
        # Determine caption based on workflow step
        if st.session_state.workflow_step == 5 and st.session_state.analysis_results:
            caption = "Lot Analysis Visualization - Showing Lot Numbers and Areas"
        else:
            caption = "Current Processing Stage"
        
        st.image(st.session_state.current_display_image, 
                caption=caption, 
                use_container_width=True)
    else:
        st.info("📷 Please upload a site survey image to begin")
    st.markdown("</div>", unsafe_allow_html=True)

# Main workflow area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Step 1: Upload Site Survey
    if st.session_state.workflow_step == 0:
        st.header("📤 Step 1: Upload Site Survey Image")
        
        uploaded_file = st.file_uploader(
            "Choose a site survey image",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload a site plan or survey image for processing"
        )
        
        if uploaded_file is not None:
            print(f"[PROGRESS] File uploaded: {uploaded_file.name}")
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.session_state.current_display_image = st.session_state.uploaded_image
            
            st.success("✅ Image uploaded successfully!")
            print(f"[SUCCESS] Image loaded successfully: {uploaded_file.name}")
            
            if st.button("Proceed to Boundary Extraction", type="primary"):
                print("[WORKFLOW] Moving to Step 2: Boundary Extraction")
                st.session_state.workflow_step = 1
                st.rerun()
    
    # Step 2: Extract Site Boundary
    elif st.session_state.workflow_step == 1:
        st.header("🗺️ Step 2: Extract Site Boundary")
        
        if st.button("Extract Site Boundary", type="primary"):
            print("[BUTTON] Extract Site Boundary clicked")
            boundary_image = extract_site_boundary(st.session_state.uploaded_image)
            
            if boundary_image:
                st.session_state.boundary_image = boundary_image
                st.session_state.current_display_image = boundary_image
                st.success("✅ Site boundary extracted and saved as site_boundary.png")
                print("[WORKFLOW] Moving to Step 3: Road Type Selection")
                time.sleep(2)
                st.session_state.workflow_step = 2
                st.rerun()
            else:
                st.error("❌ Failed to extract boundary. Please try again.")
                print("[ERROR] Boundary extraction failed")
        
        if st.button("⬅️ Back to Upload"):
            st.session_state.workflow_step = 0
            st.rerun()
    
    # Step 3: Select Road Type
    elif st.session_state.workflow_step == 2:
        st.header("🛣️ Step 3: Select Road Type")
        
        road_type = st.selectbox(
            "Choose road network type:",
            options=['grid', 'organic', 'cul-de-sac', 'hybrid'],
            index=['grid', 'organic', 'cul-de-sac', 'hybrid'].index(st.session_state.road_type),
            help="Select the type of road network to generate"
        )
        st.session_state.road_type = road_type
        print(f"[SELECTION] Road type selected: {road_type}")
        
        st.info(f"Selected: **{road_type.capitalize()}** road network")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ Back to Boundary"):
                st.session_state.workflow_step = 1
                st.rerun()
        with col_b:
            if st.button("Generate Road Network", type="primary"):
                print(f"[WORKFLOW] Moving to Step 4: Generate {st.session_state.road_type} Road Network")
                st.session_state.workflow_step = 3
                st.rerun()
    
    # Step 4: Generate Road Network
    elif st.session_state.workflow_step == 3:
        st.header("🏗️ Step 4: Generate Road Network")
        
        st.info(f"Generating **{st.session_state.road_type}** road network...")
        
        if st.button("Generate Roads", type="primary"):
            print("[BUTTON] Generate Roads clicked")
            road_network = generate_road_network(
                st.session_state.boundary_image, 
                st.session_state.road_type
            )
            
            if road_network:
                st.session_state.road_network_image = road_network
                st.session_state.current_display_image = road_network
                st.success("✅ Road network generated and saved as road_network.png")
                print("[WORKFLOW] Moving to Step 5: Finalization")
                time.sleep(2)
                st.session_state.workflow_step = 4
                st.rerun()
            else:
                st.error("❌ Failed to generate road network. Please try again.")
                print("[ERROR] Road network generation failed")
        
        if st.button("⬅️ Back to Road Type"):
            st.session_state.workflow_step = 2
            st.rerun()
    
    # Step 5: Finalize Site Plan
    elif st.session_state.workflow_step == 4:
        st.header("✅ Step 5: Finalize Site Plan")
        
        if st.button("Finalize with Lotting", type="primary"):
            print("[BUTTON] Finalize with Lotting clicked")
            final_plan = finalize_with_lotting(st.session_state.road_network_image)
            
            if final_plan:
                st.session_state.final_image = final_plan
                st.session_state.current_display_image = final_plan
                st.success("🎉 Site plan finalized and saved as finalized_plan.png")
                print("[WORKFLOW] Moving to Step 6: Analysis")
                time.sleep(2)
                st.session_state.workflow_step = 5
                st.rerun()
            else:
                st.error("❌ Failed to finalize site plan. Please try again.")
                print("[ERROR] Site plan finalization failed")
        
        if st.button("⬅️ Back to Road Network"):
            st.session_state.workflow_step = 3
            st.rerun()
    
    # Step 6: Analyze Site Plan
    elif st.session_state.workflow_step == 5:
        st.header("📊 Step 6: Analyze Site Plan")
        
        # Pixel per foot input
        col_a, col_b = st.columns(2)
        with col_a:
            pixel_per_foot = st.number_input(
                "Pixels per foot (for area calculations)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.pixel_per_foot,
                step=0.1,
                help="Enter the scale conversion rate (pixels per foot)"
            )
            st.session_state.pixel_per_foot = pixel_per_foot
        
        with col_b:
            st.info(f"📏 1 square foot = {pixel_per_foot**2:.1f} pixels")
        
        if st.button("Run Site Analysis", type="primary"):
            print("[BUTTON] Run Site Analysis clicked")
            
            # Run the comprehensive analysis
            analysis_results = analyze_site_plan("finalized_plan.png", pixel_per_foot)
            
            if analysis_results:
                st.session_state.analysis_results = analysis_results
                
                # Load and display the matplotlib visualization with lot numbers
                if analysis_results.get('visualization_path'):
                    viz_image = Image.open(analysis_results['visualization_path'])
                    st.session_state.current_display_image = viz_image
                    print(f"[SUCCESS] Loaded visualization from {analysis_results['visualization_path']}")
                
                st.success("✅ Analysis completed successfully!")
                st.rerun()  # Rerun to update the display image
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            analysis_results = st.session_state.analysis_results
            
            # Display analysis results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Site summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Lots", analysis_results['num_buildable_lots'])
                st.metric("Site Area", f"{analysis_results['site_area']:,.0f} sf")
            
            with col2:
                st.metric("Buildable Area", f"{analysis_results['total_buildable_area']:,.0f} sf")
                st.metric("Usage Rate", f"{analysis_results['usage_rate']:.1f}%")
            
            with col3:
                if analysis_results['road_stats']:
                    st.metric("Road Length", f"{analysis_results['road_stats']['Total_Road_Length_Ft']:,.0f} ft")
                    st.metric("Road Area", f"{analysis_results['road_stats']['Total_Road_Area_SF']:,.0f} sf")
            
            # Lot details table
            st.markdown("---")
            st.subheader("📋 Lot Details")
            
            # Create DataFrame for display
            import pandas as pd
            buildable_lots = analysis_results['buildable_lots']
            if buildable_lots:
                lot_df = pd.DataFrame([{
                    'Lot ID': lot['Lot_ID'],
                    'Area (sf)': f"{lot['Area_SqFt']:.0f}",
                    'Width (ft)': f"{lot['Width_Ft']:.1f}",
                    'Depth (ft)': f"{lot['Depth_Ft']:.1f}",
                    'Buildable': '✓' if lot['Buildable'] else '✗',
                    'Orientation': lot['Orientation'],
                    'Est. House (sf)': f"{lot['Est_House_SF']:.0f}" if lot['Est_House_SF'] > 0 else "N/A"
                } for lot in buildable_lots[:20]])  # Show first 20 lots
                
                st.dataframe(lot_df, use_container_width=True)
            
            # Road network details
            if analysis_results['road_stats']:
                st.markdown("---")
                st.subheader("🛣️ Road Network Analysis")
                
                road_col1, road_col2 = st.columns(2)
                with road_col1:
                    st.metric("Average Road Width", f"{analysis_results['road_stats']['Avg_Road_Width_Ft']:.1f} ft")
                    st.metric("Min Road Width", f"{analysis_results['road_stats']['Min_Road_Width_Ft']:.1f} ft")
                    st.metric("Max Road Width", f"{analysis_results['road_stats']['Max_Road_Width_Ft']:.1f} ft")
                
                with road_col2:
                    st.metric("ROW Width", f"{analysis_results['road_stats']['ROW_Width_Ft']:.1f} ft")
                    st.metric("Road Segments", f"{analysis_results['road_stats']['Num_Road_Segments']}")
                    main_pct = analysis_results['road_stats']['Main_Road_Coverage'] * 100
                    st.metric("Main Roads", f"{main_pct:.1f}%")
            
            # Export options
            st.markdown("---")
            st.subheader("📥 Export Options")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                # Export lot data as CSV
                if buildable_lots:
                    csv_data = "Lot_ID,Area_SqFt,Width_Ft,Depth_Ft,Buildable,Orientation,Est_House_SF\n"
                    for lot in buildable_lots:
                        csv_data += f"{lot['Lot_ID']},{lot['Area_SqFt']:.0f},{lot['Width_Ft']:.1f},{lot['Depth_Ft']:.1f},{lot['Buildable']},{lot['Orientation']},{lot['Est_House_SF']:.0f}\n"
                    
                    st.download_button(
                        label="📊 Download Lot Data (CSV)",
                        data=csv_data,
                        file_name="lot_analysis.csv",
                        mime="text/csv"
                    )
            
            with col_exp2:
                # Export analysis report as JSON
                json_data = json.dumps({
                    'site_area': analysis_results['site_area'],
                    'total_buildable_area': analysis_results['total_buildable_area'],
                    'usage_rate': analysis_results['usage_rate'],
                    'num_buildable_lots': analysis_results['num_buildable_lots'],
                    'road_stats': analysis_results['road_stats']
                }, indent=2)
                
                st.download_button(
                    label="📄 Download Report (JSON)",
                    data=json_data,
                    file_name="site_analysis_report.json",
                    mime="application/json"
                )
            
            with col_exp3:
                # Download final image
                if st.session_state.final_image:
                    buf = BytesIO()
                    st.session_state.final_image.save(buf, format='PNG')
                    byte_data = buf.getvalue()
                    
                    st.download_button(
                        label="🖼️ Download Site Plan",
                        data=byte_data,
                        file_name="final_site_plan.png",
                        mime="image/png"
                    )
            
            # Download visualization with lot numbers
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            with col_viz2:
                if analysis_results.get('visualization_path'):
                    viz_image = Image.open(analysis_results['visualization_path'])
                    buf_viz = BytesIO()
                    viz_image.save(buf_viz, format='PNG')
                    viz_byte_data = buf_viz.getvalue()
                    
                    st.download_button(
                        label="📐 Download Lot Analysis Map",
                        data=viz_byte_data,
                        file_name="lot_analysis_map.png",
                        mime="image/png"
                    )
            
            # Reset option
            st.markdown("---")
            if st.button("🔄 Start New Project"):
                print("[WORKFLOW] Resetting for new project")
                for key in st.session_state.keys():
                    del st.session_state[key]
                print("[RESET] Session state cleared")
                st.rerun()
        else:
            # Show Run Analysis button again if no results yet
            st.info("Click 'Run Site Analysis' to analyze the subdivision plan")
        
        col_back1, col_back2 = st.columns(2)
        with col_back1:
            if st.button("⬅️ Back to Finalization"):
                st.session_state.workflow_step = 4
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Powered by Google Gemini AI | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
