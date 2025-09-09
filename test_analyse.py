import cv2
import numpy as np

# Import platform configuration for cross-platform compatibility
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.platform_config import platform_config, configure_for_deployment
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage import morphology
import datetime as import_datetime

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
    
    # Define WIDER range for red color in HSV to capture more variations
    # Including lighter reds, pinks, and orange-reds
    
    # Lower red range (0-20 degrees in hue) - extended from 10 to 20
    lower_red1 = np.array([0, 30, 30])    # Reduced saturation and value thresholds
    upper_red1 = np.array([20, 255, 255]) # Extended hue range to include orange-reds
    
    # Upper red range (160-180 degrees in hue) - extended from 170 to 160
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
    
    # Apply morphological operations to enhance edges
    # Create kernels for different operations
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    
    # Step 1: Close small gaps in lines (connect broken segments)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Step 2: Dilate to strengthen and thicken edges
    red_mask = cv2.dilate(red_mask, kernel_small, iterations=1)
    
    # Step 3: Apply median filter to reduce noise while preserving edges
    red_mask = cv2.medianBlur(red_mask, 3)
    
    # Step 4: Final closing to ensure continuity
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # Optional: Apply edge detection and add back to enhance boundaries
    edges = cv2.Canny(red_mask, 50, 150)
    red_mask = cv2.bitwise_or(red_mask, edges)
    
    # The mask is already binary (255 for red, 0 for others)
    # Save the binary image
    cv2.imwrite(output_path, red_mask)
    print(f"Binary image saved to {output_path}")
    
    return True

def detect_bounding_boxes(binary_path, pixel_per_foot=1.0, output_vis_path="lot_detection.png", output_csv_path="lot_data.csv", return_labels=False):
    """
    Detect irregular bounding boxes using connected components with pixel-accurate area measurement
    
    Args:
        binary_path: Path to binary image
        pixel_per_foot: Number of pixels per foot for conversion to square feet
        output_vis_path: Path for output visualization
        output_csv_path: Path for output CSV data
    """
    # Step 1: Read binary image (white lines on black background)
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    if binary_img is None:
        print(f"Error: Could not read binary image from {binary_path}")
        return None
    
    print(f"Processing image of size: {binary_img.shape}")
    
    # Step 2: Apply morphological closing to ensure closed boundaries
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 3: Invert image (regions become white, boundaries become black)
    inverted = cv2.bitwise_not(closed_img)
    
    # Step 4: Use connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    print(f"Found {num_labels} components (including background)")
    
    # Step 5: Process each component (skip label 0 which is background)
    lot_data = []
    min_area_threshold = 500  # Minimum pixels to be considered a valid lot
    
    for label in range(1, num_labels):
        # Extract statistics for this component
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
        
        # Calculate additional metrics
        # Compactness (circularity): 4π × area / perimeter²
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate area in square feet
        area_sq_ft = area / (pixel_per_foot ** 2)
        perimeter_ft = perimeter / pixel_per_foot
        
        # Store all data
        lot_data.append({
            'Lot_ID': len(lot_data) + 1,
            'Label': label,
            'Area_Pixels': area,  # Exact pixel count!
            'Area_SqFt': area_sq_ft,  # Area in square feet
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
    print("\n" + "="*60)
    print("PERFORMING EXTENDED LOT ANALYSIS")
    print("="*60)
    
    # Analyze each lot for additional properties
    for lot in lot_data:
        # Calculate width and depth in feet
        width_px = lot['BBox_Width']
        height_px = lot['BBox_Height']
        
        # Width is typically the shorter dimension facing the road
        # Depth is the longer dimension perpendicular to the road
        if width_px <= height_px:
            lot['Width_Ft'] = width_px / pixel_per_foot
            lot['Depth_Ft'] = height_px / pixel_per_foot
            lot['Orientation'] = 'North-South' if height_px > width_px * 1.2 else 'Square'
        else:
            lot['Width_Ft'] = height_px / pixel_per_foot
            lot['Depth_Ft'] = width_px / pixel_per_foot
            lot['Orientation'] = 'East-West' if width_px > height_px * 1.2 else 'Square'
        
        # Check if lot faces road (simplified - checking if lot is not the largest one)
        lot['Faces_Road'] = lot['Area_Pixels'] != max(lot_data, key=lambda x: x['Area_Pixels'])['Area_Pixels']
        
        # Determine if buildable (lots larger than minimum and not the site boundary)
        min_buildable_area = 2000  # Minimum 2000 sf for buildable lot
        lot['Buildable'] = lot['Area_SqFt'] >= min_buildable_area and lot['Area_SqFt'] < 50000
        
        # Estimate house size based on lot area (typical 20-30% coverage)
        if lot['Buildable']:
            coverage_ratio = 0.25  # 25% coverage
            lot['Est_House_SF'] = lot['Area_SqFt'] * coverage_ratio
        else:
            lot['Est_House_SF'] = 0
    
    print(f"Detected {len(lot_data)} valid lots (area > {min_area_threshold} pixels)")
    
    # Step 6: Visualize with different colors
    # Create colored visualization
    color_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    
    # Generate random colors for each lot
    np.random.seed(42)  # For reproducible colors
    colors = np.random.randint(50, 255, size=(num_labels, 3))
    colors[0] = [0, 0, 0]  # Background stays black
    
    # Apply colors to labeled regions
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            color_img[y, x] = colors[label]
    
    # Draw boundaries from original binary image
    boundaries = binary_img.copy()
    color_img[boundaries > 0] = [255, 255, 255]  # White boundaries
    
    # Add lot numbers and area text
    for lot in lot_data:
        cx = int(lot['Centroid_X'])
        cy = int(lot['Centroid_Y'])
        
        # Create text with lot ID and area in square feet
        text = f"Lot {lot['Lot_ID']}"
        area_text = f"{lot['Area_SqFt']:.0f} sf"
        
        # Draw text with background for visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6  # Slightly larger for better visibility
        thickness = 2
        
        # Lot ID
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(color_img, (cx - text_w//2 - 2, cy - text_h - 2),
                     (cx + text_w//2 + 2, cy + 2), (255, 255, 255), -1)
        cv2.putText(color_img, text, (cx - text_w//2, cy), font, font_scale, (0, 0, 0), thickness)
        
        # Area below lot ID
        (area_w, area_h), _ = cv2.getTextSize(area_text, font, font_scale*0.8, thickness)
        cv2.rectangle(color_img, (cx - area_w//2 - 2, cy + 5),
                     (cx + area_w//2 + 2, cy + area_h + 8), (255, 255, 255), -1)
        cv2.putText(color_img, area_text, (cx - area_w//2, cy + area_h + 5), 
                   font, font_scale*0.8, (0, 0, 0), thickness)
    
    # Save visualization
    cv2.imwrite(output_vis_path, color_img)
    print(f"Visualization saved to {output_vis_path}")
    
    # Step 7: Export detailed data to CSV
    if lot_data:
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Lot_ID', 'Label', 'Area_Pixels', 'Area_SqFt', 
                         'Perimeter_Pixels', 'Perimeter_Ft', 'Centroid_X', 'Centroid_Y', 
                         'BBox_X', 'BBox_Y', 'BBox_Width', 'BBox_Height', 
                         'Width_Ft', 'Depth_Ft', 'Faces_Road', 'Buildable', 
                         'Orientation', 'Est_House_SF', 'Compactness', 'Aspect_Ratio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lot_data)
        
        print(f"Lot data saved to {output_csv_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("LOT DETECTION SUMMARY")
        print("="*60)
        print(f"Conversion rate: {pixel_per_foot} pixels per foot")
        print("\n--- Area Statistics ---")
        
        areas_px = [lot['Area_Pixels'] for lot in lot_data]
        areas_sf = [lot['Area_SqFt'] for lot in lot_data]
        
        print(f"Total lots detected: {len(lot_data)}")
        print(f"\nIn Square Feet:")
        print(f"  Total area: {sum(areas_sf):,.0f} sf")
        print(f"  Average area: {np.mean(areas_sf):,.0f} sf")
        print(f"  Median area: {np.median(areas_sf):,.0f} sf")
        print(f"  Min area: {min(areas_sf):,.0f} sf")
        print(f"  Max area: {max(areas_sf):,.0f} sf")
        print(f"  Std deviation: {np.std(areas_sf):,.0f} sf")
        
        print(f"\nIn Pixels:")
        print(f"  Total area: {sum(areas_px):,} pixels")
        print(f"  Average area: {np.mean(areas_px):.1f} pixels")
        
        print("\n" + "="*60)
        print("INDIVIDUAL LOT DETAILS")
        print("="*60)
        for lot in sorted(lot_data, key=lambda x: x['Area_SqFt'], reverse=True):
            print(f"Lot #{lot['Lot_ID']:3d}: {lot['Area_SqFt']:8,.0f} sf ({lot['Area_Pixels']:7,} px) | "
                  f"Perimeter: {lot['Perimeter_Ft']:.0f} ft | "
                  f"Compactness: {lot['Compactness']:.2f}")
    
    if return_labels:
        return lot_data, labels
    return lot_data

def analyze_road_network(original_image_path, pixel_per_foot, output_path="road_analysis.png"):
    """
    Analyze the road network from the black areas in the original image
    """
    print("\n" + "="*60)
    print("ANALYZING ROAD NETWORK")
    print("="*60)
    
    # Read the original image
    img = cv2.imread(original_image_path)
    if img is None:
        print(f"Error: Could not read image from {original_image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect black areas (roads) - threshold for very dark pixels
    # Black roads typically have values < 50
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
    
    # Classify roads by width (main vs residential)
    main_road_threshold = 30  # Roads wider than 30 ft are main roads
    main_road_pixels = np.sum(road_widths_px > (main_road_threshold * pixel_per_foot))
    residential_road_pixels = np.sum(road_widths_px <= (main_road_threshold * pixel_per_foot))
    
    # Estimate number of road segments using connected components on skeleton
    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
    num_road_segments = num_labels - 1  # Subtract background
    
    # Calculate road area coverage
    total_road_area_px = np.sum(road_mask > 0)
    total_road_area_sf = total_road_area_px / (pixel_per_foot ** 2)
    
    # Estimate ROW and pavement widths (typical ratios)
    row_width_ft = avg_road_width_ft * 1.5  # ROW is typically 1.5x pavement width
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
    
    # Print road network statistics
    print("\nRoad Network Statistics:")
    print(f"  Total Road Length: {total_road_length_ft:.0f} ft")
    print(f"  Total Road Area: {total_road_area_sf:,.0f} sf")
    print(f"  Average Road Width: {avg_road_width_ft:.1f} ft")
    print(f"  Min Road Width: {min_road_width_ft:.1f} ft")
    print(f"  Max Road Width: {max_road_width_ft:.1f} ft")
    print(f"  ROW Width (estimated): {row_width_ft:.1f} ft")
    print(f"  Pavement Width: {pavement_width_ft:.1f} ft")
    print(f"  Number of Road Segments: {num_road_segments}")
    print(f"  Main Roads: {main_road_pixels / max(road_pixels, 1) * 100:.1f}%")
    print(f"  Residential Roads: {residential_road_pixels / max(road_pixels, 1) * 100:.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nRoad analysis visualization saved to {output_path}")
    plt.show()
    
    return road_stats

def create_matplotlib_visualization(original_image_path, binary_path, lot_data, labels_img, pixel_per_foot, road_stats=None, output_path="lot_matplotlib_viz.png"):
    """
    Create matplotlib visualization overlaying lot numbers on the original site plan image.
    Skip the largest lot (site boundary) and calculate usage rate.
    """
    print("\nCreating matplotlib visualization with original site plan...")
    
    # Read the original site plan image
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        print(f"Error: Could not read original image from {original_image_path}")
        return
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
    fig, ax = plt.subplots(1, 1, figsize=(18, 12), dpi=100)
    
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
    
    # Set title with usage rate
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
    
    # Create detailed statistics box including road stats
    stats_text = (
        f"╔══════════════════════════╗\n"
        f"║   SITE STATISTICS        ║\n"
        f"╠══════════════════════════╣\n"
        f"║ Buildable Lots: {len(buildable_lots):>9}║\n"
        f"║ Site Area:  {site_area_sf:>10,.0f} sf║\n"
        f"║ Buildable:  {total_buildable_area:>10,.0f} sf║\n"
        f"║ Usage Rate:     {usage_rate:>8.1f}%║\n"
        f"╠══════════════════════════╣\n"
        f"║ Avg Lot: {np.mean([lot['Area_SqFt'] for lot in buildable_lots]):>13,.0f} sf║\n"
        f"║ Min Lot: {min([lot['Area_SqFt'] for lot in buildable_lots]):>13,.0f} sf║\n"
        f"║ Max Lot: {max([lot['Area_SqFt'] for lot in buildable_lots]):>13,.0f} sf║\n"
    )
    
    if road_stats:
        stats_text += (
            f"╠══════════════════════════╣\n"
            f"║     ROAD NETWORK         ║\n"
            f"╠══════════════════════════╣\n"
            f"║ Total Length: {road_stats['Total_Road_Length_Ft']:>8.0f} ft║\n"
            f"║ Avg Width:    {road_stats['Avg_Road_Width_Ft']:>8.1f} ft║\n"
        )
    
    stats_text += f"╚══════════════════════════╝"
    
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
    
    # Add scale reference
    scale_text = f"Scale: 1 foot = {pixel_per_foot:.2f} pixels"
    ax.text(0.98, 0.02, scale_text, 
           transform=ax.transAxes, 
           fontsize=10,
           ha='right',
           bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', 
                    edgecolor='gray', 
                    alpha=0.9))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nMatplotlib visualization saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    return fig

def generate_detailed_report(lot_data, road_stats, pixel_per_foot, output_path="subdivision_report.txt"):
    """
    Generate a detailed text report with all lot and road information
    """
    # Find buildable lots (exclude the largest lot which is site boundary)
    max_area_lot = max(lot_data, key=lambda x: x['Area_Pixels'])
    buildable_lots = [lot for lot in lot_data if lot['Label'] != max_area_lot['Label']]
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SUBDIVISION ANALYSIS DETAILED REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {import_datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Scale: {pixel_per_foot:.2f} pixels per foot\n")
    
    # SECTION 1: LOT INFORMATION
    report_lines.append("="*80)
    report_lines.append("LOT INFORMATION")
    report_lines.append("="*80)
    report_lines.append(f"\nTotal Lots (excluding site boundary): {len(buildable_lots)}")
    report_lines.append(f"Site Boundary Area: {max_area_lot['Area_SqFt']:,.0f} sf\n")
    
    # Create lot table header
    report_lines.append("-"*80)
    report_lines.append(f"{'Lot#':<6} {'Area(sf)':<10} {'Width(ft)':<10} {'Depth(ft)':<10} {'Road':<6} {'Build':<7} {'Orient':<12} {'House(sf)':<10}")
    report_lines.append("-"*80)
    
    # Sort lots by ID for consistent display
    for lot in sorted(buildable_lots, key=lambda x: x['Lot_ID']):
        lot_id = lot['Lot_ID']
        area_sf = f"{lot['Area_SqFt']:.0f}"
        width_ft = f"{lot['Width_Ft']:.1f}"
        depth_ft = f"{lot['Depth_Ft']:.1f}"
        faces_road = "Yes" if lot['Faces_Road'] else "No"
        buildable = "Yes" if lot['Buildable'] else "No"
        orientation = lot['Orientation'][:10]  # Truncate if too long
        house_sf = f"{lot['Est_House_SF']:.0f}" if lot['Est_House_SF'] > 0 else "N/A"
        
        report_lines.append(f"{lot_id:<6} {area_sf:<10} {width_ft:<10} {depth_ft:<10} {faces_road:<6} {buildable:<7} {orientation:<12} {house_sf:<10}")
    
    # Lot Statistics Summary
    report_lines.append("-"*80)
    report_lines.append("\nLOT STATISTICS SUMMARY:")
    areas = [lot['Area_SqFt'] for lot in buildable_lots]
    widths = [lot['Width_Ft'] for lot in buildable_lots]
    depths = [lot['Depth_Ft'] for lot in buildable_lots]
    buildable_only = [lot for lot in buildable_lots if lot['Buildable']]
    
    report_lines.append(f"  Average Lot Area: {np.mean(areas):,.0f} sf")
    report_lines.append(f"  Median Lot Area: {np.median(areas):,.0f} sf")
    report_lines.append(f"  Min/Max Lot Area: {min(areas):,.0f} / {max(areas):,.0f} sf")
    report_lines.append(f"  Average Dimensions: {np.mean(widths):.1f} ft (W) x {np.mean(depths):.1f} ft (D)")
    report_lines.append(f"  Buildable Lots: {len(buildable_only)} of {len(buildable_lots)}")
    report_lines.append(f"  Total Buildable Area: {sum(lot['Area_SqFt'] for lot in buildable_only):,.0f} sf")
    report_lines.append(f"  Total Est. House Area: {sum(lot['Est_House_SF'] for lot in buildable_only):,.0f} sf")
    
    # Orientation breakdown
    orientations = {}
    for lot in buildable_lots:
        ori = lot['Orientation']
        orientations[ori] = orientations.get(ori, 0) + 1
    
    report_lines.append("\n  Lot Orientations:")
    for ori, count in orientations.items():
        report_lines.append(f"    {ori}: {count} lots ({count/len(buildable_lots)*100:.1f}%)")
    
    # SECTION 2: ROAD NETWORK INFORMATION
    if road_stats:
        report_lines.append("\n" + "="*80)
        report_lines.append("ROAD NETWORK INFORMATION")
        report_lines.append("="*80)
        
        report_lines.append(f"\nTotal Road Length: {road_stats['Total_Road_Length_Ft']:,.0f} ft")
        report_lines.append(f"Total Road Area: {road_stats['Total_Road_Area_SF']:,.0f} sf")
        report_lines.append(f"ROW Width (estimated): {road_stats['ROW_Width_Ft']:.1f} ft")
        report_lines.append(f"Pavement Width (average): {road_stats['Pavement_Width_Ft']:.1f} ft")
        report_lines.append(f"Minimum Road Width: {road_stats['Min_Road_Width_Ft']:.1f} ft")
        report_lines.append(f"Maximum Road Width: {road_stats['Max_Road_Width_Ft']:.1f} ft")
        report_lines.append(f"Number of Road Segments: {road_stats['Num_Road_Segments']}")
        
        # Road classification
        main_pct = road_stats['Main_Road_Coverage'] * 100
        res_pct = road_stats['Residential_Road_Coverage'] * 100
        report_lines.append(f"\nRoad Classification:")
        report_lines.append(f"  Main Roads: {main_pct:.1f}%")
        report_lines.append(f"  Residential Roads: {res_pct:.1f}%")
        
        # Note about grade
        report_lines.append(f"\nNote: Road grade information requires elevation data (not available in 2D plan)")
    
    # SECTION 3: SITE EFFICIENCY
    report_lines.append("\n" + "="*80)
    report_lines.append("SITE EFFICIENCY ANALYSIS")
    report_lines.append("="*80)
    
    total_buildable_area = sum(lot['Area_SqFt'] for lot in buildable_only)
    site_area = max_area_lot['Area_SqFt']
    road_area = road_stats['Total_Road_Area_SF'] if road_stats else 0
    
    usage_rate = (total_buildable_area / site_area) * 100
    road_coverage = (road_area / site_area) * 100
    open_space = 100 - usage_rate - road_coverage
    
    report_lines.append(f"\nTotal Site Area: {site_area:,.0f} sf")
    report_lines.append(f"Buildable Lot Area: {total_buildable_area:,.0f} sf ({usage_rate:.1f}%)")
    report_lines.append(f"Road Area: {road_area:,.0f} sf ({road_coverage:.1f}%)")
    report_lines.append(f"Open Space/Other: ({open_space:.1f}%)")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print("\n" + report_text)
    
    print(f"\n✓ Detailed report saved to {output_path}")
    
    return report_text

def create_table_visualization(lot_data, road_stats, pixel_per_foot, output_path="data_tables.png"):
    """
    Create a matplotlib figure with tables showing lot and road data
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Find buildable lots
    max_area_lot = max(lot_data, key=lambda x: x['Area_Pixels'])
    buildable_lots = [lot for lot in lot_data if lot['Label'] != max_area_lot['Label']]
    
    # Limit to first 20 lots for display
    display_lots = sorted(buildable_lots, key=lambda x: x['Lot_ID'])[:20]
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create lot data for table
    lot_headers = ['Lot#', 'Area(sf)', 'Width(ft)', 'Depth(ft)', 'Road', 'Build', 'Orient', 'House(sf)']
    lot_rows = []
    
    for lot in display_lots:
        row = [
            f"{lot['Lot_ID']}",
            f"{lot['Area_SqFt']:.0f}",
            f"{lot['Width_Ft']:.1f}",
            f"{lot['Depth_Ft']:.1f}",
            "Yes" if lot['Faces_Road'] else "No",
            "Yes" if lot['Buildable'] else "No",
            lot['Orientation'][:8],
            f"{lot['Est_House_SF']:.0f}" if lot['Est_House_SF'] > 0 else "N/A"
        ]
        lot_rows.append(row)
    
    # Create road data for table
    road_headers = ['Property', 'Value', 'Unit']
    road_rows = []
    
    if road_stats:
        road_rows = [
            ['Total Length', f"{road_stats['Total_Road_Length_Ft']:.0f}", 'ft'],
            ['Total Area', f"{road_stats['Total_Road_Area_SF']:,.0f}", 'sf'],
            ['ROW Width', f"{road_stats['ROW_Width_Ft']:.1f}", 'ft'],
            ['Pavement Width', f"{road_stats['Pavement_Width_Ft']:.1f}", 'ft'],
            ['Min Width', f"{road_stats['Min_Road_Width_Ft']:.1f}", 'ft'],
            ['Max Width', f"{road_stats['Max_Road_Width_Ft']:.1f}", 'ft'],
            ['Road Segments', f"{road_stats['Num_Road_Segments']}", 'count'],
            ['Main Roads', f"{road_stats['Main_Road_Coverage']*100:.1f}", '%'],
            ['Residential', f"{road_stats['Residential_Road_Coverage']*100:.1f}", '%']
        ]
    
    # Create subplots
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 2), (2, 0), colspan=1)
    ax3 = plt.subplot2grid((3, 2), (2, 1), colspan=1)
    
    # Lot table
    ax1.axis('tight')
    ax1.axis('off')
    lot_table = ax1.table(cellText=lot_rows, colLabels=lot_headers, 
                         cellLoc='center', loc='center',
                         colWidths=[0.08, 0.12, 0.12, 0.12, 0.08, 0.08, 0.15, 0.12])
    lot_table.auto_set_font_size(False)
    lot_table.set_fontsize(9)
    lot_table.scale(1, 1.5)
    
    # Style the header
    for i in range(len(lot_headers)):
        lot_table[(0, i)].set_facecolor('#4CAF50')
        lot_table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(lot_rows) + 1):
        for j in range(len(lot_headers)):
            if i % 2 == 0:
                lot_table[(i, j)].set_facecolor('#f0f0f0')
    
    ax1.set_title(f'LOT INFORMATION (Showing {len(display_lots)} of {len(buildable_lots)} lots)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Road table
    ax2.axis('tight')
    ax2.axis('off')
    road_table = ax2.table(cellText=road_rows, colLabels=road_headers,
                          cellLoc='center', loc='center')
    road_table.auto_set_font_size(False)
    road_table.set_fontsize(9)
    road_table.scale(1, 1.5)
    
    # Style the header
    for i in range(len(road_headers)):
        road_table[(0, i)].set_facecolor('#2196F3')
        road_table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('ROAD NETWORK DATA', fontsize=12, fontweight='bold', pad=10)
    
    # Summary statistics
    ax3.axis('off')
    
    buildable_only = [lot for lot in buildable_lots if lot['Buildable']]
    total_buildable_area = sum(lot['Area_SqFt'] for lot in buildable_only)
    site_area = max_area_lot['Area_SqFt']
    usage_rate = (total_buildable_area / site_area) * 100
    
    summary_text = (
        f"SITE SUMMARY\n\n"
        f"Total Lots: {len(buildable_lots)}\n"
        f"Buildable Lots: {len(buildable_only)}\n"
        f"Site Area: {site_area:,.0f} sf\n"
        f"Buildable Area: {total_buildable_area:,.0f} sf\n"
        f"Usage Rate: {usage_rate:.1f}%\n\n"
        f"Avg Lot Size: {np.mean([lot['Area_SqFt'] for lot in buildable_lots]):,.0f} sf\n"
        f"Avg Width: {np.mean([lot['Width_Ft'] for lot in buildable_lots]):.1f} ft\n"
        f"Avg Depth: {np.mean([lot['Depth_Ft'] for lot in buildable_lots]):.1f} ft"
    )
    
    ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax3.set_title('SUMMARY STATISTICS', fontsize=12, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle('SUBDIVISION DATA ANALYSIS', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Data tables saved to {output_path}")
    plt.show()
    
    return fig

def main():
    # Get pixel to foot conversion rate from user
    print("="*60)
    print("SUBDIVISION LOT ANALYSIS")
    print("="*60)
    
    # Default conversion rate (can be changed)
    default_pixel_per_foot = 2.0  # Example: 2 pixels = 1 foot
    
    try:
        user_input = input(f"Enter pixels per foot conversion rate (default={default_pixel_per_foot}): ").strip()
        if user_input:
            pixel_per_foot = float(user_input)
        else:
            pixel_per_foot = default_pixel_per_foot
    except ValueError:
        print(f"Invalid input. Using default: {default_pixel_per_foot} pixels/foot")
        pixel_per_foot = default_pixel_per_foot
    
    print(f"Using conversion rate: {pixel_per_foot} pixels per foot")
    print(f"This means 1 square foot = {pixel_per_foot**2:.1f} pixels")
    
    # Step 1: Convert generated_image4.png to binary
    input_file = "finalized_plan.png"
    binary_file = "red_binary.png"
    
    success = convert_red_to_binary(input_file, binary_file)
    
    if success:
        print(f"Successfully converted {input_file} to binary image")
        print(f"Red pixels -> White (255)")
        print(f"Other pixels -> Black (0)")
        
        # Step 2: Detect bounding boxes with exact pixel counting
        print("\n" + "="*60)
        print("DETECTING BOUNDING BOXES WITH CONNECTED COMPONENTS")
        print("="*60)
        
        lot_data, labels_img = detect_bounding_boxes(
            binary_file,
            pixel_per_foot=pixel_per_foot,
            output_vis_path="lot_detection_visualization.png",
            output_csv_path="lot_analysis.csv",
            return_labels=True
        )
        
        if lot_data:
            print(f"\n✓ Successfully detected {len(lot_data)} lots with pixel-accurate measurements!")
            
            # Step 3: Analyze road network
            road_stats = analyze_road_network(
                input_file,
                pixel_per_foot,
                output_path="road_network_analysis.png"
            )
            
            # Step 4: Create matplotlib visualization overlaying on original site plan
            print("\n" + "="*60)
            print("CREATING MATPLOTLIB VISUALIZATION WITH USAGE RATE")
            print("="*60)
            
            create_matplotlib_visualization(
                input_file,  # Use original site plan image
                binary_file,
                lot_data,
                labels_img,
                pixel_per_foot,
                road_stats=road_stats,
                output_path="lot_matplotlib_overlay.png"
            )
            
            # Step 5: Generate comprehensive report
            print("\n" + "="*60)
            print("GENERATING COMPREHENSIVE REPORT")
            print("="*60)
            
            # Export extended lot data
            buildable_lots = [lot for lot in lot_data if lot['Buildable']]
            
            print(f"\nSummary of Buildable Lots:")
            print(f"  Total: {len(buildable_lots)} lots")
            print(f"  Average Width: {np.mean([lot['Width_Ft'] for lot in buildable_lots]):.1f} ft")
            print(f"  Average Depth: {np.mean([lot['Depth_Ft'] for lot in buildable_lots]):.1f} ft")
            print(f"  Total Est. House SF: {sum([lot['Est_House_SF'] for lot in buildable_lots]):,.0f} sf")
            
            # Orientation distribution
            orientations = {}
            for lot in buildable_lots:
                ori = lot['Orientation']
                orientations[ori] = orientations.get(ori, 0) + 1
            
            print(f"\nLot Orientations:")
            for ori, count in orientations.items():
                print(f"  {ori}: {count} lots ({count/len(buildable_lots)*100:.1f}%)")
            
            print(f"\n✓ Analysis complete! Check output files for detailed results.")
            
            # Step 6: Generate detailed report and data tables
            print("\n" + "="*60)
            print("GENERATING DETAILED DATA REPORT AND TABLES")
            print("="*60)
            
            # Generate text report
            generate_detailed_report(lot_data, road_stats, pixel_per_foot, 
                                    output_path="subdivision_detailed_report.txt")
            
            # Generate table visualization
            create_table_visualization(lot_data, road_stats, pixel_per_foot,
                                     output_path="subdivision_data_tables.png")
    else:
        print("Conversion failed")

if __name__ == "__main__":
    main()
