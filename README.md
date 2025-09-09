# SmartLandPlanner

An AI-powered subdivision layout system that transforms property boundaries into comprehensive subdivision plans. From deed parsing to road network generation, SmartLandPlanner creates optimized land development layouts using advanced AI and geometric calculations.

## Features

### Core Subdivision Planning
- **AI-Powered Site Analysis**: Uses Google Gemini AI to analyze site plans and extract boundaries
- **Intelligent Road Network Generation**: Automatically creates optimized road layouts for subdivisions
- **Lot Configuration**: Generates buildable lots with automatic setback calculations
- **Site Plan Visualization**: Interactive maps showing roads, lots, and infrastructure

### Deed Processing & Boundary Extraction
- **AI-Powered Parsing**: Uses OpenAI GPT models to extract survey calls from deed text
- **Interactive Editing**: Review and correct parsed data with an editable table interface  
- **Geometry Calculation**: Converts bearings/distances to coordinates and calculates polygon geometry
- **SVG Visualization**: Generate scalable vector graphics of property boundaries

### Export & Analysis
- **Multiple Export Formats**: Export as SVG, CSV, JSON, PNG, and more
- **Cost Analysis**: Calculate development costs and lot valuations
- **Quality Metrics**: Analyze lot sizes, road efficiency, and development feasibility
- **Flexible Configuration**: Support for different units, bearing conventions, and coordinate systems

## Architecture

### Frontend
- **Streamlit** web application with comprehensive subdivision planning interface
- **5-Tab Workflow**: PDF Processing → Text Parsing → Review & Edit → Visualization → Site Planning
- Interactive data tables and real-time visualization
- File upload support for PDFs, images, and text files

### AI Services
- **OpenAI Integration** for deed text parsing with structured outputs
- **Google Gemini AI** for site plan analysis and boundary extraction
- **Computer Vision** for image processing and lot detection
- **Intelligent Algorithms** for road network optimization

### Backend Services
- **Geometry Calculator** using Shapely for coordinate transformations
- **SVG Generator** for creating detailed boundary and subdivision visualizations
- **Cost Calculator** for development cost analysis
- **Bearing Parser** supporting multiple format conventions

### Data Models
- Pydantic models for type safety and validation
- Canonical JSON schema for survey calls and subdivision data
- Project settings, quality thresholds, and development parameters

## Installation

### Quick Start (Cross-Platform)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd smart-land-planner
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies:**
   
   **Windows:**
   - Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
   
   **Linux/Ubuntu:**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx
   ```

4. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key and Google Gemini API key
   ```

5. **Test platform compatibility:**
   ```bash
   python test_platform.py
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🚀 Deployment

### Streamlit Cloud

This application is optimized for deployment on Streamlit Cloud with automatic cross-platform compatibility:

1. **Required Files:**
   - ✅ `requirements.txt` - Python dependencies (includes opencv-python-headless)
   - ✅ `packages.txt` - System dependencies for Linux (Tesseract OCR)
   - ✅ `.streamlit/secrets.toml` - API keys and configuration

2. **Deploy to Streamlit Cloud:**
   - Push your code to GitHub
   - Connect repository to Streamlit Cloud
   - Set up secrets in the dashboard
   - Deploy automatically!

3. **Platform Detection:**
   - Automatically detects Windows vs Linux environment
   - Configures Tesseract OCR paths appropriately
   - Uses headless OpenCV for server environments

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

## Usage

### Complete Subdivision Planning Workflow

#### 1. Configuration
- Enter your **OpenAI API key** for deed text parsing
- Enter your **Google Gemini API key** for site plan analysis
- Select AI models and configure project settings
- Set project units (feet/meters) and bearing convention

#### 2. Property Boundary Extraction
**Option A: Deed Text Processing**
- Paste deed text or upload a PDF file
- Click "Parse Deed" to extract survey calls using AI
- Review and edit extracted boundary data

**Option B: Site Plan Image Analysis**
- Upload a site plan image (PNG, JPG)
- Use AI to automatically extract property boundaries
- Review and refine boundary polygon

#### 3. Review & Edit
- Use interactive tables to review boundary data
- Edit bearings, distances, and other parameters
- Add or remove boundary calls as needed
- Recalculate geometry after changes

#### 4. Visualization & Export
- View the generated property boundary polygon
- See closure statistics and area calculations
- Export boundary as PNG for site planning
- Export as SVG, CSV, or JSON formats

#### 5. AI-Powered Site Planning
- **Road Network Generation**: AI creates optimized road layouts
- **Lot Configuration**: Automatic lot creation with setbacks
- **Site Analysis**: Analyze lot sizes, road efficiency, and costs
- **Final Plan Export**: Generate comprehensive subdivision plans

## Supported Formats

### Bearing Formats
- Quadrant: `N 45°30'15" E`, `S 22°10' W`
- Azimuth: `123°45'30"`, `123.75°`
- Variations: `N45-30-15E`, `N 45 30 E`

### Distance Units
- Feet (`ft`), Meters (`m`)
- Chains (`ch`), Rods (`rd`)
- Automatic unit conversion

### Curve Types
- Radius and arc length
- Chord bearing and chord length
- Left/right curve direction
- Automatic chord-to-arc calculations

## Technical Details

### OpenAI Integration
- Uses structured outputs with function calling for reliable JSON extraction
- Custom prompts optimized for deed parsing
- Confidence scoring and fallback regex parsing
- Support for multiple model types

### Geometry Calculations
- Coordinate system: Local grid (X=East, Y=North)
- Azimuth convention: Degrees clockwise from North
- Curve discretization with configurable segments
- Closure error calculation and validation

### Data Validation
- Real-time bearing format validation
- Unit consistency checking
- Geometry closure analysis
- Confidence-based quality indicators

## Development

### Project Structure
```
smart-land-planner/
├── app.py                 # Main Streamlit application with 5-tab workflow
├── src/
│   ├── models/           # Pydantic data models for deeds and subdivisions
│   ├── parsers/          # OpenAI and Gemini AI integration
│   ├── geometry/         # Coordinate calculations and transformations
│   ├── visualization/    # SVG generation and site plan visualization
│   ├── components/       # UI components and data tables
│   └── utils/           # Utilities (bearing parser, cost calculator, etc.)
├── tests/               # Unit tests
└── requirements.txt     # Dependencies
```

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features
1. Update data models in `src/models/` for new subdivision features
2. Extend AI parsers (OpenAI/Gemini) for enhanced analysis
3. Add new visualization options for site plans and lot layouts
4. Update Streamlit UI components and workflow tabs
5. Enhance cost calculation and development analysis tools

## Configuration Options

### Project Settings
- Distance units and bearing conventions
- Point of Beginning coordinates
- Coordinate Reference System (CRS)
- Quality thresholds and tolerances

### AI Settings
- **OpenAI**: Model selection (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
- **Google Gemini**: API key management for site plan analysis
- Parsing confidence thresholds and quality settings

## Troubleshooting

### Common Issues
1. **Low parsing confidence**: Review and manually edit calls in the table
2. **Closure errors**: Check bearing/distance accuracy and units
3. **Missing curve data**: Verify radius, arc length, and chord information
4. **API errors**: Confirm OpenAI and Gemini API keys and model availability
5. **Site plan analysis issues**: Ensure clear site plan images with visible boundaries
6. **Road generation problems**: Check boundary polygon quality and site constraints

### Error Handling
- Graceful degradation for incomplete data
- Regex fallback for AI parsing failures
- Validation warnings for data quality issues
- Detailed error messages and stack traces

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
