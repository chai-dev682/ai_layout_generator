# Deed Parser System

A comprehensive system for parsing legal deed descriptions and visualizing property boundaries using AI-powered text extraction and geometric calculations.

## Features

- **AI-Powered Parsing**: Uses OpenAI GPT models to extract survey calls from deed text
- **Interactive Editing**: Review and correct parsed data with an editable table interface  
- **Geometry Calculation**: Converts bearings/distances to coordinates and calculates polygon geometry
- **SVG Visualization**: Generate scalable vector graphics of property boundaries
- **Multiple Export Formats**: Export as SVG, CSV, JSON, and more
- **Flexible Configuration**: Support for different units, bearing conventions, and coordinate systems

## Architecture

### Frontend
- **Streamlit** web application with tabbed interface
- Editable data tables using `st.data_editor`
- Real-time SVG preview and export options
- File upload support for PDFs and text files

### Backend Services
- **OpenAI Integration** with structured outputs and function calling
- **Geometry Calculator** using Shapely for coordinate transformations
- **SVG Generator** for creating detailed boundary visualizations
- **Bearing Parser** supporting multiple format conventions

### Data Models
- Pydantic models for type safety and validation
- Canonical JSON schema for survey calls
- Project settings and quality thresholds

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd deed-parser
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Configuration
- Enter your OpenAI API key in the sidebar
- Select the desired OpenAI model (gpt-4o recommended)
- Set project units (feet/meters) and bearing convention
- Configure Point of Beginning coordinates

### 2. Text Input & Parsing
- Paste deed text or upload a PDF file
- Click "Parse Deed" to extract survey calls using AI
- Review parsing confidence and any warnings

### 3. Review & Edit
- Use the interactive table to review extracted calls
- Edit bearings, distances, and other parameters
- Add or remove calls as needed
- Recalculate geometry after changes

### 4. Visualization
- View the generated property boundary polygon
- See closure statistics and area calculations
- Export as SVG, CSV, or JSON formats

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
deed-parser/
├── app.py                 # Main Streamlit application
├── src/
│   ├── models/           # Pydantic data models
│   ├── parsers/          # OpenAI integration
│   ├── geometry/         # Coordinate calculations  
│   ├── visualization/    # SVG generation
│   └── utils/           # Utilities (bearing parser, etc.)
├── tests/               # Unit tests
└── requirements.txt     # Dependencies
```

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features
1. Update data models in `src/models/`
2. Extend parsers or geometry calculators
3. Add new visualization options
4. Update Streamlit UI components

## Configuration Options

### Project Settings
- Distance units and bearing conventions
- Point of Beginning coordinates
- Coordinate Reference System (CRS)
- Quality thresholds and tolerances

### OpenAI Settings
- Model selection (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
- API key management
- Parsing confidence thresholds

## Troubleshooting

### Common Issues
1. **Low parsing confidence**: Review and manually edit calls in the table
2. **Closure errors**: Check bearing/distance accuracy and units
3. **Missing curve data**: Verify radius, arc length, and chord information
4. **API errors**: Confirm OpenAI API key and model availability

### Error Handling
- Graceful degradation for incomplete data
- Regex fallback for AI parsing failures
- Validation warnings for data quality issues
- Detailed error messages and stack traces

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
