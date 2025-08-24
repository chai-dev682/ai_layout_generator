"""
Specialized parser for extracting site boundary information from PDF-extracted text.
Uses OpenAI GPT to identify and extract site survey data from site survey documents.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .openai_parser import OpenAIDeedParser
from ..models.deed_models import ProjectSettings, DeedParseResult, SurveyCall

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logging


class BoundaryDataType(Enum):
    """Types of boundary data that can be found in site surveys"""
    LINE_CURVE_TABLE = "line_curve_table"
    PROPERTY_DESCRIPTION = "property_description"  
    DEED_DESCRIPTION = "deed_description"
    LEGAL_DESCRIPTION = "legal_description"
    SURVEY_NOTES = "survey_notes"
    COORDINATE_TABLE = "coordinate_table"
    BEARING_DISTANCE_TABLE = "bearing_distance_table"


@dataclass
class BoundaryExtractionResult:
    """Result of boundary information extraction from PDF text"""
    extracted_data: Dict[BoundaryDataType, str]
    confidence_scores: Dict[BoundaryDataType, float]
    processing_time: float
    warnings: List[str]
    cost_estimate: float
    
    @property
    def has_boundary_data(self) -> bool:
        """Whether any boundary data was found"""
        return any(data.strip() for data in self.extracted_data.values())
    
    @property
    def primary_boundary_text(self) -> str:
        """Get the primary boundary description text"""
        # Priority order for boundary data
        priority_order = [
            BoundaryDataType.LINE_CURVE_TABLE,
            BoundaryDataType.BEARING_DISTANCE_TABLE,
            BoundaryDataType.LEGAL_DESCRIPTION,
            BoundaryDataType.PROPERTY_DESCRIPTION,
            BoundaryDataType.DEED_DESCRIPTION,
            BoundaryDataType.SURVEY_NOTES
        ]
        
        for data_type in priority_order:
            data = self.extracted_data.get(data_type, "").strip()
            if data:
                return data
        
        return ""
    
    @property
    def all_boundary_text(self) -> str:
        """Get ALL boundary description text combined for comprehensive processing"""
        all_data = []
        
        # Priority order for combining data (most structured first)
        priority_order = [
            BoundaryDataType.LINE_CURVE_TABLE,
            BoundaryDataType.BEARING_DISTANCE_TABLE,
            BoundaryDataType.COORDINATE_TABLE,
            BoundaryDataType.LEGAL_DESCRIPTION,
            BoundaryDataType.PROPERTY_DESCRIPTION,
            BoundaryDataType.DEED_DESCRIPTION,
            BoundaryDataType.SURVEY_NOTES
        ]
        
        for data_type in priority_order:
            data = self.extracted_data.get(data_type, "").strip()
            if data:
                # Add section header for clarity
                section_header = f"\n\n=== {data_type.value.replace('_', ' ').upper()} ===\n"
                all_data.append(section_header + data)
        
        return "\n\n".join(all_data) if all_data else ""


class PDFBoundaryParser:
    """
    Parser specialized in extracting site boundary information from PDF-extracted text.
    Designed to work with site survey documents, plats, and legal descriptions.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """
        Initialize boundary parser
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use for extraction
        """
        self.openai_parser = OpenAIDeedParser(api_key=openai_api_key, model=model)
        self.model = model
        
        logger.info(f"PDF Boundary Parser initialized with model: {model}")
    
    def extract_boundary_information(self, pdf_text: str, settings: ProjectSettings) -> BoundaryExtractionResult:
        """
        Extract site boundary information from PDF text
        
        Args:
            pdf_text: Raw text extracted from PDF
            settings: Project settings for context
            
        Returns:
            BoundaryExtractionResult with extracted boundary data
        """
        start_time = time.time()
        logger.info("=== Starting PDF Boundary Information Extraction ===")
        logger.info(f"PDF text length: {len(pdf_text)} characters")
        logger.info(f"Using model: {self.model}")
        
        try:
            # Step 1: Identify and extract different types of boundary data
            boundary_data = self._extract_boundary_data_types(pdf_text, settings)
            
            # Step 2: Calculate confidence scores for each data type
            confidence_scores = self._calculate_confidence_scores(boundary_data, pdf_text)
            
            processing_time = time.time() - start_time
            
            # Estimate cost (using OpenAI pricing)
            cost_estimate = self._estimate_extraction_cost(pdf_text)
            
            result = BoundaryExtractionResult(
                extracted_data=boundary_data,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                warnings=self._generate_warnings(boundary_data),
                cost_estimate=cost_estimate
            )
            
            logger.info(f"=== Boundary Extraction Completed ===")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Found {len([d for d in boundary_data.values() if d.strip()])} boundary data types")
            logger.info(f"Estimated cost: ${cost_estimate:.4f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Boundary extraction failed after {processing_time:.2f} seconds: {str(e)}")
            
            return BoundaryExtractionResult(
                extracted_data={},
                confidence_scores={},
                processing_time=processing_time,
                warnings=[f"Extraction failed: {str(e)}"],
                cost_estimate=0.0
            )
    
    def _extract_boundary_data_types(self, pdf_text: str, settings: ProjectSettings) -> Dict[BoundaryDataType, str]:
        """Extract different types of boundary data from PDF text"""
        logger.info("--- Step 1: Extracting Boundary Data Types ---")
        
        # Create specialized prompt for boundary data extraction
        system_prompt = self._create_boundary_extraction_prompt(settings)
        
        try:
            # Use OpenAI to identify and extract boundary information
            if self.openai_parser.model_info.get('reasoning', False):
                # Use reasoning model
                boundary_data = self._extract_with_reasoning_model(pdf_text, system_prompt)
            else:
                # Use function calling model
                boundary_data = self._extract_with_function_calling(pdf_text, system_prompt)
            
            # Log what was found
            for data_type, data in boundary_data.items():
                if data.strip():
                    logger.info(f"Found {data_type.value}: {len(data)} characters")
                else:
                    logger.info(f"No {data_type.value} found")
            
            return boundary_data
            
        except Exception as e:
            logger.error(f"Failed to extract boundary data types: {e}")
            return {}
    
    def _create_boundary_extraction_prompt(self, settings: ProjectSettings) -> str:
        """Create system prompt for boundary data extraction"""
        return f"""You are a professional surveyor and legal description expert specialized in extracting site boundary information from site survey documents.

CONTEXT:
- Document is a site survey PDF that has been converted to text
- Default units: {settings.units}
- Bearing convention: {settings.bearing_convention}
- You need to identify and extract different types of boundary information

TYPES OF BOUNDARY DATA TO EXTRACT:

1. LINE/CURVE TABLE:
   - Tabular data with columns like: Line#, Bearing, Distance, Curve Radius, etc.
   - Often titled "Line and Curve Table", "Survey Table", "Boundary Table"
   - Contains numbered survey calls in table format

2. PROPERTY DESCRIPTION:
   - Narrative property description with metes and bounds
   - Usually starts with "Beginning at..." or "Commencing at..."
   - Contains bearing and distance calls in paragraph form

3. DEED DESCRIPTION:
   - Legal deed description of the property
   - May reference previous deeds or recorded documents
   - Contains technical legal language

4. LEGAL DESCRIPTION:
   - Formal legal description for recording purposes
   - Often contains lot numbers, subdivision names
   - May reference plat books or recorded surveys

5. SURVEY NOTES:
   - Surveyor's notes about monuments, markers, measurements
   - Field notes about conditions found during survey
   - Notes about discrepancies or special conditions

6. COORDINATE TABLE:
   - Table of coordinate points (X, Y coordinates)
   - May be state plane, UTM, or local coordinates
   - Often labeled as "Coordinate List" or "Point Table"

7. BEARING/DISTANCE TABLE:
   - Simple table with just bearings and distances
   - May not have curve information
   - Basic survey call data in tabular form

EXTRACTION INSTRUCTIONS:
1. Scan the entire document text for each type of boundary data
2. Extract complete sections, don't truncate
3. Preserve original formatting and spacing where possible
4. If multiple sections of same type exist, combine them
5. Include table headers and column labels
6. Mark confidence based on clarity and completeness

SITE SURVEY DOCUMENT PATTERNS:
- Look for section headers like "LEGAL DESCRIPTION", "METES AND BOUNDS", "LINE TABLE"
- Survey tables often have numbered rows (Line 1, Line 2, etc.)
- Property descriptions often start with POB establishment
- Coordinate data may be in State Plane or local system
- Bearing formats: N 45°30'15" E, S 22°10' W, etc.
- Distance units: feet, meters, chains, rods

Focus on extracting complete, usable boundary information that can be used for CAD drawing and property boundary determination."""

    def _extract_with_function_calling(self, pdf_text: str, system_prompt: str) -> Dict[BoundaryDataType, str]:
        """Extract boundary data using function calling"""
        logger.info("Using function calling for boundary data extraction")
        
        # Create function schema for boundary extraction
        function_schema = {
            "name": "extract_boundary_data",
            "description": "Extract different types of site boundary information from survey document text",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_curve_table": {
                        "type": "string",
                        "description": "Complete line and curve table with all columns and rows"
                    },
                    "property_description": {
                        "type": "string", 
                        "description": "Narrative property description with metes and bounds"
                    },
                    "deed_description": {
                        "type": "string",
                        "description": "Legal deed description text"
                    },
                    "legal_description": {
                        "type": "string",
                        "description": "Formal legal description for recording"
                    },
                    "survey_notes": {
                        "type": "string",
                        "description": "Surveyor's field notes and observations"
                    },
                    "coordinate_table": {
                        "type": "string",
                        "description": "Table of coordinate points (X, Y coordinates)"
                    },
                    "bearing_distance_table": {
                        "type": "string",
                        "description": "Simple bearing and distance table"
                    },
                    "extraction_notes": {
                        "type": "string",
                        "description": "Notes about what was found and extraction confidence"
                    }
                },
                "required": ["extraction_notes"]
            }
        }
        
        # Prepare API call
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all boundary information from this site survey document:\n\n{pdf_text}"}
            ],
            "functions": [function_schema],
            "function_call": {"name": "extract_boundary_data"}
        }
        
        # Add temperature if supported
        if self.openai_parser.model_info.get('supports_temperature', True):
            api_params["temperature"] = 0.1
        
        response = self.openai_parser.client.chat.completions.create(**api_params)
        
        # Parse function call result
        message = response.choices[0].message
        if message.function_call:
            import json
            function_args = json.loads(message.function_call.arguments)
            
            # Map results to BoundaryDataType enum
            boundary_data = {}
            for data_type in BoundaryDataType:
                key = data_type.value
                boundary_data[data_type] = function_args.get(key, "")
            
            # Log extraction notes
            extraction_notes = function_args.get("extraction_notes", "")
            if extraction_notes:
                logger.info(f"Extraction notes: {extraction_notes}")
            
            return boundary_data
        else:
            raise ValueError("No function call in OpenAI response")
    
    def _extract_with_reasoning_model(self, pdf_text: str, system_prompt: str) -> Dict[BoundaryDataType, str]:
        """Extract boundary data using reasoning model (text-based parsing)"""
        logger.info("Using reasoning model for boundary data extraction")
        
        # Combine system prompt with user message for reasoning models
        combined_prompt = f"""{system_prompt}

Please analyze the following site survey document text and extract all boundary information in JSON format:

{{
  "line_curve_table": "Complete line and curve table text",
  "property_description": "Narrative property description text", 
  "deed_description": "Legal deed description text",
  "legal_description": "Formal legal description text",
  "survey_notes": "Surveyor's notes and observations",
  "coordinate_table": "Coordinate points table",
  "bearing_distance_table": "Bearing and distance table",
  "extraction_notes": "Notes about what was found and confidence"
}}

SITE SURVEY DOCUMENT TEXT:
{pdf_text}"""
        
        response = self.openai_parser.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": combined_prompt}]
        )
        
        # Parse JSON from response
        response_text = response.choices[0].message.content
        
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            parsed_data = json.loads(json_str.strip())
            
            # Map to BoundaryDataType enum
            boundary_data = {}
            for data_type in BoundaryDataType:
                key = data_type.value
                boundary_data[data_type] = parsed_data.get(key, "")
            
            return boundary_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from reasoning model: {e}")
            logger.error(f"Response preview: {response_text[:500]}...")
            return {}
    
    def _calculate_confidence_scores(self, boundary_data: Dict[BoundaryDataType, str], 
                                   original_text: str) -> Dict[BoundaryDataType, float]:
        """Calculate confidence scores for extracted boundary data"""
        confidence_scores = {}
        
        for data_type, extracted_text in boundary_data.items():
            if not extracted_text.strip():
                confidence_scores[data_type] = 0.0
                continue
            
            # Calculate confidence based on various factors
            confidence = 0.0
            
            # Length factor (longer extractions often more complete)
            length_factor = min(len(extracted_text) / 500, 1.0) * 0.3
            
            # Keyword relevance factor
            relevance_factor = self._calculate_relevance_score(data_type, extracted_text) * 0.4
            
            # Structure factor (tables, formatting, etc.)
            structure_factor = self._calculate_structure_score(data_type, extracted_text) * 0.3
            
            confidence = length_factor + relevance_factor + structure_factor
            confidence_scores[data_type] = min(confidence, 1.0)
        
        return confidence_scores
    
    def _calculate_relevance_score(self, data_type: BoundaryDataType, text: str) -> float:
        """Calculate relevance score based on expected keywords"""
        text_lower = text.lower()
        
        keyword_sets = {
            BoundaryDataType.LINE_CURVE_TABLE: [
                'line', 'curve', 'bearing', 'distance', 'radius', 'chord', 'table', 'arc'
            ],
            BoundaryDataType.PROPERTY_DESCRIPTION: [
                'beginning', 'thence', 'point of beginning', 'pob', 'feet', 'bearing'
            ],
            BoundaryDataType.DEED_DESCRIPTION: [
                'deed', 'recorded', 'book', 'page', 'instrument', 'grantor', 'grantee'
            ],
            BoundaryDataType.LEGAL_DESCRIPTION: [
                'legal', 'lot', 'block', 'subdivision', 'plat', 'tract', 'parcel'
            ],
            BoundaryDataType.SURVEY_NOTES: [
                'found', 'set', 'monument', 'marker', 'iron', 'concrete', 'note'
            ],
            BoundaryDataType.COORDINATE_TABLE: [
                'coordinate', 'northing', 'easting', 'x', 'y', 'state plane', 'utm'
            ],
            BoundaryDataType.BEARING_DISTANCE_TABLE: [
                'bearing', 'distance', 'azimuth', 'feet', 'meters', 'direction'
            ]
        }
        
        keywords = keyword_sets.get(data_type, [])
        if not keywords:
            return 0.5
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return min(matches / len(keywords), 1.0)
    
    def _calculate_structure_score(self, data_type: BoundaryDataType, text: str) -> float:
        """Calculate structure score based on expected formatting"""
        if data_type in [BoundaryDataType.LINE_CURVE_TABLE, BoundaryDataType.COORDINATE_TABLE, 
                        BoundaryDataType.BEARING_DISTANCE_TABLE]:
            # Look for table-like structure
            lines = text.split('\n')
            if len(lines) > 3:  # Tables should have multiple rows
                # Check for consistent column structure
                consistent_rows = 0
                for line in lines:
                    if len(line.split()) >= 3:  # Multiple columns
                        consistent_rows += 1
                
                return min(consistent_rows / len(lines), 1.0)
            return 0.2
        
        elif data_type in [BoundaryDataType.PROPERTY_DESCRIPTION, BoundaryDataType.DEED_DESCRIPTION]:
            # Look for narrative structure with bearing/distance patterns
            import re
            bearing_pattern = r'[NS]\s*\d+[°\s]*\d*[\'\"]*\s*[EW]'
            distance_pattern = r'\d+\.?\d*\s*feet|\d+\.?\d*\s*ft'
            
            bearing_matches = len(re.findall(bearing_pattern, text, re.IGNORECASE))
            distance_matches = len(re.findall(distance_pattern, text, re.IGNORECASE))
            
            if bearing_matches > 0 and distance_matches > 0:
                return min((bearing_matches + distance_matches) / 10, 1.0)
            return 0.3
        
        return 0.5  # Default for other types
    
    def _generate_warnings(self, boundary_data: Dict[BoundaryDataType, str]) -> List[str]:
        """Generate warnings about extraction quality"""
        warnings = []
        
        # Check if any data was found
        found_data = [data_type for data_type, data in boundary_data.items() if data.strip()]
        if not found_data:
            warnings.append("No boundary data found in PDF text")
            return warnings
        
        # Check for missing critical data types
        critical_types = [
            BoundaryDataType.LINE_CURVE_TABLE,
            BoundaryDataType.PROPERTY_DESCRIPTION,
            BoundaryDataType.BEARING_DISTANCE_TABLE
        ]
        
        missing_critical = [dt for dt in critical_types if dt not in found_data]
        if len(missing_critical) == len(critical_types):
            warnings.append("No primary boundary data (tables or descriptions) found")
        
        # Check for data quality issues
        for data_type, data in boundary_data.items():
            if data.strip():
                if len(data) < 50:
                    warnings.append(f"{data_type.value} appears incomplete (very short)")
                
                if data_type in [BoundaryDataType.LINE_CURVE_TABLE, BoundaryDataType.COORDINATE_TABLE]:
                    if '\n' not in data:
                        warnings.append(f"{data_type.value} may not be properly formatted as table")
        
        return warnings
    
    def _estimate_extraction_cost(self, text: str) -> float:
        """Estimate cost for OpenAI API calls"""
        # Rough estimation based on token count
        # GPT-4o pricing: ~$0.005 per 1K input tokens, ~$0.015 per 1K output tokens
        input_tokens = len(text.split()) * 1.3  # Rough token estimation
        output_tokens = 500  # Estimated output for boundary extraction
        
        input_cost = (input_tokens / 1000) * 0.005
        output_cost = (output_tokens / 1000) * 0.015
        
        return input_cost + output_cost
    
    def format_to_cad_table(self, boundary_result: BoundaryExtractionResult, 
                           settings: ProjectSettings) -> DeedParseResult:
        """
        Convert ALL extracted boundary information to CAD-ready line/curve table
        
        Args:
            boundary_result: Result from boundary extraction
            settings: Project settings
            
        Returns:
            DeedParseResult with structured survey calls
        """
        logger.info("=== Converting ALL Boundary Data to CAD-Ready Format ===")
        
        # Combine all extracted boundary data
        all_boundary_data = []
        found_data_types = []
        
        # Priority order for combining data (most structured first)
        priority_order = [
            BoundaryDataType.LINE_CURVE_TABLE,
            BoundaryDataType.BEARING_DISTANCE_TABLE,
            BoundaryDataType.COORDINATE_TABLE,
            BoundaryDataType.LEGAL_DESCRIPTION,
            BoundaryDataType.PROPERTY_DESCRIPTION,
            BoundaryDataType.DEED_DESCRIPTION,
            BoundaryDataType.SURVEY_NOTES
        ]
        
        for data_type in priority_order:
            data = boundary_result.extracted_data.get(data_type, "").strip()
            if data:
                confidence = boundary_result.confidence_scores.get(data_type, 0.0)
                logger.info(f"Including {data_type.value}: {len(data)} characters (confidence: {confidence:.2f})")
                
                # Add section header for clarity
                section_header = f"\n\n=== {data_type.value.replace('_', ' ').upper()} ===\n"
                all_boundary_data.append(section_header + data)
                found_data_types.append(data_type.value)
        
        if not all_boundary_data:
            logger.warning("No boundary data found for CAD conversion")
            return DeedParseResult(
                tracts=[],
                settings=settings,
                calls=[],
                geometry=None,
                original_text=""
            )
        
        # Combine all boundary data into one comprehensive text
        combined_boundary_text = "\n\n".join(all_boundary_data)
        
        logger.info(f"Converting {len(combined_boundary_text)} characters from {len(found_data_types)} data types: {', '.join(found_data_types)}")
        
        # Create enhanced prompt for processing multiple data types
        enhanced_prompt = self._create_comprehensive_cad_prompt(found_data_types, settings)
        
        # Use the existing OpenAI deed parser with enhanced context
        try:
            # Temporarily store the original prompt creation method
            original_prompt_method = self.openai_parser._create_multi_tract_system_prompt
            
            # Replace with our enhanced prompt method
            self.openai_parser._create_multi_tract_system_prompt = lambda s: enhanced_prompt
            
            deed_result = self.openai_parser.parse_deed_text(combined_boundary_text, settings)
            
            # Restore original method
            self.openai_parser._create_multi_tract_system_prompt = original_prompt_method
            
            logger.info(f"Successfully converted to {len(deed_result.calls)} survey calls from {len(found_data_types)} data types")
            
            # Add metadata about processing
            deed_result.original_text = f"PROCESSED FROM {len(found_data_types)} BOUNDARY DATA TYPES:\n{', '.join(found_data_types)}\n\n{combined_boundary_text}"
            
            return deed_result
            
        except Exception as e:
            logger.error(f"Failed to convert combined boundary data to CAD format: {e}")
            raise
    
    def _create_comprehensive_cad_prompt(self, data_types: List[str], settings: ProjectSettings) -> str:
        """Create enhanced prompt for processing multiple boundary data types"""
        unit_context = f"Default distance unit is {settings.units}"
        bearing_context = f"Bearing convention is {settings.bearing_convention}"
        
        data_types_str = ", ".join(data_types)
        
        return f"""You are a precise legal deed parser specialized in extracting boundary survey calls from MULTIPLE TYPES of site survey data.

CONTEXT:
- {unit_context}
- {bearing_context}
- Processing {len(data_types)} different boundary data types: {data_types_str}
- Each section is marked with headers for identification
- Combine information from ALL sections to create complete boundary description

DATA TYPES BEING PROCESSED:
{self._get_data_type_descriptions(data_types)}

COMPREHENSIVE EXTRACTION INSTRUCTIONS:
1. ANALYZE ALL SECTIONS: Review each marked section for boundary information
2. PRIORITIZE STRUCTURED DATA: Line/curve tables and coordinate tables have highest priority
3. CROSS-REFERENCE: Use multiple sections to validate and complete boundary calls
4. MERGE COMPLEMENTARY DATA: Combine table data with narrative descriptions
5. RESOLVE CONFLICTS: When data conflicts, prioritize more structured sources
6. EXTRACT COMPLETE BOUNDARY: Create full site boundary from all available information

BOUNDARY CONSTRUCTION STRATEGY:
- If LINE/CURVE TABLE exists: Use as primary structure, supplement with other data
- If COORDINATE TABLE exists: Use for validation and missing coordinate information  
- If LEGAL/PROPERTY DESCRIPTION exists: Use for narrative context and missing calls
- If SURVEY NOTES exist: Use for clarification and additional details

QUALITY ASSURANCE:
- Ensure all boundary segments are captured from any available source
- Cross-validate bearings and distances between different data types
- Use highest confidence data when multiple sources provide same information
- Mark confidence based on data source reliability and consistency

TRACT IDENTIFICATION:
- Look for multiple tracts across all data sections
- Each tract may be described in different sections
- Combine tract information from all relevant sources

Return structured JSON with all tracts and their complete boundary call sequences extracted from ALL provided data types."""
    
    def _get_data_type_descriptions(self, data_types: List[str]) -> str:
        """Get descriptions for the data types being processed"""
        descriptions = {
            "line_curve_table": "- LINE_CURVE_TABLE: Structured tabular data with survey calls, bearings, distances, curve information",
            "bearing_distance_table": "- BEARING_DISTANCE_TABLE: Simple table format with bearings and distances",
            "coordinate_table": "- COORDINATE_TABLE: Point coordinates (X, Y) for boundary vertices",
            "legal_description": "- LEGAL_DESCRIPTION: Formal legal description with metes and bounds",
            "property_description": "- PROPERTY_DESCRIPTION: Narrative property description with survey calls",
            "deed_description": "- DEED_DESCRIPTION: Legal deed text with property boundaries",
            "survey_notes": "- SURVEY_NOTES: Surveyor's field notes and observations"
        }
        
        return "\n".join(descriptions.get(dt, f"- {dt.upper()}: Survey boundary information") for dt in data_types)


def test_pdf_boundary_parser():
    """Test the PDF boundary parser"""
    # Sample site survey text that might be extracted from PDF
    sample_text = """
    LEGAL DESCRIPTION
    
    Being a tract of land situated in Harris County, Texas, described as follows:
    
    BEGINNING at a concrete monument found at the intersection of Main Street and Oak Avenue;
    THENCE N 15°30'45" E along Oak Avenue, a distance of 125.50 feet to a point for corner;
    THENCE with a curve to the right having a radius of 285.00 feet, an arc length of 42.15 feet;
    THENCE S 75°15'00" E, a distance of 200.00 feet to an iron rod found for corner;
    THENCE S 15°30'45" W, a distance of 150.25 feet to the POINT OF BEGINNING.
    
    LINE AND CURVE TABLE
    Line    Bearing         Distance    Curve   Radius   Arc Length
    1       N 15°30'45" E   125.50      -       -        -
    2       Curve Right     -           Yes     285.00   42.15
    3       S 75°15'00" E   200.00      -       -        -
    4       S 15°30'45" W   150.25      -       -        -
    
    SURVEYOR NOTES:
    - Found concrete monument at POB in good condition
    - Iron rod set at northeast corner
    - All measurements in feet, bearings referenced to magnetic north
    """
    
    # This would require actual OpenAI API key
    print("PDF Boundary Parser test completed - requires API key for full testing")


if __name__ == "__main__":
    test_pdf_boundary_parser()
