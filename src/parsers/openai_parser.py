"""
OpenAI-based deed parsing using structured outputs and function calling.
"""
import os
import json
import math
import time
import logging
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pydantic import ValidationError

from ..models.deed_models import SurveyCall, PARSE_CALLS_SCHEMA, PARSE_TRACTS_SCHEMA, ProjectSettings, Tract, DeedParseResult
from ..utils.bearing_parser import BearingParser, DistanceParser
import re

# Configure logger
logger = logging.getLogger(__name__)
# Prevent duplicate handlers
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger


class OpenAIDeedParser:
    """Parse deed text using OpenAI with structured outputs and comprehensive logging"""
    
    # Supported OpenAI models with their capabilities
    SUPPORTED_MODELS = {
        # Standard GPT models (support custom temperature)
        "gpt-4o": {"supports_functions": True, "max_tokens": 128000, "reasoning": False, "supports_temperature": True},
        "gpt-4o-mini": {"supports_functions": True, "max_tokens": 128000, "reasoning": False, "supports_temperature": True},
        "gpt-4-turbo": {"supports_functions": True, "max_tokens": 128000, "reasoning": False, "supports_temperature": True},
        "gpt-4": {"supports_functions": True, "max_tokens": 8192, "reasoning": False, "supports_temperature": True},
        "gpt-3.5-turbo": {"supports_functions": True, "max_tokens": 16384, "reasoning": False, "supports_temperature": True},
        
        # New GPT models (may not support custom temperature)
        "gpt-4.1": {"supports_functions": True, "max_tokens": 200000, "reasoning": False, "supports_temperature": False},
        "gpt-4.1-mini": {"supports_functions": True, "max_tokens": 200000, "reasoning": False, "supports_temperature": False},
        "gpt-5": {"supports_functions": True, "max_tokens": 300000, "reasoning": False, "supports_temperature": False},
        
        # O1-series reasoning models (no custom temperature)
        "o1-preview": {"supports_functions": False, "max_tokens": 32768, "reasoning": True, "supports_temperature": False},
        "o1-mini": {"supports_functions": False, "max_tokens": 65536, "reasoning": True, "supports_temperature": False},
        
        # New O-series reasoning models (no custom temperature)
        "o3": {"supports_functions": False, "max_tokens": 100000, "reasoning": True, "supports_temperature": False},
        "o3-pro": {"supports_functions": False, "max_tokens": 150000, "reasoning": True, "supports_temperature": False},
        "o4-mini": {"supports_functions": False, "max_tokens": 80000, "reasoning": True, "supports_temperature": False},
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize parser with OpenAI client and logging.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: OpenAI model to use
        """
        logger.info(f"Initializing OpenAI Deed Parser with model: {model}")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        # Validate model
        if model not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported list. Proceeding with caution.")
            
        self.model = model
        self.model_info = self.SUPPORTED_MODELS.get(model, {"supports_functions": True, "reasoning": False})
        self.client = OpenAI(api_key=self.api_key)
        self.bearing_parser = BearingParser()
        self.distance_parser = DistanceParser()
        
        logger.info(f"Parser initialized successfully. Model supports functions: {self.model_info.get('supports_functions', True)}")
        logger.info(f"Reasoning model: {self.model_info.get('reasoning', False)}")
        logger.info(f"Temperature support: {self.model_info.get('supports_temperature', True)}")
        
        # Log model limitations
        if self.model_info.get('reasoning', False):
            logger.info("Reasoning model limitations: No system messages, no function calling, no custom temperature")
            logger.info("Reasoning model advantages: Advanced reasoning capabilities for complex deed analysis")
        elif not self.model_info.get('supports_temperature', True):
            logger.info("New model limitation: No custom temperature support (uses default temperature 1.0)")
            logger.info("New model advantages: Latest capabilities and improved performance")
    
    def parse_deed_text(self, deed_text: str, settings: ProjectSettings) -> DeedParseResult:
        """
        Parse deed text into structured survey calls, always using multi-tract approach.
        Single tracts are handled as arrays with one element.
        
        Args:
            deed_text: Raw deed or legal description text
            settings: Project settings for parsing context
            
        Returns:
            DeedParseResult containing tracts and calls
        """
        start_time = time.time()
        logger.info("=== Starting Deed Parsing Process ===")
        logger.info(f"Deed text length: {len(deed_text)} characters")
        logger.info(f"Using model: {self.model}")
        logger.info(f"Settings: {settings.units} units, {settings.bearing_convention} bearings")
        
        try:
            # Always use multi-tract parser - single tracts become arrays with 1 element
            result = self._parse_multiple_tracts(deed_text, settings)
            
            total_time = time.time() - start_time
            logger.info(f"=== Deed Parsing Completed Successfully ===")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Parsed {len(result.tracts)} tract(s)")
            logger.info(f"Total calls: {sum(len(tract.calls) for tract in result.tracts)}")
            logger.info(f"Average confidence: {result.total_confidence:.2f}")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Deed parsing failed after {total_time:.2f} seconds: {str(e)}")
            raise
    
    
    def _parse_multiple_tracts(self, deed_text: str, settings: ProjectSettings) -> DeedParseResult:
        """Parse deed extracting POB first, then line/curve information for each tract"""
        logger.info("--- Step 1: POB Coordinate Extraction ---")
        step1_start = time.time()
        
        # Step 1: Extract POB information for all tracts first
        pob_system_prompt = self._create_pob_extraction_prompt(settings)
        tract_pobs = self._extract_tract_pobs(deed_text, pob_system_prompt)
        
        step1_time = time.time() - step1_start
        logger.info(f"POB extraction completed in {step1_time:.2f} seconds")
        logger.info(f"Found {len(tract_pobs)} tract POBs")
        
        logger.info("--- Step 2: Line/Curve Information Extraction ---")
        step2_start = time.time()
        
        # Step 2: Extract line/curve information for each tract
        system_prompt = self._create_multi_tract_system_prompt(settings)
        raw_tracts_data = self._call_openai_multi_tract_parser(deed_text, system_prompt)
        
        step2_time = time.time() - step2_start
        logger.info(f"Line/curve extraction completed in {step2_time:.2f} seconds")
        logger.info(f"Extracted boundary data for {len(raw_tracts_data)} tracts")
        
        logger.info("--- Step 3: Data Merging and Processing ---")
        step3_start = time.time()
        
        # Step 3: Merge POB information with tract data
        merged_tracts_data = self._merge_pob_with_tracts(tract_pobs, raw_tracts_data)
        logger.info(f"Merged data for {len(merged_tracts_data)} tracts")
        
        # Process each tract
        tracts = []
        all_calls = []
        
        for i, tract_data in enumerate(merged_tracts_data):
            logger.info(f"Processing tract {i+1}/{len(merged_tracts_data)}: {tract_data.get('tract_id', 'Unknown')}")
            
            # Skip tracts with no calls to avoid geometry calculation issues
            if not tract_data.get('calls') or len(tract_data.get('calls', [])) == 0:
                logger.warning(f"Skipping tract '{tract_data.get('tract_id', 'Unknown')}' - no boundary calls found")
                continue
            
            tract = self._process_tract_data(tract_data, settings)
            tracts.append(tract)
            all_calls.extend(tract.calls)
            logger.info(f"Tract {tract.tract_id}: {len(tract.calls)} calls, POB=({tract.pob_x:.2f}, {tract.pob_y:.2f})")
        
        step3_time = time.time() - step3_start
        logger.info(f"Data processing completed in {step3_time:.2f} seconds")
        
        # Create result with detailed error handling
        try:
            logger.info("Creating DeedParseResult...")
            logger.info(f"Tracts: {len(tracts)}")
            logger.info(f"Calls: {len(all_calls)}")
            logger.info(f"Settings type: {type(settings)}")
            
            # Try to create without geometry first to isolate the issue
            result = DeedParseResult(
                tracts=tracts,
                settings=settings,
                calls=all_calls,
                geometry=None,  # Skip geometry initially
                original_text=deed_text
            )
            
            # Now try to add geometry if available
            if tracts and tracts[0].geometry:
                try:
                    result.geometry = tracts[0].geometry
                    logger.info("Successfully added geometry to result")
                except Exception as geom_error:
                    logger.error(f"Failed to add geometry to result: {geom_error}")
                    logger.error(f"Geometry type: {type(tracts[0].geometry)}")
                    # Keep result without geometry
                    
        except Exception as e:
            logger.error(f"Failed to create DeedParseResult: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try with minimal data
            try:
                result = DeedParseResult(
                    tracts=[],
                    settings=settings,
                    calls=[],
                    geometry=None,
                    original_text=deed_text
                )
                logger.info("Created minimal DeedParseResult as fallback")
            except Exception as minimal_error:
                logger.error(f"Even minimal DeedParseResult failed: {minimal_error}")
                raise
        
        logger.info(f"Final result: {len(result.tracts)} tracts, {len(all_calls)} total calls")
        return result
    

    

    
    def _create_multi_tract_system_prompt(self, settings: ProjectSettings) -> str:
        """Create system prompt for multi-tract parsing"""
        unit_context = f"Default distance unit is {settings.units}"
        bearing_context = f"Bearing convention is {settings.bearing_convention}"
        
        return f"""You are a precise legal deed parser specialized in extracting multiple tracts with boundary survey calls.

CONTEXT:
- {unit_context}
- {bearing_context}
- Multiple tracts may be present in this deed

INSTRUCTIONS:
1. Identify ALL separate tracts/parcels in the deed
2. For each tract, extract:
   - Tract identifier (TRACT 1, PARCEL A, etc.)
   - Point of Beginning description and coordinates if given
   - ALL boundary calls in sequence order
3. Preserve the exact raw text for each tract and call
4. Parse bearings into both text format and azimuth degrees (0-360)
5. Extract distances with units
6. Identify curves vs. lines based on keywords
7. Set confidence based on clarity

TRACT IDENTIFICATION:
Look for keywords like: "TRACT", "PARCEL", "LOT", "PORTION", "UNIT", "BEING TRACT", etc.
Each tract typically has its own Point of Beginning and set of boundary calls.

POINT OF BEGINNING:
- Extract POB description for each tract
- Look for coordinate information if provided
- Note references to monuments, intersections, etc.

Return structured JSON with all tracts and their respective calls."""
    
    def _process_tract_data(self, tract_data: Dict[str, Any], settings: ProjectSettings) -> Tract:
        """Process raw tract data from OpenAI into Tract object"""
        # Extract calculated POB coordinates
        pob_coords = tract_data.get("pob_coordinates", {})
        pob_x = pob_coords.get("x", 0.0)
        pob_y = pob_coords.get("y", 0.0)
        
        # Process calls for this tract
        raw_calls = tract_data.get("calls", [])
        processed_calls = self._post_process_calls(raw_calls, settings)
        
        # Create tract with error handling
        try:
            tract = Tract(
                tract_id=tract_data.get("tract_id", "Unknown Tract"),
                description=tract_data.get("description", ""),
                pob_x=pob_x,
                pob_y=pob_y,
                pob_description=tract_data.get("pob_description", "Point of Beginning"),
                calls=processed_calls,
                raw_text=tract_data.get("raw_text", "")
            )
            logger.info(f"Successfully created tract: {tract.tract_id}")
        except Exception as e:
            logger.error(f"Failed to create tract: {e}")
            # Create a minimal tract (Tract is already imported at module level)
            tract = Tract(
                tract_id="Error Tract",
                description="Failed to parse tract",
                pob_x=0.0,
                pob_y=0.0,
                pob_description="Point of Beginning",
                calls=[],
                raw_text=""
            )
        
        # Calculate geometry for this tract
        try:
            from ..geometry.calculator import GeometryCalculator
            tract_settings = ProjectSettings(
                units=settings.units,
                bearing_convention=settings.bearing_convention,
                pob_x=tract.pob_x,
                pob_y=tract.pob_y,
                pob_description=tract.pob_description,
                confidence_threshold=settings.confidence_threshold,
                closure_tolerance=settings.closure_tolerance
            )
            
            calculator = GeometryCalculator(tract_settings)
            geometry = calculator.calculate_polygon(processed_calls)
            
            # Ensure geometry is properly serializable by converting to dict and back
            from ..models.deed_models import PolygonGeometry
            geometry_dict = geometry.model_dump()
            tract.geometry = PolygonGeometry(**geometry_dict)
            logger.info(f"Successfully calculated geometry for tract {tract.tract_id}")
        except Exception as e:
            logger.error(f"Failed to calculate geometry for tract {tract.tract_id}: {e}")
            tract.geometry = None
        
        return tract
    
    def _create_pob_extraction_prompt(self, settings: ProjectSettings) -> str:
        """Create system prompt focused on POB coordinate calculation"""
        unit_context = f"Default distance unit is {settings.units}"
        bearing_context = f"Bearing convention is {settings.bearing_convention}"
        
        return f"""You are a surveyor and legal deed parser specialized in calculating Point of Beginning (POB) coordinates for each tract.

CONTEXT:
- {unit_context}
- {bearing_context}
- You must CALCULATE POB coordinates from deed descriptions, not look for explicit coordinates
- Establish a local coordinate system based on reference points in the deed
- IMPORTANT: Total Tract = sum of all individual tracts (understand this relationship)

INSTRUCTIONS:
1. FIRST PRIORITY: Identify ALL tracts/parcels in the deed, including "Total Tract"
2. UNDERSTAND TRACT RELATIONSHIPS:
   - "Total Tract" encompasses all individual tracts
   - Individual tracts are subdivisions of the Total Tract
   - Use this relationship to improve coordinate accuracy
3. For each tract, analyze the POB establishment process:
   - Find the reference point (intersection, monument, etc.)
   - Follow the courses and distances to reach the POB
   - Calculate the POB coordinates in a local system
   - Consider relationships between tract POBs
4. COORDINATE CALCULATION PROCESS:
   - Establish origin (0,0) at a logical reference point (e.g., road intersection)
   - Set coordinate axes (typically N-S and E-W)
   - Follow bearing and distance instructions step by step
   - Calculate final POB coordinates (X, Y)
   - Validate that individual tract POBs are consistent with Total Tract

POB ESTABLISHMENT PATTERNS:
- "To find the Point of Beginning commence at..."
- "Beginning at... thence... to the Point of Beginning"
- "Point of Beginning", "POB", "Point of Commencement"

COORDINATE SYSTEM ASSUMPTIONS:
- Origin at major intersection or reference point
- X-axis = East-West (positive East)
- Y-axis = North-South (positive North)
- Use survey bearing conventions (N/S reference, angles toward E/W)

BEARING CALCULATIONS:
- N θ° W: ΔX = -L·sin(θ), ΔY = +L·cos(θ)
- N θ° E: ΔX = +L·sin(θ), ΔY = +L·cos(θ)
- S θ° W: ΔX = -L·sin(θ), ΔY = -L·cos(θ)
- S θ° E: ΔX = +L·sin(θ), ΔY = -L·cos(θ)

Return calculated POB coordinates (X, Y) for each tract."""
    
    def _extract_tract_pobs(self, deed_text: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Extract and calculate POB coordinates for all tracts"""
        start_time = time.time()
        logger.info(f"Starting POB extraction using model: {self.model}")
        
        try:
            # Handle o-series models differently (they don't support function calling or system messages)
            if self.model_info.get('reasoning', False):
                logger.info("Using reasoning model - text-based response parsing")
                tract_pobs = self._extract_pobs_with_reasoning_model(deed_text, system_prompt)
                
                response_time = time.time() - start_time
                logger.info(f"POB extraction completed in {response_time:.2f} seconds")
                logger.info(f"Extracted {len(tract_pobs)} POB coordinates")
                
                # Log each POB
                for pob in tract_pobs:
                    coords = pob.get('pob_coordinates', {})
                    logger.info(f"  {pob.get('tract_id', 'Unknown')}: ({coords.get('x', 0):.2f}, {coords.get('y', 0):.2f})")
                
                return tract_pobs
                
            else:
                logger.info("Using standard model with function calling")
                
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract all POBs' coordinates (X, Y) from this deed:\n\n{deed_text}"}
                    ],
                    "functions": [self._get_pob_extraction_schema()],
                    "function_call": {"name": "extract_tract_pobs"}
                }
                
                # Only add temperature if the model supports it
                if self.model_info.get('supports_temperature', True):
                    api_params["temperature"] = 0.0
                    logger.info("Using temperature=0.0 for consistent results")
                else:
                    logger.info("Using default temperature (model doesn't support custom temperature)")
                
                response = self.client.chat.completions.create(**api_params)
                
                message = response.choices[0].message
                if message.function_call:
                    function_args = json.loads(message.function_call.arguments)
                    
                    # Extract coordinate system info for logging
                    coord_system = function_args.get("coordinate_system", {})
                    tract_pobs = function_args.get("tract_pobs", [])
                    
                    response_time = time.time() - start_time
                    logger.info(f"POB extraction completed in {response_time:.2f} seconds")
                    logger.info(f"Coordinate System: {coord_system.get('origin_description', 'Unknown origin')}")
                    logger.info(f"Axes: X={coord_system.get('x_axis_direction', 'East')}, Y={coord_system.get('y_axis_direction', 'North')}")
                    logger.info(f"Extracted {len(tract_pobs)} POB coordinates")
                    
                    # Log each POB
                    for pob in tract_pobs:
                        coords = pob.get('pob_coordinates', {})
                        logger.info(f"  {pob.get('tract_id', 'Unknown')}: ({coords.get('x', 0):.2f}, {coords.get('y', 0):.2f})")
                    
                    return tract_pobs
                else:
                    raise ValueError("No function call in OpenAI POB extraction response")
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"POB extraction failed after {response_time:.2f} seconds: {str(e)}")
            raise ValueError(f"OpenAI POB extraction failed: {str(e)}")
    
    def _get_pob_extraction_schema(self) -> Dict[str, Any]:
        """Schema for POB coordinate calculation function calling"""
        return {
            "name": "extract_tract_pobs",
            "description": "Calculate Point of Beginning coordinates (X, Y) for each tract by analyzing deed description",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate_system": {
                        "type": "object",
                        "properties": {
                            "origin_description": {"type": "string", "description": "Description of the coordinate system origin"},
                            "x_axis_direction": {"type": "string", "description": "Direction of X-axis (typically East)"},
                            "y_axis_direction": {"type": "string", "description": "Direction of Y-axis (typically North)"}
                        },
                        "required": ["origin_description", "x_axis_direction", "y_axis_direction"]
                    },
                    "tract_pobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tract_id": {"type": "string", "description": "Tract identifier (e.g., 'Total Tract', 'Tract 1', 'Tract 2')"},
                                "pob_description": {"type": "string", "description": "Full POB establishment description from deed"},
                                "calculation_steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "step": {"type": "string", "description": "Description of calculation step"},
                                            "bearing": {"type": "string", "description": "Bearing if applicable"},
                                            "distance": {"type": "number", "description": "Distance if applicable"},
                                            "from_point": {"type": "string", "description": "Starting point description"},
                                            "to_point": {"type": "string", "description": "Ending point description"}
                                        },
                                        "required": ["step"]
                                    }
                                },
                                "pob_coordinates": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number", "description": "Calculated X coordinate"},
                                        "y": {"type": "number", "description": "Calculated Y coordinate"}
                                    },
                                    "required": ["x", "y"]
                                },
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Calculation confidence"}
                            },
                            "required": ["tract_id", "pob_description", "calculation_steps", "pob_coordinates", "confidence"]
                        }
                    }
                },
                "required": ["coordinate_system", "tract_pobs"]
            }
        }
    
    def _merge_pob_with_tracts(self, tract_pobs: List[Dict[str, Any]], 
                              raw_tracts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge calculated POB coordinates with tract data, using Total Tract as primary reference"""
        # Create a mapping of tract_id to POB data
        pob_map = {pob["tract_id"]: pob for pob in tract_pobs}
        
        # Find Total Tract for POB reference hierarchy
        total_tract_pob = None
        for pob in tract_pobs:
            if "total" in pob["tract_id"].lower() or "total tract" in pob["tract_id"].lower():
                total_tract_pob = pob
                logger.info(f"Found Total Tract POB reference: {pob['tract_id']} at ({pob['pob_coordinates']['x']:.2f}, {pob['pob_coordinates']['y']:.2f})")
                break
        
        merged_tracts = []
        for tract_data in raw_tracts_data:
            tract_id = tract_data.get("tract_id", "Unknown Tract")
            
            # Find matching POB data
            pob_data = pob_map.get(tract_id)
            if pob_data:
                # Use Total Tract POB as reference if available and this is not the Total Tract
                if total_tract_pob and tract_id != total_tract_pob["tract_id"]:
                    # Calculate relative offset from Total Tract POB
                    offset_x = pob_data["pob_coordinates"]["x"] - total_tract_pob["pob_coordinates"]["x"]
                    offset_y = pob_data["pob_coordinates"]["y"] - total_tract_pob["pob_coordinates"]["y"]
                    
                    logger.info(f"Tract {tract_id}: Using Total Tract as reference")
                    logger.info(f"  Offset from Total Tract: ({offset_x:.2f}, {offset_y:.2f})")
                    
                    # Store both absolute and relative coordinates
                    tract_data["pob_coordinates"] = {
                        "x": pob_data["pob_coordinates"]["x"],
                        "y": pob_data["pob_coordinates"]["y"],
                        "has_coordinates": True,
                        "reference_tract": total_tract_pob["tract_id"],
                        "offset_x": offset_x,
                        "offset_y": offset_y
                    }
                else:
                    # Use direct coordinates (for Total Tract or when no Total Tract found)
                    tract_data["pob_coordinates"] = {
                        "x": pob_data["pob_coordinates"]["x"],
                        "y": pob_data["pob_coordinates"]["y"],
                        "has_coordinates": True
                    }
                
                tract_data["pob_description"] = pob_data["pob_description"]
                tract_data["pob_calculation_steps"] = pob_data.get("calculation_steps", [])
            
            merged_tracts.append(tract_data)
        
        # Handle case where POB extraction found more tracts than tract parser
        # Use fuzzy matching to avoid duplicates (e.g., "TOTAL TRACT" vs "Total Tract")
        for pob_data in tract_pobs:
            pob_tract_id = pob_data["tract_id"]
            
            # Check for exact match first
            exact_match = any(t.get("tract_id") == pob_tract_id for t in merged_tracts)
            
            # Check for fuzzy match (case-insensitive, ignore spaces/punctuation)
            fuzzy_match = any(
                self._normalize_tract_id(t.get("tract_id", "")) == self._normalize_tract_id(pob_tract_id) 
                for t in merged_tracts
            )
            
            if not exact_match and not fuzzy_match:
                # Only create tract if no similar tract exists
                merged_tracts.append({
                    "tract_id": pob_tract_id,
                    "description": f"Tract identified by POB calculation: {pob_data['pob_description'][:50]}...",
                    "pob_description": pob_data["pob_description"],
                    "pob_coordinates": {
                        "x": pob_data["pob_coordinates"]["x"],
                        "y": pob_data["pob_coordinates"]["y"],
                        "has_coordinates": True
                    },
                    "pob_calculation_steps": pob_data.get("calculation_steps", []),
                    "raw_text": "",
                    "calls": []
                })
            elif fuzzy_match and not exact_match:
                # Update the existing tract with the POB tract ID for consistency
                for tract in merged_tracts:
                    if self._normalize_tract_id(tract.get("tract_id", "")) == self._normalize_tract_id(pob_tract_id):
                        logger.info(f"Updating tract ID from '{tract['tract_id']}' to '{pob_tract_id}' for consistency")
                        tract["tract_id"] = pob_tract_id
                        break
        
        # Sort tracts to put Total Tract first for better reference
        merged_tracts.sort(key=lambda t: (0 if "total" in t.get("tract_id", "").lower() else 1, t.get("tract_id", "")))
        
        return merged_tracts
    
    def _normalize_tract_id(self, tract_id: str) -> str:
        """Normalize tract ID for fuzzy matching (case-insensitive, no spaces/punctuation)"""
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', tract_id.lower())
    
    def _extract_pobs_with_reasoning_model(self, deed_text: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Extract POBs using o-series reasoning models (text-based parsing)"""
        logger.info("Using reasoning model for POB extraction - parsing text response")
        
        # O1 models don't support system messages, so combine everything into user message
        combined_prompt = f"""You are a surveyor and legal deed parser specialized in calculating Point of Beginning (POB) coordinates for each tract.

INSTRUCTIONS:
1. FIRST PRIORITY: Identify ALL tracts/parcels in the deed
2. For each tract, analyze the POB establishment process:
   - Find the reference point (intersection, monument, etc.)
   - Follow the courses and distances to reach the POB
   - Calculate the POB coordinates in a local system
3. COORDINATE CALCULATION PROCESS:
   - Establish origin (0,0) at a logical reference point (e.g., road intersection)
   - Set coordinate axes (typically N-S and E-W)
   - Follow bearing and distance instructions step by step
   - Calculate final POB coordinates (X, Y)

COORDINATE SYSTEM ASSUMPTIONS:
- Origin at major intersection or reference point
- X-axis = East-West (positive East)
- Y-axis = North-South (positive North)
- Use survey bearing conventions (N/S reference, angles toward E/W)

BEARING CALCULATIONS:
- N θ° W: ΔX = -L·sin(θ), ΔY = +L·cos(θ)
- N θ° E: ΔX = +L·sin(θ), ΔY = +L·cos(θ)
- S θ° W: ΔX = -L·sin(θ), ΔY = -L·cos(θ)
- S θ° E: ΔX = +L·sin(θ), ΔY = -L·cos(θ)

TRACT RELATIONSHIP ANALYSIS:
- Identify if there is a "Total Tract" that encompasses all individual tracts
- If Total Tract exists, use it as the primary reference for coordinate consistency
- Validate that individual tract POBs are logically positioned relative to Total Tract
- Consider that Total Tract boundary should encompass all individual tract boundaries

Please analyze the deed and return your response in the following JSON format:
{{
  "coordinate_system": {{
    "origin_description": "Description of coordinate system origin",
    "x_axis_direction": "East",
    "y_axis_direction": "North"
  }},
  "tract_relationships": {{
    "has_total_tract": "boolean - whether a Total Tract exists",
    "total_tract_id": "ID of the Total Tract if it exists",
    "relationship_notes": "Notes about how tracts relate to each other"
  }},
  "tract_pobs": [
    {{
      "tract_id": "Tract identifier",
      "is_total_tract": "boolean - whether this is the encompassing Total Tract",
      "pob_description": "Full POB description",
      "calculation_steps": [
        {{
          "step": "Step description",
          "bearing": "Bearing if applicable",
          "distance": "Distance if applicable",
          "from_point": "Starting point",
          "to_point": "Ending point"
        }}
      ],
      "pob_coordinates": {{
        "x": "calculated X coordinate",
        "y": "calculated Y coordinate"
      }},
      "confidence": "confidence value between 0 and 1",
      "relationship_to_total": "how this tract relates to the Total Tract if applicable"
    }}
  ]
}}

DEED TEXT TO ANALYZE:
{deed_text}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
            # Note: o1 models only support default temperature (1.0)
        )
        
        # Parse JSON from response
        response_text = response.choices[0].message.content
        logger.info(f"Reasoning model response length: {len(response_text)} characters")
        
        try:
            # Extract JSON from response with multiple strategies
            import re
            
            # Strategy 1: Look for JSON code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("Found JSON in code block")
            else:
                # Strategy 2: Look for JSON object starting with {
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.info("Found JSON object in response")
                else:
                    # Strategy 3: Use entire response
                    json_str = response_text
                    logger.info("Using entire response as JSON")
            
            # Clean up common issues
            json_str = json_str.strip()
            
            parsed_data = json.loads(json_str)
            tract_pobs = parsed_data.get("tract_pobs", [])
            logger.info(f"Successfully parsed {len(tract_pobs)} tract POBs")
            
            return tract_pobs
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from reasoning model response: {e}")
            logger.error(f"Response text preview: {response_text[:1000]}...")
            
            # Try to extract basic information even if JSON parsing fails
            logger.warning("Attempting fallback parsing...")
            return self._fallback_pob_parsing(response_text)
    
    def _fallback_pob_parsing(self, response_text: str) -> List[Dict[str, Any]]:
        """Fallback parsing when JSON extraction fails"""
        logger.warning("Using fallback POB parsing")
        
        # Create a minimal POB entry
        return [{
            "tract_id": "TRACT 1",
            "pob_description": "Point of Beginning (extracted from fallback parsing)",
            "calculation_steps": [{"step": "Fallback parsing - manual review required"}],
            "pob_coordinates": {"x": 0.0, "y": 0.0},
            "confidence": 0.1
        }]
    
    def _parse_tracts_with_reasoning_model(self, deed_text: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Parse tracts using o-series reasoning models (text-based parsing)"""
        logger.info("Using reasoning model for tract parsing - parsing text response")
        
        # O1 models don't support system messages, so combine everything into user message
        combined_prompt = f"""You are a precise legal deed parser specialized in extracting boundary survey calls for site boundary drawing.

INSTRUCTIONS:
1. For each tract, extract ALL boundary calls in sequence order starting from POB
2. Site boundary is drawn by connecting lines and curves from POB
3. Preserve exact raw text for each call
4. Parse bearings into both text format and azimuth degrees (0-360)
5. Extract distances with units
6. Identify curves vs. lines based on keywords
7. For curves, extract: direction (L/R), radius, arc length, chord bearing, chord length
8. Set confidence based on clarity:
   - 0.9-1.0: Clear, unambiguous
   - 0.7-0.9: Minor ambiguity
   - 0.5-0.7: Significant uncertainty
   - <0.5: Very unclear or missing data

SITE BOUNDARY CONSTRUCTION:
- Each call represents a segment of the site boundary
- Lines: straight segments with bearing and distance
- Curves: curved segments with radius, arc length, chord information
- Sequence is critical for proper boundary drawing

CURVE INDICATORS:
- "curve", "arc", "radius", "chord"
- "thence curving", "with a curve"
- "R=", "radius =", "arc length"
- "curve to the left/right"

BEARING FORMATS:
- Quadrant: N 45°30'15" E, S 22°10' W, N45-30-15E
- Azimuth: 123°45'30", 123.75°

Please analyze the deed and return your response in JSON format with the following structure:
{{
  "tracts": [
    {{
      "tract_id": "Tract identifier",
      "description": "Brief description of the tract",
      "pob_description": "Point of Beginning description",
      "raw_text": "Raw text for this tract",
      "calls": [
        {{
          "sequence": "Order in boundary",
          "type": "line or curve or tie_line or tie_curve",
          "raw_text": "Original text segment",
          "bearing": "Bearing text (e.g. 'N 45°30' E')",
          "azimuth_deg": "Azimuth 0-360 degrees",
          "distance": "Distance value",
          "distance_unit": "ft or m or ch or rd",
          "curve_direction": "L or R",
          "radius": "Curve radius",
          "arc_length": "Arc length",
          "chord_bearing": "Chord bearing text",
          "chord_length": "Chord length",
          "confidence": "confidence value between 0 and 1"
        }}
      ]
    }}
  ]
}}

Focus on extracting boundary calls for each tract in sequence order.

DEED TEXT TO ANALYZE:
{deed_text}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        )
        
        # Parse JSON from response
        response_text = response.choices[0].message.content
        logger.info(f"Multi-tract reasoning model response length: {len(response_text)} characters")
        
        try:
            import re
            
            # Multiple parsing strategies
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("Found JSON in code block")
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.info("Found JSON object in response")
                else:
                    json_str = response_text
                    logger.info("Using entire response as JSON")
            
            json_str = json_str.strip()
            parsed_data = json.loads(json_str)
            tracts = parsed_data.get("tracts", [])
            logger.info(f"Successfully parsed {len(tracts)} tracts")
            
            return tracts
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from reasoning model response: {e}")
            logger.error(f"Response text preview: {response_text[:1000]}...")
            
            # Fallback: create a minimal tract with empty calls
            logger.warning("Using fallback tract parsing")
            return [{
                "tract_id": "TRACT 1",
                "description": "Tract from fallback parsing",
                "pob_description": "Point of Beginning",
                "raw_text": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "calls": []
            }]
    
    def _call_openai_multi_tract_parser(self, deed_text: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Call OpenAI with multi-tract function calling"""
        start_time = time.time()
        logger.info(f"Starting multi-tract parsing using model: {self.model}")
        
        try:
            # Handle o-series models differently (they don't support function calling or system messages)
            if self.model_info.get('reasoning', False):
                logger.info("Using reasoning model for multi-tract parsing")
                tracts_data = self._parse_tracts_with_reasoning_model(deed_text, system_prompt)
                
                response_time = time.time() - start_time
                logger.info(f"Multi-tract parsing completed in {response_time:.2f} seconds")
                logger.info(f"Extracted data for {len(tracts_data)} tracts")
                
                # Log tract summary
                for i, tract in enumerate(tracts_data):
                    calls_count = len(tract.get('calls', []))
                    logger.info(f"  Tract {i+1}: {tract.get('tract_id', 'Unknown')} - {calls_count} calls")
                
                return tracts_data
                
            else:
                logger.info("Using standard model with function calling for multi-tract parsing")
                
                # Prepare API call parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                    {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Parse this multi-tract deed description:\n\n{deed_text}"}
                    ],
                    "functions": [PARSE_TRACTS_SCHEMA],
                    "function_call": {"name": "parse_deed_tracts"}
                }
                
                # Only add temperature if the model supports it
                if self.model_info.get('supports_temperature', True):
                    api_params["temperature"] = 0.1
                    logger.info("Using temperature=0.1 for multi-tract parsing")
                else:
                    logger.info("Using default temperature for multi-tract parsing")
                
                response = self.client.chat.completions.create(**api_params)
            
            # Extract function call result
            message = response.choices[0].message
            if message.function_call:
                function_args = json.loads(message.function_call.arguments)
                tracts_data = function_args.get("tracts", [])
                
                response_time = time.time() - start_time
                logger.info(f"Multi-tract parsing completed in {response_time:.2f} seconds")
                logger.info(f"Extracted data for {len(tracts_data)} tracts")
                
                # Log tract summary
                for i, tract in enumerate(tracts_data):
                    calls_count = len(tract.get('calls', []))
                    logger.info(f"  Tract {i+1}: {tract.get('tract_id', 'Unknown')} - {calls_count} calls")
                
                return tracts_data
            else:
                raise ValueError("No function call in OpenAI response")
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Multi-tract parsing failed after {response_time:.2f} seconds: {str(e)}")
            raise ValueError(f"OpenAI multi-tract parsing failed: {str(e)}")
    
    def _create_multi_tract_system_prompt(self, settings: ProjectSettings) -> str:
        """Create system prompt for line/curve extraction after POB identification"""
        unit_context = f"Default distance unit is {settings.units}"
        bearing_context = f"Bearing convention is {settings.bearing_convention}"
        
        return f"""You are a precise legal deed parser specialized in extracting boundary survey calls for site boundary drawing.

CONTEXT:
- {unit_context}
- {bearing_context}
- POB information has been pre-extracted
- Focus on line/curve sequence for site boundary construction

INSTRUCTIONS:
1. For each tract, extract ALL boundary calls in sequence order starting from POB
2. Site boundary is drawn by connecting lines and curves from POB
3. Preserve exact raw text for each call
4. Parse bearings into both text format and azimuth degrees (0-360)
5. Extract distances with units
6. Identify curves vs. lines based on keywords
7. For curves, extract: direction (L/R), radius, arc length, chord bearing, chord length
8. Set confidence based on clarity:
   - 0.9-1.0: Clear, unambiguous
   - 0.7-0.9: Minor ambiguity
   - 0.5-0.7: Significant uncertainty
   - <0.5: Very unclear or missing data

SITE BOUNDARY CONSTRUCTION:
- Each call represents a segment of the site boundary
- Lines: straight segments with bearing and distance
- Curves: curved segments with radius, arc length, chord information
- Sequence is critical for proper boundary drawing

CURVE INDICATORS:
- "curve", "arc", "radius", "chord"
- "thence curving", "with a curve"
- "R=", "radius =", "arc length"
- "curve to the left/right"

BEARING FORMATS:
- Quadrant: N 45°30'15" E, S 22°10' W, N45-30-15E
- Azimuth: 123°45'30", 123.75°

Return structured JSON with all tracts and their boundary call sequences."""
    
    def _post_process_calls(self, raw_calls: List[Dict[str, Any]], settings: ProjectSettings) -> List[SurveyCall]:
        """Post-process and validate OpenAI results"""
        processed_calls = []
        
        for i, call_data in enumerate(raw_calls):
            try:
                logger.info(f"Processing call {i+1}: {call_data.get('type', 'unknown')} - {call_data.get('raw_text', '')[:50]}...")
                
                # Ensure call_data has proper string values for distance_unit
                if 'distance_unit' in call_data and call_data['distance_unit']:
                    # Convert enum to string if needed
                    if hasattr(call_data['distance_unit'], 'value'):
                        call_data['distance_unit'] = call_data['distance_unit'].value
                    # Ensure it's a string
                    call_data['distance_unit'] = str(call_data['distance_unit'])
                
                # Create base call
                call = SurveyCall(**call_data)
                
                # Post-process with deterministic parsing
                call = self._enhance_call_with_regex(call, settings)
                
                # Validate and normalize
                call = self._validate_and_normalize(call, settings)
                
                # Additional validation for required fields
                call = self._ensure_required_fields(call, call_data, settings)
                
                processed_calls.append(call)
                logger.info(f"Call {i+1} processed: bearing={call.bearing}, azimuth={call.azimuth_deg}, distance={call.distance}")
                
            except ValidationError as e:
                logger.error(f"Validation error for call {i+1}: {str(e)}")
                # Create a low-confidence call for invalid data
                fallback_call = SurveyCall(
                    sequence=i + 1,
                    type="line",  # Default to line
                    raw_text=call_data.get("raw_text", ""),
                    confidence=0.1,
                    source="validation_error",
                    notes=f"Validation error: {str(e)}"
                )
                processed_calls.append(fallback_call)
            except Exception as e:
                logger.error(f"Unexpected error processing call {i+1}: {str(e)}")
                # Create fallback call for any other errors
                fallback_call = SurveyCall(
                    sequence=i + 1,
                    type="line",
                    raw_text=call_data.get("raw_text", "Unknown call"),
                    confidence=0.1,
                    source="error",
                    notes=f"Processing error: {str(e)}"
                )
                processed_calls.append(fallback_call)
        
        return processed_calls
    
    def _enhance_call_with_regex(self, call: SurveyCall, settings: ProjectSettings) -> SurveyCall:
        """Enhance call data with deterministic regex parsing"""
        # Parse bearing if missing or low confidence
        if not call.bearing or call.confidence < 0.7:
            regex_bearing = self._extract_bearing_with_regex(call.raw_text)
            if regex_bearing:
                if not call.bearing:
                    call.bearing = regex_bearing
                    call.source = "regex"
                elif call.confidence < 0.7:
                    # Compare with LLM result
                    call.notes = f"LLM: {call.bearing}, Regex: {regex_bearing}"
        
        # Parse distance if missing
        if not call.distance or call.confidence < 0.7:
            regex_distance, regex_unit = self._extract_distance_with_regex(call.raw_text)
            if regex_distance:
                if not call.distance:
                    call.distance = regex_distance
                    call.distance_unit = regex_unit
                    call.source = "regex"
                elif call.confidence < 0.7:
                    call.notes = f"LLM dist: {call.distance}, Regex: {regex_distance}"
        
        return call
    
    def _extract_bearing_with_regex(self, text: str) -> Optional[str]:
        """Extract bearing using regex patterns"""
        # Try to find bearing patterns in text
        azimuth = self.bearing_parser.parse_bearing(text)
        if azimuth is not None:
            # Convert back to quadrant format for consistency
            return self.bearing_parser.azimuth_to_quadrant(azimuth)
        return None
    
    def _extract_distance_with_regex(self, text: str) -> tuple[Optional[float], Optional[str]]:
        """Extract distance using regex patterns"""
        return self.distance_parser.parse_distance(text)
    
    def _validate_and_normalize(self, call: SurveyCall, settings: ProjectSettings) -> SurveyCall:
        """Validate and normalize call data"""
        # Calculate azimuth from bearing if missing
        if call.bearing and not call.azimuth_deg:
            azimuth = self.bearing_parser.parse_bearing(call.bearing)
            if azimuth is not None:
                call.azimuth_deg = azimuth
        
        # Normalize distance units
        if call.distance and call.distance_unit:
            if call.distance_unit != settings.units:
                # Convert to project units
                if settings.units == "ft" and call.distance_unit == "m":
                    call.distance = call.distance * 3.28084
                    call.distance_unit = "ft"
                elif settings.units == "m" and call.distance_unit == "ft":
                    call.distance = call.distance / 3.28084
                    call.distance_unit = "m"
        
        # Validate curve data consistency
        if call.type in ["curve", "tie_curve"]:
            if call.radius and call.chord_length and not call.arc_length:
                # Calculate arc length from radius and chord
                if call.radius > call.chord_length / 2:
                    central_angle = 2 * math.asin(call.chord_length / (2 * call.radius))
                    call.arc_length = call.radius * central_angle
        
        # Update confidence based on completeness
        required_fields = self._get_required_fields(call.type)
        missing_fields = [field for field in required_fields if getattr(call, field) is None]
        
        if missing_fields:
            call.confidence = min(call.confidence, 0.5)
            if not call.notes:
                call.notes = f"Missing: {', '.join(missing_fields)}"
        
        return call
    
    def _ensure_required_fields(self, call: SurveyCall, call_data: Dict[str, Any], settings: ProjectSettings) -> SurveyCall:
        """Ensure required fields are present and valid for geometry calculations"""
        
        # For line calls, we need bearing/azimuth and distance
        if call.type in ["line", "tie_line"]:
            # Calculate azimuth from bearing if missing
            if call.azimuth_deg is None and call.bearing:
                try:
                    azimuth = self.bearing_parser.parse_bearing(call.bearing)
                    if azimuth is not None:
                        call.azimuth_deg = azimuth
                        logger.info(f"Calculated azimuth {azimuth:.1f}° from bearing '{call.bearing}'")
                except Exception as e:
                    logger.warning(f"Failed to calculate azimuth from bearing '{call.bearing}': {e}")
            
            # Try to extract bearing from raw text if still missing
            if call.azimuth_deg is None and call.raw_text:
                try:
                    azimuth = self.bearing_parser.parse_bearing(call.raw_text)
                    if azimuth is not None:
                        call.azimuth_deg = azimuth
                        # Also set bearing in standard format
                        call.bearing = self.bearing_parser.azimuth_to_quadrant(azimuth)
                        logger.info(f"Extracted azimuth {azimuth:.1f}° from raw text")
                except Exception as e:
                    logger.warning(f"Failed to extract bearing from raw text: {e}")
            
            # Try to extract distance from raw text if missing
            if call.distance is None and call.raw_text:
                try:
                    distance, unit = self.distance_parser.parse_distance(call.raw_text)
                    if distance is not None:
                        call.distance = distance
                        call.distance_unit = unit
                        logger.info(f"Extracted distance {distance} {unit} from raw text")
                except Exception as e:
                    logger.warning(f"Failed to extract distance from raw text: {e}")
            
            # Final validation - if we still don't have required fields, mark as low confidence
            if call.azimuth_deg is None or call.distance is None:
                missing_fields = []
                if call.azimuth_deg is None:
                    missing_fields.append("azimuth/bearing")
                if call.distance is None:
                    missing_fields.append("distance")
                
                call.confidence = min(call.confidence, 0.2)
                call.notes = f"Missing required fields: {', '.join(missing_fields)}. {call.notes or ''}"
                logger.warning(f"Call {call.sequence} missing required fields: {missing_fields}")
        
        return call
    
    def _get_required_fields(self, call_type: str) -> List[str]:
        """Get required fields for each call type"""
        base_fields = ["bearing", "distance"]
        
        if call_type in ["curve", "tie_curve"]:
            return base_fields + ["radius", "chord_length"]
        else:
            return base_fields


# Utility function for testing
def test_parser():
    """Test the parser with sample deed text"""
    sample_deed = """
    Beginning at a point on the north line of Main Street;
    thence N 45°30' E, 120.00 feet to a point;
    thence with a curve to the right having a radius of 300.00 feet,
    an arc length of 47.12 feet, chord bearing N 50°15' E, chord length 47.08 feet;
    thence S 30° E, 85.5 feet to the point of beginning.
    """
    
    settings = ProjectSettings()
    parser = OpenAIDeedParser()
    
    try:
        result = parser.parse_deed_text(sample_deed, settings)
        print(f"Parsed {len(result.tracts)} tract(s):")
        
        for i, tract in enumerate(result.tracts):
            print(f"  Tract {i+1}: {tract.tract_id}")
            print(f"    POB: {tract.pob_description}")
            print(f"    POB Coordinates: ({tract.pob_x:.2f}, {tract.pob_y:.2f})")
            print(f"    Calls: {len(tract.calls)}")
            for call in tract.calls:
                print(f"      {call.sequence}: {call.type} - {call.raw_text[:50]}...")
                print(f"        Confidence: {call.confidence}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_parser()
