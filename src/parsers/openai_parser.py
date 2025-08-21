"""
OpenAI-based deed parsing using structured outputs and function calling.
"""
import os
import json
import math
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pydantic import ValidationError

from ..models.deed_models import SurveyCall, PARSE_CALLS_SCHEMA, PARSE_TRACTS_SCHEMA, ProjectSettings, Tract, DeedParseResult
from ..utils.bearing_parser import BearingParser, DistanceParser
import re


class OpenAIDeedParser:
    """Parse deed text using OpenAI with structured outputs"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize parser with OpenAI client.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.bearing_parser = BearingParser()
        self.distance_parser = DistanceParser()
    
    def parse_deed_text(self, deed_text: str, settings: ProjectSettings) -> DeedParseResult:
        """
        Parse deed text into structured survey calls, handling multiple tracts.
        
        Args:
            deed_text: Raw deed or legal description text
            settings: Project settings for parsing context
            
        Returns:
            DeedParseResult containing tracts and calls
        """
        # Check if deed contains multiple tracts
        has_multiple_tracts = self._detect_multiple_tracts(deed_text)
        
        if has_multiple_tracts:
            return self._parse_multiple_tracts(deed_text, settings)
        else:
            return self._parse_single_tract(deed_text, settings)
    
    def _detect_multiple_tracts(self, deed_text: str) -> bool:
        """
        Detect if deed text contains multiple tracts.
        
        Args:
            deed_text: Raw deed text
            
        Returns:
            True if multiple tracts detected
        """
        # Common patterns for multiple tracts
        tract_patterns = [
            r'\bTRACT\s+[A-Z0-9]+\b',
            r'\bPARCEL\s+[A-Z0-9]+\b',
            r'\bLOT\s+[A-Z0-9]+\b',
            r'\bPORTION\s+[A-Z0-9]+\b',
            r'\bUNIT\s+[A-Z0-9]+\b',
            r'\bBEING\s+TRACT\b',
            r'\bBEING\s+PARCEL\b',
            r'\bFIRST\s+TRACT\b',
            r'\bSECOND\s+TRACT\b',
            r'\bTRACT\s+ONE\b',
            r'\bTRACT\s+TWO\b'
        ]
        
        matches = 0
        for pattern in tract_patterns:
            matches += len(re.findall(pattern, deed_text, re.IGNORECASE))
        
        # If we find 2 or more tract indicators, likely multiple tracts
        return matches >= 2
    
    def _parse_multiple_tracts(self, deed_text: str, settings: ProjectSettings) -> DeedParseResult:
        """Parse deed with multiple tracts"""
        system_prompt = self._create_multi_tract_system_prompt(settings)
        
        # Parse with OpenAI structured outputs for multiple tracts
        raw_tracts_data = self._call_openai_multi_tract_parser(deed_text, system_prompt)
        
        # Process each tract
        tracts = []
        all_calls = []
        
        for tract_data in raw_tracts_data:
            tract = self._process_tract_data(tract_data, settings)
            tracts.append(tract)
            all_calls.extend(tract.calls)
        
        # Create result
        result = DeedParseResult(
            tracts=tracts,
            settings=settings,
            calls=all_calls,  # Legacy support
            geometry=tracts[0].geometry if tracts else None,  # Primary tract geometry
            original_text=deed_text
        )
        
        return result
    
    def _parse_single_tract(self, deed_text: str, settings: ProjectSettings) -> DeedParseResult:
        """Parse deed with single tract (legacy behavior)"""
        system_prompt = self._create_system_prompt(settings)
        
        # Parse with OpenAI structured outputs
        raw_calls = self._call_openai_parser(deed_text, system_prompt)
        
        # Post-process and validate
        processed_calls = self._post_process_calls(raw_calls, settings)
        
        # Create single tract
        tract = Tract(
            tract_id="TRACT 1",
            description="Primary tract",
            pob_x=settings.pob_x,
            pob_y=settings.pob_y,
            pob_description=settings.pob_description,
            calls=processed_calls,
            raw_text=deed_text
        )
        
        # Calculate geometry for the tract
        from ..geometry.calculator import GeometryCalculator
        calculator = GeometryCalculator(settings)
        tract.geometry = calculator.calculate_polygon(processed_calls)
        
        # Create result
        result = DeedParseResult(
            tracts=[tract],
            settings=settings,
            calls=processed_calls,  # Legacy support
            geometry=tract.geometry,  # Legacy support
            original_text=deed_text
        )
        
        return result
    
    def _create_system_prompt(self, settings: ProjectSettings) -> str:
        """Create system prompt with project context"""
        unit_context = f"Default distance unit is {settings.units}"
        bearing_context = f"Bearing convention is {settings.bearing_convention}"
        
        return f"""You are a precise legal deed parser specialized in extracting boundary survey calls.

CONTEXT:
- {unit_context}
- {bearing_context}
- Point of Beginning: {settings.pob_description}

INSTRUCTIONS:
1. Extract ALL boundary calls in sequence order
2. Preserve the exact raw text for each call
3. Parse bearings into both text format and azimuth degrees (0-360)
4. Extract distances with units
5. Identify curves vs. lines based on keywords (curve, arc, radius, chord)
6. For curves, extract: direction (L/R), radius, arc length, chord bearing, chord length
7. Set confidence based on clarity:
   - 0.9-1.0: Clear, unambiguous
   - 0.7-0.9: Minor ambiguity
   - 0.5-0.7: Significant uncertainty
   - <0.5: Very unclear or missing data

CURVE INDICATORS:
- "curve", "arc", "radius", "chord"
- "thence curving", "with a curve"
- "R=", "radius =", "arc length"
- "curve to the left/right"

BEARING FORMATS TO RECOGNIZE:
- Quadrant: N 45°30'15" E, S 22°10' W, N45-30-15E
- Azimuth: 123°45'30", 123.75°

Return only the structured JSON matching the function schema."""
    
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
    
    def _call_openai_multi_tract_parser(self, deed_text: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Call OpenAI with multi-tract function calling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this multi-tract deed description:\n\n{deed_text}"}
                ],
                functions=[PARSE_TRACTS_SCHEMA],
                function_call={"name": "parse_deed_tracts"},
                temperature=0.1
            )
            
            # Extract function call result
            message = response.choices[0].message
            if message.function_call:
                function_args = json.loads(message.function_call.arguments)
                return function_args.get("tracts", [])
            else:
                raise ValueError("No function call in OpenAI response")
                
        except Exception as e:
            raise ValueError(f"OpenAI multi-tract parsing failed: {str(e)}")
    
    def _process_tract_data(self, tract_data: Dict[str, Any], settings: ProjectSettings) -> Tract:
        """Process raw tract data from OpenAI into Tract object"""
        # Extract POB coordinates if provided
        pob_coords = tract_data.get("pob_coordinates", {})
        pob_x = pob_coords.get("x", 0.0) if pob_coords.get("has_coordinates", False) else 0.0
        pob_y = pob_coords.get("y", 0.0) if pob_coords.get("has_coordinates", False) else 0.0
        
        # Process calls for this tract
        raw_calls = tract_data.get("calls", [])
        processed_calls = self._post_process_calls(raw_calls, settings)
        
        # Create tract
        tract = Tract(
            tract_id=tract_data.get("tract_id", "Unknown Tract"),
            description=tract_data.get("description", ""),
            pob_x=pob_x,
            pob_y=pob_y,
            pob_description=tract_data.get("pob_description", "Point of Beginning"),
            calls=processed_calls,
            raw_text=tract_data.get("raw_text", "")
        )
        
        # Calculate geometry for this tract
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
        tract.geometry = calculator.calculate_polygon(processed_calls)
        
        return tract
    
    def _call_openai_parser(self, deed_text: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Call OpenAI with function calling for structured output"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this deed description:\n\n{deed_text}"}
                ],
                functions=[PARSE_CALLS_SCHEMA],
                function_call={"name": "parse_deed_calls"},
                temperature=0.1  # Low temperature for consistent parsing
            )
            
            # Extract function call result
            message = response.choices[0].message
            if message.function_call:
                function_args = json.loads(message.function_call.arguments)
                return function_args.get("calls", [])
            else:
                raise ValueError("No function call in OpenAI response")
                
        except Exception as e:
            raise ValueError(f"OpenAI parsing failed: {str(e)}")
    
    def _post_process_calls(self, raw_calls: List[Dict[str, Any]], settings: ProjectSettings) -> List[SurveyCall]:
        """Post-process and validate OpenAI results"""
        processed_calls = []
        
        for i, call_data in enumerate(raw_calls):
            try:
                # Create base call
                call = SurveyCall(**call_data)
                
                # Post-process with deterministic parsing
                call = self._enhance_call_with_regex(call, settings)
                
                # Validate and normalize
                call = self._validate_and_normalize(call, settings)
                
                processed_calls.append(call)
                
            except ValidationError as e:
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
        calls = parser.parse_deed_text(sample_deed, settings)
        print(f"Parsed {len(calls)} calls:")
        for call in calls:
            print(f"  {call.sequence}: {call.type} - {call.raw_text[:50]}...")
            print(f"    Confidence: {call.confidence}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_parser()
