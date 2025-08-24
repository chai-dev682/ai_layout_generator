"""
Data models for deed parsing system following the canonical JSON schema.
"""
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, computed_field
import math
from enum import Enum


class CallType(str, Enum):
    """Types of survey calls"""
    LINE = "line"
    CURVE = "curve"
    TIE_LINE = "tie_line"
    TIE_CURVE = "tie_curve"


class CurveDirection(str, Enum):
    """Curve direction indicators"""
    LEFT = "L"
    RIGHT = "R"


class DistanceUnit(str, Enum):
    """Supported distance units"""
    FEET = "ft"
    METERS = "m"
    CHAINS = "ch"
    RODS = "rd"


class BearingConvention(str, Enum):
    """Bearing format conventions"""
    QUADRANT = "quadrant"  # N 45°30' E
    AZIMUTH = "azimuth"    # 45.5°


class SurveyCall(BaseModel):
    """Individual survey call (line or curve) following canonical schema"""
    model_config = {"extra": "allow", "validate_assignment": False}
    
    sequence: int = Field(..., description="Order in the boundary description")
    type: CallType = Field(..., description="Type of survey call")
    raw_text: str = Field(..., description="Original text from deed")
    
    # Line fields
    bearing: Optional[str] = Field(None, description="Bearing as text (e.g., 'N 45°30' E')")
    azimuth_deg: Optional[float] = Field(None, description="Azimuth in decimal degrees (0-360)")
    distance: Optional[float] = Field(None, description="Distance value")
    distance_unit: Optional[str] = Field(None, description="Distance unit")
    
    # Curve fields
    curve_direction: Optional[CurveDirection] = Field(None, description="Curve direction L/R")
    radius: Optional[float] = Field(None, description="Curve radius")
    arc_length: Optional[float] = Field(None, description="Arc length")
    chord_bearing: Optional[str] = Field(None, description="Chord bearing as text")
    chord_length: Optional[float] = Field(None, description="Chord length")
    
    # Quality indicators
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Parsing confidence")
    source: str = Field(default="llm", description="Extraction source: llm, regex, manual")
    notes: Optional[str] = Field(None, description="Additional notes or warnings")
    
    @field_validator('azimuth_deg')
    @classmethod
    def validate_azimuth(cls, v):
        if v is not None and not (0 <= v <= 360):
            raise ValueError('Azimuth must be between 0 and 360 degrees')
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))


class ProjectSettings(BaseModel):
    """Project-level settings and conventions"""
    model_config = {"extra": "allow", "validate_assignment": False}
    
    units: str = Field(default="ft", description="Primary distance unit")
    bearing_convention: str = Field(default="quadrant", description="Bearing convention")
    
    # Point of Beginning
    pob_x: float = Field(default=0.0, description="POB X coordinate")
    pob_y: float = Field(default=0.0, description="POB Y coordinate")
    pob_description: str = Field(default="Point of Beginning", description="POB description")
    
    # Coordinate system
    crs: Optional[str] = Field(None, description="Coordinate Reference System (EPSG code)")
    
    # Quality thresholds
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence for auto-accept")
    closure_tolerance: float = Field(default=0.1, description="Acceptable closure error")
    
    # OpenAI settings
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")


class GeometryPoint(BaseModel):
    """2D coordinate point"""
    x: float
    y: float
    description: Optional[str] = None


class PolygonGeometry(BaseModel):
    """Computed polygon geometry"""
    vertices: List[GeometryPoint] = Field(..., description="Polygon vertices in order")
    closure_error: float = Field(..., description="Distance between last point and POB")
    closure_percentage: float = Field(..., description="Closure error as % of perimeter")
    perimeter: float = Field(..., description="Total perimeter length")
    area: Optional[float] = Field(None, description="Polygon area if closed")


class Tract(BaseModel):
    """Individual tract within a deed"""
    model_config = {"extra": "allow", "validate_assignment": False}
    
    tract_id: str = Field(..., description="Tract identifier (e.g., 'TRACT 1', 'PARCEL A')")
    description: str = Field(..., description="Tract description from deed")
    pob_x: float = Field(default=0.0, description="Point of Beginning X coordinate")
    pob_y: float = Field(default=0.0, description="Point of Beginning Y coordinate")
    pob_description: str = Field(default="Point of Beginning", description="POB description from deed")
    calls: List[SurveyCall] = Field(default_factory=list, description="Survey calls for this tract")
    geometry: Optional[PolygonGeometry] = Field(None, description="Computed geometry for this tract")
    raw_text: str = Field(..., description="Raw text for this tract from deed")
    
    @property
    def total_confidence(self) -> float:
        """Calculate average confidence for this tract's calls"""
        if not self.calls:
            return 0.0
        return sum(call.confidence for call in self.calls) / len(self.calls)


class DeedParseResult(BaseModel):
    """Complete result from deed parsing - supports multiple tracts"""
    model_config = {"extra": "allow", "validate_assignment": False}
    
    tracts: List[Tract] = Field(default_factory=list, description="Individual tracts")
    settings: ProjectSettings = Field(..., description="Project settings used")
    
    # Legacy support for single tract (backwards compatibility)
    calls: List[SurveyCall] = Field(default_factory=list, description="All calls combined (legacy)")
    geometry: Optional[PolygonGeometry] = Field(None, description="Combined geometry (legacy)")
    
    # Metadata
    original_text: str = Field(..., description="Original deed text")
    parse_timestamp: Optional[str] = Field(None, description="When parsing was performed")
    @computed_field
    @property
    def total_confidence(self) -> float:
        """Average confidence across all tracts"""
        if not self.tracts:
            return 0.0
        all_calls = []
        for tract in self.tracts:
            all_calls.extend(tract.calls)
        if not all_calls:
            return 0.0
        return sum(call.confidence for call in all_calls) / len(all_calls)
    
    @property
    def has_multiple_tracts(self) -> bool:
        """Check if deed contains multiple tracts"""
        return len(self.tracts) > 1
    
    @property
    def primary_tract(self) -> Optional[Tract]:
        """Get the primary (first) tract"""
        return self.tracts[0] if self.tracts else None


# OpenAI Function Schema for structured outputs - supports multiple tracts
PARSE_TRACTS_SCHEMA = {
    "name": "parse_deed_tracts",
    "description": "Extract multiple tracts with boundary survey calls from legal deed description",
    "parameters": {
        "type": "object",
        "properties": {
            "tracts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tract_id": {"type": "string", "description": "Tract identifier (e.g., 'TRACT 1', 'PARCEL A')"},
                        "description": {"type": "string", "description": "Brief description of the tract"},
                        "pob_description": {"type": "string", "description": "Point of Beginning description"},
                        "pob_coordinates": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number", "description": "X coordinate if specified"},
                                "y": {"type": "number", "description": "Y coordinate if specified"},
                                "has_coordinates": {"type": "boolean", "description": "Whether coordinates were found"}
                            }
                        },
                        "raw_text": {"type": "string", "description": "Raw text for this tract"},
                        "calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sequence": {"type": "integer", "description": "Order in boundary"},
                                    "type": {"type": "string", "enum": ["line", "curve", "tie_line", "tie_curve"]},
                                    "raw_text": {"type": "string", "description": "Original text segment"},
                                    "bearing": {"type": "string", "description": "Bearing text (e.g. 'N 45°30' E')"},
                                    "azimuth_deg": {"type": "number", "description": "Azimuth 0-360 degrees"},
                                    "distance": {"type": "number", "description": "Distance value"},
                                    "distance_unit": {"type": "string", "enum": ["ft", "m", "ch", "rd"]},
                                    "curve_direction": {"type": "string", "enum": ["L", "R"]},
                                    "radius": {"type": "number", "description": "Curve radius"},
                                    "arc_length": {"type": "number", "description": "Arc length"},
                                    "chord_bearing": {"type": "string", "description": "Chord bearing text"},
                                    "chord_length": {"type": "number", "description": "Chord length"},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                                },
                                "required": ["sequence", "type", "raw_text", "confidence"]
                            }
                        }
                    },
                    "required": ["tract_id", "description", "pob_description", "raw_text", "calls"]
                }
            }
        },
        "required": ["tracts"]
    }
}

# Legacy schema for backwards compatibility
PARSE_CALLS_SCHEMA = {
    "name": "parse_deed_calls",
    "description": "Extract boundary survey calls from legal deed description (single tract)",
    "parameters": {
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sequence": {"type": "integer", "description": "Order in boundary"},
                        "type": {"type": "string", "enum": ["line", "curve", "tie_line", "tie_curve"]},
                        "raw_text": {"type": "string", "description": "Original text segment"},
                        "bearing": {"type": "string", "description": "Bearing text (e.g. 'N 45°30' E')"},
                        "azimuth_deg": {"type": "number", "description": "Azimuth 0-360 degrees"},
                        "distance": {"type": "number", "description": "Distance value"},
                        "distance_unit": {"type": "string", "enum": ["ft", "m", "ch", "rd"]},
                        "curve_direction": {"type": "string", "enum": ["L", "R"]},
                        "radius": {"type": "number", "description": "Curve radius"},
                        "arc_length": {"type": "number", "description": "Arc length"},
                        "chord_bearing": {"type": "string", "description": "Chord bearing text"},
                        "chord_length": {"type": "number", "description": "Chord length"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["sequence", "type", "raw_text", "confidence"]
                }
            }
        },
        "required": ["calls"]
    }
}
