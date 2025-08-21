"""
Configuration settings for the Deed Parser application.
"""
import os
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration settings"""
    
    # Application metadata
    APP_NAME: str = "Deed Parser System"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-powered legal deed parsing and boundary visualization"
    
    # Default settings
    DEFAULT_UNITS: str = "ft"
    DEFAULT_BEARING_CONVENTION: str = "quadrant"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
    DEFAULT_CLOSURE_TOLERANCE: float = 0.1
    
    # OpenAI settings
    SUPPORTED_MODELS: List[str] = None
    DEFAULT_MODEL: str = "gpt-4o"
    DEFAULT_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 4000
    
    # Geometry settings
    DEFAULT_CURVE_SEGMENTS: int = 16
    MAX_CURVE_SEGMENTS: int = 64
    MIN_CURVE_SEGMENTS: int = 4
    
    # Visualization settings
    DEFAULT_SVG_WIDTH: int = 800
    DEFAULT_SVG_HEIGHT: int = 600
    DEFAULT_SVG_MARGIN: int = 50
    
    # File processing
    MAX_FILE_SIZE_MB: int = 10
    SUPPORTED_FILE_TYPES: List[str] = None
    
    # Quality thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    LOW_CONFIDENCE_THRESHOLD: float = 0.5
    CLOSURE_ERROR_WARNING: float = 1.0  # feet
    CLOSURE_ERROR_CRITICAL: float = 5.0  # feet
    
    def __post_init__(self):
        """Initialize default values that depend on other settings"""
        if self.SUPPORTED_MODELS is None:
            self.SUPPORTED_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        
        if self.SUPPORTED_FILE_TYPES is None:
            self.SUPPORTED_FILE_TYPES = ["pdf", "txt", "doc", "docx"]


# Global configuration instance
config = AppConfig()


# Environment-specific overrides
def load_config_from_env():
    """Load configuration overrides from environment variables"""
    
    # OpenAI settings
    if os.getenv("OPENAI_MODEL"):
        config.DEFAULT_MODEL = os.getenv("OPENAI_MODEL")
    
    if os.getenv("OPENAI_TEMPERATURE"):
        config.DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE"))
    
    # Quality thresholds
    if os.getenv("CONFIDENCE_THRESHOLD"):
        config.DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD"))
    
    if os.getenv("CLOSURE_TOLERANCE"):
        config.DEFAULT_CLOSURE_TOLERANCE = float(os.getenv("CLOSURE_TOLERANCE"))
    
    # File processing
    if os.getenv("MAX_FILE_SIZE_MB"):
        config.MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB"))


# Load environment overrides on import
load_config_from_env()


# Utility functions
def get_quality_color(confidence: float) -> str:
    """Get color code for confidence level"""
    if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
        return "#28a745"  # Green
    elif confidence >= config.LOW_CONFIDENCE_THRESHOLD:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def get_closure_status(closure_error: float) -> tuple[str, str]:
    """Get closure status and color"""
    if closure_error <= config.DEFAULT_CLOSURE_TOLERANCE:
        return "Excellent", "#28a745"
    elif closure_error <= config.CLOSURE_ERROR_WARNING:
        return "Good", "#ffc107"
    elif closure_error <= config.CLOSURE_ERROR_CRITICAL:
        return "Warning", "#fd7e14"
    else:
        return "Critical", "#dc3545"


def format_area(area_sqft: float) -> str:
    """Format area in both square feet and acres"""
    acres = area_sqft / 43560
    if acres >= 1.0:
        return f"{acres:.2f} acres ({area_sqft:,.0f} sq ft)"
    else:
        return f"{area_sqft:,.0f} sq ft ({acres:.3f} acres)"


def validate_model_name(model: str) -> bool:
    """Validate if model name is supported"""
    return model in config.SUPPORTED_MODELS


def get_default_project_settings() -> Dict:
    """Get default project settings as dictionary"""
    return {
        "units": config.DEFAULT_UNITS,
        "bearing_convention": config.DEFAULT_BEARING_CONVENTION,
        "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
        "closure_tolerance": config.DEFAULT_CLOSURE_TOLERANCE,
        "openai_model": config.DEFAULT_MODEL,
        "pob_x": 0.0,
        "pob_y": 0.0,
        "pob_description": "Point of Beginning"
    }
