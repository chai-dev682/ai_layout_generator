"""
Cross-platform configuration for OpenCV and Tesseract.
Handles Windows and Linux (Streamlit Cloud) compatibility.
"""
import os
import sys
import platform
import logging
from typing import Dict, Optional
import streamlit as st

logger = logging.getLogger(__name__)


class PlatformConfig:
    """Cross-platform configuration manager"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.is_windows = self.platform == "windows"
        self.is_linux = self.platform == "linux"
        self.is_streamlit_cloud = self._detect_streamlit_cloud()
        
        # Initialize configurations
        self._setup_tesseract()
        self._setup_opencv()
        
        logger.info(f"Platform detected: {self.platform}")
        logger.info(f"Streamlit Cloud: {self.is_streamlit_cloud}")
        logger.info(f"Tesseract path: {self.tesseract_cmd}")
    
    def _detect_streamlit_cloud(self) -> bool:
        """Detect if running on Streamlit Cloud"""
        # Streamlit Cloud sets specific environment variables
        streamlit_indicators = [
            'STREAMLIT_CLOUD',
            'STREAMLIT_SERVER_PORT',
            'STREAMLIT_SHARING_MODE'
        ]
        
        # Check for Streamlit Cloud environment variables
        for indicator in streamlit_indicators:
            if os.getenv(indicator):
                return True
        
        # Check for typical cloud environment patterns
        if os.getenv('HOME') == '/app' or '/app' in sys.path[0]:
            return True
            
        return False
    
    def _setup_tesseract(self):
        """Configure Tesseract OCR path for cross-platform compatibility"""
        # Try to get from Streamlit secrets first
        try:
            if hasattr(st, 'secrets') and 'tesseract' in st.secrets:
                self.tesseract_cmd = st.secrets['tesseract'].get('TESSERACT_CMD')
                tessdata_prefix = st.secrets['tesseract'].get('TESSDATA_PREFIX')
                if tessdata_prefix:
                    os.environ['TESSDATA_PREFIX'] = tessdata_prefix
                if self.tesseract_cmd:
                    logger.info(f"Using Tesseract from secrets: {self.tesseract_cmd}")
                    return
        except Exception as e:
            logger.debug(f"Could not read Tesseract from secrets: {e}")
        
        # Platform-specific default paths
        if self.is_windows:
            # Common Windows Tesseract installation paths
            windows_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', 'User')),
                "tesseract"  # If in PATH
            ]
            
            self.tesseract_cmd = None
            for path in windows_paths:
                if os.path.isfile(path) or self._test_tesseract_path(path):
                    self.tesseract_cmd = path
                    break
            
            if not self.tesseract_cmd:
                self.tesseract_cmd = "tesseract"  # Hope it's in PATH
                logger.warning("Tesseract not found in common Windows paths, using 'tesseract' from PATH")
        
        elif self.is_linux:
            if self.is_streamlit_cloud:
                # Streamlit Cloud / Linux container paths
                linux_paths = [
                    "/usr/bin/tesseract",
                    "/usr/local/bin/tesseract",
                    "tesseract"
                ]
            else:
                # Standard Linux paths
                linux_paths = [
                    "/usr/bin/tesseract",
                    "/usr/local/bin/tesseract",
                    "/opt/tesseract/bin/tesseract",
                    "tesseract"
                ]
            
            self.tesseract_cmd = None
            for path in linux_paths:
                if os.path.isfile(path) or self._test_tesseract_path(path):
                    self.tesseract_cmd = path
                    break
            
            if not self.tesseract_cmd:
                self.tesseract_cmd = "tesseract"
                logger.warning("Tesseract not found in common Linux paths, using 'tesseract' from PATH")
        
        else:
            # Other platforms (macOS, etc.)
            self.tesseract_cmd = "tesseract"
    
    def _test_tesseract_path(self, path: str) -> bool:
        """Test if a Tesseract path is valid"""
        try:
            import subprocess
            result = subprocess.run([path, "--version"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _setup_opencv(self):
        """Configure OpenCV for cross-platform compatibility"""
        # OpenCV configuration for headless environments
        if self.is_streamlit_cloud or os.getenv('DISPLAY') is None:
            # Set OpenCV to use headless mode
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            
            # Disable GUI-related OpenCV features
            try:
                import cv2
                # Ensure we're using headless version
                if hasattr(cv2, 'setUseOptimized'):
                    cv2.setUseOptimized(True)
                logger.info("OpenCV configured for headless operation")
            except ImportError:
                logger.warning("OpenCV not available")
    
    def configure_pytesseract(self):
        """Configure pytesseract with the correct executable path"""
        try:
            import pytesseract
            if self.tesseract_cmd and self.tesseract_cmd != "tesseract":
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                logger.info(f"Pytesseract configured with: {self.tesseract_cmd}")
            
            # Test Tesseract installation
            try:
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract version: {version}")
                return True
            except Exception as e:
                logger.error(f"Tesseract test failed: {e}")
                return False
                
        except ImportError:
            logger.error("pytesseract not available")
            return False
    
    def get_opencv_info(self) -> Dict[str, str]:
        """Get OpenCV configuration information"""
        try:
            import cv2
            return {
                "version": cv2.__version__,
                "build_info": "Available",
                "headless": "opencv-python-headless" in cv2.__file__ if hasattr(cv2, '__file__') else "Unknown",
                "platform": self.platform
            }
        except ImportError:
            return {
                "version": "Not installed",
                "build_info": "Not available",
                "headless": "Unknown",
                "platform": self.platform
            }
    
    def get_tesseract_info(self) -> Dict[str, str]:
        """Get Tesseract configuration information"""
        info = {
            "command": self.tesseract_cmd,
            "platform": self.platform,
            "streamlit_cloud": str(self.is_streamlit_cloud)
        }
        
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            info["version"] = str(version)
            info["available"] = "Yes"
        except Exception as e:
            info["version"] = "Unknown"
            info["available"] = f"No: {str(e)}"
        
        return info
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """Validate all dependencies are working"""
        results = {}
        
        # Test OpenCV
        try:
            import cv2
            # Try a simple operation
            test_array = cv2.imread if hasattr(cv2, 'imread') else None
            results["opencv"] = test_array is not None
        except Exception:
            results["opencv"] = False
        
        # Test Tesseract
        results["tesseract"] = self.configure_pytesseract()
        
        # Test PIL
        try:
            from PIL import Image
            results["pillow"] = True
        except ImportError:
            results["pillow"] = False
        
        # Test NumPy
        try:
            import numpy as np
            results["numpy"] = True
        except ImportError:
            results["numpy"] = False
        
        return results
    
    def get_info(self) -> Dict[str, any]:
        """Get platform information for this instance"""
        return {
            "platform": self.platform,
            "is_windows": self.is_windows,
            "is_linux": self.is_linux,
            "is_streamlit_cloud": self.is_streamlit_cloud,
            "tesseract": self.get_tesseract_info(),
            "opencv": self.get_opencv_info(),
            "dependencies": self.validate_dependencies()
        }


# Global platform configuration instance
platform_config = PlatformConfig()


def get_platform_info() -> Dict[str, any]:
    """Get comprehensive platform information"""
    return {
        "platform": platform_config.platform,
        "is_windows": platform_config.is_windows,
        "is_linux": platform_config.is_linux,
        "is_streamlit_cloud": platform_config.is_streamlit_cloud,
        "tesseract": platform_config.get_tesseract_info(),
        "opencv": platform_config.get_opencv_info(),
        "dependencies": platform_config.validate_dependencies()
    }


def configure_for_deployment():
    """Configure the application for deployment"""
    logger.info("Configuring application for deployment...")
    
    # Configure Tesseract
    tesseract_ok = platform_config.configure_pytesseract()
    
    # Validate dependencies
    deps = platform_config.validate_dependencies()
    
    # Log status
    logger.info(f"Tesseract configured: {tesseract_ok}")
    logger.info(f"Dependencies status: {deps}")
    
    return tesseract_ok and all(deps.values())


if __name__ == "__main__":
    # Test the configuration
    info = get_platform_info()
    print("Platform Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nDeployment ready: {configure_for_deployment()}")
