#!/usr/bin/env python3
"""
Platform compatibility test script.
Run this to verify that all dependencies are working correctly.
"""
import sys
import os
import platform

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    results = {}
    
    # Test basic Python packages
    try:
        import numpy as np
        results['numpy'] = f"✅ {np.__version__}"
    except ImportError as e:
        results['numpy'] = f"❌ {e}"
    
    try:
        from PIL import Image
        results['pillow'] = f"✅ Available"
    except ImportError as e:
        results['pillow'] = f"❌ {e}"
    
    # Test OpenCV
    try:
        import cv2
        results['opencv'] = f"✅ {cv2.__version__}"
        
        # Test a simple operation
        test_array = cv2.imread
        if test_array is None:
            results['opencv'] += " (Warning: imread not available)"
    except ImportError as e:
        results['opencv'] = f"❌ {e}"
    
    # Test Tesseract
    try:
        import pytesseract
        results['pytesseract'] = f"✅ Available"
    except ImportError as e:
        results['pytesseract'] = f"❌ {e}"
    
    # Test PDF libraries
    try:
        import PyPDF2
        results['pypdf2'] = f"✅ {PyPDF2.__version__}"
    except ImportError as e:
        results['pypdf2'] = f"❌ {e}"
    
    try:
        import pdfplumber
        results['pdfplumber'] = f"✅ {pdfplumber.__version__}"
    except ImportError as e:
        results['pdfplumber'] = f"❌ {e}"
    
    try:
        import fitz
        results['pymupdf'] = f"✅ {fitz.__version__}"
    except ImportError as e:
        results['pymupdf'] = f"❌ {e}"
    
    # Test Streamlit
    try:
        import streamlit
        results['streamlit'] = f"✅ {streamlit.__version__}"
    except ImportError as e:
        results['streamlit'] = f"❌ {e}"
    
    return results

def test_platform_config():
    """Test platform-specific configuration"""
    print("\nTesting platform configuration...")
    
    try:
        from src.utils.platform_config import platform_config, get_platform_info, configure_for_deployment
        
        # Get platform info
        info = get_platform_info()
        
        print(f"Platform: {info['platform']}")
        print(f"Windows: {info['is_windows']}")
        print(f"Linux: {info['is_linux']}")
        print(f"Streamlit Cloud: {info['is_streamlit_cloud']}")
        
        # Test Tesseract configuration
        tesseract_info = info['tesseract']
        print(f"\nTesseract:")
        print(f"  Command: {tesseract_info['command']}")
        print(f"  Version: {tesseract_info['version']}")
        print(f"  Available: {tesseract_info['available']}")
        
        # Test OpenCV configuration
        opencv_info = info['opencv']
        print(f"\nOpenCV:")
        print(f"  Version: {opencv_info['version']}")
        print(f"  Headless: {opencv_info['headless']}")
        
        # Test dependencies
        deps = info['dependencies']
        print(f"\nDependency Status:")
        for dep, status in deps.items():
            status_icon = "✅" if status else "❌"
            print(f"  {dep}: {status_icon}")
        
        # Test deployment configuration
        deployment_ready = configure_for_deployment()
        print(f"\nDeployment Ready: {'✅' if deployment_ready else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Platform configuration test failed: {e}")
        return False

def test_tesseract_ocr():
    """Test Tesseract OCR functionality"""
    print("\nTesting Tesseract OCR...")
    
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image with text
        img_array = np.ones((100, 300, 3), dtype=np.uint8) * 255
        test_img = Image.fromarray(img_array)
        
        # Try to run OCR (this will fail on the blank image but tests the setup)
        try:
            text = pytesseract.image_to_string(test_img)
            print("✅ Tesseract OCR is working")
            return True
        except Exception as e:
            print(f"⚠️  Tesseract OCR test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Tesseract OCR import failed: {e}")
        return False

def test_opencv_functionality():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV functionality...")
    
    try:
        import cv2
        import numpy as np
        
        # Test basic OpenCV operations
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test color conversion
        gray = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
        
        # Test morphological operations
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        print("✅ OpenCV basic operations working")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV functionality test failed: {e}")
        return False

def main():
    """Run all platform tests"""
    print("🔧 Platform Compatibility Test")
    print("=" * 50)
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Test imports
    import_results = test_imports()
    print("\nImport Results:")
    for package, result in import_results.items():
        print(f"  {package}: {result}")
    
    # Test platform configuration
    platform_ok = test_platform_config()
    
    # Test Tesseract
    tesseract_ok = test_tesseract_ocr()
    
    # Test OpenCV
    opencv_ok = test_opencv_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    
    all_imports_ok = all("✅" in result for result in import_results.values())
    print(f"Imports: {'✅' if all_imports_ok else '❌'}")
    print(f"Platform Config: {'✅' if platform_ok else '❌'}")
    print(f"Tesseract OCR: {'✅' if tesseract_ok else '❌'}")
    print(f"OpenCV: {'✅' if opencv_ok else '❌'}")
    
    overall_status = all([all_imports_ok, platform_ok, tesseract_ok, opencv_ok])
    print(f"\nOverall Status: {'✅ READY FOR DEPLOYMENT' if overall_status else '❌ NEEDS ATTENTION'}")
    
    if not overall_status:
        print("\n💡 Tips:")
        print("- Check DEPLOYMENT_GUIDE.md for setup instructions")
        print("- Ensure all system dependencies are installed")
        print("- Verify Tesseract is in PATH (Windows) or installed via package manager (Linux)")

if __name__ == "__main__":
    main()
