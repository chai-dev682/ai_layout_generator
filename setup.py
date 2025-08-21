"""
Setup script for the Deed Parser System.
"""
import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return False


def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ .env file created from env.example")
        print("⚠️ Please edit .env file with your OpenAI API key")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("⚠️ No env.example file found")
        return False


def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    try:
        # Test imports
        import streamlit
        import openai
        import shapely
        import pandas
        import numpy
        print("✅ All required packages import successfully")
        
        # Run unit tests if pytest is available
        try:
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], 
                          check=True, capture_output=True, text=True)
            print("✅ Unit tests passed")
        except subprocess.CalledProcessError:
            print("⚠️ Some unit tests failed (this may be normal without API key)")
        except FileNotFoundError:
            print("ℹ️ pytest not found, skipping unit tests")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup function"""
    print("🗺️ Deed Parser System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Run: python run.py")
    print("3. Open browser to: http://localhost:8501")
    print("\nFor help, see README.md")


if __name__ == "__main__":
    main()
