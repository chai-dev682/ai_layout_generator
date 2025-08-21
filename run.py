#!/usr/bin/env python3
"""
Simple runner script for the Deed Parser application.
"""
import subprocess
import sys
import os


def main():
    """Run the Streamlit application"""
    print("🗺️ Starting Deed Parser System...")
    print("=" * 50)
    
    # Check if requirements are installed
    try:
        import streamlit
        import openai
        import shapely
        print("✅ Dependencies verified")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not found in environment")
        print("You can set it in the Streamlit sidebar or create a .env file")
    
    print("🚀 Launching Streamlit application...")
    print("📱 Open your browser to: http://localhost:8501")
    print("=" * 50)
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
