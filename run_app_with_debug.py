#!/usr/bin/env python3
"""
Enhanced Flask Application Runner with Better Console Output
This script ensures all print statements and logging appear immediately in the console.
"""

import sys
import os

# Force immediate output flushing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Set environment variable to force immediate output
os.environ['PYTHONUNBUFFERED'] = '1'

# Configure logging to show in console immediately
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cashflow_app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# Create a custom print function that forces flush
def debug_print(*args, **kwargs):
    """Custom print function that forces immediate output"""
    print(*args, **kwargs, flush=True)
    sys.stdout.flush()

# Replace print with debug_print for better console output
print = debug_print

if __name__ == '__main__':
    print("🚀 Starting Cash Flow SAP Bank System with Enhanced Console Output...")
    print("📝 All print statements will now appear immediately in the console")
    print("🔍 Debug mode enabled - you'll see detailed processing information")
    
    # Import and run the main application
    from app1 import app
    
    print("✅ Application imported successfully")
    print("🌐 Starting Flask server on http://localhost:5000")
    print("💡 Click buttons in the web interface to see real-time console output")
    print("=" * 60)
    
    # Run the Flask app with enhanced debugging
    app.run(
        debug=True, 
        use_reloader=False, 
        threaded=True,
        host='0.0.0.0',
        port=5000
    ) 