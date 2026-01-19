"""
Run script for the Flask application
Run this file to start the web server
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the Flask app
from app.app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

