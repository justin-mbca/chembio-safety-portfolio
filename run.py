#!/usr/bin/env python3
"""
ChemBio SafeGuard - Main Application Entry Point
"""

import sys
import os
from pathlib import Path
import uvicorn

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point for the ChemBio SafeGuard application"""
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 3000))
    
    uvicorn.run(
        "src.api.simple_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )

if __name__ == "__main__":
    main()
