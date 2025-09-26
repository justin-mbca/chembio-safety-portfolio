#!/usr/bin/env python3
"""
ChemBio SafeGuard - Main Application Entry Point
"""

import sys
from pathlib import Path
import uvicorn

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point for the ChemBio SafeGuard application"""
    uvicorn.run(
        "src.api.simple_api:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
