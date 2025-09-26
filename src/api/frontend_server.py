#!/usr/bin/env python3
"""
Static file server for ChemBio SafeGuard Frontend
Serves the HTML interface for the Chemical & Biological Safety System
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve files from frontend directory"""
    
    def __init__(self, *handler_args, **handler_kwargs):
        super().__init__(*handler_args, directory=str(Path(__file__).parent / "frontend"), **handler_kwargs)
    
    def end_headers(self):
        # Add CORS headers to allow API communication
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def start_frontend_server(port=3000):
    """Start the frontend development server"""
    
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return False
    
    try:
        with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            print("\n" + "="*60)
            print("🚀 ChemBio SafeGuard Frontend Server Starting")
            print("="*60)
            print(f"📁 Serving from: {frontend_dir}")
            print(f"🌐 Local URL: http://localhost:{port}")
            print(f"🔗 Network URL: http://127.0.0.1:{port}")
            print("="*60)
            print("📋 Instructions:")
            print("1. Keep this server running")
            print("2. Start the API server: python simple_api.py")
            print("3. Open the web interface in your browser")
            print("4. Press Ctrl+C to stop the server")
            print("="*60)
            
            # Open browser automatically
            try:
                webbrowser.open(f'http://localhost:{port}')
                print("🌐 Opening browser...")
            except (OSError, webbrowser.Error):
                print(f"💡 Manually open: http://localhost:{port}")
            
            print(f"\n✅ Server running on port {port}")
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 48:  # Port already in use
            print(f"❌ Port {port} is already in use. Try a different port.")
            print("   Example: python frontend_server.py --port 3001")
        else:
            print(f"❌ Error starting server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped.")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ChemBio SafeGuard Frontend Server")
    parser.add_argument("--port", "-p", type=int, default=3000, 
                       help="Port to run the frontend server on (default: 3000)")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = start_frontend_server(args.port)
    sys.exit(0 if success else 1)
