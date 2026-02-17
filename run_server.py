#!/usr/bin/env python3
"""
DeepAMR Server Runner

Start the DeepAMR prediction server with frontend.

Usage:
    python run_server.py
    python run_server.py --port 8080
    python run_server.py --reload  # Development mode
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import sleep


def main():
    parser = argparse.ArgumentParser(description="Run DeepAMR Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")

    args = parser.parse_args()

    # Check if uvicorn is installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ██████╗ ███████╗███████╗██████╗  █████╗ ███╗   ███╗██████╗ ║
    ║   ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔══██╗║
    ║   ██║  ██║█████╗  █████╗  ██████╔╝███████║██╔████╔██║██████╔╝║
    ║   ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══██║██║╚██╔╝██║██╔══██╗║
    ║   ██████╔╝███████╗███████╗██║     ██║  ██║██║ ╚═╝ ██║██║  ██║║
    ║   ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝║
    ║                                                              ║
    ║   Antimicrobial Resistance Prediction System                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print(f"  Starting server at http://{args.host}:{args.port}")
    print(f"  API docs available at http://localhost:{args.port}/docs")
    print()
    print("  Press Ctrl+C to stop the server")
    print()

    # Open browser after a short delay
    if not args.no_browser:
        def open_browser():
            sleep(2)
            webbrowser.open(f"http://localhost:{args.port}")

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    # Run uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
