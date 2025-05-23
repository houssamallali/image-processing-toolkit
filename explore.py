#!/usr/bin/env python3
"""
Quick launcher for the Image Processing CLI Explorer.
Simply run: python explore.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Image Processing CLI."""
    cli_path = Path(__file__).parent / "image_processing_cli.py"
    
    if not cli_path.exists():
        print("❌ CLI file not found. Please ensure image_processing_cli.py exists.")
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, str(cli_path)], check=True)
    except KeyboardInterrupt:
        print("\n✨ Thanks for exploring! ✨")
    except Exception as e:
        print(f"❌ Error launching CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
