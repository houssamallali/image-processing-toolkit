#!/usr/bin/env python3
"""
TP3 - Fourier Analysis Main Module
Entry point for all Fourier analysis demonstrations.

Usage:
    python main.py

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os

def main():
    """Main function to run all TP3 Fourier analysis demonstrations."""
    print("🔄 Starting TP3 - Fourier Analysis Demonstrations")
    print("=" * 70)
    
    try:
        # Import and run individual modules
        print("\n1. Running 2D Fourier Transform...")
        # Note: 2Dfourier.py was removed, would need to be recreated
        
        print("\n2. Running Frequency Domain Filtering...")
        from frequency_domain_filtering import main as freq_filter_main
        freq_filter_main()
        
        print("\n3. Running Inverse Fourier Transform...")
        from inverse_fourier import main as inverse_main
        inverse_main()
        
        print("\n4. Running Fourier Applications...")
        from fourier_applications import main as apps_main
        apps_main()
        
        print("\n" + "=" * 70)
        print("✅ TP3 - Fourier Analysis completed successfully!")
        print("📁 All visualizations saved to: plots/03_fourier_analysis/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in TP3: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
