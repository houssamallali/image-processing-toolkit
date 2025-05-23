#!/usr/bin/env python3
"""
TP2 - Spatial Filtering Main Module
Professional implementation of spatial filtering techniques.

This module provides a clean interface to run all spatial filtering operations
including convolution, low-pass, high-pass, and enhancement filters.

Usage:
    python main.py

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os
sys.path.append('..')

from common_utils import clear_cache
from convolution import main as convolution_main
from lowpass import main as lowpass_main

def main():
    """Main function to run all TP2 spatial filtering demonstrations."""
    print("🔄 Starting TP2 - Spatial Filtering Demonstrations")
    print("=" * 70)

    try:
        clear_cache()

        # Run convolution demonstrations
        print("\n1. Running Convolution Operations...")
        convolution_main()

        # Run low-pass filtering demonstrations
        print("\n2. Running Low-Pass Filtering...")
        lowpass_main()

        # TODO: Add other filtering modules as they are refactored
        # highpass_main()
        # enhancement_main()
        # aliasing_main()

        print("\n" + "=" * 70)
        print("✅ TP2 - Spatial Filtering completed successfully!")
        print("📁 All visualizations saved to: plots/02_spatial_filtering/")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error in TP2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
