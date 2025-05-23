#!/usr/bin/env python3
"""
TP8 - Noise and Filtering Main Module
Entry point for all noise and filtering demonstrations.

Usage:
    python main.py

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os

def main():
    """Main function to run all TP8 noise and filtering demonstrations."""
    print("🔄 Starting TP8 - Noise and Filtering Demonstrations")
    print("=" * 70)
    
    try:
        print("\n1. Running Noise Generation...")
        from noise_generation import main as noise_gen_main
        noise_gen_main()
        
        print("\n2. Running Noise Analysis...")
        from noise_analysis import main as noise_analysis_main
        noise_analysis_main()
        
        print("\n3. Running Spatial Filtering...")
        from spatial_filtering import main as spatial_main
        spatial_main()
        
        print("\n" + "=" * 70)
        print("✅ TP8 - Noise and Filtering completed successfully!")
        print("📁 All visualizations saved to: plots/08_noise_and_filtering/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in TP8: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
