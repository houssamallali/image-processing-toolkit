#!/usr/bin/env python3
"""
TP6 - Image Restoration Main Module
Entry point for all image restoration demonstrations.

Usage:
    python main.py

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os

def main():
    """Main function to run all TP6 image restoration demonstrations."""
    print("🔄 Starting TP6 - Image Restoration Demonstrations")
    print("=" * 70)
    
    try:
        print("\n1. Running Image Restoration...")
        from image_restoration import main as restoration_main
        restoration_main()
        
        print("\n2. Running Motion PSF Generator...")
        from motion_psf_generator import main as psf_main
        psf_main()
        
        print("\n3. Running Astronomy Sample Generator...")
        from astronomy_sample_generator import main as astro_gen_main
        astro_gen_main()
        
        print("\n4. Running Astronomy Restoration...")
        from astronomy_restoration import main as astro_rest_main
        astro_rest_main()
        
        print("\n5. Running Iterative Restoration...")
        from iterative_restoration import main as iter_main
        iter_main()
        
        print("\n6. Running Iterative Restoration Demo...")
        from iterative_restoration_demo import main as iter_demo_main
        iter_demo_main()
        
        print("\n" + "=" * 70)
        print("✅ TP6 - Image Restoration completed successfully!")
        print("📁 All visualizations saved to: plots/06_image_restoration/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in TP6: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
