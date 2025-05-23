#!/usr/bin/env python3
"""
TP4 - Image Segmentation Main Module
Entry point for all image segmentation demonstrations.

Usage:
    python main.py

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os

def main():
    """Main function to run all TP4 image segmentation demonstrations."""
    print("🔄 Starting TP4 - Image Segmentation Demonstrations")
    print("=" * 70)
    
    try:
        print("\n1. Running Basic Thresholding...")
        from thresholding import main as threshold_main
        threshold_main()
        
        print("\n2. Running K-Means Segmentation...")
        from k_means_segmentation import main as kmeans_main
        kmeans_main()
        
        print("\n3. Running Otsu Thresholding...")
        from otsu_thresholding import main as otsu_main
        otsu_main()
        
        print("\n4. Running Advanced K-Means...")
        from advanced_k_means import main as advanced_main
        advanced_main()
        
        print("\n" + "=" * 70)
        print("✅ TP4 - Image Segmentation completed successfully!")
        print("📁 All visualizations saved to: plots/04_image_segmentation/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in TP4: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
