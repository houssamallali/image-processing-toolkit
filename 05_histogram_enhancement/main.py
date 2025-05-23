#!/usr/bin/env python3
"""
TP5 - Histogram Enhancement Main Module
Entry point for all histogram enhancement demonstrations.

Usage:
    python main.py

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os

def main():
    """Main function to run all TP5 histogram enhancement demonstrations."""
    print("🔄 Starting TP5 - Histogram Enhancement Demonstrations")
    print("=" * 70)
    
    try:
        print("\n1. Running Gamma Correction...")
        from gamma_correction import main as gamma_main
        gamma_main()
        
        print("\n2. Running Contrast Stretching...")
        from contrast_stretching import main as contrast_main
        contrast_main()
        
        print("\n3. Running Histogram Enhancement...")
        from histogram_enhancement import main as hist_enh_main
        hist_enh_main()
        
        print("\n4. Running Histogram Equalization...")
        from histogram_equalization import main as hist_eq_main
        hist_eq_main()
        
        print("\n5. Running Histogram Matching...")
        from histogram_matching import main as hist_match_main
        hist_match_main()
        
        print("\n6. Running LUT Transformations...")
        from lut_transformations import main as lut_main
        lut_main()
        
        print("\n7. Running Combined Enhancement...")
        from combined_enhancement import main as combined_main
        combined_main()
        
        print("\n8. Running Histogram Comparison...")
        from histogram_comparison import main as comparison_main
        comparison_main()
        
        print("\n9. Running Phobos Matching...")
        from phobos_matching import main as phobos_match_main
        phobos_match_main()
        
        print("\n10. Running Phobos Synthesis...")
        from phobos_synthesis import main as phobos_synth_main
        phobos_synth_main()
        
        print("\n" + "=" * 70)
        print("✅ TP5 - Histogram Enhancement completed successfully!")
        print("📁 All visualizations saved to: plots/05_histogram_enhancement/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in TP5: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
