#!/usr/bin/env python3
"""
TP7 - Image Registration Main Module
Entry point for all image registration demonstrations.

Usage:
    python main.py

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os

def main():
    """Main function to run all TP7 image registration demonstrations."""
    print("🔄 Starting TP7 - Image Registration Demonstrations")
    print("=" * 70)
    
    try:
        print("\n1. Running Image Registration...")
        from image_registration import main as registration_main
        registration_main()
        
        print("\n2. Running Automatic Registration Demo...")
        from automatic_registration_demo import main as auto_demo_main
        auto_demo_main()
        
        print("\n3. Running Manual Registration Demo...")
        from manual_registration_demo import main as manual_demo_main
        manual_demo_main()
        
        print("\n4. Running Mixed Registration Demo...")
        from mixed_registration_demo import main as mixed_demo_main
        mixed_demo_main()
        
        print("\n" + "=" * 70)
        print("✅ TP7 - Image Registration completed successfully!")
        print("📁 All visualizations saved to: plots/07_image_registration/")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in TP7: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
