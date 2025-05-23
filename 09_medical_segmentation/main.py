import os
import sys

# Add the current directory to the path to handle imports from both locations
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from follicle_segmentation import process_follicle_image

def run_follicle_segmentation():
    """
    Run the follicle segmentation practical work
    """
    print("=" * 50)
    print("TP9: Segmentation of Follicles")
    print("=" * 50)
    
    # Try multiple possible paths to handle different run contexts
    possible_paths = [
        os.path.join("..", "images", "follicule.bmp"),  # When run from TP9 directory
        os.path.join("images", "follicule.bmp"),        # When run from project root
    ]
    
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    # Check if the image exists
    if not image_path:
        print(f"Error: Image not found. Please check that the follicule.bmp file exists in the images directory.")
        return
    
    print(f"Using image: {image_path}")
    print("Processing follicle image...")
    print("-" * 50)
    
    # Process the image and get measurements
    measurements = process_follicle_image(image_path)
    
    print("-" * 50)
    print("Segmentation complete!")
    print("Results saved in the 'plots/TP9_Follicle_Segmentation' directory")
    
    return measurements

if __name__ == "__main__":
    run_follicle_segmentation() 