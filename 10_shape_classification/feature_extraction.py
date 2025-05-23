#!/usr/bin/env python3
"""
TP 11 - Feature Extraction Module
Implementation of feature extraction from binary images as shown in the assignment.

This module implements the exact feature extraction process described in the TP11 images:
- Extract 9 geometrical features from each binary image
- Organize features in an array of nbFeatures lines and 216 columns
- Each column represents the different features of image i

Author: Generated for TP11 Assignment
"""

import glob
from skimage import measure, io
import numpy as np
import matplotlib.pyplot as plt
import os

def load_kimia_database(rep=None):
    """
    Load the Kimia database following the exact structure from the assignment.

    Args:
        rep (str): Repository path containing the Kimia images

    Returns:
        tuple: (classes, nbClasses, nbImages)
    """
    # Try different possible paths for the Kimia database (prioritize Kimia216)
    possible_paths = [
        "images/images_Kimia216/",
        "../images/images_Kimia216/",
        "../../images/images_Kimia216/",
        "../TP10_Kimia_Classification/images_Kimia216/",
        "TP10_Kimia_Classification/images_Kimia216/",
        "images/images_Kimia/",
        "../images/images_Kimia/",
        "../../images/images_Kimia/"
    ]

    if rep is None:
        # Find the correct path
        for path in possible_paths:
            if os.path.exists(path):
                rep = path
                print(f"Found Kimia database at: {rep}")
                break
        else:
            print("Warning: Kimia database not found in any expected location")
            rep = "images/images_Kimia/"  # Default fallback

    # Check what classes are actually available in the database
    if os.path.exists(rep):
        # Get actual class names from the directory
        import glob
        all_files = glob.glob(os.path.join(rep, "*.bmp"))
        available_classes = set()
        for file in all_files:
            basename = os.path.basename(file)
            if '-' in basename:
                class_name = basename.split('-')[0]
                available_classes.add(class_name)

        classes = sorted(list(available_classes))
        print(f"Found {len(classes)} classes in database: {classes}")
    else:
        # Fallback to expected classes
        classes = ['bird', 'bone', 'brick', 'camel', 'car', 'children',
                   'classic', 'elephant', 'face', 'fork', 'fountain',
                   'glass', 'hammer', 'heart', 'key', 'misk', 'ray', 'turtle']

    nbClasses = len(classes)
    nbImages = 12

    return classes, nbClasses, nbImages, rep

def extract_features_from_database(rep=None):
    """
    Extract features from the entire Kimia database.

    This function implements the exact feature extraction process shown in the assignment:
    1. For each image in the database, extract nbFeatures different features
    2. Organize these features in an array of nbFeatures lines and 216 columns
    3. Each column i represents the different features of image i

    Args:
        rep (str): Repository path containing the Kimia images

    Returns:
        tuple: (properties, target, classes)
    """
    # Load database structure and find correct path
    classes, nbClasses, nbImages, rep = load_kimia_database(rep)

    # The features are manually computed
    total_images = nbClasses * nbImages  # 18 * 12 = 216
    nbFeatures = 9  # 9 geometrical features

    properties = np.zeros((total_images, nbFeatures))
    target = np.zeros(total_images)

    print(f"Extracting features from {total_images} images...")
    print(f"Classes: {classes}")

    index = 0

    # Process each class
    for class_idx, class_name in enumerate(classes):
        print(f"Processing class {class_idx + 1}/{nbClasses}: {class_name}")

        # Get all files for this class using glob
        pattern = rep + class_name + "-*.bmp"
        filelist = glob.glob(pattern)

        if len(filelist) == 0:
            print(f"Warning: No files found for pattern {pattern}")
            continue

        # Sort filelist to ensure consistent ordering
        filelist.sort()

        # Process each image in the class (limit to nbImages)
        for i, filename in enumerate(filelist[:nbImages]):
            try:
                print(f"  Processing: {filename}")

                # Read the image
                I = io.imread(filename)

                # Convert to binary image if needed
                if len(I.shape) == 3:
                    # Convert RGB to grayscale then to binary
                    I = I[:, :, 0] > 128
                else:
                    # Already grayscale, convert to binary
                    I = I > 128

                # Extract region properties using skimage.measure.regionprops
                props = measure.regionprops(I.astype(int))

                if len(props) > 0:
                    # Use the largest region (main object)
                    largest_region = max(props, key=lambda x: x.area)

                    # Extract the 9 geometrical features as specified in the assignment:
                    # area, convex area, eccentricity, equivalent diameter, extent,
                    # major axis length, minor axis length, perimeter, solidity

                    properties[index, 0] = largest_region.area
                    properties[index, 1] = largest_region.convex_area
                    properties[index, 2] = largest_region.eccentricity
                    properties[index, 3] = largest_region.equivalent_diameter
                    properties[index, 4] = largest_region.extent
                    properties[index, 5] = largest_region.major_axis_length
                    properties[index, 6] = largest_region.minor_axis_length
                    properties[index, 7] = largest_region.perimeter
                    properties[index, 8] = largest_region.solidity

                    # Set target class
                    target[index] = class_idx

                    index += 1

                else:
                    print(f"    Warning: No regions found in {filename}")

            except Exception as e:
                print(f"    Error processing {filename}: {e}")
                continue

    # Trim arrays to actual number of processed images
    properties = properties[:index]
    target = target[:index]

    print(f"\nFeature extraction completed!")
    print(f"Total images processed: {index}")
    print(f"Feature matrix shape: {properties.shape}")
    print(f"Target array shape: {target.shape}")

    return properties, target, classes

def visualize_sample_images(rep=None, num_samples=4):
    """
    Visualize sample images from different classes as shown in Figure 38.1.

    Args:
        rep (str): Repository path containing the Kimia images
        num_samples (int): Number of sample classes to display
    """
    classes, _, _, rep = load_kimia_database(rep)

    # Select available classes for display (up to 4)
    sample_classes = classes[:min(4, len(classes))]
    print(f"Displaying sample classes: {sample_classes}")

    fig, axes = plt.subplots(1, len(sample_classes), figsize=(12, 3))
    fig.suptitle('Figure 38.1: Sample images from the Kimia database', fontsize=14)

    for i, class_name in enumerate(sample_classes):
        # Find the first image of this class
        pattern = rep + class_name + "-*.bmp"
        filelist = glob.glob(pattern)

        if len(filelist) > 0:
            # Load and display the first image
            filename = sorted(filelist)[0]  # Take the first image
            image = io.imread(filename)

            # Convert to binary for display
            if len(image.shape) == 3:
                binary_image = image[:, :, 0] < 128  # Invert for better visualization
            else:
                binary_image = image < 128

            axes[i].imshow(binary_image, cmap='gray')
            axes[i].set_title(f'({chr(97+i)}) {class_name}')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'No {class_name}\nimage found',
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'({chr(97+i)}) {class_name}')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('tp11/sample_kimia_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_statistics(properties, target, classes):
    """
    Analyze and display statistics of the extracted features.

    Args:
        properties (np.array): Feature matrix
        target (np.array): Target classes
        classes (list): Class names
    """
    feature_names = ['area', 'convex_area', 'eccentricity',
                    'equivalent_diameter', 'extent', 'major_axis_length',
                    'minor_axis_length', 'perimeter', 'solidity']

    print("\nFeature Statistics:")
    print("=" * 50)

    for i, feature_name in enumerate(feature_names):
        feature_values = properties[:, i]
        print(f"{feature_name:20s}: mean={np.mean(feature_values):.4f}, "
              f"std={np.std(feature_values):.4f}, "
              f"min={np.min(feature_values):.4f}, "
              f"max={np.max(feature_values):.4f}")

    # Plot feature distributions by class
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    for i, feature_name in enumerate(feature_names):
        for class_idx, class_name in enumerate(classes):
            class_mask = target == class_idx
            if np.sum(class_mask) > 0:
                class_features = properties[class_mask, i]
                axes[i].hist(class_features, alpha=0.6, label=class_name, bins=10)

        axes[i].set_title(f'Distribution of {feature_name}')
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('Frequency')
        if i == 0:  # Only show legend for first subplot to avoid clutter
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('tp11/feature_distributions_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_features_to_file(properties, target, classes, filename="tp11/extracted_features.npz"):
    """
    Save extracted features to a file for later use.

    Args:
        properties (np.array): Feature matrix
        target (np.array): Target classes
        classes (list): Class names
        filename (str): Output filename
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save features
    np.savez(filename,
             properties=properties,
             target=target,
             classes=classes)

    print(f"Features saved to {filename}")

def load_features_from_file(filename="tp11/extracted_features.npz"):
    """
    Load previously extracted features from a file.

    Args:
        filename (str): Input filename

    Returns:
        tuple: (properties, target, classes)
    """
    data = np.load(filename, allow_pickle=True)
    return data['properties'], data['target'], data['classes']

def main():
    """
    Main function to demonstrate feature extraction.
    """
    print("TP 11 - Feature Extraction from Kimia Database")
    print("=" * 50)

    # Visualize sample images
    print("1. Visualizing sample images...")
    visualize_sample_images()

    # Extract features
    print("\n2. Extracting features from all images...")
    properties, target, classes = extract_features_from_database()

    # Analyze features
    print("\n3. Analyzing feature statistics...")
    analyze_feature_statistics(properties, target, classes)

    # Save features
    print("\n4. Saving features to file...")
    save_features_to_file(properties, target, classes)

    print("\nFeature extraction completed successfully!")
    return properties, target, classes

if __name__ == "__main__":
    main()
