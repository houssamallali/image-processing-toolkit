#!/usr/bin/env python3
"""
TP 13 - Multiscale Analysis Main Module
Complete implementation of multiscale analysis techniques.

This module provides a comprehensive analysis pipeline that includes:
- Pyramidal decomposition and reconstruction
- Morphological multiscale decomposition
- Error analysis and quality metrics
- Professional visualization

Author: TP13 Implementation
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.morphology import disk
import os
import sys

# Import our modules
from config import *
from pyramidal_decomposition import (
    LaplacianPyramidDecomposition,
    LaplacianPyramidDecomposition_function,
    LaplacianPyramidReconstruction
)
from morphological_multiscale import MorphologicalDecomposition
from visualization import MultiscaleVisualizer

def load_cerveau_image():
    """
    Load the cerveau image for analysis.

    Returns:
        np.ndarray: Loaded and preprocessed cerveau image
    """
    image_path = get_image_path()

    if image_path is None:
        print("Creating synthetic cerveau-like image for demonstration...")
        # Create a synthetic cerveau-like image
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        r = np.sqrt(x**2 + y**2)

        # Create cerveau-like structure
        cerveau = np.zeros_like(r)
        cerveau[r < 0.8] = 0.7 + 0.3 * np.sin(5 * np.pi * r[r < 0.8])
        cerveau[r < 0.6] = 0.9 + 0.1 * np.cos(8 * np.pi * r[r < 0.6])
        cerveau[r < 0.3] = 0.5 + 0.2 * np.sin(10 * np.pi * r[r < 0.3])

        # Add some noise
        cerveau += 0.05 * np.random.randn(*cerveau.shape)
        cerveau = np.clip(cerveau, 0, 1)

        return cerveau.astype(IMAGE_DTYPE)

    try:
        # Load the image
        image = io.imread(image_path)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Convert to float and normalize
        image = image.astype(IMAGE_DTYPE)
        if NORMALIZE_INPUT:
            image = image / np.max(image) if np.max(image) > 0 else image

        if VERBOSE:
            print(f"Loaded cerveau image from: {image_path}")
            print(f"Image shape: {image.shape}")
            print(f"Image range: [{np.min(image):.3f}, {np.max(image):.3f}]")

        return image

    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        print("Creating synthetic image instead...")
        return load_cerveau_image()  # Recursive call to create synthetic

def run_pyramidal_analysis(image, levels=DEFAULT_PYRAMID_LEVELS):
    """
    Run complete pyramidal decomposition and reconstruction analysis.

    Args:
        image (np.ndarray): Input image
        levels (int): Number of pyramid levels

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("PYRAMIDAL DECOMPOSITION AND RECONSTRUCTION ANALYSIS")
    print("="*60)

    # Initialize decomposer
    decomposer = LaplacianPyramidDecomposition(levels=levels)

    # Perform decomposition
    print(f"\n1. Performing {levels}-level pyramidal decomposition...")
    gaussian_pyramid, laplacian_pyramid = decomposer.decompose(image)

    # Perform reconstruction with details
    print("\n2. Reconstructing image with details...")
    reconstructed_with_details = decomposer.reconstruct()

    # Perform reconstruction without details
    print("\n3. Reconstructing image without details...")
    reconstructed_smooth = decomposer.reconstruct_without_details()

    # Compute error metrics
    print("\n4. Computing reconstruction error metrics...")
    error_metrics = decomposer.compute_reconstruction_error(image, reconstructed_with_details)
    error_metrics_smooth = decomposer.compute_reconstruction_error(image, reconstructed_smooth)

    # Print results
    print("\nReconstruction Quality Metrics:")
    print("-" * 40)
    print("With Details:")
    for metric, value in error_metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")

    print("\nWithout Details (Smooth):")
    for metric, value in error_metrics_smooth.items():
        print(f"  {metric.upper()}: {value:.6f}")

    # Test specification functions
    print("\n5. Testing specification functions...")
    pyrL_spec, pyrG_spec = LaplacianPyramidDecomposition_function(image, levels)
    reconstructed_spec = LaplacianPyramidReconstruction(pyrL_spec)

    error_spec = decomposer.compute_reconstruction_error(image, reconstructed_spec)
    print("\nSpecification Function Results:")
    for metric, value in error_spec.items():
        print(f"  {metric.upper()}: {value:.6f}")

    return {
        'original': image,
        'gaussian_pyramid': gaussian_pyramid,
        'laplacian_pyramid': laplacian_pyramid,
        'reconstructed_with_details': reconstructed_with_details,
        'reconstructed_smooth': reconstructed_smooth,
        'reconstructed_spec': reconstructed_spec,
        'error_metrics': error_metrics,
        'error_metrics_smooth': error_metrics_smooth,
        'error_metrics_spec': error_spec,
        'pyrL_spec': pyrL_spec,
        'pyrG_spec': pyrG_spec
    }

def run_morphological_analysis(image, levels=DEFAULT_MORPHOLOGICAL_LEVELS):
    """
    Run complete morphological multiscale analysis.

    Args:
        image (np.ndarray): Input image
        levels (int): Number of decomposition levels

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("MORPHOLOGICAL MULTISCALE ANALYSIS")
    print("="*60)

    # Initialize decomposer with disk structuring element
    se = disk(DEFAULT_SE_RADIUS)
    decomposer = MorphologicalDecomposition(se)

    # Perform scale-space decomposition
    print(f"\n1. Performing {levels}-level morphological decomposition...")
    scale_space = decomposer.decompose(image, levels)

    # Perform opening and closing decomposition
    print("\n2. Performing opening and closing decomposition...")
    opening_space, closing_space = decomposer.decompose_with_closing(image, levels)

    # Apply Kramer-Bruckner filter
    print("\n3. Applying Kramer-Bruckner filter...")
    kb_sequence = decomposer.kramer_bruckner_filter(image, KB_ITERATIONS)

    # Compute residues
    print("\n4. Computing residues between levels...")
    residues = decomposer.compute_residues(image, scale_space)

    # Analyze scale space properties
    print("\n5. Analyzing scale space properties...")
    analysis = decomposer.analyze_scale_space(image, levels)

    print("\nScale Space Analysis:")
    print("-" * 40)
    for level in range(min(5, len(analysis['energy']))):
        print(f"Level {level}:")
        print(f"  Energy: {analysis['energy'][level]:.2f}")
        print(f"  Entropy: {analysis['entropy'][level]:.2f}")
        print(f"  Contrast: {analysis['contrast'][level]:.4f}")

    return {
        'original': image,
        'scale_space': scale_space,
        'opening_space': opening_space,
        'closing_space': closing_space,
        'kb_sequence': kb_sequence,
        'residues': residues,
        'analysis': analysis,
        'filtered': scale_space[-1]  # Final filtered result
    }

def run_complete_analysis(image_path=None, pyramid_levels=DEFAULT_PYRAMID_LEVELS,
                         morphological_levels=DEFAULT_MORPHOLOGICAL_LEVELS):
    """
    Run complete multiscale analysis pipeline.

    Args:
        image_path (str): Path to image (optional)
        pyramid_levels (int): Number of pyramid levels
        morphological_levels (int): Number of morphological levels

    Returns:
        dict: Complete analysis results
    """
    print("TP 13 - MULTISCALE ANALYSIS")
    print("="*60)
    print("Comprehensive implementation of pyramidal and morphological multiscale methods")
    print()

    # Validate configuration
    if not validate_config():
        print("Configuration validation failed. Please check config.py")
        return None

    # Load image
    if image_path:
        try:
            image = io.imread(image_path)
            if len(image.shape) == 3:
                image = color.rgb2gray(image)
            image = image.astype(IMAGE_DTYPE)
            if NORMALIZE_INPUT:
                image = image / np.max(image)
        except Exception as e:
            print(f"Error loading custom image: {e}")
            image = load_cerveau_image()
    else:
        image = load_cerveau_image()

    print(f"Working with image of shape: {image.shape}")

    # Run pyramidal analysis
    pyramidal_results = run_pyramidal_analysis(image, pyramid_levels)

    # Run morphological analysis
    morphological_results = run_morphological_analysis(image, morphological_levels)

    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    visualizer = MultiscaleVisualizer()

    # Pyramid decomposition visualization
    print("\n1. Creating pyramid decomposition visualization...")
    visualizer.visualize_pyramid_decomposition(
        pyramidal_results['gaussian_pyramid'],
        pyramidal_results['laplacian_pyramid'],
        pyramidal_results['original']
    )

    # Reconstruction comparison
    print("\n2. Creating reconstruction comparison...")
    visualizer.visualize_reconstruction_comparison(
        pyramidal_results['original'],
        pyramidal_results['reconstructed_with_details'],
        pyramidal_results['reconstructed_smooth']
    )

    # Morphological decomposition
    print("\n3. Creating morphological decomposition visualization...")
    visualizer.visualize_morphological_decomposition(
        morphological_results['scale_space']
    )

    # Error analysis
    print("\n4. Creating error analysis...")
    visualizer.visualize_error_analysis(pyramidal_results['error_metrics'])

    # Method comparison
    print("\n5. Creating method comparison...")
    visualizer.visualize_multiscale_comparison(pyramidal_results, morphological_results)

    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nPyramidal Decomposition ({pyramid_levels} levels):")
    print(f"  Perfect reconstruction MSE: {pyramidal_results['error_metrics']['mse']:.2e}")
    print(f"  Perfect reconstruction PSNR: {pyramidal_results['error_metrics']['psnr']:.2f} dB")
    print(f"  Perfect reconstruction SSIM: {pyramidal_results['error_metrics']['ssim']:.4f}")

    print(f"\nMorphological Decomposition ({morphological_levels} levels):")
    print(f"  Final energy: {morphological_results['analysis']['energy'][-1]:.2f}")
    print(f"  Final contrast: {morphological_results['analysis']['contrast'][-1]:.4f}")

    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("Analysis completed successfully!")

    return {
        'pyramidal': pyramidal_results,
        'morphological': morphological_results,
        'image': image
    }

def demonstrate_cerveau_decomposition():
    """
    Demonstrate the specific cerveau image decomposition as shown in the specification.
    """
    print("\n" + "="*60)
    print("CERVEAU IMAGE DECOMPOSITION DEMONSTRATION")
    print("="*60)

    # Load cerveau image
    cerveau_image = load_cerveau_image()

    # Perform 4-level decomposition as specified
    print("\nPerforming 4-level pyramidal decomposition of cerveau image...")

    decomposer = LaplacianPyramidDecomposition(levels=4)
    gaussian_pyramid, laplacian_pyramid = decomposer.decompose(cerveau_image)

    # Show pyramid levels
    print(f"\nGaussian Pyramid Levels:")
    for i, level in enumerate(gaussian_pyramid):
        print(f"  Level {i}: {level.shape}")

    print(f"\nLaplacian Pyramid Levels:")
    for i, level in enumerate(laplacian_pyramid):
        print(f"  Level {i}: {level.shape}")

    # Reconstruct and analyze
    reconstructed = decomposer.reconstruct()
    reconstructed_smooth = decomposer.reconstruct_without_details()

    # Compute errors
    error_with_details = decomposer.compute_reconstruction_error(cerveau_image, reconstructed)
    error_smooth = decomposer.compute_reconstruction_error(cerveau_image, reconstructed_smooth)

    print(f"\nReconstruction Results:")
    print(f"  With details - MSE: {error_with_details['mse']:.2e}")
    print(f"  Without details - MSE: {error_smooth['mse']:.2e}")
    print(f"  Perfect reconstruction achieved: {error_with_details['mse'] < 1e-10}")

    # Create visualization
    visualizer = MultiscaleVisualizer()
    visualizer.visualize_pyramid_decomposition(gaussian_pyramid, laplacian_pyramid, cerveau_image)
    visualizer.visualize_reconstruction_comparison(cerveau_image, reconstructed, reconstructed_smooth)

    return {
        'cerveau_image': cerveau_image,
        'gaussian_pyramid': gaussian_pyramid,
        'laplacian_pyramid': laplacian_pyramid,
        'reconstructed': reconstructed,
        'reconstructed_smooth': reconstructed_smooth
    }

if __name__ == "__main__":
    # Print configuration summary
    print_config_summary()
    print()

    # Run complete analysis
    try:
        results = run_complete_analysis()

        # Demonstrate specific cerveau decomposition
        cerveau_demo = demonstrate_cerveau_decomposition()

        print("\n" + "="*60)
        print("TP 13 MULTISCALE ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nAll visualizations have been generated and saved.")
        print("Check the outputs/ directory for the generated figures.")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
