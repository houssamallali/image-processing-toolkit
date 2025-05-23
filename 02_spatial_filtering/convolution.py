#!/usr/bin/env python3
"""
TP2 - Spatial Filtering: Convolution Operations
Professional implementation of convolution-based image filtering techniques.

This module demonstrates:
- Basic convolution operations with standard kernels
- Mean, Gaussian, and Laplacian filtering
- Professional visualization of filtering results
- Quantitative analysis of filter effects

Physics Background:
Convolution is a mathematical operation that combines two functions to produce a third.
In image processing, convolution applies a kernel (small matrix) to each pixel neighborhood,
creating effects like blurring, sharpening, or edge detection.

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from common_utils import ImageProcessor, print_image_info, clear_cache, apply_kernel, create_standard_kernels

class ConvolutionProcessor(ImageProcessor):
    """Class for convolution-based image filtering operations."""

    def __init__(self):
        super().__init__('02_spatial_filtering')
        clear_cache()
        self.kernels = create_standard_kernels()

    def demonstrate_basic_convolution(self, filename='blood.jpg'):
        """
        Demonstrate basic convolution operations with standard kernels.

        Args:
            filename (str): Input image filename

        Returns:
            dict: Dictionary containing original image and filtered results
        """
        print("=" * 60)
        print("SPATIAL FILTERING - CONVOLUTION OPERATIONS")
        print("=" * 60)

        # Load image
        image = self.load_image(filename, as_gray=True)
        print_image_info(image, "Original Image")

        # Apply different kernels
        results = {'original': image}

        # Select key kernels for demonstration
        demo_kernels = {
            'Mean Filter (3x3)': self.kernels['mean_3x3'],
            'Gaussian Filter (3x3)': self.kernels['gaussian_3x3'],
            'Laplacian Filter': self.kernels['laplacian']
        }

        print("\nApplying convolution filters...")
        for name, kernel in demo_kernels.items():
            print(f"  Processing: {name}")
            filtered = apply_kernel(image, kernel)
            results[name] = filtered

            # Print kernel information
            print(f"    Kernel shape: {kernel.shape}")
            print(f"    Kernel sum: {kernel.sum():.3f}")

        return results

    def visualize_convolution_results(self, results):
        """
        Create professional visualization of convolution results.

        Args:
            results (dict): Dictionary of images and filtered results
        """
        # Create main comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original image
        axes[0, 0].imshow(results['original'], cmap='gray')
        axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')

        # Filtered images
        filter_names = ['Mean Filter (3x3)', 'Gaussian Filter (3x3)', 'Laplacian Filter']
        positions = [(0, 1), (1, 0), (1, 1)]

        for name, pos in zip(filter_names, positions):
            axes[pos].imshow(results[name], cmap='gray')
            axes[pos].set_title(name, fontweight='bold', fontsize=14)
            axes[pos].axis('off')

        plt.tight_layout()
        self.save_plot('convolution_comparison.png', 'Convolution Filter Comparison')

    def analyze_filter_effects(self, results):
        """
        Analyze and visualize the effects of different filters.

        Args:
            results (dict): Dictionary of filtered images
        """
        original = results['original']

        # Create analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Filtered images
        filter_names = ['Mean Filter (3x3)', 'Gaussian Filter (3x3)', 'Laplacian Filter']
        for i, name in enumerate(filter_names):
            axes[0, i].imshow(results[name], cmap='gray')
            axes[0, i].set_title(f'{name}', fontweight='bold')
            axes[0, i].axis('off')

        # Row 2: Difference images (showing what was removed/enhanced)
        for i, name in enumerate(filter_names):
            if 'Laplacian' in name:
                # For Laplacian, show the absolute values (edges detected)
                diff = np.abs(results[name])
                title = 'Edge Detection Result'
            else:
                # For smoothing filters, show what was removed
                diff = np.abs(original - results[name])
                title = 'Removed Details'

            axes[1, i].imshow(diff, cmap='hot')
            axes[1, i].set_title(title, fontweight='bold')
            axes[1, i].axis('off')

        plt.tight_layout()
        self.save_plot('filter_effects_analysis.png', 'Analysis of Filter Effects')

    def demonstrate_kernel_properties(self):
        """
        Visualize different kernels and their properties.
        """
        kernels_to_show = {
            'Mean (3x3)': self.kernels['mean_3x3'],
            'Gaussian (3x3)': self.kernels['gaussian_3x3'],
            'Laplacian': self.kernels['laplacian'],
            'Sobel X': self.kernels['sobel_x'],
            'Sobel Y': self.kernels['sobel_y'],
            'Sharpen': self.kernels['sharpen']
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (name, kernel) in enumerate(kernels_to_show.items()):
            # Visualize kernel
            im = axes[i].imshow(kernel, cmap='RdBu', interpolation='nearest')
            axes[i].set_title(f'{name}\nSum: {kernel.sum():.2f}', fontweight='bold')

            # Add values as text
            for y in range(kernel.shape[0]):
                for x in range(kernel.shape[1]):
                    axes[i].text(x, y, f'{kernel[y, x]:.2f}',
                               ha='center', va='center', fontsize=10)

            axes[i].set_xticks([])
            axes[i].set_yticks([])
            plt.colorbar(im, ax=axes[i], shrink=0.6)

        plt.tight_layout()
        self.save_plot('kernel_visualization.png', 'Convolution Kernels and Properties')

def main():
    """Main function to run convolution demonstrations."""
    try:
        # Initialize processor
        processor = ConvolutionProcessor()

        # Demonstrate basic convolution
        results = processor.demonstrate_basic_convolution('blood.jpg')

        # Visualize results
        processor.visualize_convolution_results(results)

        # Analyze filter effects
        processor.analyze_filter_effects(results)

        # Show kernel properties
        processor.demonstrate_kernel_properties()

        print("\n" + "=" * 60)
        print("✅ TP2 - Convolution Operations completed successfully!")
        print(f"📁 All visualizations saved to: plots/02_spatial_filtering/")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error in TP2 Convolution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()