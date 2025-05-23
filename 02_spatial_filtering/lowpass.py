#!/usr/bin/env python3
"""
TP2 - Spatial Filtering: Low-Pass Filters
Professional implementation of low-pass filtering techniques for noise reduction.

This module demonstrates:
- Mean filtering with different kernel sizes
- Median filtering for salt-and-pepper noise
- Morphological filters (minimum, maximum)
- Gaussian filtering for smooth noise reduction
- Quantitative comparison of filter performance

Physics Background:
Low-pass filters remove high-frequency components (noise, fine details) while preserving
low-frequency components (main structures). Different filters have different characteristics:
- Mean: Simple averaging, can blur edges
- Median: Preserves edges while removing impulse noise
- Gaussian: Smooth blurring with controlled spread
- Morphological: Shape-based filtering

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.filters.rank import mean, median, minimum, maximum
from skimage.morphology import disk, square
from skimage.util import img_as_ubyte
from common_utils import ImageProcessor, print_image_info, clear_cache, calculate_image_metrics

class LowPassFilterProcessor(ImageProcessor):
    """Class for low-pass filtering operations."""

    def __init__(self):
        super().__init__('02_spatial_filtering')
        clear_cache()

    def apply_lowpass_filters(self, filename='blood.jpg'):
        """
        Apply various low-pass filters to an image.

        Args:
            filename (str): Input image filename

        Returns:
            dict: Dictionary containing original and filtered images
        """
        print("=" * 60)
        print("SPATIAL FILTERING - LOW-PASS FILTERS")
        print("=" * 60)

        # Load image
        image = self.load_image(filename, as_gray=True)
        image_ubyte = img_as_ubyte(image)  # Convert to uint8 for rank filters
        print_image_info(image, "Original Image")

        results = {'original': image}

        print("\nApplying low-pass filters...")

        # Mean filters with different sizes
        print("  Processing: Mean filters...")
        results['mean_3x3'] = mean(image_ubyte, square(3)) / 255.0
        results['mean_25x25'] = mean(image_ubyte, square(25)) / 255.0

        # Median filter
        print("  Processing: Median filter...")
        results['median_5x5'] = median(image_ubyte, square(5)) / 255.0

        # Morphological filters
        print("  Processing: Morphological filters...")
        results['minimum_5x5'] = minimum(image_ubyte, square(5)) / 255.0
        results['maximum_5x5'] = maximum(image_ubyte, square(5)) / 255.0

        # Gaussian filter
        print("  Processing: Gaussian filter...")
        results['gaussian_sigma5'] = gaussian(image, sigma=5)

        return results

    def visualize_lowpass_results(self, results):
        """
        Create professional visualization of low-pass filtering results.

        Args:
            results (dict): Dictionary of filtered images
        """
        # Main comparison plot
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Original image (larger subplot)
        axes[0, 0].imshow(results['original'], cmap='gray')
        axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=14)
        axes[0, 0].axis('off')

        # Filtered images
        filter_configs = [
            ('mean_3x3', 'Mean Filter 3×3', (0, 1)),
            ('mean_25x25', 'Mean Filter 25×25', (0, 2)),
            ('median_5x5', 'Median Filter 5×5', (1, 0)),
            ('minimum_5x5', 'Minimum Filter 5×5', (1, 1)),
            ('maximum_5x5', 'Maximum Filter 5×5', (1, 2)),
            ('gaussian_sigma5', 'Gaussian Filter σ=5', (2, 0))
        ]

        for key, title, pos in filter_configs:
            axes[pos].imshow(results[key], cmap='gray')
            axes[pos].set_title(title, fontweight='bold', fontsize=12)
            axes[pos].axis('off')

        # Hide unused subplots
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')

        plt.tight_layout()
        self.save_plot('lowpass_filters_comparison.png', 'Low-Pass Filters Comparison')

    def analyze_filter_performance(self, results):
        """
        Analyze and compare the performance of different filters.

        Args:
            results (dict): Dictionary of filtered images
        """
        original = results['original']

        # Calculate metrics for each filter
        print("\nFilter Performance Analysis:")
        print("-" * 50)

        metrics_data = []
        filter_names = []

        for key, filtered in results.items():
            if key == 'original':
                continue

            metrics = calculate_image_metrics(original, filtered)
            metrics_data.append(metrics)
            filter_names.append(key.replace('_', ' ').title())

            print(f"{key.replace('_', ' ').title()}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            if 'ssim' in metrics:
                print(f"  SSIM: {metrics['ssim']:.4f}")
            print()

        # Create performance comparison plot
        self._plot_performance_metrics(metrics_data, filter_names)

    def _plot_performance_metrics(self, metrics_data, filter_names):
        """
        Plot performance metrics comparison.

        Args:
            metrics_data (list): List of metric dictionaries
            filter_names (list): List of filter names
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # MSE comparison
        mse_values = [m['mse'] for m in metrics_data]
        axes[0].bar(range(len(filter_names)), mse_values, color='lightcoral')
        axes[0].set_title('Mean Squared Error (Lower is Better)', fontweight='bold')
        axes[0].set_ylabel('MSE')
        axes[0].set_xticks(range(len(filter_names)))
        axes[0].set_xticklabels(filter_names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)

        # PSNR comparison
        psnr_values = [m['psnr'] for m in metrics_data]
        axes[1].bar(range(len(filter_names)), psnr_values, color='lightblue')
        axes[1].set_title('Peak Signal-to-Noise Ratio (Higher is Better)', fontweight='bold')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_xticks(range(len(filter_names)))
        axes[1].set_xticklabels(filter_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)

        # SSIM comparison (if available)
        if 'ssim' in metrics_data[0]:
            ssim_values = [m['ssim'] for m in metrics_data]
            axes[2].bar(range(len(filter_names)), ssim_values, color='lightgreen')
            axes[2].set_title('Structural Similarity Index (Higher is Better)', fontweight='bold')
            axes[2].set_ylabel('SSIM')
            axes[2].set_xticks(range(len(filter_names)))
            axes[2].set_xticklabels(filter_names, rotation=45, ha='right')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'SSIM not available\n(requires scikit-image)',
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('SSIM Comparison', fontweight='bold')

        plt.tight_layout()
        self.save_plot('filter_performance_metrics.png', 'Low-Pass Filter Performance Comparison')

    def demonstrate_noise_reduction(self, filename='blood.jpg'):
        """
        Demonstrate noise reduction capabilities of different filters.

        Args:
            filename (str): Input image filename
        """
        # Load clean image
        clean_image = self.load_image(filename, as_gray=True)

        # Add different types of noise
        print("Adding noise to demonstrate filter effectiveness...")

        # Gaussian noise
        gaussian_noise = np.random.normal(0, 0.05, clean_image.shape)
        noisy_gaussian = np.clip(clean_image + gaussian_noise, 0, 1)

        # Salt and pepper noise
        noisy_sp = clean_image.copy()
        noise_mask = np.random.random(clean_image.shape)
        noisy_sp[noise_mask < 0.05] = 0  # Salt
        noisy_sp[noise_mask > 0.95] = 1  # Pepper

        # Apply best filters for each noise type
        clean_ubyte = img_as_ubyte(clean_image)
        noisy_gaussian_ubyte = img_as_ubyte(noisy_gaussian)
        noisy_sp_ubyte = img_as_ubyte(noisy_sp)

        # Gaussian filter for Gaussian noise
        denoised_gaussian = gaussian(noisy_gaussian, sigma=1)

        # Median filter for salt-and-pepper noise
        denoised_sp = median(noisy_sp_ubyte, square(3)) / 255.0

        # Visualize noise reduction
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Gaussian noise
        axes[0, 0].imshow(clean_image, cmap='gray')
        axes[0, 0].set_title('Original Clean Image', fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(noisy_gaussian, cmap='gray')
        axes[0, 1].set_title('With Gaussian Noise', fontweight='bold')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(denoised_gaussian, cmap='gray')
        axes[0, 2].set_title('Denoised (Gaussian Filter)', fontweight='bold')
        axes[0, 2].axis('off')

        # Row 2: Salt-and-pepper noise
        axes[1, 0].imshow(clean_image, cmap='gray')
        axes[1, 0].set_title('Original Clean Image', fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(noisy_sp, cmap='gray')
        axes[1, 1].set_title('With Salt & Pepper Noise', fontweight='bold')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(denoised_sp, cmap='gray')
        axes[1, 2].set_title('Denoised (Median Filter)', fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()
        self.save_plot('noise_reduction_demonstration.png', 'Noise Reduction with Low-Pass Filters')

def main():
    """Main function to run low-pass filtering demonstrations."""
    try:
        # Initialize processor
        processor = LowPassFilterProcessor()

        # Apply low-pass filters
        results = processor.apply_lowpass_filters('blood.jpg')

        # Visualize results
        processor.visualize_lowpass_results(results)

        # Analyze performance
        processor.analyze_filter_performance(results)

        # Demonstrate noise reduction
        processor.demonstrate_noise_reduction('blood.jpg')

        print("\n" + "=" * 60)
        print("✅ TP2 - Low-Pass Filters completed successfully!")
        print(f"📁 All visualizations saved to: plots/02_spatial_filtering/")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error in TP2 Low-Pass Filters: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()