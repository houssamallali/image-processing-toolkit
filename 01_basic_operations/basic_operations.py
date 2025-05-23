#!/usr/bin/env python3
"""
TP1 - Basic Image Operations
Fundamental image processing operations including loading, display, and channel analysis.

This module demonstrates:
- Image loading and basic properties
- RGB channel separation and visualization
- Image compression with different quality levels
- Professional visualization techniques

Author: Professional Image Processing Project
Date: 2024
"""

import sys
import os
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from common_utils import ImageProcessor, print_image_info, clear_cache

class BasicImageOperations(ImageProcessor):
    """Class for basic image processing operations."""

    def __init__(self):
        super().__init__('01_basic_operations')
        clear_cache()

    def load_and_analyze_image(self, filename='retina.jpg'):
        """
        Load an image and analyze its basic properties.

        Args:
            filename (str): Image filename

        Returns:
            np.ndarray: Loaded image
        """
        print("=" * 60)
        print("BASIC IMAGE OPERATIONS - TP1")
        print("=" * 60)

        # Load image
        image = self.load_image(filename, normalize=False)

        # Print comprehensive image information
        print_image_info(image, "Loaded Image")

        return image

    def visualize_original_image(self, image):
        """
        Create a professional visualization of the original image.

        Args:
            image (np.ndarray): Input image
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title("Original Retina Image", fontsize=16, fontweight='bold')
        plt.axis('off')

        # Add image information as text
        info_text = f"Shape: {image.shape}\nData type: {image.dtype}\nRange: [{image.min()}, {image.max()}]"
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        self.save_plot('original_image.png', 'Original Retina Image Analysis')

    def analyze_rgb_channels(self, image):
        """
        Separate and analyze RGB channels of the image.

        Args:
            image (np.ndarray): Input RGB image

        Returns:
            tuple: (red, green, blue) channel arrays
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB (3-channel)")

        # Extract RGB channels
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]

        # Print channel statistics
        print("\nRGB Channel Analysis:")
        print(f"Red Channel   - Mean: {red_channel.mean():.2f}, Std: {red_channel.std():.2f}")
        print(f"Green Channel - Mean: {green_channel.mean():.2f}, Std: {green_channel.std():.2f}")
        print(f"Blue Channel  - Mean: {blue_channel.mean():.2f}, Std: {blue_channel.std():.2f}")

        return red_channel, green_channel, blue_channel

    def visualize_rgb_channels(self, image):
        """
        Create professional visualization of RGB channels.

        Args:
            image (np.ndarray): Input RGB image
        """
        red, green, blue = self.analyze_rgb_channels(image)

        # Create channel visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original RGB Image', fontweight='bold')
        axes[0, 0].axis('off')

        # Individual channels in grayscale
        channels = [red, green, blue]
        channel_names = ['Red Channel', 'Green Channel', 'Blue Channel']
        colors = ['Reds', 'Greens', 'Blues']

        for i, (channel, name, cmap) in enumerate(zip(channels, channel_names, colors)):
            # Grayscale representation
            axes[0, i].imshow(channel, cmap='gray')
            axes[0, i].set_title(f'{name} (Grayscale)', fontweight='bold')
            axes[0, i].axis('off')

            # Colored representation
            axes[1, i].imshow(channel, cmap=cmap)
            axes[1, i].set_title(f'{name} (Colored)', fontweight='bold')
            axes[1, i].axis('off')

        plt.tight_layout()
        self.save_plot('rgb_channels_analysis.png', 'RGB Channel Decomposition Analysis')

    def create_channel_histograms(self, image):
        """
        Create histograms for each RGB channel.

        Args:
            image (np.ndarray): Input RGB image
        """
        red, green, blue = self.analyze_rgb_channels(image)

        plt.figure(figsize=(12, 8))

        # Combined histogram
        plt.subplot(2, 2, 1)
        plt.hist(red.flatten(), bins=50, alpha=0.7, color='red', label='Red')
        plt.hist(green.flatten(), bins=50, alpha=0.7, color='green', label='Green')
        plt.hist(blue.flatten(), bins=50, alpha=0.7, color='blue', label='Blue')
        plt.title('Combined RGB Histograms', fontweight='bold')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Individual channel histograms
        channels = [red, green, blue]
        colors = ['red', 'green', 'blue']
        titles = ['Red Channel Histogram', 'Green Channel Histogram', 'Blue Channel Histogram']

        for i, (channel, color, title) in enumerate(zip(channels, colors, titles)):
            plt.subplot(2, 2, i + 2)
            plt.hist(channel.flatten(), bins=50, color=color, alpha=0.7)
            plt.title(title, fontweight='bold')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_plot('rgb_histograms.png', 'RGB Channel Histogram Analysis')

    def demonstrate_compression_effects(self, image, qualities=[10, 50, 90]):
        """
        Demonstrate JPEG compression effects at different quality levels.

        Args:
            image (np.ndarray): Input image
            qualities (list): List of JPEG quality levels
        """
        print(f"\nDemonstrating JPEG compression at quality levels: {qualities}")

        compressed_images = []
        file_sizes = []

        for quality in qualities:
            # Save with specific quality
            temp_filename = f'temp_q{quality}.jpg'
            io.imsave(temp_filename, image, quality=quality)

            # Load back the compressed image
            compressed = io.imread(temp_filename)
            compressed_images.append(compressed)

            # Get file size
            file_size = os.path.getsize(temp_filename) / 1024  # KB
            file_sizes.append(file_size)

            # Clean up temporary file
            os.remove(temp_filename)

            print(f"Quality {quality}: File size = {file_size:.1f} KB")

        # Visualize compression effects
        fig, axes = plt.subplots(2, len(qualities), figsize=(15, 10))

        for i, (compressed, quality, size) in enumerate(zip(compressed_images, qualities, file_sizes)):
            # Show compressed image
            axes[0, i].imshow(compressed)
            axes[0, i].set_title(f'Quality {quality}\nSize: {size:.1f} KB', fontweight='bold')
            axes[0, i].axis('off')

            # Show difference from original
            diff = np.abs(image.astype(float) - compressed.astype(float))
            axes[1, i].imshow(diff, cmap='hot')
            axes[1, i].set_title(f'Difference (Q{quality})', fontweight='bold')
            axes[1, i].axis('off')

        plt.tight_layout()
        self.save_plot('compression_comparison.png', 'JPEG Compression Quality Comparison')

def main():
    """Main function to run all basic image operations."""
    try:
        # Initialize processor
        processor = BasicImageOperations()

        # Load and analyze image
        image = processor.load_and_analyze_image('retina.jpg')

        # Visualize original image
        processor.visualize_original_image(image)

        # Analyze and visualize RGB channels
        processor.visualize_rgb_channels(image)

        # Create channel histograms
        processor.create_channel_histograms(image)

        # Demonstrate compression effects
        processor.demonstrate_compression_effects(image)

        print("\n" + "=" * 60)
        print("✅ TP1 - Basic Image Operations completed successfully!")
        print(f"📁 All visualizations saved to: plots/01_basic_operations/")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error in TP1: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()