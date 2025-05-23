#!/usr/bin/env python3
"""
Common Utilities for Image Processing Project
Centralized utilities for image loading, saving, and visualization.

This module provides consistent functionality across all TPs for:
- Image loading and preprocessing
- Plot generation and saving
- Error handling and validation
- Performance optimization

Author: Professional Image Processing Project
Date: 2024
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, color, img_as_float
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global configuration
IMAGES_DIR = "images"
PLOTS_DIR = "plots"
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (12, 8)

class ImageProcessor:
    """Base class for image processing operations with common utilities."""

    def __init__(self, tp_name):
        """
        Initialize the image processor.

        Args:
            tp_name (str): Name of the TP (e.g., '01_basic_operations')
        """
        self.tp_name = tp_name
        self.plots_dir = Path(PLOTS_DIR) / tp_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Clear any existing matplotlib cache
        plt.clf()
        plt.close('all')

    def load_image(self, filename, as_gray=False, normalize=True):
        """
        Load an image with consistent preprocessing.

        Args:
            filename (str): Image filename
            as_gray (bool): Convert to grayscale
            normalize (bool): Normalize to [0, 1] range

        Returns:
            np.ndarray: Processed image
        """
        # Try multiple possible paths
        possible_paths = [
            Path(IMAGES_DIR) / filename,
            Path("images") / filename,
            Path("../images") / filename,
            Path(filename)
        ]

        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break

        if image_path is None:
            raise FileNotFoundError(f"Image {filename} not found in any expected location")

        try:
            # Load image
            image = io.imread(str(image_path))

            # Convert to grayscale if requested
            if as_gray and len(image.shape) == 3:
                image = color.rgb2gray(image)

            # Convert to float and normalize
            if normalize:
                image = img_as_float(image)

            print(f"✓ Loaded image: {filename} (shape: {image.shape})")
            return image

        except Exception as e:
            raise IOError(f"Error loading image {filename}: {e}")

    def save_plot(self, filename, title=None, dpi=DEFAULT_DPI, bbox_inches='tight'):
        """
        Save a plot with consistent formatting.

        Args:
            filename (str): Output filename
            title (str): Plot title
            dpi (int): Resolution
            bbox_inches (str): Bounding box setting
        """
        if title:
            plt.suptitle(title, fontsize=16, fontweight='bold')

        output_path = self.plots_dir / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches,
                   facecolor='white', edgecolor='none')

        print(f"✓ Saved plot: {output_path}")

        # Clear the current figure
        plt.clf()

    def create_comparison_plot(self, images, titles, filename, suptitle=None,
                             cmap='gray', figsize=DEFAULT_FIGSIZE):
        """
        Create a comparison plot with multiple images.

        Args:
            images (list): List of images to display
            titles (list): List of titles for each image
            filename (str): Output filename
            suptitle (str): Overall title
            cmap (str): Colormap
            figsize (tuple): Figure size
        """
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, (img, title) in enumerate(zip(images, titles)):
            if i < len(axes):
                axes[i].imshow(img, cmap=cmap)
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        if suptitle:
            plt.suptitle(suptitle, fontsize=16, fontweight='bold')

        plt.tight_layout()
        self.save_plot(filename)

    def create_histogram_plot(self, image, filename, title=None, bins=256):
        """
        Create a histogram plot for an image.

        Args:
            image (np.ndarray): Input image
            filename (str): Output filename
            title (str): Plot title
            bins (int): Number of histogram bins
        """
        plt.figure(figsize=(10, 6))

        if len(image.shape) == 3:
            # Color image
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                plt.hist(image[:, :, i].flatten(), bins=bins, alpha=0.7,
                        color=color, label=f'{color.capitalize()} channel')
            plt.legend()
        else:
            # Grayscale image
            plt.hist(image.flatten(), bins=bins, alpha=0.7, color='gray')

        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        if title:
            plt.title(title)

        self.save_plot(filename)

def validate_image(image, name="image"):
    """
    Validate image properties and provide informative feedback.

    Args:
        image (np.ndarray): Image to validate
        name (str): Name for error messages

    Returns:
        bool: True if valid
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")

    if image.size == 0:
        raise ValueError(f"{name} is empty")

    if len(image.shape) not in [2, 3]:
        raise ValueError(f"{name} must be 2D or 3D array")

    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        raise ValueError(f"{name} has invalid number of channels: {image.shape[2]}")

    return True

def normalize_image(image, method='minmax'):
    """
    Normalize image to [0, 1] range using different methods.

    Args:
        image (np.ndarray): Input image
        method (str): Normalization method ('minmax', 'zscore', 'robust')

    Returns:
        np.ndarray: Normalized image
    """
    validate_image(image)

    if method == 'minmax':
        # Min-max normalization
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        else:
            return image

    elif method == 'zscore':
        # Z-score normalization
        mean, std = image.mean(), image.std()
        if std > 0:
            normalized = (image - mean) / std
            # Rescale to [0, 1]
            return normalize_image(normalized, 'minmax')
        else:
            return image

    elif method == 'robust':
        # Robust normalization using percentiles
        p2, p98 = np.percentile(image, [2, 98])
        if p98 > p2:
            clipped = np.clip(image, p2, p98)
            return (clipped - p2) / (p98 - p2)
        else:
            return image

    else:
        raise ValueError(f"Unknown normalization method: {method}")

def setup_matplotlib():
    """Configure matplotlib for consistent, high-quality plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': DEFAULT_DPI,
        'savefig.dpi': DEFAULT_DPI,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

def clear_cache():
    """Clear matplotlib and other caches for clean execution."""
    plt.clf()
    plt.close('all')

    # Clear any numpy warnings
    np.seterr(all='ignore')

def print_image_info(image, name="Image"):
    """
    Print comprehensive information about an image.

    Args:
        image (np.ndarray): Image to analyze
        name (str): Name for display
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {image.shape}")
    print(f"  Data type: {image.dtype}")
    print(f"  Range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Mean: {image.mean():.3f}")
    print(f"  Std: {image.std():.3f}")

    if len(image.shape) == 2:
        print(f"  Type: Grayscale")
    elif len(image.shape) == 3:
        print(f"  Type: Color ({image.shape[2]} channels)")

def create_output_directory(tp_name):
    """
    Create output directory for a specific TP.

    Args:
        tp_name (str): TP directory name

    Returns:
        Path: Created directory path
    """
    output_dir = Path(PLOTS_DIR) / tp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def apply_kernel(image, kernel, mode='same', boundary='symm'):
    """
    Apply a convolution kernel to an image with proper error handling.

    Args:
        image (np.ndarray): Input image
        kernel (np.ndarray): Convolution kernel
        mode (str): Convolution mode
        boundary (str): Boundary condition

    Returns:
        np.ndarray: Convolved image
    """
    from scipy.signal import convolve2d

    validate_image(image)

    if len(image.shape) == 3:
        # Apply to each channel separately
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = convolve2d(image[:, :, i], kernel, mode=mode, boundary=boundary)
        return result
    else:
        return convolve2d(image, kernel, mode=mode, boundary=boundary)

def create_standard_kernels():
    """
    Create a dictionary of standard convolution kernels.

    Returns:
        dict: Dictionary of kernels with descriptive names
    """
    kernels = {
        'mean_3x3': (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        'gaussian_3x3': (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        'laplacian': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'edge_detection': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        'sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    }
    return kernels

def calculate_image_metrics(image1, image2):
    """
    Calculate common image quality metrics between two images.

    Args:
        image1 (np.ndarray): Reference image
        image2 (np.ndarray): Comparison image

    Returns:
        dict: Dictionary of metrics (MSE, PSNR, SSIM if available)
    """
    validate_image(image1, "image1")
    validate_image(image2, "image2")

    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape")

    # Mean Squared Error
    mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)

    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 1.0 if image1.max() <= 1.0 else 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    metrics = {
        'mse': mse,
        'psnr': psnr,
        'mae': np.mean(np.abs(image1.astype(float) - image2.astype(float)))
    }

    # Try to calculate SSIM if skimage is available
    try:
        from skimage.metrics import structural_similarity as ssim
        if len(image1.shape) == 3:
            metrics['ssim'] = ssim(image1, image2, multichannel=True, channel_axis=2)
        else:
            metrics['ssim'] = ssim(image1, image2)
    except ImportError:
        pass

    return metrics

def robust_image_load(filename, fallback_paths=None):
    """
    Robustly load an image from multiple possible locations.

    Args:
        filename (str): Image filename
        fallback_paths (list): Additional paths to try

    Returns:
        np.ndarray: Loaded image
    """
    if fallback_paths is None:
        fallback_paths = []

    # Standard paths to try
    standard_paths = [
        Path(IMAGES_DIR) / filename,
        Path("images") / filename,
        Path("../images") / filename,
        Path(filename)
    ]

    all_paths = standard_paths + [Path(p) / filename for p in fallback_paths]

    for path in all_paths:
        if path.exists():
            try:
                image = io.imread(str(path))
                print(f"✓ Successfully loaded: {path}")
                return image
            except Exception as e:
                print(f"⚠ Failed to load {path}: {e}")
                continue

    raise FileNotFoundError(f"Could not load {filename} from any location")

# Initialize matplotlib settings
setup_matplotlib()
