#!/usr/bin/env python3
"""
TP 13 - Multiscale Analysis Configuration
Centralized configuration management for pyramidal decomposition and multiscale analysis.

Author: TP13 Implementation
Date: 2024
"""

import os
import numpy as np

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Verbose output
VERBOSE = True

# =============================================================================
# IMAGE CONFIGURATION
# =============================================================================

# Default image paths
DEFAULT_IMAGE_PATHS = [
    "imagekimia216/cerveau.jpg",
    "../imagekimia216/cerveau.jpg",
    "../../imagekimia216/cerveau.jpg",
    "imagekimia216/cerveau.jpeg",
    "../imagekimia216/cerveau.jpeg",
    "images/cerveau.jpg",
    "../images/cerveau.jpg",
    "../../images/cerveau.jpg",
    "images/cerveau.jpeg",
    "../images/cerveau.jpeg",
    # Fallback to brain images if cerveau not found
    "imagekimia216/BrainT1.bmp",
    "../imagekimia216/BrainT1.bmp",
    "../../imagekimia216/BrainT1.bmp",
    "imagekimia216/brain.bmp",
    "../imagekimia216/brain.bmp"
]

# Image processing parameters
IMAGE_DTYPE = np.float64
NORMALIZE_INPUT = True
CLIP_OUTPUT = True

# =============================================================================
# PYRAMIDAL DECOMPOSITION CONFIGURATION
# =============================================================================

# Default number of pyramid levels
DEFAULT_PYRAMID_LEVELS = 4

# Gaussian filter parameters
GAUSSIAN_SIGMA = 1.0
GAUSSIAN_TRUNCATE = 3.0

# Subsampling/upsampling parameters
SUBSAMPLING_FACTOR = 0.5
UPSAMPLING_FACTOR = 2.0

# Interpolation method for resizing
INTERPOLATION_ORDER = 1  # Bilinear interpolation

# =============================================================================
# MORPHOLOGICAL MULTISCALE CONFIGURATION
# =============================================================================

# Default structuring element parameters
DEFAULT_SE_RADIUS = 3
DEFAULT_SE_TYPE = 'disk'  # 'disk', 'square', 'diamond'

# Morphological decomposition levels
DEFAULT_MORPHOLOGICAL_LEVELS = 5

# Kramer-Bruckner filter parameters
KB_ITERATIONS = 3
KB_CONVERGENCE_THRESHOLD = 1e-6

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
SAVE_FIGURES = True

# Color maps
PYRAMID_COLORMAP = 'gray'
ERROR_COLORMAP = 'hot'
DIFFERENCE_COLORMAP = 'RdBu_r'

# Font sizes
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
SUPTITLE_FONTSIZE = 16

# Figure sizes
PYRAMID_FIGURE_SIZE = (15, 10)
COMPARISON_FIGURE_SIZE = (12, 8)
ERROR_FIGURE_SIZE = (10, 6)

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output directory
OUTPUT_DIR = "outputs/"

# Output filenames
OUTPUT_FILES = {
    'pyramid_decomposition': 'pyramid_decomposition.png',
    'reconstruction_comparison': 'reconstruction_comparison.png',
    'morphological_decomposition': 'morphological_decomposition.png',
    'error_analysis': 'error_analysis.png',
    'multiscale_comparison': 'multiscale_comparison.png',
    'gaussian_pyramid': 'gaussian_pyramid.png',
    'laplacian_pyramid': 'laplacian_pyramid.png',
    'reconstruction_error': 'reconstruction_error.png',
    'scale_space': 'scale_space_decomposition.png'
}

# =============================================================================
# ERROR ANALYSIS CONFIGURATION
# =============================================================================

# Error metrics to compute
ERROR_METRICS = ['mse', 'psnr', 'ssim', 'mae']

# PSNR calculation parameters
PSNR_MAX_VALUE = 1.0  # For normalized images

# SSIM parameters
SSIM_WIN_SIZE = 7
SSIM_K1 = 0.01
SSIM_K2 = 0.03

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Memory management
MAX_IMAGE_SIZE = 1024  # Maximum dimension for processing
ENABLE_MEMORY_OPTIMIZATION = True

# Parallel processing
N_JOBS = -1  # Use all available cores

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_image_path():
    """
    Get the correct image path, trying alternatives if needed.

    Returns:
        str: Valid image path or None if not found
    """
    for path in DEFAULT_IMAGE_PATHS:
        if os.path.exists(path):
            return path

    print("Warning: Brain image not found in any expected location")
    return None

def create_output_directory():
    """
    Create output directory if it doesn't exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_output_path(filename_key):
    """
    Get full output path for a given filename key.

    Args:
        filename_key (str): Key from OUTPUT_FILES dict

    Returns:
        str: Full output path
    """
    create_output_directory()
    return os.path.join(OUTPUT_DIR, OUTPUT_FILES[filename_key])

def validate_config():
    """
    Validate configuration parameters.

    Returns:
        bool: True if configuration is valid
    """
    errors = []

    # Check pyramid levels
    if DEFAULT_PYRAMID_LEVELS < 1:
        errors.append("DEFAULT_PYRAMID_LEVELS must be >= 1")

    # Check Gaussian parameters
    if GAUSSIAN_SIGMA <= 0:
        errors.append("GAUSSIAN_SIGMA must be positive")

    # Check subsampling factor
    if not 0 < SUBSAMPLING_FACTOR < 1:
        errors.append("SUBSAMPLING_FACTOR must be between 0 and 1")

    # Check upsampling factor
    if UPSAMPLING_FACTOR <= 1:
        errors.append("UPSAMPLING_FACTOR must be > 1")

    # Check morphological levels
    if DEFAULT_MORPHOLOGICAL_LEVELS < 1:
        errors.append("DEFAULT_MORPHOLOGICAL_LEVELS must be >= 1")

    # Check structuring element radius
    if DEFAULT_SE_RADIUS < 1:
        errors.append("DEFAULT_SE_RADIUS must be >= 1")

    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True

def print_config_summary():
    """
    Print a summary of the current configuration.
    """
    print("TP 13 - Multiscale Analysis Configuration Summary")
    print("=" * 50)
    print(f"Image path: {get_image_path()}")
    print(f"Pyramid levels: {DEFAULT_PYRAMID_LEVELS}")
    print(f"Morphological levels: {DEFAULT_MORPHOLOGICAL_LEVELS}")
    print(f"Gaussian sigma: {GAUSSIAN_SIGMA}")
    print(f"Structuring element: {DEFAULT_SE_TYPE} (radius {DEFAULT_SE_RADIUS})")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figure DPI: {FIGURE_DPI}")
    print("=" * 50)

# =============================================================================
# ALGORITHM-SPECIFIC CONFIGURATIONS
# =============================================================================

# Laplacian pyramid specific settings
LAPLACIAN_CONFIG = {
    'sigma': GAUSSIAN_SIGMA,
    'truncate': GAUSSIAN_TRUNCATE,
    'preserve_dtype': True,
    'anti_aliasing': True
}

# Morphological decomposition specific settings
MORPHOLOGICAL_CONFIG = {
    'structuring_element_type': DEFAULT_SE_TYPE,
    'radius': DEFAULT_SE_RADIUS,
    'preserve_range': True,
    'iterations': KB_ITERATIONS
}

# Reconstruction specific settings
RECONSTRUCTION_CONFIG = {
    'interpolation_order': INTERPOLATION_ORDER,
    'preserve_range': True,
    'anti_aliasing': True,
    'clip_output': CLIP_OUTPUT
}

if __name__ == "__main__":
    print_config_summary()
    if validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
