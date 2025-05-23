#!/usr/bin/env python3
"""
TP 11 - Configuration File
Centralized configuration for the image classification system.

This file contains all configurable parameters for easy modification
without changing the main code.

Author: Generated for TP11 Assignment
"""

import os

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Path to Kimia database
DATABASE_PATH = "images/images_Kimia216/"

# Alternative paths to try if main path doesn't exist
ALTERNATIVE_PATHS = [
    "images/images_Kimia/",
    "../images/images_Kimia216/",
    "../../images/images_Kimia216/",
    "./images/images_Kimia216/",
    "../TP10_Kimia_Classification/images_Kimia216/",
    "TP10_Kimia_Classification/images_Kimia216/"
]

# Database structure
KIMIA_CLASSES = [
    'bird', 'bone', 'brick', 'camel', 'car', 'children',
    'classic', 'elephant', 'face', 'fork', 'fountain',
    'glass', 'hammer', 'heart', 'key', 'misk', 'ray', 'turtle'
]

# Number of images per class in Kimia database
IMAGES_PER_CLASS = 12

# Image file pattern
IMAGE_PATTERN = "{class_name}-*.bmp"

# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================

# Features to extract (must match skimage.measure.regionprops attributes)
FEATURE_NAMES = [
    'area',
    'convex_area',
    'eccentricity',
    'equivalent_diameter',
    'extent',
    'major_axis_length',
    'minor_axis_length',
    'perimeter',
    'solidity'
]

# Number of features
NUM_FEATURES = len(FEATURE_NAMES)

# Binary threshold for image conversion
BINARY_THRESHOLD = 128

# Whether to invert binary images (True for white objects on black background)
INVERT_BINARY = False

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

# Data splitting
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# Feature preprocessing
USE_STANDARDIZATION = True
USE_NORMALIZATION = False
USE_QUANTILE_TRANSFORM = False

# =============================================================================
# SVM CONFIGURATION
# =============================================================================

SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'random_state': RANDOM_STATE,
    'probability': True  # Enable probability estimates
}

# Alternative SVM configurations to try
SVM_ALTERNATIVES = [
    {'kernel': 'linear', 'C': 1.0},
    {'kernel': 'poly', 'degree': 3, 'C': 1.0},
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}
]

# =============================================================================
# MLP CONFIGURATION
# =============================================================================

MLP_CONFIG = {
    'hidden_layer_sizes': (100, 50),
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'learning_rate': 'adaptive',
    'alpha': 0.0001
}

# Alternative MLP configurations
MLP_ALTERNATIVES = [
    {'hidden_layer_sizes': (50,), 'max_iter': 500},
    {'hidden_layer_sizes': (100,), 'max_iter': 1000},
    {'hidden_layer_sizes': (200, 100), 'max_iter': 1000},
    {'hidden_layer_sizes': (100, 50, 25), 'max_iter': 1500}
]

# =============================================================================
# RANDOM FOREST CONFIGURATION
# =============================================================================

RF_CONFIG = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# =============================================================================
# KNN CONFIGURATION
# =============================================================================

KNN_CONFIG = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto',
    'metric': 'minkowski'
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
SAVE_FIGURES = True

# Plot colors
PLOT_COLORS = ['skyblue', 'lightgreen', 'coral', 'gold', 'lightpink', 'lightgray']

# Confusion matrix settings
CONFUSION_MATRIX_COLORS = ['Blues', 'Greens', 'Oranges', 'Purples']

# Font sizes
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output directory
OUTPUT_DIR = "tp11/"

# Output filenames
OUTPUT_FILES = {
    'sample_images': 'sample_kimia_images.png',
    'feature_distributions': 'feature_distributions_by_class.png',
    'feature_correlations': 'feature_correlations.png',
    'accuracy_comparison': 'accuracy_comparison.png',
    'confusion_matrices': 'all_confusion_matrices.png',
    'feature_importance': 'feature_importance.png',
    'extracted_features': 'extracted_features.npz',
    'classification_report': 'classification_report.txt',
    'model_comparison': 'model_comparison.csv',
    'statistical_significance': 'statistical_significance.png',
    'cv_comprehensive_results': 'cv_comprehensive_results.png',
    'learning_curve': 'learning_curve.png',
    'hyperparameter_optimization': 'hyperparameter_optimization.png',
    'validation_curves': 'validation_curves.png'
}

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Cross-validation settings
CV_FOLDS = 5
CV_SCORING = 'accuracy'

# Grid search settings
GRID_SEARCH_CV = 3
GRID_SEARCH_SCORING = 'accuracy'

# Parallel processing
N_JOBS = -1  # Use all available cores

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging level
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# Log to file
LOG_TO_FILE = True
LOG_FILENAME = 'tp11/classification.log'

# Verbose output
VERBOSE = True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_database_path():
    """
    Get the correct database path, trying alternatives if needed.

    Returns:
        str: Valid database path or None if not found
    """
    # Try main path first
    if os.path.exists(DATABASE_PATH):
        return DATABASE_PATH

    # Try alternative paths
    for alt_path in ALTERNATIVE_PATHS:
        if os.path.exists(alt_path):
            print(f"Using alternative database path: {alt_path}")
            return alt_path

    print("Warning: Kimia database not found in any expected location")
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

    # Check test size
    if not 0 < TEST_SIZE < 1:
        errors.append("TEST_SIZE must be between 0 and 1")

    # Check number of features
    if NUM_FEATURES != len(FEATURE_NAMES):
        errors.append("NUM_FEATURES must match length of FEATURE_NAMES")

    # Check classes
    if len(KIMIA_CLASSES) == 0:
        errors.append("KIMIA_CLASSES cannot be empty")

    # Check SVM config
    if SVM_CONFIG['C'] <= 0:
        errors.append("SVM C parameter must be positive")

    # Check MLP config
    if MLP_CONFIG['max_iter'] <= 0:
        errors.append("MLP max_iter must be positive")

    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config_summary():
    """
    Print a summary of the current configuration.
    """
    print("TP 11 - Configuration Summary")
    print("=" * 40)
    print(f"Database path: {get_database_path()}")
    print(f"Number of classes: {len(KIMIA_CLASSES)}")
    print(f"Images per class: {IMAGES_PER_CLASS}")
    print(f"Number of features: {NUM_FEATURES}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"SVM kernel: {SVM_CONFIG['kernel']}")
    print(f"MLP hidden layers: {MLP_CONFIG['hidden_layer_sizes']}")
    print("=" * 40)

if __name__ == "__main__":
    print_config_summary()
    if validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
