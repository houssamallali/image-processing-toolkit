#!/usr/bin/env python3
"""
TP 13 - Multiscale Analysis Test Suite
Comprehensive testing for pyramidal decomposition and morphological multiscale analysis.

This module provides thorough testing of:
- Pyramidal decomposition and reconstruction algorithms
- Morphological multiscale decomposition
- Error analysis and quality metrics
- Visualization components

Author: TP13 Implementation
Date: 2024
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.morphology import disk
import warnings

# Import modules to test
from config import *
from pyramidal_decomposition import (
    LaplacianPyramidDecomposition,
    LaplacianPyramidDecomposition_function,
    LaplacianPyramidReconstruction
)
from morphological_multiscale import MorphologicalDecomposition
from visualization import MultiscaleVisualizer
from main import load_cerveau_image

class TestPyramidalDecomposition(unittest.TestCase):
    """Test cases for pyramidal decomposition and reconstruction."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test image
        self.test_image = np.random.rand(64, 64).astype(IMAGE_DTYPE)
        self.decomposer = LaplacianPyramidDecomposition(levels=3)

        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore")

    def test_decomposition_basic(self):
        """Test basic decomposition functionality."""
        gaussian_pyramid, laplacian_pyramid = self.decomposer.decompose(self.test_image)

        # Check pyramid structure
        self.assertEqual(len(gaussian_pyramid), 4)  # 3 levels + original
        self.assertEqual(len(laplacian_pyramid), 3)  # 3 detail levels

        # Check that each level is smaller than the previous
        for i in range(1, len(gaussian_pyramid)):
            self.assertLess(gaussian_pyramid[i].size, gaussian_pyramid[i-1].size)

    def test_perfect_reconstruction(self):
        """Test that Laplacian pyramid enables perfect reconstruction."""
        gaussian_pyramid, laplacian_pyramid = self.decomposer.decompose(self.test_image)
        reconstructed = self.decomposer.reconstruct()

        # Check reconstruction quality
        mse = np.mean((self.test_image - reconstructed) ** 2)
        self.assertLess(mse, 1e-6, "Perfect reconstruction should have very low MSE")

    def test_reconstruction_without_details(self):
        """Test smooth reconstruction without details."""
        gaussian_pyramid, laplacian_pyramid = self.decomposer.decompose(self.test_image)
        smooth_reconstruction = self.decomposer.reconstruct_without_details()

        # Smooth reconstruction should be different from original
        mse = np.mean((self.test_image - smooth_reconstruction) ** 2)
        self.assertGreater(mse, 1e-6, "Smooth reconstruction should differ from original")

        # Should have same shape as original
        self.assertEqual(smooth_reconstruction.shape, self.test_image.shape)

    def test_error_metrics(self):
        """Test error metric computation."""
        gaussian_pyramid, laplacian_pyramid = self.decomposer.decompose(self.test_image)
        reconstructed = self.decomposer.reconstruct()

        error_metrics = self.decomposer.compute_reconstruction_error(
            self.test_image, reconstructed
        )

        # Check that all expected metrics are present
        expected_metrics = ['mse', 'mae', 'psnr', 'ssim', 'max_diff']
        for metric in expected_metrics:
            self.assertIn(metric, error_metrics)
            self.assertIsInstance(error_metrics[metric], (int, float))

    def test_specification_functions(self):
        """Test the specification-format functions."""
        pyrL, pyrG = LaplacianPyramidDecomposition_function(self.test_image, 3)
        reconstructed = LaplacianPyramidReconstruction(pyrL)

        # Check pyramid structure
        self.assertEqual(len(pyrL), 4)  # 3 levels + coarsest
        self.assertEqual(len(pyrG), 4)  # 3 levels + coarsest

        # Check reconstruction shape
        self.assertEqual(reconstructed.shape, self.test_image.shape)

    def test_different_levels(self):
        """Test decomposition with different numbers of levels."""
        for levels in [1, 2, 4, 5]:
            decomposer = LaplacianPyramidDecomposition(levels=levels)
            gaussian_pyramid, laplacian_pyramid = decomposer.decompose(self.test_image)

            self.assertEqual(len(gaussian_pyramid), levels + 1)
            self.assertEqual(len(laplacian_pyramid), levels)

class TestMorphologicalDecomposition(unittest.TestCase):
    """Test cases for morphological multiscale decomposition."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.rand(64, 64).astype(IMAGE_DTYPE)
        self.se = disk(3)
        self.decomposer = MorphologicalDecomposition(self.se)

        warnings.filterwarnings("ignore")

    def test_scale_space_decomposition(self):
        """Test basic scale-space decomposition."""
        scale_space = self.decomposer.decompose(self.test_image, levels=3)

        # Check structure
        self.assertEqual(len(scale_space), 4)  # Original + 3 levels

        # All images should have same shape
        for img in scale_space:
            self.assertEqual(img.shape, self.test_image.shape)

    def test_opening_closing_decomposition(self):
        """Test opening and closing decomposition."""
        opening_space, closing_space = self.decomposer.decompose_with_closing(
            self.test_image, levels=3
        )

        # Check structures
        self.assertEqual(len(opening_space), 4)
        self.assertEqual(len(closing_space), 4)

        # All images should have same shape
        for img in opening_space + closing_space:
            self.assertEqual(img.shape, self.test_image.shape)

    def test_kramer_bruckner_filter(self):
        """Test Kramer-Bruckner filter."""
        kb_sequence = self.decomposer.kramer_bruckner_filter(
            self.test_image, iterations=3
        )

        # Check sequence
        self.assertGreaterEqual(len(kb_sequence), 2)  # At least original + 1 iteration

        # All images should have same shape
        for img in kb_sequence:
            self.assertEqual(img.shape, self.test_image.shape)

    def test_residue_computation(self):
        """Test residue computation."""
        scale_space = self.decomposer.decompose(self.test_image, levels=3)
        residues = self.decomposer.compute_residues(self.test_image, scale_space)

        # Should have one less residue than scale space levels
        self.assertEqual(len(residues), len(scale_space) - 1)

        # All residues should have same shape as original
        for residue in residues:
            self.assertEqual(residue.shape, self.test_image.shape)

    def test_scale_space_analysis(self):
        """Test scale space analysis."""
        analysis = self.decomposer.analyze_scale_space(self.test_image, max_levels=3)

        # Check analysis structure
        expected_keys = ['levels', 'energy', 'entropy', 'contrast',
                        'mean_intensity', 'std_intensity']
        for key in expected_keys:
            self.assertIn(key, analysis)

        # Check that we have the right number of measurements
        self.assertEqual(analysis['levels'], 4)  # Original + 3 levels
        self.assertEqual(len(analysis['energy']), 4)

    def test_different_structuring_elements(self):
        """Test with different structuring elements."""
        se_types = ['disk', 'square', 'diamond']

        for se_type in se_types:
            decomposer = MorphologicalDecomposition(se_type=se_type, radius=2)
            scale_space = decomposer.decompose(self.test_image, levels=2)

            self.assertEqual(len(scale_space), 3)  # Original + 2 levels

class TestVisualization(unittest.TestCase):
    """Test cases for visualization components."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.rand(32, 32).astype(IMAGE_DTYPE)
        self.visualizer = MultiscaleVisualizer(save_figures=False)  # Don't save during tests

        warnings.filterwarnings("ignore")
        plt.ioff()  # Turn off interactive plotting for tests

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        self.assertIsInstance(self.visualizer, MultiscaleVisualizer)
        self.assertFalse(self.visualizer.save_figures)

    def test_pyramid_visualization(self):
        """Test pyramid visualization (without displaying)."""
        decomposer = LaplacianPyramidDecomposition(levels=2)
        gaussian_pyramid, laplacian_pyramid = decomposer.decompose(self.test_image)

        # This should not raise an exception
        try:
            self.visualizer.visualize_pyramid_decomposition(
                gaussian_pyramid, laplacian_pyramid, self.test_image
            )
            plt.close('all')  # Clean up
        except Exception as e:
            self.fail(f"Pyramid visualization failed: {e}")

    def test_reconstruction_visualization(self):
        """Test reconstruction comparison visualization."""
        decomposer = LaplacianPyramidDecomposition(levels=2)
        gaussian_pyramid, laplacian_pyramid = decomposer.decompose(self.test_image)
        reconstructed = decomposer.reconstruct()
        smooth = decomposer.reconstruct_without_details()

        try:
            self.visualizer.visualize_reconstruction_comparison(
                self.test_image, reconstructed, smooth
            )
            plt.close('all')
        except Exception as e:
            self.fail(f"Reconstruction visualization failed: {e}")

    def test_morphological_visualization(self):
        """Test morphological decomposition visualization."""
        decomposer = MorphologicalDecomposition(disk(2))
        scale_space = decomposer.decompose(self.test_image, levels=2)

        try:
            self.visualizer.visualize_morphological_decomposition(scale_space)
            plt.close('all')
        except Exception as e:
            self.fail(f"Morphological visualization failed: {e}")

class TestConfiguration(unittest.TestCase):
    """Test cases for configuration management."""

    def test_config_validation(self):
        """Test configuration validation."""
        self.assertTrue(validate_config(), "Configuration should be valid")

    def test_output_directory_creation(self):
        """Test output directory creation."""
        create_output_directory()
        self.assertTrue(os.path.exists(OUTPUT_DIR))

    def test_image_path_detection(self):
        """Test image path detection."""
        # This might return None if no image is found, which is acceptable
        image_path = get_image_path()
        if image_path is not None:
            self.assertTrue(os.path.exists(image_path))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up integration test fixtures."""
        warnings.filterwarnings("ignore")
        plt.ioff()

    def test_cerveau_image_loading(self):
        """Test cerveau image loading functionality."""
        cerveau_image = load_cerveau_image()

        self.assertIsInstance(cerveau_image, np.ndarray)
        self.assertEqual(len(cerveau_image.shape), 2)  # Should be grayscale
        self.assertEqual(cerveau_image.dtype, IMAGE_DTYPE)

    def test_complete_pyramidal_pipeline(self):
        """Test complete pyramidal analysis pipeline."""
        # Use a small test image for speed
        test_image = np.random.rand(32, 32).astype(IMAGE_DTYPE)

        # Run pyramidal decomposition
        decomposer = LaplacianPyramidDecomposition(levels=2)
        gaussian_pyramid, laplacian_pyramid = decomposer.decompose(test_image)

        # Reconstruct
        reconstructed = decomposer.reconstruct()

        # Compute errors
        errors = decomposer.compute_reconstruction_error(test_image, reconstructed)

        # Check that reconstruction is nearly perfect
        self.assertLess(errors['mse'], 1e-6)

    def test_complete_morphological_pipeline(self):
        """Test complete morphological analysis pipeline."""
        test_image = np.random.rand(32, 32).astype(IMAGE_DTYPE)

        # Run morphological decomposition
        decomposer = MorphologicalDecomposition(disk(2))
        scale_space = decomposer.decompose(test_image, levels=2)

        # Run Kramer-Bruckner filter
        kb_sequence = decomposer.kramer_bruckner_filter(test_image, iterations=2)

        # Analyze scale space
        analysis = decomposer.analyze_scale_space(test_image, max_levels=2)

        # Basic checks
        self.assertEqual(len(scale_space), 3)
        self.assertGreaterEqual(len(kb_sequence), 2)
        self.assertIn('energy', analysis)

    def tearDown(self):
        """Clean up after integration tests."""
        plt.close('all')

def run_performance_tests():
    """Run performance tests for the algorithms."""
    print("\nRunning performance tests...")

    import time

    # Test with different image sizes
    sizes = [64, 128, 256]

    for size in sizes:
        print(f"\nTesting with {size}x{size} image:")

        test_image = np.random.rand(size, size).astype(IMAGE_DTYPE)

        # Pyramidal decomposition timing
        start_time = time.time()
        decomposer = LaplacianPyramidDecomposition(levels=3)
        gaussian_pyramid, laplacian_pyramid = decomposer.decompose(test_image)
        reconstructed = decomposer.reconstruct()
        pyramidal_time = time.time() - start_time

        # Morphological decomposition timing
        start_time = time.time()
        morph_decomposer = MorphologicalDecomposition(disk(3))
        scale_space = morph_decomposer.decompose(test_image, levels=3)
        morphological_time = time.time() - start_time

        print(f"  Pyramidal decomposition: {pyramidal_time:.3f}s")
        print(f"  Morphological decomposition: {morphological_time:.3f}s")

def run_all_tests():
    """Run all test suites."""
    print("TP 13 - Multiscale Analysis Test Suite")
    print("="*50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestPyramidalDecomposition,
        TestMorphologicalDecomposition,
        TestVisualization,
        TestConfiguration,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Run performance tests
    run_performance_tests()

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")

    return success

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
