#!/usr/bin/env python3
"""
TP 13 - Visualization Module
Professional visualization tools for multiscale analysis results.

This module provides comprehensive visualization capabilities for:
- Pyramidal decomposition results
- Morphological multiscale analysis
- Reconstruction comparisons and error analysis
- Publication-quality figures

Author: TP13 Implementation
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import warnings

from config import *

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

class MultiscaleVisualizer:
    """
    Professional visualization class for multiscale analysis results.
    """

    def __init__(self, save_figures=SAVE_FIGURES, dpi=FIGURE_DPI, format=FIGURE_FORMAT):
        """
        Initialize the visualizer.

        Args:
            save_figures (bool): Whether to save figures
            dpi (int): Figure DPI for saving
            format (str): Figure format for saving
        """
        self.save_figures = save_figures
        self.dpi = dpi
        self.format = format

        # Create output directory
        create_output_directory()

    def visualize_pyramid_decomposition(self, gaussian_pyramid, laplacian_pyramid,
                                      original_image=None, title="Pyramidal Decomposition"):
        """
        Visualize Gaussian and Laplacian pyramid decomposition.

        Args:
            gaussian_pyramid (list): Gaussian pyramid levels
            laplacian_pyramid (list): Laplacian pyramid levels
            original_image (np.ndarray): Original image (optional)
            title (str): Figure title
        """
        n_levels = len(gaussian_pyramid)

        # Create figure with custom layout
        fig = plt.figure(figsize=PYRAMID_FIGURE_SIZE)
        gs = gridspec.GridSpec(3, n_levels, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.2)

        # Original image (if provided)
        if original_image is not None:
            ax_orig = fig.add_subplot(gs[0, :2])
            ax_orig.imshow(original_image, cmap=PYRAMID_COLORMAP)
            ax_orig.set_title("(a) Original Image", fontsize=LABEL_FONTSIZE, fontweight='bold')
            ax_orig.axis('off')

        # Gaussian pyramid
        for i, img in enumerate(gaussian_pyramid):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(img, cmap=PYRAMID_COLORMAP)
            if i == 0:
                ax.set_title(f"(b) Gaussian Pyramid Level {i}", fontsize=LABEL_FONTSIZE)
            else:
                ax.set_title(f"Level {i}", fontsize=LABEL_FONTSIZE)
            ax.axis('off')

            # Add size annotation
            ax.text(0.02, 0.98, f"{img.shape[0]}×{img.shape[1]}",
                   transform=ax.transAxes, fontsize=TICK_FONTSIZE-1,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor='white', alpha=0.8))

        # Laplacian pyramid (details)
        for i, img in enumerate(laplacian_pyramid):
            ax = fig.add_subplot(gs[2, i])

            # Normalize for better visualization
            img_norm = img - np.min(img)
            if np.max(img_norm) > 0:
                img_norm = img_norm / np.max(img_norm)

            ax.imshow(img_norm, cmap=DIFFERENCE_COLORMAP)
            if i == 0:
                ax.set_title(f"(c) Laplacian Pyramid (Details) Level {i}", fontsize=LABEL_FONTSIZE)
            else:
                ax.set_title(f"Level {i}", fontsize=LABEL_FONTSIZE)
            ax.axis('off')

        plt.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight='bold')

        if self.save_figures:
            output_path = get_output_path('pyramid_decomposition')
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       format=self.format, facecolor='white')
            if VERBOSE:
                print(f"Saved pyramid decomposition visualization: {output_path}")

        plt.show()

    def visualize_reconstruction_comparison(self, original, reconstructed,
                                          reconstructed_smooth=None):
        """
        Visualize reconstruction comparison with error analysis.

        Args:
            original (np.ndarray): Original image
            reconstructed (np.ndarray): Reconstructed image with details
            reconstructed_smooth (np.ndarray): Reconstructed without details (optional)
        """
        if reconstructed_smooth is not None:
            fig, axes = plt.subplots(2, 3, figsize=COMPARISON_FIGURE_SIZE)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()

        # Original image
        axes[0].imshow(original, cmap=PYRAMID_COLORMAP)
        axes[0].set_title("(a) Original Image", fontsize=LABEL_FONTSIZE, fontweight='bold')
        axes[0].axis('off')

        # Reconstructed with details
        axes[1].imshow(reconstructed, cmap=PYRAMID_COLORMAP)
        axes[1].set_title("(b) Reconstruction with Details", fontsize=LABEL_FONTSIZE, fontweight='bold')
        axes[1].axis('off')

        # Error map
        error_map = np.abs(original - reconstructed)
        im_error = axes[2].imshow(error_map, cmap=ERROR_COLORMAP)
        axes[2].set_title("(c) Reconstruction Error", fontsize=LABEL_FONTSIZE, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im_error, ax=axes[2], shrink=0.8)

        if reconstructed_smooth is not None:
            # Smooth reconstruction
            axes[3].imshow(reconstructed_smooth, cmap=PYRAMID_COLORMAP)
            axes[3].set_title("(d) Reconstruction without Details", fontsize=LABEL_FONTSIZE, fontweight='bold')
            axes[3].axis('off')

            # Difference between reconstructions
            diff_map = np.abs(reconstructed - reconstructed_smooth)
            im_diff = axes[4].imshow(diff_map, cmap=DIFFERENCE_COLORMAP)
            axes[4].set_title("(e) Detail Contribution", fontsize=LABEL_FONTSIZE, fontweight='bold')
            axes[4].axis('off')
            plt.colorbar(im_diff, ax=axes[4], shrink=0.8)

            # Error statistics
            axes[5].axis('off')
            self._add_error_statistics(axes[5], original, reconstructed, reconstructed_smooth)
        else:
            # Error statistics
            axes[3].axis('off')
            self._add_error_statistics(axes[3], original, reconstructed)

        plt.suptitle("Pyramidal Reconstruction Analysis", fontsize=SUPTITLE_FONTSIZE, fontweight='bold')
        plt.tight_layout()

        if self.save_figures:
            output_path = get_output_path('reconstruction_comparison')
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       format=self.format, facecolor='white')
            if VERBOSE:
                print(f"Saved reconstruction comparison: {output_path}")

        plt.show()

    def visualize_morphological_decomposition(self, scale_space, title="Morphological Scale-Space"):
        """
        Visualize morphological multiscale decomposition.

        Args:
            scale_space (list): Scale space decomposition
            title (str): Figure title
        """
        n_levels = len(scale_space)
        cols = min(5, n_levels)
        rows = (n_levels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, img in enumerate(scale_space):
            row, col = i // cols, i % cols

            axes[row, col].imshow(img, cmap=PYRAMID_COLORMAP)
            if i == 0:
                axes[row, col].set_title("Original", fontsize=LABEL_FONTSIZE, fontweight='bold')
            else:
                axes[row, col].set_title(f"Level {i}", fontsize=LABEL_FONTSIZE)
            axes[row, col].axis('off')

        # Hide unused subplots
        for i in range(n_levels, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')

        plt.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight='bold')
        plt.tight_layout()

        if self.save_figures:
            output_path = get_output_path('morphological_decomposition')
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       format=self.format, facecolor='white')
            if VERBOSE:
                print(f"Saved morphological decomposition: {output_path}")

        plt.show()

    def visualize_error_analysis(self, error_metrics, levels=None):
        """
        Visualize error analysis across pyramid levels.

        Args:
            error_metrics (dict or list): Error metrics
            levels (list): Level names (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=ERROR_FIGURE_SIZE)
        axes = axes.flatten()

        if isinstance(error_metrics, dict):
            # Single reconstruction error
            metrics = error_metrics
            x_labels = list(metrics.keys())
            values = list(metrics.values())

            axes[0].bar(x_labels, values, color='skyblue', alpha=0.7)
            axes[0].set_title("Reconstruction Error Metrics", fontweight='bold')
            axes[0].set_ylabel("Error Value")
            axes[0].tick_params(axis='x', rotation=45)

            # Hide other subplots
            for i in range(1, 4):
                axes[i].axis('off')

        else:
            # Multiple level analysis
            if levels is None:
                levels = [f"Level {i}" for i in range(len(error_metrics))]

            # Extract metrics
            mse_values = [m.get('mse', 0) for m in error_metrics]
            psnr_values = [m.get('psnr', 0) for m in error_metrics]
            ssim_values = [m.get('ssim', 0) for m in error_metrics]
            mae_values = [m.get('mae', 0) for m in error_metrics]

            # MSE plot
            axes[0].plot(levels, mse_values, 'o-', linewidth=2, markersize=6)
            axes[0].set_title("Mean Squared Error", fontweight='bold')
            axes[0].set_ylabel("MSE")
            axes[0].grid(True, alpha=0.3)

            # PSNR plot
            axes[1].plot(levels, psnr_values, 's-', linewidth=2, markersize=6, color='orange')
            axes[1].set_title("Peak Signal-to-Noise Ratio", fontweight='bold')
            axes[1].set_ylabel("PSNR (dB)")
            axes[1].grid(True, alpha=0.3)

            # SSIM plot
            axes[2].plot(levels, ssim_values, '^-', linewidth=2, markersize=6, color='green')
            axes[2].set_title("Structural Similarity Index", fontweight='bold')
            axes[2].set_ylabel("SSIM")
            axes[2].grid(True, alpha=0.3)

            # MAE plot
            axes[3].plot(levels, mae_values, 'd-', linewidth=2, markersize=6, color='red')
            axes[3].set_title("Mean Absolute Error", fontweight='bold')
            axes[3].set_ylabel("MAE")
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_figures:
            output_path = get_output_path('error_analysis')
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       format=self.format, facecolor='white')
            if VERBOSE:
                print(f"Saved error analysis: {output_path}")

        plt.show()

    def visualize_multiscale_comparison(self, pyramidal_results, morphological_results):
        """
        Compare pyramidal and morphological multiscale methods.

        Args:
            pyramidal_results (dict): Results from pyramidal decomposition
            morphological_results (dict): Results from morphological decomposition
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(pyramidal_results['original'], cmap=PYRAMID_COLORMAP)
        axes[0, 0].set_title("Original Image", fontweight='bold')
        axes[0, 0].axis('off')

        # Pyramidal reconstruction
        axes[0, 1].imshow(pyramidal_results['reconstructed_with_details'], cmap=PYRAMID_COLORMAP)
        axes[0, 1].set_title("Pyramidal Reconstruction", fontweight='bold')
        axes[0, 1].axis('off')

        # Morphological result
        axes[0, 2].imshow(morphological_results['filtered'], cmap=PYRAMID_COLORMAP)
        axes[0, 2].set_title("Morphological Filtering", fontweight='bold')
        axes[0, 2].axis('off')

        # Error comparisons
        pyr_error = np.abs(pyramidal_results['original'] - pyramidal_results['reconstructed_with_details'])
        morph_error = np.abs(pyramidal_results['original'] - morphological_results['filtered'])

        im1 = axes[1, 0].imshow(pyr_error, cmap=ERROR_COLORMAP)
        axes[1, 0].set_title("Pyramidal Error", fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)

        im2 = axes[1, 1].imshow(morph_error, cmap=ERROR_COLORMAP)
        axes[1, 1].set_title("Morphological Error", fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)

        # Comparison statistics
        axes[1, 2].axis('off')
        self._add_comparison_statistics(axes[1, 2], pyramidal_results, morphological_results)

        plt.suptitle("Multiscale Methods Comparison", fontsize=SUPTITLE_FONTSIZE, fontweight='bold')
        plt.tight_layout()

        if self.save_figures:
            output_path = get_output_path('multiscale_comparison')
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       format=self.format, facecolor='white')
            if VERBOSE:
                print(f"Saved multiscale comparison: {output_path}")

        plt.show()

    def _add_error_statistics(self, ax, original, reconstructed, smooth=None):
        """Add error statistics text to an axis."""
        # Compute metrics
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        max_error = np.max(np.abs(original - reconstructed))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psnr = peak_signal_noise_ratio(original, reconstructed, data_range=1.0)
            ssim = structural_similarity(original, reconstructed, data_range=1.0)

        stats_text = f"""Error Statistics:

MSE: {mse:.6f}
MAE: {mae:.6f}
Max Error: {max_error:.6f}
PSNR: {psnr:.2f} dB
SSIM: {ssim:.4f}"""

        if smooth is not None:
            mse_smooth = np.mean((original - smooth) ** 2)
            stats_text += f"\n\nSmooth Reconstruction:\nMSE: {mse_smooth:.6f}"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=TICK_FONTSIZE, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

    def _add_comparison_statistics(self, ax, pyr_results, morph_results):
        """Add comparison statistics text to an axis."""
        pyr_mse = np.mean((pyr_results['original'] - pyr_results['reconstructed_with_details']) ** 2)
        morph_mse = np.mean((pyr_results['original'] - morph_results['filtered']) ** 2)

        stats_text = f"""Method Comparison:

Pyramidal MSE: {pyr_mse:.6f}
Morphological MSE: {morph_mse:.6f}

Better Method: {'Pyramidal' if pyr_mse < morph_mse else 'Morphological'}
Improvement: {abs(pyr_mse - morph_mse):.6f}"""

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=TICK_FONTSIZE, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

# Convenience functions
def visualize_pyramid(gaussian_pyramid, laplacian_pyramid, original=None):
    """Convenience function for pyramid visualization."""
    visualizer = MultiscaleVisualizer()
    visualizer.visualize_pyramid_decomposition(gaussian_pyramid, laplacian_pyramid, original)

def visualize_reconstruction(original, reconstructed, smooth=None):
    """Convenience function for reconstruction visualization."""
    visualizer = MultiscaleVisualizer()
    visualizer.visualize_reconstruction_comparison(original, reconstructed, smooth)

def visualize_morphological(scale_space):
    """Convenience function for morphological visualization."""
    visualizer = MultiscaleVisualizer()
    visualizer.visualize_morphological_decomposition(scale_space)
