#!/usr/bin/env python3
"""
TP 13 - Pyramidal Decomposition and Reconstruction
Implementation of Gaussian and Laplacian pyramids for multiscale image analysis.

This module implements the algorithms described in the TP13 specification:
- Algorithm 3: Pyramidal decomposition
- Algorithm 4: Pyramidal reconstruction

Author: TP13 Implementation
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import rescale, resize
from skimage.filters import gaussian
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings

from config import *

class LaplacianPyramidDecomposition:
    """
    Implementation of Laplacian pyramid decomposition and reconstruction.

    This class provides methods for:
    - Gaussian pyramid construction
    - Laplacian pyramid decomposition
    - Perfect reconstruction from pyramid
    - Error analysis and quality metrics
    """

    def __init__(self, levels=DEFAULT_PYRAMID_LEVELS, sigma=GAUSSIAN_SIGMA):
        """
        Initialize the pyramid decomposition.

        Args:
            levels (int): Number of pyramid levels
            sigma (float): Gaussian filter standard deviation
        """
        self.levels = levels
        self.sigma = sigma
        self.gaussian_pyramid = []
        self.laplacian_pyramid = []

    def decompose(self, image):
        """
        Perform Laplacian pyramid decomposition.

        Implementation of Algorithm 3 from the specification:
        Data: image A₀
        Result: pyramid of approximations {Aᵢ}, pyramid of details {Dᵢ}
        for i=1 to 3 do
            filtering: F = filt(Aᵢ₋₁);
            subsampling: Aᵢ = ech(F, 0.5);
            details: Dᵢ = Aᵢ₋₁ - ech(Aᵢ, 2);
        end

        Args:
            image (np.ndarray): Input image

        Returns:
            tuple: (gaussian_pyramid, laplacian_pyramid)
        """
        # Ensure image is float and normalized
        if image.dtype != IMAGE_DTYPE:
            image = image.astype(IMAGE_DTYPE)

        if NORMALIZE_INPUT:
            image = image / np.max(image) if np.max(image) > 0 else image

        # Initialize pyramids
        self.gaussian_pyramid = [image.copy()]
        self.laplacian_pyramid = []

        current_image = image.copy()

        if VERBOSE:
            print(f"Starting pyramidal decomposition with {self.levels} levels")
            print(f"Original image shape: {image.shape}")

        # Build Gaussian pyramid and compute Laplacian details
        for level in range(self.levels):
            if VERBOSE:
                print(f"Processing level {level + 1}/{self.levels}")

            # Step 1: Gaussian filtering
            filtered = gaussian(current_image, sigma=self.sigma,
                              truncate=GAUSSIAN_TRUNCATE, preserve_range=True)

            # Step 2: Subsampling (ech(F, 0.5))
            subsampled = rescale(filtered, SUBSAMPLING_FACTOR,
                               order=INTERPOLATION_ORDER,
                               preserve_range=True,
                               anti_aliasing=True)

            # Add to Gaussian pyramid
            self.gaussian_pyramid.append(subsampled)

            # Step 3: Compute details (Dᵢ = Aᵢ₋₁ - ech(Aᵢ, 2))
            # Upsample the subsampled image back to original size
            upsampled = resize(subsampled, current_image.shape,
                             order=INTERPOLATION_ORDER,
                             preserve_range=True,
                             anti_aliasing=True)

            # Compute detail (residue)
            detail = current_image - upsampled
            self.laplacian_pyramid.append(detail)

            # Update current image for next iteration
            current_image = subsampled

            if VERBOSE:
                print(f"  Level {level + 1}: {current_image.shape} -> {subsampled.shape}")

        if VERBOSE:
            print("Pyramidal decomposition completed")
            print(f"Gaussian pyramid levels: {len(self.gaussian_pyramid)}")
            print(f"Laplacian pyramid levels: {len(self.laplacian_pyramid)}")

        return self.gaussian_pyramid, self.laplacian_pyramid

    def reconstruct(self, gaussian_pyramid=None, laplacian_pyramid=None,
                   interpolation='bilinear'):
        """
        Reconstruct image from Laplacian pyramid.

        Implementation of Algorithm 4 from the specification:
        Data: image A₃, pyramid of details {Dᵢ}
        Result: reconstructed pyramid {Bᵢ}
        initialization: B₃ = A₃;
        for i=3 to 1 do
            oversampling: R = ech(Bᵢ, 2);
            adding details: Bᵢ₋₁ = R + Dᵢ
        end

        Args:
            gaussian_pyramid (list): Gaussian pyramid (optional, uses stored if None)
            laplacian_pyramid (list): Laplacian pyramid (optional, uses stored if None)
            interpolation (str): Interpolation method for reconstruction

        Returns:
            np.ndarray: Reconstructed image
        """
        if gaussian_pyramid is None:
            gaussian_pyramid = self.gaussian_pyramid
        if laplacian_pyramid is None:
            laplacian_pyramid = self.laplacian_pyramid

        if not gaussian_pyramid or not laplacian_pyramid:
            raise ValueError("No pyramid data available. Run decompose() first.")

        if VERBOSE:
            print("Starting pyramidal reconstruction")

        # Initialize with the coarsest level (B₃ = A₃)
        reconstructed = gaussian_pyramid[-1].copy()

        if VERBOSE:
            print(f"Starting from coarsest level: {reconstructed.shape}")

        # Reconstruct from coarse to fine (i=3 to 1)
        for level in range(len(laplacian_pyramid) - 1, -1, -1):
            if VERBOSE:
                print(f"Reconstructing level {level}")

            # Get target shape from the corresponding detail level
            target_shape = laplacian_pyramid[level].shape

            # Step 1: Oversampling (R = ech(Bᵢ, 2))
            upsampled = resize(reconstructed, target_shape,
                             order=INTERPOLATION_ORDER,
                             preserve_range=True,
                             anti_aliasing=True)

            # Step 2: Adding details (Bᵢ₋₁ = R + Dᵢ)
            reconstructed = upsampled + laplacian_pyramid[level]

            if VERBOSE:
                print(f"  Level {level}: {upsampled.shape} + details -> {reconstructed.shape}")

        # Clip output if requested
        if CLIP_OUTPUT:
            reconstructed = np.clip(reconstructed, 0, 1)

        if VERBOSE:
            print("Pyramidal reconstruction completed")
            print(f"Final reconstructed image shape: {reconstructed.shape}")

        return reconstructed

    def reconstruct_without_details(self, gaussian_pyramid=None):
        """
        Reconstruct image without adding details (smooth reconstruction).

        This creates a smooth version by only upsampling the coarsest level
        without adding the detail information.

        Args:
            gaussian_pyramid (list): Gaussian pyramid (optional, uses stored if None)

        Returns:
            np.ndarray: Smooth reconstructed image
        """
        if gaussian_pyramid is None:
            gaussian_pyramid = self.gaussian_pyramid

        if not gaussian_pyramid:
            raise ValueError("No Gaussian pyramid data available.")

        # Start from coarsest level
        smooth_reconstruction = gaussian_pyramid[-1].copy()

        # Get target shape from original image
        target_shape = gaussian_pyramid[0].shape

        # Simple upsampling to original size
        smooth_reconstruction = resize(smooth_reconstruction, target_shape,
                                     order=INTERPOLATION_ORDER,
                                     preserve_range=True,
                                     anti_aliasing=True)

        if CLIP_OUTPUT:
            smooth_reconstruction = np.clip(smooth_reconstruction, 0, 1)

        return smooth_reconstruction

    def compute_reconstruction_error(self, original, reconstructed):
        """
        Compute various error metrics between original and reconstructed images.

        Args:
            original (np.ndarray): Original image
            reconstructed (np.ndarray): Reconstructed image

        Returns:
            dict: Dictionary containing error metrics
        """
        # Ensure same shape
        if original.shape != reconstructed.shape:
            reconstructed = resize(reconstructed, original.shape,
                                 preserve_range=True, anti_aliasing=True)

        # Ensure same data type and range
        original = original.astype(IMAGE_DTYPE)
        reconstructed = reconstructed.astype(IMAGE_DTYPE)

        # Normalize to [0, 1] range for metrics
        orig_norm = (original - np.min(original)) / (np.max(original) - np.min(original))
        recon_norm = (reconstructed - np.min(reconstructed)) / (np.max(reconstructed) - np.min(reconstructed))

        errors = {}

        # Mean Squared Error
        errors['mse'] = np.mean((orig_norm - recon_norm) ** 2)

        # Mean Absolute Error
        errors['mae'] = np.mean(np.abs(orig_norm - recon_norm))

        # Peak Signal-to-Noise Ratio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            errors['psnr'] = peak_signal_noise_ratio(orig_norm, recon_norm,
                                                   data_range=1.0)

        # Structural Similarity Index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            errors['ssim'] = structural_similarity(orig_norm, recon_norm,
                                                 win_size=min(SSIM_WIN_SIZE,
                                                            min(orig_norm.shape) // 2),
                                                 data_range=1.0)

        # Maximum absolute difference
        errors['max_diff'] = np.max(np.abs(orig_norm - recon_norm))

        return errors

def LaplacianPyramidDecomposition_function(image, levels):
    """
    Function implementation matching the specification format.

    This function replicates the exact algorithm shown in the specification.

    Args:
        image (np.ndarray): Original image, float32
        levels (int): Number of levels of decomposition

    Returns:
        tuple: (pyrL, pyrG) - Laplacian and Gaussian pyramids as lists of arrays
    """
    pyrL = []
    pyrG = []

    sigma = 3.0
    Image = image.copy()

    for i in range(levels):
        prevImage = Image.copy()
        g = ndimage.gaussian_filter(Image, sigma)

        Image = rescale(g, 0.5, preserve_range=True, anti_aliasing=True)
        primeImage = resize(Image, prevImage.shape, preserve_range=True, anti_aliasing=True)

        pyrL.append(prevImage - primeImage)
        pyrG.append(prevImage)

    pyrL.append(Image)
    pyrG.append(Image)

    return pyrL, pyrG


def LaplacianPyramidReconstruction(pyr, interp='bilinear'):
    """
    Reconstruction of the Laplacian pyramid, starting from the last image.

    This function replicates the exact algorithm shown in the specification.

    Args:
        pyr (list): Pyramid of images (list of arrays)
        interp (str): Interpolation mode for upsizing the image

    Returns:
        np.ndarray: Reconstructed image
    """
    Image = pyr[-1].copy()

    for i in range(len(pyr) - 2, -1, -1):
        Image = pyr[i] + resize(Image, pyr[i].shape, preserve_range=True, anti_aliasing=True)

    return Image
