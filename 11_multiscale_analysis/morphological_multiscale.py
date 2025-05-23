#!/usr/bin/env python3
"""
TP 13 - Morphological Multiscale Decomposition
Implementation of scale-space decomposition using morphological operators.

This module implements:
- Morphological multiscale decomposition with dilation/erosion
- Kramer and Bruckner multiscale filter
- Scale-space analysis without sampling changes

Author: TP13 Implementation
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import disk, square, diamond, dilation, erosion
from scipy import ndimage
import warnings

from config import *

class MorphologicalDecomposition:
    """
    Implementation of morphological multiscale decomposition.
    
    This class provides methods for:
    - Scale-space decomposition using morphological operators
    - Kramer-Bruckner filter implementation
    - Multi-scale analysis with different structuring elements
    """
    
    def __init__(self, structuring_element=None, se_type=DEFAULT_SE_TYPE, 
                 radius=DEFAULT_SE_RADIUS):
        """
        Initialize the morphological decomposition.
        
        Args:
            structuring_element (np.ndarray): Custom structuring element
            se_type (str): Type of structuring element ('disk', 'square', 'diamond')
            radius (int): Radius of the structuring element
        """
        if structuring_element is not None:
            self.structuring_element = structuring_element
        else:
            self.structuring_element = self._create_structuring_element(se_type, radius)
        
        self.se_type = se_type
        self.radius = radius
        self.scale_space = []
        
    def _create_structuring_element(self, se_type, radius):
        """
        Create a structuring element of specified type and radius.
        
        Args:
            se_type (str): Type of structuring element
            radius (int): Radius of the structuring element
            
        Returns:
            np.ndarray: Structuring element
        """
        if se_type == 'disk':
            return disk(radius)
        elif se_type == 'square':
            return square(2 * radius + 1)
        elif se_type == 'diamond':
            return diamond(radius)
        else:
            raise ValueError(f"Unknown structuring element type: {se_type}")
    
    def decompose(self, image, levels=DEFAULT_MORPHOLOGICAL_LEVELS, 
                  increasing_radius=True):
        """
        Perform morphological multiscale decomposition.
        
        Build scale-space decomposition with morphological operators.
        The resulting images will have the same size as the original.
        
        Args:
            image (np.ndarray): Input image
            levels (int): Number of decomposition levels
            increasing_radius (bool): Whether to increase radius at each level
            
        Returns:
            list: Scale-space decomposition (list of images)
        """
        # Ensure image is float and normalized
        if image.dtype != IMAGE_DTYPE:
            image = image.astype(IMAGE_DTYPE)
        
        if NORMALIZE_INPUT:
            image = image / np.max(image) if np.max(image) > 0 else image
        
        # Initialize scale space
        self.scale_space = [image.copy()]
        current_image = image.copy()
        
        if VERBOSE:
            print(f"Starting morphological decomposition with {levels} levels")
            print(f"Structuring element: {self.se_type} (radius {self.radius})")
            print(f"Original image shape: {image.shape}")
        
        # Build scale space
        for level in range(levels):
            if VERBOSE:
                print(f"Processing level {level + 1}/{levels}")
            
            # Create structuring element for this level
            if increasing_radius:
                current_radius = self.radius * (level + 1)
                se = self._create_structuring_element(self.se_type, current_radius)
            else:
                se = self.structuring_element
            
            # Apply morphological opening (erosion followed by dilation)
            eroded = erosion(current_image, se)
            opened = dilation(eroded, se)
            
            # Store the result
            self.scale_space.append(opened)
            current_image = opened
            
            if VERBOSE:
                print(f"  Level {level + 1}: SE radius {current_radius if increasing_radius else self.radius}")
        
        if VERBOSE:
            print("Morphological decomposition completed")
            print(f"Scale space levels: {len(self.scale_space)}")
        
        return self.scale_space
    
    def decompose_with_closing(self, image, levels=DEFAULT_MORPHOLOGICAL_LEVELS):
        """
        Perform morphological decomposition using both opening and closing.
        
        Args:
            image (np.ndarray): Input image
            levels (int): Number of decomposition levels
            
        Returns:
            tuple: (opening_scale_space, closing_scale_space)
        """
        # Opening scale space
        opening_space = self.decompose(image, levels)
        
        # Closing scale space
        closing_space = [image.copy()]
        current_image = image.copy()
        
        for level in range(levels):
            current_radius = self.radius * (level + 1)
            se = self._create_structuring_element(self.se_type, current_radius)
            
            # Apply morphological closing (dilation followed by erosion)
            dilated = dilation(current_image, se)
            closed = erosion(dilated, se)
            
            closing_space.append(closed)
            current_image = closed
        
        return opening_space, closing_space
    
    def kramer_bruckner_filter(self, image, iterations=KB_ITERATIONS):
        """
        Apply Kramer and Bruckner multiscale filter.
        
        The iterative filter is defined as:
        MK^n_B(f) = K_B(MK^{n-1}_B(f))
        
        where K_B(f)(x) = {
            D_B(f)(x)  if D_B(f)(x) - f ≤ f - E_B(f)(x)
            E_B(f)(x)  otherwise
        }
        
        Args:
            image (np.ndarray): Input image
            iterations (int): Number of iterations
            
        Returns:
            list: Sequence of filtered images
        """
        if VERBOSE:
            print(f"Applying Kramer-Bruckner filter with {iterations} iterations")
        
        # Ensure image is float and normalized
        if image.dtype != IMAGE_DTYPE:
            image = image.astype(IMAGE_DTYPE)
        
        if NORMALIZE_INPUT:
            image = image / np.max(image) if np.max(image) > 0 else image
        
        filtered_sequence = [image.copy()]
        current_image = image.copy()
        
        for iteration in range(iterations):
            if VERBOSE:
                print(f"Iteration {iteration + 1}/{iterations}")
            
            # Compute dilation and erosion
            dilated = dilation(current_image, self.structuring_element)
            eroded = erosion(current_image, self.structuring_element)
            
            # Apply Kramer-Bruckner rule
            # K_B(f)(x) = D_B(f)(x) if D_B(f)(x) - f ≤ f - E_B(f)(x), else E_B(f)(x)
            condition = (dilated - current_image) <= (current_image - eroded)
            filtered_image = np.where(condition, dilated, eroded)
            
            filtered_sequence.append(filtered_image)
            current_image = filtered_image
            
            # Check for convergence
            if iteration > 0:
                diff = np.mean(np.abs(filtered_sequence[-1] - filtered_sequence[-2]))
                if diff < KB_CONVERGENCE_THRESHOLD:
                    if VERBOSE:
                        print(f"Converged after {iteration + 1} iterations (diff: {diff:.2e})")
                    break
        
        return filtered_sequence
    
    def compute_residues(self, original, scale_space):
        """
        Compute residues (details) between consecutive levels.
        
        Args:
            original (np.ndarray): Original image
            scale_space (list): Scale space decomposition
            
        Returns:
            list: Residues between levels
        """
        residues = []
        
        for i in range(len(scale_space) - 1):
            residue = scale_space[i] - scale_space[i + 1]
            residues.append(residue)
        
        return residues
    
    def analyze_scale_space(self, image, max_levels=10):
        """
        Analyze the scale space properties of an image.
        
        Args:
            image (np.ndarray): Input image
            max_levels (int): Maximum number of levels to analyze
            
        Returns:
            dict: Analysis results including energy, entropy, and contrast
        """
        scale_space = self.decompose(image, max_levels)
        
        analysis = {
            'levels': len(scale_space),
            'energy': [],
            'entropy': [],
            'contrast': [],
            'mean_intensity': [],
            'std_intensity': []
        }
        
        for level, img in enumerate(scale_space):
            # Energy (sum of squared intensities)
            energy = np.sum(img ** 2)
            analysis['energy'].append(energy)
            
            # Entropy
            hist, _ = np.histogram(img.flatten(), bins=256, density=True)
            hist = hist[hist > 0]  # Remove zero entries
            entropy = -np.sum(hist * np.log2(hist))
            analysis['entropy'].append(entropy)
            
            # Contrast (standard deviation)
            contrast = np.std(img)
            analysis['contrast'].append(contrast)
            
            # Mean and standard deviation
            analysis['mean_intensity'].append(np.mean(img))
            analysis['std_intensity'].append(np.std(img))
        
        return analysis

class AdvancedMorphologicalFilters:
    """
    Advanced morphological filters for multiscale analysis.
    """
    
    @staticmethod
    def alternating_sequential_filter(image, se_sizes, operation='opening'):
        """
        Apply alternating sequential filter.
        
        Args:
            image (np.ndarray): Input image
            se_sizes (list): List of structuring element sizes
            operation (str): 'opening' or 'closing'
            
        Returns:
            np.ndarray: Filtered image
        """
        result = image.copy()
        
        for size in se_sizes:
            se = disk(size)
            if operation == 'opening':
                result = morphology.opening(result, se)
            elif operation == 'closing':
                result = morphology.closing(result, se)
            else:
                raise ValueError("Operation must be 'opening' or 'closing'")
        
        return result
    
    @staticmethod
    def morphological_gradient(image, se):
        """
        Compute morphological gradient.
        
        Args:
            image (np.ndarray): Input image
            se (np.ndarray): Structuring element
            
        Returns:
            np.ndarray: Morphological gradient
        """
        dilated = dilation(image, se)
        eroded = erosion(image, se)
        return dilated - eroded
    
    @staticmethod
    def top_hat_transform(image, se, variant='white'):
        """
        Apply top-hat transform.
        
        Args:
            image (np.ndarray): Input image
            se (np.ndarray): Structuring element
            variant (str): 'white' or 'black'
            
        Returns:
            np.ndarray: Top-hat transformed image
        """
        if variant == 'white':
            opened = morphology.opening(image, se)
            return image - opened
        elif variant == 'black':
            closed = morphology.closing(image, se)
            return closed - image
        else:
            raise ValueError("Variant must be 'white' or 'black'")

def create_multiscale_structuring_elements(base_radius, levels, se_type='disk'):
    """
    Create a series of structuring elements with increasing sizes.
    
    Args:
        base_radius (int): Base radius for the structuring element
        levels (int): Number of levels
        se_type (str): Type of structuring element
        
    Returns:
        list: List of structuring elements
    """
    elements = []
    
    for level in range(levels):
        radius = base_radius * (level + 1)
        
        if se_type == 'disk':
            se = disk(radius)
        elif se_type == 'square':
            se = square(2 * radius + 1)
        elif se_type == 'diamond':
            se = diamond(radius)
        else:
            raise ValueError(f"Unknown structuring element type: {se_type}")
        
        elements.append(se)
    
    return elements
