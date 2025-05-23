# TP 13 - Multiscale Analysis

## Overview

This practical work implements advanced multiscale analysis techniques for image processing, focusing on pyramidal decomposition and reconstruction methods. The implementation covers both Gaussian/Laplacian pyramids and morphological multiscale decomposition.

## Theoretical Background

### Pyramidal Decomposition and Reconstruction

**Gaussian Pyramid**: A sequence of images where each level is obtained by:
1. Gaussian filtering of the previous level
2. Subsampling by a factor of 2
3. Detail extraction (residues) for reconstruction

**Laplacian Pyramid**: Enables exact reconstruction by storing detail information at each level, allowing perfect reconstruction of the original image.

### Scale-space Decomposition

**Morphological Multiscale**: Uses morphological operators (dilation/erosion) to decompose images at different scales without sampling changes.

**Kramer and Bruckner Filter**: An iterative filter defined as:
- MK^n_B(f) = K_B(MK^{n-1}_B(f))
- Where K_B combines dilation and erosion operations

## Implementation Features

### Core Modules

1. **`pyramidal_decomposition.py`**
   - Gaussian pyramid construction
   - Laplacian pyramid decomposition
   - Perfect reconstruction algorithms
   - Error analysis and quality metrics

2. **`morphological_multiscale.py`**
   - Morphological scale-space decomposition
   - Kramer-Bruckner filter implementation
   - Structuring element management

3. **`visualization.py`**
   - Professional pyramid visualization
   - Comparative analysis plots
   - Error visualization and metrics

4. **`config.py`**
   - Centralized parameter management
   - Algorithm configurations
   - Output settings

### Key Algorithms Implemented

#### Algorithm 3: Pyramidal Decomposition
```
Data: image A₀
Result: pyramid of approximations {Aᵢ}, pyramid of details {Dᵢ}
for i=1 to 3 do
    filtering: F = filt(Aᵢ₋₁);
    subsampling: Aᵢ = ech(F, 0.5);
    details: Dᵢ = Aᵢ₋₁ - ech(Aᵢ, 2);
end
```

#### Algorithm 4: Pyramidal Reconstruction
```
Data: image A₃, pyramid of details {Dᵢ}
Result: reconstructed pyramid {Bᵢ}
initialization: B₃ = A₃;
for i=3 to 1 do
    oversampling: R = ech(Bᵢ, 2);
    adding details: Bᵢ₋₁ = R + Dᵢ
end
```

## Usage Examples

### Basic Pyramidal Decomposition
```python
from tp13.pyramidal_decomposition import LaplacianPyramidDecomposition
from tp13.visualization import visualize_pyramid

# Load cerveau image
image = load_cerveau_image()

# Create 4-level decomposition
decomposer = LaplacianPyramidDecomposition(levels=4)
gaussian_pyramid, laplacian_pyramid = decomposer.decompose(image)

# Visualize results
visualize_pyramid(gaussian_pyramid, laplacian_pyramid)
```

### Morphological Multiscale Analysis
```python
from tp13.morphological_multiscale import MorphologicalDecomposition
from skimage.morphology import disk

# Create structuring element
structuring_element = disk(3)

# Perform decomposition
decomposer = MorphologicalDecomposition(structuring_element)
scale_space = decomposer.decompose(image, levels=5)
```

### Complete Analysis Pipeline
```python
from tp13.main import run_complete_analysis

# Run full multiscale analysis
results = run_complete_analysis(
    image_path='images/brain.bmp',
    pyramid_levels=4,
    morphological_levels=5
)
```

## Expected Outputs

### Generated Visualizations
1. **`pyramid_decomposition.png`** - Gaussian and Laplacian pyramid levels
2. **`reconstruction_comparison.png`** - Original vs reconstructed images
3. **`morphological_decomposition.png`** - Scale-space decomposition results
4. **`error_analysis.png`** - Reconstruction error visualization
5. **`multiscale_comparison.png`** - Comparative analysis of methods

### Quantitative Results
- **Perfect Reconstruction**: Laplacian pyramid enables exact reconstruction
- **Error Metrics**: PSNR, MSE, and visual quality assessment
- **Scale Analysis**: Multi-resolution feature preservation

## Technical Implementation

### Key Features
- **Memory Efficient**: Optimized pyramid storage and computation
- **Flexible Parameters**: Configurable decomposition levels and filters
- **Professional Visualization**: Publication-quality figures
- **Error Analysis**: Comprehensive reconstruction quality metrics
- **Multiple Methods**: Both pyramidal and morphological approaches

### Performance Optimizations
- Efficient convolution using scipy.ndimage
- Memory-conscious pyramid storage
- Vectorized operations for speed
- Parallel processing support

## Installation and Setup

```bash
# Install required dependencies
pip install numpy scipy scikit-image matplotlib seaborn

# Run the complete analysis
python tp13/main.py

# Run specific components
python tp13/pyramidal_decomposition.py
python tp13/morphological_multiscale.py
```

## File Structure
```
tp13/
├── README.md                          # This documentation
├── config.py                          # Configuration management
├── pyramidal_decomposition.py         # Gaussian/Laplacian pyramids
├── morphological_multiscale.py        # Morphological decomposition
├── visualization.py                   # Visualization utilities
├── main.py                           # Complete analysis pipeline
├── test_multiscale.py                # Comprehensive testing
├── requirements.txt                   # Python dependencies
└── outputs/                          # Generated results
    ├── pyramid_decomposition.png
    ├── reconstruction_comparison.png
    ├── morphological_decomposition.png
    ├── error_analysis.png
    └── multiscale_comparison.png
```

## References

- Burt, P. J., & Adelson, E. H. (1983). The Laplacian pyramid as a compact image code
- Kramer, H. P., & Bruckner, J. B. (1975). Iterations of a non-linear transformation
- Lindeberg, T. (1994). Scale-space theory in computer vision

---

**Author**: TP13 Implementation
**Date**: 2024
**Course**: Advanced Image Processing - Multiscale Analysis
