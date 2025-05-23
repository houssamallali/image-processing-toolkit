# TP5: Image Enhancement Techniques

This module implements various image enhancement methods based on intensity transformations, including gamma correction, contrast stretching, and histogram modifications.

## Overview

Image enhancement aims to improve the visual quality of images or to highlight specific features. This module focuses on:

1. **Intensity Transformations (LUT)**: Pixel-by-pixel mapping using mathematical functions
2. **Histogram-based Methods**: Manipulating the distribution of pixel intensities

## Techniques Implemented

### 1. Gamma Correction (γ)

Gamma correction follows the power-law transformation:
```
s = r^γ
```
where:
- `s` is the output intensity
- `r` is the input intensity (0-1 range)
- `γ` is the gamma value:
  - `γ < 1`: Increases brightness of darker regions
  - `γ > 1`: Increases contrast in brighter regions
  - `γ = 1`: No change

This is implemented in `gamma_correction.py` using scikit-image's `adjust_gamma` function.

### 2. Contrast Stretching

Our contrast stretching implementation uses the formula:
```
s = T(r) = 1 / (1 + (m/r)^E)
```
where:
- `m` is the mean gray value
- `E` controls the steepness of the transition
- Higher `E` values produce more binary-like images

This is implemented in `contrast_stretching.py` with a custom function.

### 3. Histogram Equalization

Histogram equalization redistributes pixel intensities to spread them more uniformly across the available range, enhancing global contrast. The transformation follows:
```
T(x_k) = L * cdf_I(k)
```
where:
- `L` is the maximum intensity value (e.g., 255 for 8-bit images)
- `cdf_I` is the cumulative distribution function of the image histogram

Both custom implementation and scikit-image's built-in function are demonstrated in `histogram_equalization.py`.

### 4. Histogram Matching

Histogram matching transforms an image to match the histogram of a target image. The transformation is:
```
x₂ = cdf₂⁻¹(cdf₁(x₁))
```
where:
- `x₁` is the input pixel intensity
- `x₂` is the output pixel intensity
- `cdf₁` is the CDF of the source image
- `cdf₂` is the CDF of the target image

This is implemented in `histogram_matching.py`, including a custom implementation that creates a bimodal target histogram.

## Scripts Description

| Script | Description |
|--------|-------------|
| `gamma_correction.py` | Applies gamma correction with different γ values (0.5, 1.0, 2.0) |
| `contrast_stretching.py` | Implements the contrast stretching with different E values (10, 20, 1000) |
| `histogram_equalization.py` | Implements custom histogram equalization and compares it with scikit-image's version |
| `histogram_matching.py` | Implements histogram matching to transform an image to match a target histogram |
| `compare_histogram_methods.py` | Compares different histogram-based enhancement techniques side by side |
| `lut_transformations.py` | Visualizes Look-Up Table transformations for both gamma and contrast stretching |
| `combined_enhancement.py` | Demonstrates the pipeline of multiple enhancement techniques applied sequentially |

## Running the Scripts

Each script can be executed using the main runner:

```bash
# Run with default settings
python run.py TP5_Enhancement/gamma_correction.py

# Run and save generated plots
python run.py TP5_Enhancement/histogram_equalization.py --save-plots
```

## Generated Plots

The `plots/TP5_Enhancement/` directory contains visualizations of:

1. **Gamma Correction**: Original image alongside versions with γ=0.5, 1.0, and 2.0
2. **Contrast Stretching**: Effects of different E values showing the transition from subtle enhancement to thresholding
3. **Histogram Equalization**: Original and enhanced images with their corresponding histograms, comparing custom implementation with scikit-image
4. **Histogram Matching**: Source image, target histogram, and the matched result
5. **Comparison of Methods**: Side-by-side comparison of different enhancement techniques
6. **Combined Enhancement**: Step-by-step results of enhancement pipelines

## Theory Background

### Look-Up Tables (LUT)

A LUT is a mapping function that transforms input pixel values to output pixel values. These transformations can enhance image contrast, correct gamma, or stretch specific ranges of intensities.

The LUT transformations here follow the general form:
```
output_pixel = T(input_pixel)
```

Where T is the transformation function like gamma correction or contrast stretching.

### Histogram Processing

Histogram-based techniques analyze and modify the distribution of pixel intensities:

- **Histogram Equalization**: Transforms an image to achieve a uniform (flat) histogram
- **Adaptive Histogram Equalization (CLAHE)**: Applies equalization to local regions, improving local contrast
- **Histogram Matching**: Transforms an image to match the histogram of a reference image

## References

1. The gamma correction and contrast stretching formulas are based on standard image processing techniques
2. Implementation follows scikit-image's exposure module functionality
3. The histogram equalization algorithm follows the standard approach using cumulative distribution functions 