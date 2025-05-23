# TP6: Image Restoration with Deconvolution

This practical module explores the fundamentals of image restoration, focusing on deconvolution techniques to recover images degraded by blur and noise.

## Theoretical Background

### Damage Modeling

A damaged image can be modeled as:

```
g = D(f) + n
```

Where:
- `g` is the degraded image
- `f` is the original image
- `D` is the damage function
- `n` is additive noise

For linear, spatially invariant processes, this can be simplified to:

```
g = h * f + n
```

Where `h` is the Point Spread Function (PSF) and `*` denotes convolution.

### Frequency Domain Representation

In the frequency domain, this becomes:

```
G = H·F + N
```

Where uppercase letters represent the Fourier transforms of their lowercase counterparts.

### Restoration Methods

1. **Inverse Filtering**: Simple division in the frequency domain
   ```
   F̂ = G/H
   ```
   Issue: Division by zero or small values causes amplification of noise.

2. **Regularized Inverse Filtering**: Add a small constant to avoid division issues
   ```
   F̂ = G/(H+α)
   ```
   Where α is a small constant.

3. **Wiener Filtering**: Accounts for noise in the restoration
   ```
   F̂ = (H*/|H|²+K)·G
   ```
   Where H* is the complex conjugate of H, and K is the noise-to-signal ratio.

4. **Lucy-Richardson Deconvolution**: Iterative method based on Bayesian statistics
   ```
   f_{k+1} = f_k · [h* * (g/(h*f_k))]
   ```
   Where h* is the flipped PSF, and k is the iteration number.

5. **Van-Cittert Iterative Deconvolution**: Simple iterative approach based on residual correction
   ```
   f_{k+1} = f_k + β(g - h * f_k)
   ```
   Where β is a relaxation parameter controlling the convergence speed.

6. **Landweber Iterative Deconvolution**: Gradient-based iterative approach
   ```
   f_{k+1} = f_k + α·h* * (g - h * f_k)
   ```
   Where α is the Jansson parameter controlling step size.

## Scripts in this Directory

### 1. `image_restoration.py`

Main script that demonstrates various deconvolution techniques using a synthetic checkerboard image:

- Generates a checkerboard pattern
- Creates PSFs (Gaussian and motion blur)
- Applies blur and adds noise
- Implements several restoration methods:
  - Inverse filtering
  - Wiener filtering
  - Visualization in both spatial and frequency domains

### 2. `create_motion_psf.py`

Generates and saves various motion blur PSFs:

- Linear motion at 45°
- Horizontal motion
- Vertical motion
- Circular motion
- Saves the PSFs for reuse in other scripts
- Demonstrates the effect of each PSF on a checkerboard image

### 3. `create_sample_astronomy.py`

Generates synthetic astronomical images based on Jupiter and Saturn:

- Creates realistic-looking planetary images
- Generates Point Spread Functions (PSFs) similar to telescope observations
- Saves both original images and PSFs as PNG files and NumPy arrays
- Creates examples of images blurred with their respective PSFs

### 4. `astronomy_restoration.py`

Applies image restoration techniques to astronomical images:

- Loads Jupiter and Saturn images along with their PSFs
- Applies inverse filtering and Wiener filtering for image restoration
- Compares restoration results
- Displays and saves visualizations of the restoration process

### 5. `iterative_restoration.py`

Implements and compares iterative deconvolution algorithms for image restoration:

- Richardson-Lucy iterative deconvolution (based on Bayesian statistics)
- Van-Cittert iterative deconvolution (simple residual-based approach)
- Landweber iterative deconvolution (gradient-based method with relaxation parameter)
- Applies all methods to astronomical images and compares results
- Provides side-by-side visualization of different restoration techniques

## How to Run

1. Make sure the required dependencies are installed:
   ```
   pip install numpy scipy matplotlib scikit-image astropy
   ```

2. Run any of the scripts:
   ```
   python image_restoration.py
   python create_motion_psf.py
   python create_sample_astronomy.py
   python astronomy_restoration.py
   python iterative_restoration.py
   ```

3. Examine the generated plots in the `plots/TP6_Restoration/` directory.

## Expected Outputs

Running these scripts will generate:

- Visualization of original, blurred, and restored images
- PSF and OTF (Optical Transfer Function) representations
- Comparative results from different restoration algorithms
- Performance analysis with various noise levels

## Understanding the Results

- **Inverse filtering** performs well in noise-free scenarios but amplifies noise significantly
- **Wiener filtering** balances noise suppression and detail restoration
- For real-world applications, accurate PSF estimation is crucial
- While these methods can recover a significant amount of detail, perfect restoration is generally impossible due to information loss 