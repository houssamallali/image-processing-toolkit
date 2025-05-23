# Image Processing Project

This repository contains a collection of image processing scripts organized by topics.

## Project Structure

- **TP1_Basics**: Basic image processing operations
  - `firsttest.py`: Loading, displaying, and saving images with RGB channel manipulation

- **TP2_Filtering**: Image filtering techniques
  - `lowpass.py`: Mean, median, and Gaussian filtering
  - `highpass.py`: High-pass filtering
  - `convolution.py`: Custom kernel convolution operations
  - `enhancement.py`: Image enhancement algorithms
  - `aliasing_effect.py`: Aliasing effect demonstration

- **TP3_FourierAnalysis**: Fourier transform analysis
  - `2Dfourier.py`: 2D Fourier transforms
  - `inversefourier.py`: Inverse Fourier transforms
  - `LP_and_HP_filtering.py`: Low-pass and high-pass filtering in frequency domain
  - `application.py`: Applications of Fourier transforms

- **TP4_Segmentation**: Image segmentation techniques
  - `thresholding.py`: Manual thresholding
  - `kmeans_segmentation.py`: K-means segmentation
  - `otsu_segmentation.py`: Otsu thresholding (demonstration with generated points)
  - `advanced_segmentation.py`: Advanced K-means segmentation on color images

- **TP5_Enhancement**: Image enhancement methods
  - `gamma_correction.py`: Gamma adjustment with different gamma values
  - `contrast_stretching.py`: Contrast enhancement with s = 1/(1+(m/r)^E) transformation
  - `histogram_enhancement.py`: Histogram equalization techniques
  - `lut_transformations.py`: Visualizing Look-Up Table transformations
  - `combined_enhancement.py`: Applying multiple enhancement techniques in sequence
  - `phobos_synthetic.py`: Creating synthetic Mars moon Phobos image with craters and grooves

- **TP6_Restoration**: Image restoration techniques
  - `image_restoration.py`: Inverse and Wiener filtering for image restoration
  - `create_motion_psf.py`: Generation of motion blur point spread functions
  - `astronomy_restoration.py`: Restoration of astronomical (Jupiter and Saturn) images
  - `create_sample_astronomy.py`: Generation of synthetic astronomical images with PSFs
  - `iterative_restoration.py`: Implementation of iterative deconvolution algorithms (Richardson-Lucy, Van-Cittert, Landweber)

- **TP7_Registration**: Image registration algorithms
  - Implementation of image registration algorithms including ICP and manual point selection

- **TP8_Compression**: Image compression techniques
  - Implementation of various compression algorithms

- **TP9_Follicle_Segmentation**: Segmentation of ovarian follicles in histological images
  - `follicle_segmentation.py`: Functions for segmenting and analyzing follicle components
  - `main.py`: Main script for executing the follicle segmentation pipeline

- **images**: Contains all image resources used by the scripts

- **plots**: Contains all generated visualizations
  - `TP1_Basics/`: Plots from basic imaging operations
  - `TP2_Filtering/`: Plots showing various filtering techniques
  - `TP3_FourierAnalysis/`: Plots of Fourier transforms and filtering
  - `TP4_Segmentation/`: Plots of image segmentation results
  - `TP5_Enhancement/`: Plots demonstrating image enhancement methods
  - `TP6_Restoration/`: Plots showing image restoration techniques and PSFs
  - `TP7_Registration/`: Plots of image registration results
  - `TP8_Compression/`: Plots showing compression techniques and results
  - `TP9_Follicle_Segmentation/`: Plots of follicle segmentation and analysis

- **docs**: Documentation and reference materials
  - `python_image_processing_tutorials.pdf`: Comprehensive Python tutorials for image processing

## Running the Scripts

A convenient runner script (`run.py`) is provided to easily execute any of the Python scripts:

```bash
# List all available scripts
python run.py --list

# Run a specific script
python run.py TP2_Filtering/lowpass.py

# Run a script and save the generated plots to the 'plots' directory
python run.py TP3_FourierAnalysis/inversefourier.py --save-plots

# Run all scripts in a specific TP module (e.g., TP9)
python run.py --all-tp 9
```

Saved plots will be stored in the `plots` directory, organized into subdirectories corresponding to the TP module.

## Viewing Generated Plots

The repository includes pre-generated plots for most scripts, organized by module in the `plots` directory. You can:

1. Browse plots directly in the file explorer.
2. Use the plots as a quick reference for expected outputs.
3. Compare your own generated plots with the reference versions if you re-run the scripts.

## Usage

Each Python file can be run independently using the `run.py` script or directly with `python <script_path>`. All scripts use relative paths to access images from the centralized `images` directory.

```python
# Example of how to load an image in any script
from skimage.io import imread
# Assumes the script is inside a TP directory (e.g., TP2_Filtering)
image = imread('../images/example.jpg')
```

## Dependencies

All dependencies are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

If you prefer not to use a virtual environment, you can install the main libraries globally:

```bash
pip install opencv-python
```

Main dependencies:
- NumPy
- Matplotlib
- scikit-image
- SciPy
- scikit-learn (for K-means)
- imageio (for some image loading/saving)
- OpenCV (cv2) via `pip install opencv-python`
