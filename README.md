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
  - `otsu_segmentation.py`: Otsu thresholding
  - `advanced_segmentation.py`: Advanced segmentation methods

- **images**: Contains all image resources used by the scripts

- **docs**: Documentation and reference materials
  - `python_image_processing_tutorials.pdf`: Comprehensive Python tutorials for image processing

## Running the Scripts

A convenient runner script is provided to easily execute any of the Python scripts:

```bash
# List all available scripts
python run.py --list

# Run a specific script
python run.py TP2_Filtering/lowpass.py

# Run a script and save the generated plots
python run.py TP3_FourierAnalysis/inversefourier.py --save-plots

# Run all scripts in a specific TP module
python run.py --all-tp 2
```

Saved plots will be stored in a `plots` directory created at the root of the project.

## Usage

Each Python file can be run independently. All scripts use relative paths to access images from the centralized `images` directory.

```python
# Example of how to load an image in any script
from skimage.io import imread
image = imread('../images/example.jpg')
```

## Dependencies

All dependencies are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

Main dependencies:
- NumPy
- Matplotlib
- scikit-image
- SciPy 