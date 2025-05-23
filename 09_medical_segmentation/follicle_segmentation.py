import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage import morphology, io

def load_and_visualize_image(image_path):
    """
    Load and visualize the follicle image
    
    Parameters:
    -----------
    image_path : str
        Path to the follicle image
        
    Returns:
    --------
    img : numpy.ndarray
        Loaded image
    """
    # Load image
    img = io.imread(image_path)
    
    # Display original image
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title('Original Follicle Image')
    plt.axis('off')
    plt.show()
    
    return img

def extract_antrum(img):
    """
    Extract the antrum of the follicle by thresholding the blue component
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input follicle image
        
    Returns:
    --------
    antrum : numpy.ndarray
        Binary mask of the antrum
    """
    # Extract blue channel (in grayscale image, just use the image)
    B = img[:, :, 2] if len(img.shape) > 2 else img
    
    # Threshold to extract antrum (white region in the image)
    # Lower threshold to capture more of the antrum
    antrum = B > 180
    
    # Label connected components and keep only the largest one (the antrum)
    # Create a structure for connectivity
    struct = ndimage.generate_binary_structure(2, 2)
    labeled_antrum, num_features = ndimage.label(antrum, structure=struct)
    
    if num_features > 0:
        # Find the largest component
        sizes = ndimage.sum(antrum, labeled_antrum, range(1, num_features+1))
        largest_component = np.argmax(sizes) + 1
        antrum = labeled_antrum == largest_component
    
    # Fill holes in the antrum mask
    antrum = ndimage.binary_fill_holes(antrum)
    
    # Display result
    plt.figure(figsize=(8, 8))
    plt.imshow(antrum, cmap='gray')
    plt.title('Extracted Antrum')
    plt.axis('off')
    plt.show()
    
    return antrum

def extract_theca(antrum, disk_size=20):
    """
    Extract the theca which is the ring region around the antrum
    
    Parameters:
    -----------
    antrum : numpy.ndarray
        Binary mask of the antrum
    disk_size : int
        Size of the structuring element for dilation
        
    Returns:
    --------
    theca : numpy.ndarray
        Binary mask of the theca
    """
    # Create a disk-shaped structuring element
    selem = morphology.disk(disk_size)
    
    # Dilate the antrum to include the surrounding region
    theca = morphology.binary_dilation(antrum, selem)
    
    # The theca is the difference between the dilated region and the antrum
    theca = np.logical_and(theca, np.logical_not(antrum))
    
    # Display result
    plt.figure(figsize=(8, 8))
    plt.imshow(theca, cmap='gray')
    plt.title('Extracted Theca')
    plt.axis('off')
    plt.show()
    
    return theca

def extract_vascularization(img, antrum, theca, threshold=100):
    """
    Extract the vascularization which has a low blue component and is located in theca
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input follicle image
    antrum : numpy.ndarray
        Binary mask of the antrum
    theca : numpy.ndarray
        Binary mask of the theca
    threshold : int
        Threshold value for blue component
        
    Returns:
    --------
    vascularization : numpy.ndarray
        Binary mask of the vascularization
    """
    # Extract blue channel (in grayscale image, just use the image)
    B = img[:, :, 2] if len(img.shape) > 2 else img
    
    # Threshold to extract pixels with low blue component
    vascularization = B < threshold
    
    # The vascularization is limited to the theca region
    vascularization = np.logical_and(vascularization, theca)
    
    # Display result
    plt.figure(figsize=(8, 8))
    plt.imshow(vascularization, cmap='gray')
    plt.title('Extracted Vascularization')
    plt.axis('off')
    plt.show()
    
    return vascularization

def extract_granulosa_cells(vascularization, antrum, theca, disk_size=5):
    """
    Extract granulosa cells which are between the antrum and vascularization
    
    Parameters:
    -----------
    vascularization : numpy.ndarray
        Binary mask of the vascularization
    antrum : numpy.ndarray
        Binary mask of the antrum
    theca : numpy.ndarray
        Binary mask of the theca
    disk_size : int
        Size of the structuring element for closing operation
        
    Returns:
    --------
    granulosa : numpy.ndarray
        Binary mask of the granulosa cells
    """
    # Create a disk-shaped structuring element
    selem = morphology.disk(disk_size)
    
    # Create a filled region that includes vasculature
    # First, dilate the vascularization to connect nearby regions
    dil = morphology.binary_dilation(vascularization, selem)
    
    # Then close any remaining gaps
    dil = morphology.binary_closing(dil, selem)
    
    # Label connected components
    struct = ndimage.generate_binary_structure(2, 1)
    labeled_dil, num_features = ndimage.label(dil, structure=struct)
    
    # Find components that surround the antrum
    # The granulosa cells are in the inner part of the theca, closer to the antrum
    
    # Use the theca mask but focus on the inner part
    inner_theca = np.logical_and(theca, np.logical_not(vascularization))
    inner_theca = morphology.binary_erosion(inner_theca, morphology.disk(2))
    
    # Granulosa cells are between the antrum and vascularization/theca
    granulosa = np.logical_and(inner_theca, np.logical_not(antrum))
    
    # Display result
    plt.figure(figsize=(8, 8))
    plt.imshow(granulosa, cmap='gray')
    plt.title('Extracted Granulosa Cells')
    plt.axis('off')
    plt.show()
    
    return granulosa

def quantify_components(antrum, theca, vascularization, granulosa):
    """
    Calculate geometrical measurements for the different components
    
    Parameters:
    -----------
    antrum : numpy.ndarray
        Binary mask of the antrum
    theca : numpy.ndarray
        Binary mask of the theca
    vascularization : numpy.ndarray
        Binary mask of the vascularization
    granulosa : numpy.ndarray
        Binary mask of the granulosa cells
        
    Returns:
    --------
    measurements : dict
        Dictionary containing measurements
    """
    # Total follicle area is antrum + theca
    follicle = np.logical_or(antrum, theca)
    
    # Calculate areas (count of non-zero pixels)
    antrum_area = np.sum(antrum)
    theca_area = np.sum(theca)
    vascularization_area = np.sum(vascularization)
    granulosa_area = np.sum(granulosa)
    follicle_area = np.sum(follicle)
    
    # Calculate proportions
    q_vascularization = vascularization_area / follicle_area
    q_granulosa = granulosa_area / follicle_area
    
    # Store and print results
    measurements = {
        'antrum_area': antrum_area,
        'theca_area': theca_area,
        'vascularization_area': vascularization_area,
        'granulosa_area': granulosa_area,
        'follicle_area': follicle_area,
        'vascularization_proportion': q_vascularization,
        'granulosa_proportion': q_granulosa
    }
    
    print(f"Antrum Area: {antrum_area} pixels")
    print(f"Theca Area: {theca_area} pixels")
    print(f"Vascularization Area: {vascularization_area} pixels")
    print(f"Granulosa Area: {granulosa_area} pixels")
    print(f"Total Follicle Area: {follicle_area} pixels")
    print(f"Vascularization Proportion: {q_vascularization:.4f}")
    print(f"Granulosa Proportion: {q_granulosa:.4f}")
    
    return measurements

def visualize_segmentation(img, antrum, theca, vascularization, granulosa):
    """
    Visualize all segmented components in a single color-coded image
    
    Parameters:
    -----------
    img : numpy.ndarray
        Original image
    antrum : numpy.ndarray
        Binary mask of the antrum
    theca : numpy.ndarray
        Binary mask of the theca
    vascularization : numpy.ndarray
        Binary mask of the vascularization
    granulosa : numpy.ndarray
        Binary mask of the granulosa cells
    """
    # Create color-coded segmentation
    # Antrum: Blue, Granulosa: Green, Vascularization: Red
    result = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    
    # Create color coding
    result[antrum] = [0, 0, 255]  # Antrum: Blue
    result[granulosa] = [0, 255, 0]  # Granulosa: Green
    result[vascularization] = [255, 0, 0]  # Vascularization: Red
    
    # Display results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('(a) Original Image')
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(antrum, cmap='gray')
    plt.title('(b) Antrum')
    plt.axis('off')
    
    plt.subplot(233)
    plt.imshow(theca, cmap='gray')
    plt.title('(c) Theca')
    plt.axis('off')
    
    plt.subplot(234)
    plt.imshow(vascularization, cmap='gray')
    plt.title('(d) Vascularization')
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(granulosa, cmap='gray')
    plt.title('(e) Granulosa Cells')
    plt.axis('off')
    
    plt.subplot(236)
    plt.imshow(result)
    plt.title('(f) Segmentation of the Different Parts')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the results
    plots_dir = 'plots/TP9_Follicle_Segmentation'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'follicle_segmentation_results.png'), dpi=300)
    
    plt.show()

def process_follicle_image(image_path):
    """
    Complete pipeline for follicle segmentation
    
    Parameters:
    -----------
    image_path : str
        Path to the follicle image
    """
    # Load and visualize image
    img = load_and_visualize_image(image_path)
    
    # Extract components
    antrum = extract_antrum(img)
    theca = extract_theca(antrum)
    vascularization = extract_vascularization(img, antrum, theca)
    granulosa = extract_granulosa_cells(vascularization, antrum, theca)
    
    # Quantify results
    measurements = quantify_components(antrum, theca, vascularization, granulosa)
    
    # Visualize final segmentation
    visualize_segmentation(img, antrum, theca, vascularization, granulosa)
    
    return measurements

if __name__ == "__main__":
    # Process the follicle image
    image_path = "../images/follicule.bmp"
    process_follicle_image(image_path) 