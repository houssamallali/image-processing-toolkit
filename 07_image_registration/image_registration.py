import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import os
import cv2
from skimage import data

def load_brain_images():
    """
    Load the brain MRI test images.
    For this example, we'll use the camera image twice with different transformations
    as a placeholder for the actual brain images.
    
    Returns:
    --------
    tuple
        (brain1, brain2) - The two brain images to be registered
    """
    # Build absolute paths to the images directory
    script_dir = os.path.dirname(__file__)
    images_dir = os.path.abspath(os.path.join(script_dir, '..', 'images'))
    img1_path = os.path.join(images_dir, 'BrainT1.bmp')
    img2_path = os.path.join(images_dir, 'BrainT1bis.bmp')
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is not None and img2 is not None:
        return img1, img2

    # Fallback to synthetic camera image if the real brain images are unavailable
    print("Warning: Could not load BrainT1 images, using synthetic camera image...")
    img = data.camera()
    brain1 = img[::4, ::4]  # Downsample for faster processing
    
    # Create brain2 by applying a small rotation and translation
    rows, cols = brain1.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    M[0, 2] += 20
    M[1, 2] += 10
    brain2 = cv2.warpAffine(brain1, M, (cols, rows))
    return brain1, brain2

def rigid_registration(data1, data2):
    """
    Estimate rigid transformation (rotation + translation) between two point sets
    using SVD method.
    
    Parameters:
    -----------
    data1 : ndarray
        First set of points (nx2 array)
    data2 : ndarray
        Second set of points (nx2 array)
        
    Returns:
    --------
    tuple
        (R, t) - Rotation matrix and translation vector
    """
    # Convert to numpy arrays
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # Calculate centroids
    centroid1 = np.mean(data1, axis=0)
    centroid2 = np.mean(data2, axis=0)
    
    # Center the point sets
    centered1 = data1 - centroid1
    centered2 = data2 - centroid2
    
    # Calculate correlation matrix K
    K = np.dot(centered2.T, centered1)
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(K)
    
    # Calculate rotation matrix
    R = np.dot(U, Vt)
    
    # Ensure proper rotation matrix (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    
    # Calculate translation
    t = centroid2 - np.dot(R, centroid1)
    
    return R, t

def apply_transform(points, R, t):
    """
    Apply rigid transformation to points.
    
    Parameters:
    -----------
    points : ndarray
        Points to transform (nx2 array)
    R : ndarray
        2x2 rotation matrix
    t : ndarray
        Translation vector
        
    Returns:
    --------
    ndarray
        Transformed points
    """
    return np.dot(points, R.T) + t

def find_closest_points(points1, points2):
    """
    For each point in points1, find the closest point in points2.
    
    Parameters:
    -----------
    points1 : ndarray
        First set of points (nx2 array)
    points2 : ndarray
        Second set of points (nx2 array)
        
    Returns:
    --------
    ndarray
        Array of closest points from points2
    """
    closest_points = []
    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1) ** 2, axis=1))
        closest_idx = np.argmin(distances)
        closest_points.append(points2[closest_idx])
    return np.array(closest_points)

def icp_registration(points1, points2, max_iterations=50, tolerance=1e-6):
    """
    Perform ICP (Iterative Closest Point) registration.
    
    Parameters:
    -----------
    points1 : ndarray
        First set of points (nx2 array)
    points2 : ndarray
        Second set of points (nx2 array)
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    tuple
        (R, t, transformed_points1, error_history)
    """
    current_points = np.copy(points1)
    prev_error = float('inf')
    error_history = []
    
    R_final = np.eye(2)
    t_final = np.zeros(2)
    
    for iteration in range(max_iterations):
        # Find closest points
        closest = find_closest_points(current_points, points2)
        
        # Compute current transformation
        R, t = rigid_registration(current_points, closest)
        
        # Update points
        current_points = apply_transform(current_points, R, t)
        
        # Update final transformation
        R_final = np.dot(R, R_final)
        t_final = np.dot(R, t_final) + t
        
        # Compute error
        error = np.mean(np.sqrt(np.sum((closest - current_points) ** 2, axis=1)))
        error_history.append(error)
        
        # Check convergence
        if abs(prev_error - error) < tolerance:
            break
            
        prev_error = error
    
    return R_final, t_final, current_points, error_history

def extract_control_points(image, n_points=4):
    """
    Extract control points from image using Harris corner detector.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    n_points : int
        Number of points to extract
        
    Returns:
    --------
    ndarray
        Array of control points (nx2)
    """
    # Convert to float32 for corner detection
    gray = np.float32(image)
    # Detect corners (returns None if no corners found)
    corners = cv2.goodFeaturesToTrack(gray, n_points, 0.01, 10)
    if corners is None:
        # Return empty array if no corners detected
        return np.empty((0, 2), dtype=int)
    # Reshape and convert to integer coordinates
    points = corners.reshape(-1, 2).astype(int)
    return points

def visualize_registration(image1, image2, points1, points2, transformed_points1, output_dir):
    """
    Visualize registration results.
    
    Parameters:
    -----------
    image1, image2 : ndarray
        Input images
    points1, points2 : ndarray
        Original point sets
    transformed_points1 : ndarray
        Transformed points from image1
    output_dir : str
        Directory to save output images
    """
    plt.figure(figsize=(15, 5))
    
    # Original images with points
    plt.subplot(131)
    plt.imshow(image1, cmap='gray')
    plt.plot(points1[:, 0], points1[:, 1], 'r.', markersize=10)
    plt.title('Image 1 with Control Points')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(image2, cmap='gray')
    plt.plot(points2[:, 0], points2[:, 1], 'b.', markersize=10)
    plt.title('Image 2 with Control Points')
    plt.axis('off')
    
    # Overlay of transformed points
    plt.subplot(133)
    plt.imshow(image2, cmap='gray')
    plt.plot(points2[:, 0], points2[:, 1], 'b.', markersize=10, label='Target Points')
    plt.plot(transformed_points1[:, 0], transformed_points1[:, 1], 'r.', markersize=10, label='Transformed Points')
    plt.title('Registration Result')
    plt.legend()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'registration_result.png'), dpi=300)
    plt.close()

def main():
    """
    Main function to test image registration implementation.
    """
    # Create output directory
    output_dir = '../plots/TP7_Registration'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load test images
    print("Loading test images...")
    brain1, brain2 = load_brain_images()
    
    # Extract control points
    print("Extracting control points...")
    points1 = extract_control_points(brain1)
    points2 = extract_control_points(brain2)
    
    # Perform ICP registration
    print("Performing ICP registration...")
    R, t, transformed_points, error_history = icp_registration(points1, points2)
    
    print(f"Final rotation matrix:\n{R}")
    print(f"Final translation vector: {t}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_registration(brain1, brain2, points1, points2, transformed_points, output_dir)
    
    # Plot error history
    plt.figure(figsize=(8, 4))
    plt.plot(error_history)
    plt.title('ICP Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Error')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'convergence.png'), dpi=300)
    plt.close()
    
    print("Registration completed successfully.")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 