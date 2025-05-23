import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from image_registration import rigid_registration, apply_transform

class PointSelector:
    def __init__(self, image, title):
        self.image = image
        self.title = title
        self.points = []
        self.fig = None
        self.ax = None
        self.n_points = 0
        self.max_points = 0
        
    def onclick(self, event):
        if event.inaxes != self.ax or len(self.points) >= self.max_points:
            return
            
        x, y = event.xdata, event.ydata
        self.points.append([x, y])
        self.ax.plot(x, y, 'r.', markersize=10)
        self.fig.canvas.draw()
        
        print(f"Selected point {len(self.points)}/{self.max_points}: ({x:.1f}, {y:.1f})")
        
        if len(self.points) >= self.max_points:
            print("All points selected. Close the window to continue.")
        
    def select_points(self, n_points=4):
        self.max_points = n_points
        self.points = []  # Reset points
        
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image, cmap='gray')
        self.ax.set_title(f"{self.title}\nSelect {n_points} points")
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        plt.show()
        
        if len(self.points) != n_points:
            print(f"Warning: Selected {len(self.points)} points, expected {n_points}")
            
        # Convert points to numpy array
        return np.array(self.points)

def main():
    """
    Test manual point selection and registration
    """
    # Create output directory
    output_dir = '../plots/TP7_Registration'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load test images (using camera image with artificial transformation)
    print("Loading test images...")
    img1 = cv2.imread('../images/BrainT1.bmp', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../images/BrainT1bis.bmp', cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Warning: Could not load brain images, falling back to test image")
        # If brain images don't exist, use camera image
        from skimage import data
        img = data.camera()
        
        # Create two versions with different transformations
        img1 = img[::4, ::4]  # Downsample for faster processing
        
        # Create img2 by applying a small rotation and translation to img1
        rows, cols = img1.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)  # 15 degree rotation
        M[0, 2] += 20  # Add translation
        M[1, 2] += 10
        img2 = cv2.warpAffine(img1, M, (cols, rows))
    
    # Manual point selection
    print("\nSelect corresponding points in both images:")
    print("First image: Select 4 distinctive points")
    selector1 = PointSelector(img1, "Image 1")
    points1 = selector1.select_points(4)
    
    print("\nSecond image: Select corresponding points in the same order")
    selector2 = PointSelector(img2, "Image 2")
    points2 = selector2.select_points(4)
    
    # Perform registration
    print("\nPerforming registration...")
    R, t = rigid_registration(points1, points2)
    
    print(f"\nEstimated transformation:")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation vector: {t}")
    
    # Transform points
    transformed_points = apply_transform(points1, R, t)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original images with points
    plt.subplot(131)
    plt.imshow(img1, cmap='gray')
    plt.plot(points1[:, 0], points1[:, 1], 'r.', markersize=10)
    plt.title('Image 1 with Selected Points')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(img2, cmap='gray')
    plt.plot(points2[:, 0], points2[:, 1], 'b.', markersize=10)
    plt.title('Image 2 with Selected Points')
    plt.axis('off')
    
    # Overlay of transformed points
    plt.subplot(133)
    plt.imshow(img2, cmap='gray')
    plt.plot(points2[:, 0], points2[:, 1], 'b.', markersize=10, label='Target Points')
    plt.plot(transformed_points[:, 0], transformed_points[:, 1], 'r.', markersize=10, label='Transformed Points')
    plt.title('Registration Result')
    plt.legend()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'manual_registration_result.png'), dpi=300)
    plt.show()
    
    print(f"\nResults saved in {output_dir}")
    
    # Build affine matrix [R | t] for image warp
    M = np.hstack((R, t.reshape(2, 1)))
    rows, cols = img1.shape
    registered = cv2.warpAffine(img1, M, (cols, rows))

    # Plot before and after registration overlay
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img2, cmap='gray')
    axes[0].imshow(img1, cmap='jet', alpha=0.4)
    axes[0].set_title('Without registration')
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].imshow(registered, cmap='jet', alpha=0.4)
    axes[1].set_title('With registration')
    axes[1].axis('off')

    plt.tight_layout()
    comp_path = os.path.join(output_dir, 'registration_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.show()
    print(f"Comparison saved as {comp_path}")

if __name__ == "__main__":
    main() 