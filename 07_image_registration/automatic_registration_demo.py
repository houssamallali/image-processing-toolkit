import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from image_registration import load_brain_images, extract_control_points, icp_registration, apply_transform

# Automatic ICP-based registration test script

def main():
    # Output directory for plots
    output_dir = '../plots/TP7_Registration'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load brain images
    print("Loading test images...")
    img1, img2 = load_brain_images()

    # Number of control points to detect
    n_points = 20
    print(f"Detecting {n_points} control points in each image...")
    points1 = extract_control_points(img1, n_points)
    points2 = extract_control_points(img2, n_points)

    # Save auto-detected points visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(img1, cmap='gray')
    plt.plot(points1[:, 0], points1[:, 1], 'r.', markersize=5)
    plt.title('Auto-detected points (Image 1)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auto_points1.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(img2, cmap='gray')
    plt.plot(points2[:, 0], points2[:, 1], 'b.', markersize=5)
    plt.title('Auto-detected points (Image 2)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auto_points2.png'), dpi=300)
    plt.close()

    # Perform ICP registration
    print("Performing ICP registration on auto-detected points...")
    R, t, transformed, error_history = icp_registration(points1, points2)

    # Visualize control-point registration and convergence
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img2, cmap='gray')
    ax[0].plot(points2[:, 0], points2[:, 1], 'b.', label='Target', markersize=5)
    ax[0].plot(transformed[:, 0], transformed[:, 1], 'r.', label='Transformed', markersize=5)
    ax[0].set_title('Control Points Registration')
    ax[0].legend()
    ax[0].axis('off')

    ax[1].plot(error_history)
    ax[1].set_title('ICP Convergence')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Mean Error')
    ax[1].grid(True)

    plt.tight_layout()
    cp_path = os.path.join(output_dir, 'auto_control_registration.png')
    plt.savefig(cp_path, dpi=300)
    plt.close()
    print(f"Control registration plot saved to {cp_path}")

    # Apply warp to full image
    M = np.hstack((R, t.reshape(2, 1)))
    h, w = img1.shape
    registered = cv2.warpAffine(img1, M, (w, h))

    # Compare before/after registration overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img2, cmap='gray')
    axes[0].imshow(img1, cmap='jet', alpha=0.4)
    axes[0].set_title('Before Registration')
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].imshow(registered, cmap='jet', alpha=0.4)
    axes[1].set_title('After ICP Registration')
    axes[1].axis('off')

    plt.tight_layout()
    compare_path = os.path.join(output_dir, 'auto_registration_comparison.png')
    plt.savefig(compare_path, dpi=300)
    plt.show()
    print(f"Comparison saved to {compare_path}")

if __name__ == '__main__':
    main() 