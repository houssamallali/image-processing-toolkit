import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from image_registration import load_brain_images, extract_control_points, rigid_registration, apply_transform

"""
Simulate mixing of control points and perform single rigid registration.
This demonstrates the incorrect registration when points are permuted, illustrating Figure 25.4.
"""

def main():
    # Prepare output directory
    output_dir = '../plots/TP7_Registration'
    os.makedirs(output_dir, exist_ok=True)

    # Load test images
    img1, img2 = load_brain_images()

    # Detect a small set of control points (e.g., 4)
    n_points = 4
    points1 = extract_control_points(img1, n_points)
    points2 = extract_control_points(img2, n_points)

    # 1) Random permutation of points1
    perm = np.random.permutation(n_points)
    points1_perm = points1[perm]

    # 2) Estimate rigid transformation T once with permuted points
    R, t = rigid_registration(points1_perm, points2)

    # 3) Apply transform to permuted points
    transformed = apply_transform(points1_perm, R, t)

    # Visualize matching vs permuted registration
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img2, cmap='gray')
    axes[0].plot(points2[:,0], points2[:,1], 'b.', label='Original points')
    axes[0].plot(points1_perm[:,0], points1_perm[:,1], 'r.', label='Permuted source')
    axes[0].set_title('Permutation of Points')
    axes[0].legend()
    axes[0].axis('off')

    # 4) Warp image with incorrect single registration
    M = np.hstack((R, t.reshape(2,1)))
    h, w = img1.shape
    bad_registered = cv2.warpAffine(img1, M, (w, h))

    axes[1].imshow(img2, cmap='gray')
    axes[1].imshow(bad_registered, cmap='jet', alpha=0.5)
    axes[1].set_title('Result of Single Registration\n(With Permutation)')
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'mixed_registration_result.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f'Mixed registration result saved to {save_path}')

if __name__ == '__main__':
    main() 