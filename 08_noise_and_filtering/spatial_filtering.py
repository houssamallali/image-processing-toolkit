import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from scipy import ndimage
from PIL import Image

def add_salt_pepper_noise(A, a=0.05, b=0.05):
    nx, ny = A.shape
    B = A.copy()
    X = np.random.rand(nx, ny)
    B[X <= a] = 0
    B[(X > a) & (X <= (a + b))] = 255
    return B

def amf(I, Smax):
    """
    Adaptive median filter
    I: grayscale image
    Smax: maximal size of neighborhood
    """
    f = np.copy(I)
    nx, ny = I.shape
    sizes = np.arange(1, Smax + 1, 2)
    zmin = np.zeros((nx, ny, len(sizes)))
    zmax = np.zeros((nx, ny, len(sizes)))
    zmed = np.zeros((nx, ny, len(sizes)))
    for k, s in enumerate(sizes):
        zmin[:, :, k] = ndimage.minimum_filter(I, s)
        zmax[:, :, k] = ndimage.maximum_filter(I, s)
        zmed[:, :, k] = ndimage.median_filter(I, s)
    isMedImpulse = np.logical_or(zmin == zmed, zmax == zmed)
    for i in range(nx):
        for j in range(ny):
            k = 0
            while k < len(sizes) - 1 and isMedImpulse[i, j, k]:
                k += 1
            if I[i, j] == zmin[i, j, k] or I[i, j] == zmax[i, j, k] or k == len(sizes) - 1:
                f[i, j] = zmed[i, j, k]
            else:
                f[i, j] = I[i, j]
    return f

def main():
    plots_dir = 'plots/TP8_Compression'
    os.makedirs(plots_dir, exist_ok=True)
    img_path = 'images/jambe.tif'
    try:
        A = imageio.imread(img_path)
    except Exception as e:
        print('imageio failed, trying PIL:', e)
        A = np.array(Image.open(img_path))
    if len(A.shape) > 2:
        A = np.mean(A, axis=2).astype(np.uint8)
    # 1. Add salt-and-pepper noise
    B = add_salt_pepper_noise(A, a=0.05, b=0.05)
    plt.figure()
    plt.imshow(B, cmap='gray')
    plt.title('Salt-and-Pepper Noise')
    plt.axis('off')
    plt.show()
    imageio.imwrite(os.path.join(plots_dir, 'leg_sp.png'), B)
    # 2. Filtering
    B1 = ndimage.uniform_filter(B, 5)
    imageio.imwrite(os.path.join(plots_dir, 'leg_uniform.png'), B1)
    B2 = ndimage.minimum_filter(B, 3)
    imageio.imwrite(os.path.join(plots_dir, 'leg_minimum.png'), B2)
    B3 = ndimage.maximum_filter(B, 3)
    imageio.imwrite(os.path.join(plots_dir, 'leg_maximum.png'), B3)
    B4 = ndimage.median_filter(B, 7)
    imageio.imwrite(os.path.join(plots_dir, 'leg_median.png'), B4)
    # 3. Adaptive median filter
    print('Applying adaptive median filter (this may take a while)...')
    B5 = amf(B, 7)
    plt.figure()
    plt.imshow(B5, cmap='gray')
    plt.title('Adaptive Median Filter')
    plt.axis('off')
    plt.show()
    imageio.imwrite(os.path.join(plots_dir, 'leg_amf.png'), B5)
    print('All results saved in', plots_dir)

if __name__ == '__main__':
    main() 