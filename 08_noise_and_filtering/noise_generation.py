import matplotlib.pyplot as plt
import numpy as np
import os

def hist_stretch(I):
    """
    Histogram stretching to range [0;1]
    """
    I = I - np.min(I)
    I = I / np.max(I)
    return I

def generate_and_visualize_noise():
    """
    Generate and visualize different types of random noise
    """
    # Ensure plots directory exists
    plots_dir = 'plots/TP8_Compression'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set the size of noise images
    S = 32
    
    # 1. Uniform noise
    a = 0
    b = 255
    R1 = a + (b-a) * np.random.rand(S, S)
    
    # 2. Gaussian noise
    a = 0
    b = 1
    R2 = a + (b-a) * np.random.randn(S, S)
    R2 = hist_stretch(R2)  # Scale to [0,1] for better visualization
    
    # 3. Salt and pepper noise
    a = 0.05
    b = 0.1
    R3 = 0.5 * np.ones((S, S))
    X = np.random.rand(S, S)
    R3[X <= a] = 0
    R3[(X > a) & (X <= b)] = 1
    
    # 4. Exponential noise
    a = 1
    R4 = -1/a * np.log(1 - np.random.rand(S, S))
    R4 = hist_stretch(R4)
    
    # Create figure 1: Noise Images
    plt.figure(figsize=(10, 10))
    plt.suptitle('Different Types of Random Noise', fontsize=16)
    
    # Display images
    plt.subplot(221)
    plt.imshow(R1, cmap='gray')
    plt.title('(a) Uniform Noise')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(R2, cmap='gray')
    plt.title('(b) Gaussian Noise')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(R3, cmap='gray')
    plt.title('(c) Salt and Pepper Noise')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(R4, cmap='gray')
    plt.title('(d) Exponential Noise')
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.show()  # Display noise images
    
    # Create figure 2: Histograms
    plt.figure(figsize=(12, 8))
    plt.suptitle('Probability Distributions of Random Noise', fontsize=16)
    
    # Histogram for uniform noise
    plt.subplot(221)
    plt.hist(R1.flatten(), bins=50, density=True, color='blue', alpha=0.7)
    plt.title('Uniform Noise Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Probability Density')
    plt.grid(alpha=0.3)
    
    # Histogram for Gaussian noise
    plt.subplot(222)
    plt.hist(R2.flatten(), bins=50, density=True, color='green', alpha=0.7)
    plt.title('Gaussian Noise Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Probability Density')
    plt.grid(alpha=0.3)
    
    # Histogram for salt and pepper noise
    plt.subplot(223)
    plt.hist(R3.flatten(), bins=50, density=True, color='red', alpha=0.7)
    plt.title('Salt and Pepper Noise Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Probability Density')
    plt.grid(alpha=0.3)
    
    # Histogram for exponential noise
    plt.subplot(224)
    plt.hist(R4.flatten(), bins=50, density=True, color='purple', alpha=0.7)
    plt.title('Exponential Noise Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Probability Density')
    plt.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.show()  # Display histograms
    
    # Save the figures
    plt.figure(1)
    output_path = os.path.join(plots_dir, 'random_noise_types.png')
    plt.savefig(output_path, dpi=300)
    
    plt.figure(2)
    output_path = os.path.join(plots_dir, 'random_noise_histograms.png')
    plt.savefig(output_path, dpi=300)
    
    print(f"Random noise images and histograms generated and saved to {plots_dir}")
    print(f"- random_noise_types.png: Visualization of the four noise types")
    print(f"- random_noise_histograms.png: Histograms showing the probability distributions")

if __name__ == '__main__':
    generate_and_visualize_noise() 