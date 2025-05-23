import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse, disk
from skimage.filters import gaussian
import imageio

def create_phobos():
    """
    Create a synthetic image of Phobos based on its known characteristics.
    """
    # Create a base image (256x256 pixels)
    img = np.zeros((256, 256))
    
    # Create the basic elliptical shape of Phobos
    rr, cc = ellipse(128, 128, 90, 70, img.shape)
    img[rr, cc] = 0.6  # Set base brightness
    
    # Add large crater (Stickney)
    rr, cc = disk((100, 140), 25, shape=img.shape)
    img[rr, cc] = 0.5
    
    # Add medium craters
    rr, cc = disk((160, 120), 15, shape=img.shape)
    img[rr, cc] = 0.5
    
    rr, cc = disk((130, 80), 12, shape=img.shape)
    img[rr, cc] = 0.55
    
    # Add small craters
    for _ in range(20):
        x = np.random.randint(50, 206)
        y = np.random.randint(50, 206)
        
        # Check if point is inside the main ellipse
        if ((x-128)/90)**2 + ((y-128)/70)**2 <= 1:
            r = np.random.randint(3, 8)
            rr, cc = disk((y, x), r, shape=img.shape)
            img[rr, cc] = 0.5 + np.random.uniform(-0.1, 0.1)
    
    # Add grooves/furrows
    for i in range(5):
        thickness = np.random.randint(1, 3)
        length = np.random.randint(40, 80)
        angle = np.random.uniform(0, 2*np.pi)
        
        center_x = np.random.randint(80, 176)
        center_y = np.random.randint(80, 176)
        
        # Check if center is inside the main ellipse
        if ((center_x-128)/90)**2 + ((center_y-128)/70)**2 <= 0.8:
            x_end = center_x + length * np.cos(angle)
            y_end = center_y + length * np.sin(angle)
            
            x_coords = np.linspace(center_x, x_end, 100).astype(int)
            y_coords = np.linspace(center_y, y_end, 100).astype(int)
            
            # Keep only points inside the image and ellipse
            valid_indices = (
                (x_coords >= 0) & (x_coords < img.shape[1]) & 
                (y_coords >= 0) & (y_coords < img.shape[0]) &
                (((x_coords-128)/90)**2 + ((y_coords-128)/70)**2 <= 1)
            )
            
            x_coords = x_coords[valid_indices]
            y_coords = y_coords[valid_indices]
            
            for x, y in zip(x_coords, y_coords):
                rr, cc = disk((y, x), thickness, shape=img.shape)
                valid_indices = (
                    (cc >= 0) & (cc < img.shape[1]) & 
                    (rr >= 0) & (rr < img.shape[0]) &
                    (((cc-128)/90)**2 + ((rr-128)/70)**2 <= 1)
                )
                img[rr[valid_indices], cc[valid_indices]] = 0.45
    
    # Apply some noise and smoothing to make it more realistic
    img = img + 0.05 * np.random.randn(*img.shape)
    img = gaussian(img, sigma=1)
    
    # Add a dark background
    background = ((np.array(range(256))-128)**2 + (np.array(range(256)).reshape(-1, 1)-128)**2) > 90**2
    img[background] = 0
    
    # Normalize image to [0, 1]
    img = np.clip(img, 0, 1)
    
    return img

# Create the synthetic Phobos image
phobos = create_phobos()

# Save the image
imageio.imwrite('../plots/TP5_Enhancement/phobos.jpg', (phobos * 255).astype(np.uint8))

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(phobos, cmap='gray')
plt.title('Synthetic Phobos Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('../plots/TP5_Enhancement/phobos_display.png', dpi=300)
plt.show()

print("Synthetic Phobos image created and saved to '../plots/TP5_Enhancement/phobos.jpg'") 