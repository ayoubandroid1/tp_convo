import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducible results
np.random.seed(42)

def image_load(image_path: str, channel: int = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Load an image with automatic detection of grayscale or RGB format.
    
    Args:
        image_path (str): Path to the image file
        channel (int, optional): 1 for grayscale, 3 for RGB, None for both
    
    Returns:
        np.ndarray or tuple: Loaded image(s)
    """
    # Check if file exists
    assert os.path.exists(image_path), f"Error: Image file not found at {image_path}"
    
    # Load image in color first to check channels
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"Error: Unable to load image from {image_path}"
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Load grayscale version
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img_gray is not None, f"Error: Unable to load grayscale image from {image_path}"
    
    # Assertions to test image types
    assert len(img_gray.shape) == 2, "Grayscale image should be 2D"
    assert len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3, "RGB image should be 3D with 3 channels"
    
    if channel == 1:
        return img_gray
    elif channel == 3:
        return img_rgb
    else:
        return img_gray, img_rgb

def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply convolution filter to an image.
    
    Args:
        image (np.ndarray): Input image (2D grayscale or 3D RGB)
        kernel (np.ndarray): Convolution kernel
    
    Returns:
        np.ndarray: Filtered image
    """
    # Input validation assertions
    assert isinstance(image, np.ndarray), "Image must be a NumPy array"
    assert isinstance(kernel, np.ndarray), "Kernel must be a NumPy array"
    assert len(kernel.shape) == 2, "Kernel must be a 2D matrix"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel must have odd dimensions"
    assert len(image.shape) in [2, 3], "Image must be grayscale (2D) or RGB (3D)"
    
    if len(image.shape) == 3:  # RGB image
        assert image.shape[2] == 3, "RGB image must have 3 channels"
        height, width, channels = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        
        # Apply filter to each channel
        for c in range(channels):
            output[:, :, c] = convolve_channel(image[:, :, c], kernel)
    else:  # Grayscale image
        output = convolve_channel(image, kernel)
    
    # Output size assertion
    assert output.shape == image.shape, "Output size must match input size"
    
    # Normalize to avoid values outside [0, 255] range
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def convolve_channel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply convolution to a single channel.
    
    Args:
        image (np.ndarray): Single channel image
        kernel (np.ndarray): Convolution kernel
    
    Returns:
        np.ndarray: Convolved channel
    """
    # Dimension validation assertions
    assert image.shape[0] >= kernel.shape[0], "Image too small for kernel (height)"
    assert image.shape[1] >= kernel.shape[1], "Image too small for kernel (width)"
    assert len(image.shape) == 2, "Channel must be 2D"
    
    # Image and kernel dimensions
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    
    # Add padding to handle edges
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 
                         mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)
    
    # Apply convolution
    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    # Output size assertion
    assert output.shape == image.shape, "Output channel size must match input channel size"
    
    return output

def display_images(images: list, titles: list, main_title: str = "Image Processing Results"):
    """
    Display multiple images in a single figure with captions.
    
    Args:
        images (list): List of images to display
        titles (list): List of titles for each image
        main_title (str): Main title for the figure
    """
    assert len(images) == len(titles), "Number of images must match number of titles"
    
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 4 * rows))
    plt.suptitle(main_title, fontsize=16, fontweight='bold')
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.title(title, fontsize=12)
        
        if len(img.shape) == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:  # RGB
            plt.imshow(img)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_kernels():
    """
    Create various convolution kernels including standard and random ones.
    
    Returns:
        dict: Dictionary of kernels with their names
    """
    kernels = {}
    
    # 1. Blur filter (average)
    kernels['blur_3x3'] = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])
    
    kernels['blur_5x5'] = np.ones((5, 5)) / 25
    kernels['blur_7x7'] = np.ones((7, 7)) / 49
    
    # 2. Sobel filters
    kernels['sobel_horizontal'] = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    kernels['sobel_vertical'] = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    # 3. Edge detection filters
    kernels['laplacian'] = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    
    kernels['edge_detection'] = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    
    # 4. Sharpening filter
    kernels['sharpen'] = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    # 5. Emboss filter
    kernels['emboss'] = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])
    
    # 6. Gaussian blur
    kernels['gaussian_3x3'] = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16
    
    kernels['gaussian_5x5'] = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]) / 256
    
    # 7. Motion blur
    kernels['motion_blur'] = np.zeros((9, 9))
    kernels['motion_blur'][4, :] = 1/9  # Horizontal motion blur
    
    # 8. Box filter
    kernels['box_5x5'] = np.ones((5, 5)) / 25
    
    # 9. Random filters with seed 42
    np.random.seed(42)
    kernels['random_3x3'] = np.random.randn(3, 3)
    kernels['random_3x3'] /= np.sum(np.abs(kernels['random_3x3']))  # Normalize
    
    kernels['random_5x5'] = np.random.randn(5, 5)
    kernels['random_5x5'] /= np.sum(np.abs(kernels['random_5x5']))  # Normalize
    
    kernels['random_7x7'] = np.random.randn(7, 7)
    kernels['random_7x7'] /= np.sum(np.abs(kernels['random_7x7']))  # Normalize
    
    return kernels

def save_results(images_dict: dict, prefix: str = ""):
    """
    Save filtered images to files.
    
    Args:
        images_dict (dict): Dictionary of images with their names
        prefix (str): Prefix for filenames
    """
    for name, img in images_dict.items():
        filename = f"{prefix}{name}.jpg"
        
        if len(img.shape) == 3:  # RGB image
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img_bgr)
        else:  # Grayscale image
            cv2.imwrite(filename, img)
        
        print(f"Saved: {filename}")

def main():
    """
    Main function to demonstrate convolution filters.
    """
    # Create a simple test image if no image is provided
    test_image_path = "test_image.jpg"
    
    # Create a test image with geometric patterns
    if not os.path.exists(test_image_path):
        print("Creating test image...")
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add some patterns
        test_img[50:150, 50:150] = [255, 255, 255]  # White square
        test_img[75:125, 75:125] = [255, 0, 0]      # Red square
        cv2.rectangle(test_img, (25, 25), (175, 175), (0, 255, 0), 3)  # Green border
        cv2.circle(test_img, (100, 100), 30, (0, 0, 255), -1)  # Blue circle
        cv2.imwrite(test_image_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        print(f"Test image created: {test_image_path}")
    
    try:
        # Load images
        print("Loading images...")
        gray_image, rgb_image = image_load(test_image_path)
        print(f"Gray image shape: {gray_image.shape}")
        print(f"RGB image shape: {rgb_image.shape}")
        
        # Create all kernels
        print("Creating kernels...")
        kernels = create_kernels()
        print(f"Created {len(kernels)} kernels")
        
        # Apply filters to grayscale image
        print("Processing grayscale image...")
        gray_results = {'original': gray_image}
        for name, kernel in kernels.items():
            try:
                filtered = apply_convolution(gray_image, kernel)
                gray_results[name] = filtered
                print(f"Applied {name} filter to grayscale image")
            except Exception as e:
                print(f"Error applying {name} to grayscale: {e}")
        
        # Apply filters to RGB image
        print("Processing RGB image...")
        rgb_results = {'original': rgb_image}
        for name, kernel in kernels.items():
            try:
                filtered = apply_convolution(rgb_image, kernel)
                rgb_results[name] = filtered
                print(f"Applied {name} filter to RGB image")
            except Exception as e:
                print(f"Error applying {name} to RGB: {e}")
        
        # Display results
        print("Displaying results...")
        
        # Display grayscale results
        gray_images = list(gray_results.values())
        gray_titles = list(gray_results.keys())
        display_images(gray_images, gray_titles, "Grayscale Image Processing Results")
        
        # Display RGB results
        rgb_images = list(rgb_results.values())
        rgb_titles = list(rgb_results.keys())
        display_images(rgb_images, rgb_titles, "RGB Image Processing Results")
        
        # Save results
        print("Saving results...")
        save_results({k: v for k, v in gray_results.items() if k != 'original'}, "gray_")
        save_results({k: v for k, v in rgb_results.items() if k != 'original'}, "rgb_")
        
        # Compare Sobel horizontal and vertical
        sobel_combined_gray = np.sqrt(
            gray_results['sobel_horizontal'].astype(np.float32)**2 + 
            gray_results['sobel_vertical'].astype(np.float32)**2
        )
        sobel_combined_gray = np.clip(sobel_combined_gray, 0, 255).astype(np.uint8)
        
        sobel_combined_rgb = np.sqrt(
            rgb_results['sobel_horizontal'].astype(np.float32)**2 + 
            rgb_results['sobel_vertical'].astype(np.float32)**2
        )
        sobel_combined_rgb = np.clip(sobel_combined_rgb, 0, 255).astype(np.uint8)
        
        # Display Sobel comparison
        sobel_comparison = [
            gray_results['original'], 
            gray_results['sobel_horizontal'], 
            gray_results['sobel_vertical'], 
            sobel_combined_gray
        ]
        sobel_titles = ['Original', 'Sobel Horizontal', 'Sobel Vertical', 'Combined Sobel']
        display_images(sobel_comparison, sobel_titles, "Sobel Filter Comparison (Grayscale)")
        
        print("Processing complete!")
        
        # Print kernel information
        print("\nKernel Information:")
        for name, kernel in kernels.items():
            print(f"{name}: {kernel.shape}")
            if kernel.shape[0] <= 3:  # Only print small kernels
                print(f"  {kernel}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()