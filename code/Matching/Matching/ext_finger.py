"""
ext_finger
Extracts minutiae from a fingerprint image

Usage: [minutiae, binim, mask, cimg1, cimg2, oimg1, oimg2] = ext_finger(img, show)

Arguments:
    img - fingerprint image
    show - flag to display results (optional, default=0)

Returns:
    minutiae - extracted minutiae points [x, y, orientation, type]
    binim - binary image
    mask - binary mask
    cimg1, cimg2 - coherence images
    oimg1, oimg2 - orientation images
"""

import numpy as np
import matplotlib.pyplot as plt
from f_enhance import f_enhance
from FExtraction.p import p
from FExtraction.test_bifurcation import test_bifurcation

def ext_finger(img, show=0):
    """
    Extracts minutiae from a fingerprint image
    
    Parameters:
    img - fingerprint image
    show - flag to display results (optional, default=0)
    
    Returns:
    minutiae - extracted minutiae points [x, y, orientation, type]
    binim - binary image
    mask - binary mask
    cimg1, cimg2 - coherence images
    oimg1, oimg2 - orientation images
    """
    # Enhance the fingerprint image
    binim, mask, cimg1, cimg2, oimg1, oimg2 = f_enhance(img)
    
    # Convert to binary if not already
    if binim.dtype != bool:
        binim = binim > 0
    
    # Get image dimensions
    rows, cols = binim.shape
    
    # Initialize minutiae list
    minutiae = []
    
    # Scan the image for minutiae
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if binim[i, j] == 1:  # If current pixel is a ridge
                # Get 8-connected neighbors
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbors.append(binim[ni, nj])
                        else:
                            neighbors.append(0)
                
                # Count ridge neighbors
                ridge_count = sum(neighbors)
                
                # Check for minutiae
                if ridge_count == 1:  # Ridge ending
                    # Calculate orientation
                    orientation = calculate_ridge_orientation(binim, i, j, oimg2)
                    minutiae.append([j, i, orientation, 1])  # [x, y, orientation, type] - type 1 for ending
                    
                elif ridge_count == 3:  # Bifurcation
                    # Calculate orientation
                    orientation = calculate_ridge_orientation(binim, i, j, oimg2)
                    
                    # Test bifurcation direction
                    if test_bifurcation_direction(binim, i, j, orientation):
                        minutiae.append([j, i, orientation, 2])  # [x, y, orientation, type] - type 2 for bifurcation
    
    # Convert to numpy array
    if minutiae:
        minutiae = np.array(minutiae)
        
        # Filter false minutiae (remove minutiae too close to each other)
        minutiae = filter_false_minutiae(minutiae, min_distance=10)
        
        # Filter minutiae near image borders
        border_margin = 20
        valid_mask = (
            (minutiae[:, 0] >= border_margin) & 
            (minutiae[:, 0] < cols - border_margin) &
            (minutiae[:, 1] >= border_margin) & 
            (minutiae[:, 1] < rows - border_margin)
        )
        minutiae = minutiae[valid_mask]
    else:
        minutiae = np.array([]).reshape(0, 4)
    
    # Display results if requested
    if show:
        display_results(img, binim, minutiae, mask, cimg1, cimg2, oimg1, oimg2)
    
    return minutiae, binim, mask, cimg1, cimg2, oimg1, oimg2


def calculate_ridge_orientation(binim, i, j, oimg):
    """
    Calculate ridge orientation at a given point
    
    Parameters:
    binim - binary image
    i, j - coordinates
    oimg - orientation image
    
    Returns:
    orientation - ridge orientation in radians
    """
    # Use orientation from orientation image if available
    if oimg is not None and oimg.size > 0:
        # Map pixel coordinates to orientation image coordinates
        oi = min(int(i * oimg.shape[0] / binim.shape[0]), oimg.shape[0] - 1)
        oj = min(int(j * oimg.shape[1] / binim.shape[1]), oimg.shape[1] - 1)
        return oimg[oi, oj]
    
    # Fallback: calculate orientation from local ridge direction
    window_size = 5
    half_window = window_size // 2
    
    # Extract local window
    i_start = max(0, i - half_window)
    i_end = min(binim.shape[0], i + half_window + 1)
    j_start = max(0, j - half_window)
    j_end = min(binim.shape[1], j + half_window + 1)
    
    local_window = binim[i_start:i_end, j_start:j_end]
    
    # Calculate gradients
    gy, gx = np.gradient(local_window.astype(float))
    
    # Calculate orientation
    orientation = np.arctan2(np.mean(gy), np.mean(gx))
    
    return orientation


def test_bifurcation_direction(binim, i, j, orientation):
    """
    Test if bifurcation direction is consistent
    
    Parameters:
    binim - binary image
    i, j - coordinates
    orientation - ridge orientation
    
    Returns:
    is_valid - True if bifurcation is valid
    """
    # Use the test_bifurcation function from FExtraction
    try:
        # Get pixel values using the p function
        pixel_values, coords = p(binim, i, j)
        return test_bifurcation(pixel_values, orientation)
    except:
        # Fallback simple test
        return True


def filter_false_minutiae(minutiae, min_distance=10):
    """
    Filter out false minutiae that are too close to each other
    
    Parameters:
    minutiae - array of minutiae [x, y, orientation, type]
    min_distance - minimum distance between minutiae
    
    Returns:
    filtered_minutiae - filtered minutiae array
    """
    if len(minutiae) == 0:
        return minutiae
    
    # Calculate pairwise distances
    distances = np.sqrt(
        (minutiae[:, 0:1] - minutiae[:, 0:1].T)**2 + 
        (minutiae[:, 1:2] - minutiae[:, 1:2].T)**2
    )
    
    # Set diagonal to infinity to avoid self-comparison
    np.fill_diagonal(distances, np.inf)
    
    # Find minutiae that are too close
    too_close = distances < min_distance
    
    # Keep only one minutiae from each close pair
    to_remove = set()
    for i in range(len(minutiae)):
        if i in to_remove:
            continue
        close_indices = np.where(too_close[i])[0]
        for j in close_indices:
            if j > i:  # Only consider pairs once
                to_remove.add(j)
    
    # Remove marked minutiae
    keep_indices = [i for i in range(len(minutiae)) if i not in to_remove]
    return minutiae[keep_indices]


def display_results(img, binim, minutiae, mask, cimg1, cimg2, oimg1, oimg2):
    """
    Display the results of minutiae extraction
    
    Parameters:
    img - original image
    binim - binary image
    minutiae - extracted minutiae
    mask, cimg1, cimg2, oimg1, oimg2 - other processing results
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary image
    axes[0, 1].imshow(binim, cmap='gray')
    axes[0, 1].set_title('Binary Image')
    axes[0, 1].axis('off')
    
    # Mask
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Mask')
    axes[0, 2].axis('off')
    
    # Coherence image 1
    axes[0, 3].imshow(cimg1, cmap='jet')
    axes[0, 3].set_title('Coherence Image 1')
    axes[0, 3].axis('off')
    
    # Coherence image 2
    axes[1, 0].imshow(cimg2, cmap='jet')
    axes[1, 0].set_title('Coherence Image 2')
    axes[1, 0].axis('off')
    
    # Orientation image 1
    axes[1, 1].imshow(oimg1, cmap='hsv')
    axes[1, 1].set_title('Orientation Image 1')
    axes[1, 1].axis('off')
    
    # Orientation image 2
    axes[1, 2].imshow(oimg2, cmap='hsv')
    axes[1, 2].set_title('Orientation Image 2')
    axes[1, 2].axis('off')
    
    # Minutiae overlay
    axes[1, 3].imshow(img, cmap='gray')
    if len(minutiae) > 0:
        # Plot endings in red
        endings = minutiae[minutiae[:, 3] == 1]
        if len(endings) > 0:
            axes[1, 3].scatter(endings[:, 0], endings[:, 1], c='red', s=20, marker='o', label='Endings')
        
        # Plot bifurcations in blue
        bifurcations = minutiae[minutiae[:, 3] == 2]
        if len(bifurcations) > 0:
            axes[1, 3].scatter(bifurcations[:, 0], bifurcations[:, 1], c='blue', s=20, marker='s', label='Bifurcations')
        
        axes[1, 3].legend()
    
    axes[1, 3].set_title(f'Minutiae ({len(minutiae)} found)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    # Show the plot for 1 second then automatically close
    plt.show(block=False)
    plt.pause(1.0)  # Display for 1 second
    plt.close()  # Automatically close the figure


# Example usage
if __name__ == "__main__":
    # Create a sample fingerprint image
    img = np.random.rand(256, 256) * 255
    img = img.astype(np.float64)
    
    # Test the function
    try:
        minutiae, binim, mask, cimg1, cimg2, oimg1, oimg2 = ext_finger(img, show=1)
        print("Minutiae extraction completed successfully")
        print(f"Number of minutiae found: {len(minutiae)}")
        if len(minutiae) > 0:
            print(f"Minutiae shape: {minutiae.shape}")
            print(f"Sample minutiae: {minutiae[:5]}")  # Show first 5 minutiae
    except Exception as e:
        print(f"Error during minutiae extraction: {e}")