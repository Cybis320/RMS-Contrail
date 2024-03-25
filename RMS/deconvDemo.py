import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def demonstrate_shift_direction():
    """
    Create a definitive demonstration of the direction of the half-pixel shift
    caused by a 2x2 mean filter, and validate the correct correction.
    """
    # Create a very simple image with a single point source
    size = 7
    image = np.zeros((size, size))
    center = size // 2  # Integer division to get center pixel (3 for 7x7 image)
    image[center, center] = 1.0  # Set center pixel to 1.0
    
    # Apply the 2x2 filter
    filtered = ndimage.filters.convolve(image, weights=np.full((2, 2), 1.0/4))
    
    # Calculate center of mass for both images
    y_indices, x_indices = np.indices(image.shape)
    
    # Original image
    total_original = np.sum(image)
    com_y_original = np.sum(y_indices * image) / total_original
    com_x_original = np.sum(x_indices * image) / total_original
    
    # Filtered image
    total_filtered = np.sum(filtered)
    com_y_filtered = np.sum(y_indices * filtered) / total_filtered
    com_x_filtered = np.sum(x_indices * filtered) / total_filtered
    
    # Calculate shift
    shift_x = com_x_filtered - com_x_original
    shift_y = com_y_filtered - com_y_original
    
    # Test both possible corrections
    sub_corrected_x = com_x_filtered - 0.5
    sub_corrected_y = com_y_filtered - 0.5
    sub_error_x = sub_corrected_x - com_x_original
    sub_error_y = sub_corrected_y - com_y_original
    
    add_corrected_x = com_x_filtered + 0.5
    add_corrected_y = com_y_filtered + 0.5
    add_error_x = add_corrected_x - com_x_original
    add_error_y = add_corrected_y - com_y_original
    
    # Determine which correction is better
    sub_error = np.sqrt(sub_error_x**2 + sub_error_y**2)
    add_error = np.sqrt(add_error_x**2 + add_error_y**2)
    
    if sub_error < add_error:
        better_correction = "SUBTRACT 0.5"
        corrected_x, corrected_y = sub_corrected_x, sub_corrected_y
    else:
        better_correction = "ADD 0.5"
        corrected_x, corrected_y = add_corrected_x, add_corrected_y
    
    # Print results
    print("HALF-PIXEL SHIFT DIRECTION TEST")
    print("===============================")
    print(f"Original image center of mass: ({com_x_original:.4f}, {com_y_original:.4f})")
    print(f"Filtered image center of mass: ({com_x_filtered:.4f}, {com_y_filtered:.4f})")
    print(f"Measured shift: ({shift_x:.4f}, {shift_y:.4f})")
    print("\nCORRECTION TESTING")
    print("==================")
    print(f"If SUBTRACT 0.5: ({sub_corrected_x:.4f}, {sub_corrected_y:.4f})")
    print(f"Error from original: ({sub_error_x:.4f}, {sub_error_y:.4f}) = {sub_error:.4f}")
    print(f"If ADD 0.5: ({add_corrected_x:.4f}, {add_corrected_y:.4f})")
    print(f"Error from original: ({add_error_x:.4f}, {add_error_y:.4f}) = {add_error:.4f}")
    print(f"\nCONCLUSION: The correct correction is to {better_correction}")
    
    # Visualize the results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Helper function to display an image with pixel values and markers
    def display_image(ax, img, title, com_x=None, com_y=None, corrected_x=None, corrected_y=None):
        im = ax.imshow(img, origin='lower', cmap='viridis', interpolation='nearest')
        ax.set_title(title)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, img.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, img.shape[0], 1))
        ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add pixel values
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img[y, x] > 0.001:  # Only show non-zero values
                    ax.text(x, y, f"{img[y, x]:.3f}", ha='center', va='center', 
                            color='white' if img[y, x] > 0.3 else 'black',
                            fontsize=9, fontweight='bold')
        
        # Add coordinate labels
        for i in range(img.shape[1]):
            ax.text(i, -1, str(i), ha='center', va='center', fontsize=8)
        for i in range(img.shape[0]):
            ax.text(-1, i, str(i), ha='center', va='center', fontsize=8)
        
        # Plot center of mass
        if com_x is not None and com_y is not None:
            ax.plot(com_x, com_y, 'r+', markersize=12, label=f'CoM: ({com_x:.2f}, {com_y:.2f})')
        
        # Plot corrected position
        if corrected_x is not None and corrected_y is not None:
            ax.plot(corrected_x, corrected_y, 'bX', markersize=12, label=f'Corrected: ({corrected_x:.2f}, {corrected_y:.2f})')
        
        # Add true center marker
        ax.plot(center, center, 'wo', markersize=12, fillstyle='none', label=f'True center: ({center}, {center})')
        
        ax.legend(loc='upper right', fontsize=9)
        return im
    
    # Display the images
    display_image(axes[0, 0], image, "Original Image", com_x_original, com_y_original)
    display_image(axes[0, 1], filtered, "Filtered Image", com_x_filtered, com_y_filtered)
    display_image(axes[1, 0], filtered, "Corrected by SUBTRACTING 0.5", 
                  com_x_filtered, com_y_filtered, sub_corrected_x, sub_corrected_y)
    display_image(axes[1, 1], filtered, "Corrected by ADDING 0.5", 
                  com_x_filtered, com_y_filtered, add_corrected_x, add_corrected_y)
    
    plt.tight_layout()
    plt.savefig('definitive_shift_direction.png', dpi=150)
    plt.show()
    
    # Return the shift direction for reference
    return shift_x, shift_y, better_correction

if __name__ == "__main__":
    shift_x, shift_y, correction = demonstrate_shift_direction()
    
    # Summarize findings as a clear recommendation
    print("\nRECOMMENDATION FOR CODE CHANGE:")
    print("==============================")
    if "SUBTRACT" in correction:
        print("Add the following code before returning the star positions:")
        print("# Compensate for half-pixel shift caused by the 2×2 mean filter")
        print("x_arr = [x - 0.5 for x in x_arr]")
        print("y_arr = [y - 0.5 for y in y_arr]")
    else:
        print("Add the following code before returning the star positions:")
        print("# Compensate for half-pixel shift caused by the 2×2 mean filter")
        print("x_arr = [x + 0.5 for x in x_arr]")
        print("y_arr = [y + 0.5 for y in y_arr]")