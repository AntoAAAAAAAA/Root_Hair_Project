'''This is old root hair mask creation code that relied on connectedComponents'''

# Connected component analysis on skeletonized hairs
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeletonized_hairs, connectivity=8)
'''I think I need to consider finding alternatives to connectedComponents
It looks like it isn't able to distinguish the individual root hairs. 
Other options Google has recommended:
- cv2.findContours
- cv2.simpleBlobDetector
- skimage.measure.label
- skimage.measure.regionprops
- skan (specifically for skeletons)'''


temporaryStoreOfComponentMasksAndLengths = []
final_lengths = []
component_masks = {}
microscope_conversion_factor = 3.3  # microns per pixel. This needs to be a variable for the user
for i in range(1, num_labels):  # Skip the background label
    component_mask = np.zeros_like(skeletonized_hairs)
    component_mask[labels == i] = 1
    binary_component = (component_mask > 0).astype(np.uint8)

    # temporary for testing
    display_mask = (binary_component * 255).astype(np.uint8)
    display_mask = cv2.dilate(display_mask, np.ones((3,3), np.uint8), iterations=1) #iterations of 3 and 3 also did not help
    # temporary for testing 


    kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)
                    # kernel counts how many neighbors are equal to 1
    neighbor_count = cv2.filter2D(binary_component, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count *= binary_component # keep counts only on skeleton pixels

    if np.any(neighbor_count > 2):
        continue # Skip this component if it has any intersection points 

    endpoints = np.sum(neighbor_count == 1)
    if endpoints != 2:
        continue # Skip this component if it doesn't have exactly 2 endpoints

    # Left with components with only endpoints and boundary points. Calculate length of each 
    diag_kernel = np.array([[1, 0, 1],
                            [0, 0, 0],
                            [1, 0, 1]], dtype=np.uint8)
    diag_counts = scipy.ndimage.convolve(binary_component, diag_kernel, mode='constant') * binary_component
    n_diag = np.sum(diag_counts) / 2 # Each diagonal neighbor is counted twice in the convolution

    n_total = np.sum(neighbor_count) / 2
    n_ortho = n_total - n_diag

    length = (n_ortho * 1.0) + (n_diag * np.sqrt(2))
    length = length * microscope_conversion_factor
    # if length.any() > 100 or length.any() < 10:
    #     continue # Skip this component if the length is outside of expected range for root hairs

    final_lengths.append({
        "component_id": i,
        "length_microns": length * microscope_conversion_factor,
        "num_pixels": int(np.sum(binary_component)),
        "num_endpoints": int(endpoints)
    })
    component_masks[i] = binary_component.copy()

    temporaryStoreOfComponentMasksAndLengths.append(
        {
            'component_id': i,
            'component_mask': component_mask,
            'binary_component': binary_component,
            'display_mask': display_mask,
            'neighbor_count': neighbor_count,
            'diag_kernel': diag_kernel,
            'diag_counts': diag_counts,
            'n_diag': n_diag,
            'n_total': n_total,
            'n_ortho': n_ortho,
            'length': length
        }
    )
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  
    axs[0].imshow(root_hair_mask, cmap='gray')
    axs[0].set_title('Root Hair Mask')
    axs[0].axis('off')
    axs[1].imshow(skeletonized_hairs, cmap='gray')
    axs[1].set_title('Skeletonized Hairs')
    axs[1].axis('off')
    axs[2].imshow(display_mask, cmap='gray')
    axs[2].set_title('Current Component Mask')
    axs[2].axis('off')
    axs[3].imshow(neighbor_count, cmap='gray')
    axs[3].set_title('Neighbor Count')
    axs[3].axis('off')
    plt.tight_layout()
    plt.show()

    plt.imshow(skeletonized_hairs, cmap='gray')
    plt.title('Skeletonized Hairs')
    plt.axis('off')
    plt.show()
    plt.imshow(binary_component, cmap='gray')
    plt.title('Binary Component')
    plt.axis('off')
    plt.show()
    # answer = input("Put continue to continue or exit:")
    # if answer == "continue":
    #     continue
    # elif answer == "exit":
    #     break