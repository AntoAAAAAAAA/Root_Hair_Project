import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import os


def skeletonizeRootHairMask(root_hair_mask):
    'This function skeletonizes the individual root hairs found in a mask that contains only root hairs.'

    skeletonized_hairs = sk.morphology.skeletonize(root_hair_mask // 255) # Turn image binary (0's and 1's)
    skeletonized_hairs = (skeletonized_hairs * 255).astype(np.uint8) # convert back to 0 and 255

    return skeletonized_hairs

def findBranchLength(y, x, neighbors, length_per_branch, root_hair_pixel_set, 
                     uniquePixelsTraversed, branchID, current_length): 
    '''A recursive function used to find the lengths of branches starting from a branchpoint (a pixel with more than 2 neighbors in it's 8-pixel radius).'''     
    
    # Add current coordinates to storage so we know that we have already traversed this pixel
    uniquePixelsTraversed.add((y,x))
    
    # Find the number of neighbors surrounding a pixel
    num_neighbors = 0
    for dy, dx in neighbors:
        if (y+dy, x+dx) in root_hair_pixel_set:
            num_neighbors += 1
    
    # If we the pixel only has 1 neighbor, we know we are at the end of a branch
    if num_neighbors == 1: 
        return current_length
    
    # If there are >1 neighbors, we recursively run the function on surrounding pixels 
    for dy, dx in neighbors:
        if (y+dy, x+dx) in root_hair_pixel_set and (y+dy, x+dx) not in uniquePixelsTraversed:
            return findBranchLength(y+dy, x+dx, neighbors, length_per_branch, 
                                    root_hair_pixel_set, uniquePixelsTraversed, branchID, 
                                    current_length+1)
    # Assuming none of the surrounding pixels meet our criteria, we can end the function 
    return current_length               

def makeValidRootHairAnalysis(skeletonized_hairs, contours, microscope_conversion_factor, upper, lower):
    ''''This function takes each component of the skeletonized hair mask and filters valid root hairs. '
    The function chooses root hairs based on connectivity to main root, length parameters, the validity of endpoints, 
    and lack of branching, and measures them.

    Returns:
        tuple:
            valid_root_hair_masks (list[dict]): List of dictionaries describing
                each valid root hair. Each dictionary contains the component ID,
                the binary root hair mask, a thicker dilated mask for
                visualization, the estimated length in pixels, and the estimated
                length in microns.
                
            components_masks (np.ndarray): Labeled connected-component image of
                the skeletonized root hair mask, where each nonzero integer
                corresponds to a candidate root hair component.

    '''

    skeletonized_hairs_binary = (skeletonized_hairs > 0).astype(np.uint8)
    components_masks = sk.measure.label(skeletonized_hairs_binary, connectivity=2)
    num_objects = components_masks.max()
    print(f"     Number of components found: {num_objects}")

    valid_root_hair_masks = []

    # filtered = 0 # for testing

    for i in range(1, num_objects + 1):

        # choose a single
        root_hair_mask = (components_masks == i).astype(np.uint8)

        # Check that root hair mask has pixels 
        if root_hair_mask.sum() == 0:
            continue

        # Crop mask to zoom in on the individual pixels that make up the root hair 
        y, x = np.where(root_hair_mask > 0)
        adjustment = 3
        h , w = root_hair_mask.shape
        y_min = max(0, y.min() - adjustment)
        y_max = min(h-1, y.max() + adjustment)
        x_min = max(0, x.min() - adjustment)
        x_max = min(w-1, x.max() + adjustment)
        cropped_component_mask = root_hair_mask[y_min:y_max+1, x_min:x_max+1]

        # Create a set of every pixel that makes up the root hair 
        coords = np.argwhere(cropped_component_mask > 0)
        root_hair_pixel_set = set(tuple(coordinate) for coordinate in coords)

        # Make kernels that represent neighboring pixels 
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        ortho_neighbors = [(0, 1), (1, 0)]
        diag_neighbors = [(1, 1), (1, -1)]

        # --- Identify all neighbors, orthogonal neighbors, and diagonal neighbors -----
        list_of_degrees = []
        num_ortho_neighbors = 0
        num_diag_neighbors = 0
        branch_coords = []

        # Loop through each pixel in the root hair 
        for y, x in coords:
            num_neighbors = 0

            # Check number of neighbors using 8-neighbor kernel
            for dy, dx in neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_neighbors += 1
            list_of_degrees.append(num_neighbors)
            if num_neighbors >= 3:
                branch_coords.append((y,x))

            # Check number of orthogonal neighbors using orthogonal kernel
            for dy, dx in ortho_neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_ortho_neighbors += 1

            # Check number of diagonal neighbors using diagonal kernel
            for dy, dx in diag_neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_diag_neighbors += 1

        # Find # of endpoints
        endpoints = list_of_degrees.count(1)

        # Exclude root hairs that only have 1 endpoint  
        if endpoints == 1:
            continue
            
        # Filter root hairs that have branches that are significantly long  
        valid_branches = True
        if endpoints > 2:
            for y, x in branch_coords:
                length_per_branch = {}
                for dy, dx in neighbors: 
                    if (y + dy, x + dx) in root_hair_pixel_set:
                        # Traverse pixels that make up root hair and recursively find how long the branches are
                        uniquePixelsTraversed = set()
                        uniquePixelsTraversed.add((y,x))
                        branchID = (y+dy, x+dx)
                        length_per_branch[branchID]= findBranchLength(y+dy, 
                                                                      x+dx, 
                                                                      neighbors,
                                                                      length_per_branch, 
                                                                      root_hair_pixel_set, 
                                                                      uniquePixelsTraversed, 
                                                                      branchID, 
                                                                      current_length = 0)

                # Find how many short branches there are   
                short_branches = 0
                for length in length_per_branch.values():
                    if length < 5:
                        short_branches += 1
                
                # Filter out root hairs that have significant branching 
                if len(length_per_branch.keys()) == 4 and short_branches < 2:
                    valid_branches = False 
                    break
                if len(length_per_branch.keys()) == 3 and short_branches < 1:
                    valid_branches = False  
                    break
        
        if not valid_branches:
            continue 

        # Calculate the length of the root hair using ortho and diag neighbors
        length = num_ortho_neighbors + (num_diag_neighbors * np.sqrt(2))
        length_in_microns = length * microscope_conversion_factor

        # Exclude root hairs that are too long/too short  
        if length_in_microns > upper or length_in_microns < lower: 
            continue

        # Check if root hair is within a reasonable distance to the main root 
        coords_of_endpoints = [tuple(coords[i]) for i, d in enumerate(list_of_degrees) if d == 1]
        coords_of_endpoints = [(y + y_min, x + x_min) for y,x in coords_of_endpoints]
        all_contours = np.vstack(contours)
        coords_of_contours = all_contours.reshape(-1, 2)
        coords_of_contours = set(map(tuple, coords_of_contours.tolist()))

        endpoint_near_root = False
        for y,x in coords_of_endpoints:
            for n in range(1,8): # check if any of the pixels within a 8 pixel radius of the endpoint are in the contours
                if (x+n, y) in coords_of_contours or (x-n, y) in coords_of_contours:
                    endpoint_near_root = True
                    break
            if endpoint_near_root:
                break
        
        if not endpoint_near_root:
            continue 
        
        # Create a dilated/thicker mask of the root hair (for visualization purposes)
        kernel = np.ones((4,4), np.uint8)
        thicker_mask = cv2.dilate(root_hair_mask, kernel, iterations=1)


        valid_root_hair_masks.append({
            'id': i,
            'mask': root_hair_mask,
            'thicker mask': thicker_mask,
            'cropped_component_mask': cropped_component_mask,
            'length': length,
            'length in microns': length_in_microns,
        })

        # testing
        # for mask_dict in valid_root_hair_masks:
        #     cropped_component_mask = mask_dict['cropped_component_mask']
        #     i = mask_dict['id']

        #     debugFolder = 'debug_images'
        #     os.makedirs(debugFolder, exist_ok= True)
        #     plt.figure(figsize=(5,5))
        #     plt.imshow(cropped_component_mask, cmap='gray')
        #     plt.title(f"Component {i}")
        #     plt.axis('off')
        #     plt.savefig(f'{debugFolder}/component_{i}.png', bbox_inches='tight')
        #     plt.close()

        

    print("     Root hair analysis complete")
    return valid_root_hair_masks, components_masks

def makeFinalMaskWithFinalRootHairs(image_grey, valid_root_hair_masks):
    '(Used primarily in testing) This function creates an overlay of the filtered root hairs on top of the original grayscale image.'

    
    color_root = cv2.cvtColor(image_grey, cv2.COLOR_GRAY2RGB)
    overlay = color_root.copy()
    for i in range(0, len(valid_root_hair_masks)):
        valid_dict = valid_root_hair_masks[i]
        ind_mask = (valid_dict['thicker mask'] * 255).astype(np.uint8)
        overlay[ind_mask == 255] = [0, 0 , 255]
    root_hair_overlay = cv2.addWeighted(color_root, 0.2, overlay, 0.8, 0)

    return root_hair_overlay