import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import os

def createNewRootHairMask(image_grey, main_root):
    '''This function creates a new mask for root hairs using adaptive thresholding.'''

    better_adapt = cv2.adaptiveThreshold(image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)
    root_hair_mask = better_adapt.copy()
    expanded_root_mask = cv2.dilate(main_root, np.ones((7,7), np.uint8), iterations=1)
    root_hair_mask[expanded_root_mask > 0] = 0

    return root_hair_mask

def skeletonizeRootHairMask(root_hair_mask):
    'This function skeletonizes the root hair mask.'

    skeletonized_hairs = sk.morphology.skeletonize(root_hair_mask // 255)
    skeletonized_hairs = (skeletonized_hairs * 255).astype(np.uint8)

    return skeletonized_hairs

def addMainRootToSkeletonizedHairs(skeletonized_hairs, contour):
    'This function adds the main root boundary countour to skeletonized hairs to make a version with contours.'

    skeletonized_hairs_with_contours = skeletonized_hairs.copy()
    skeletonized_hairs_with_contours = cv2.drawContours(skeletonized_hairs_with_contours, contour, -1, 255, 1)

    return skeletonized_hairs_with_contours

def findBranchLength(y, x, neighbors, length_per_branch, root_hair_pixel_set, 
                     uniquePixelsTraversed, branchID, current_length): 
    '''A recursive function used to find the lengths of branches starting from a branchpoint.'''     
    uniquePixelsTraversed.add((y,x))
    num_neighbors = 0
    for dy, dx in neighbors:
        if (y+dy, x+dx) in root_hair_pixel_set:
            num_neighbors += 1
    if num_neighbors == 1:
        return current_length
    for dy, dx in neighbors:
        if (y+dy, x+dx) in root_hair_pixel_set and (y+dy, x+dx) not in uniquePixelsTraversed:
            return findBranchLength(y+dy, x+dx, neighbors, length_per_branch, 
                                    root_hair_pixel_set, uniquePixelsTraversed, branchID, 
                                    current_length+1)
    return current_length               

def makeValidRootHairMasks(skeletonized_hairs, contour, microscope_conversion_factor, upper, lower):
    ''''This function takes each component of the skeletonized hair mask and filters valid root hairs '
    The function chooses root hairs based on connectivity to main root, length parameters, the validity of endpoints, 
    and lack of branching.
    '''

    skeletonized_hairs_binary = (skeletonized_hairs > 0).astype(np.uint8)
    components_masks = sk.measure.label(skeletonized_hairs_binary, connectivity=2)
    num_objects = components_masks.max()
    print(f"Number of connected components: {num_objects}")

    valid_root_hair_masks = []

    filtered = 0 # for testing

    for i in range(1, num_objects + 1):
        one_component_mask = (components_masks == i).astype(np.uint8)

        # # for testing 
        # if one_component_mask.sum() < 10:  # Filter out small components
        #     continue
        # if one_component_mask.sum() > 500:  # Filter out large components
        #     continue

        if one_component_mask.sum() == 0:
            continue
        y, x = np.where(one_component_mask > 0)
        adjustment = 3
        h , w = one_component_mask.shape
        y_min = max(0, y.min() - adjustment)
        y_max = min(h-1, y.max() + adjustment)
        x_min = max(0, x.min() - adjustment)
        x_max = min(w-1, x.max() + adjustment)

        cropped_component_mask = one_component_mask[y_min:y_max+1, x_min:x_max+1]

        # for testing
        # filtered += 1
        # debugFolder = 'debug_images'
        # os.makedirs(debugFolder, exist_ok= True)
        # plt.figure(figsize=(5,5))
        # plt.imshow(cropped_component_mask, cmap='gray')
        # plt.title(f"Component {i}")
        # plt.axis('off')
        # plt.savefig(f'{debugFolder}/component_{i}.png', bbox_inches='tight')
        # plt.close()
        # print(f'''Component {i}, length: {length:.2f} pixels, 
        #        length: {length_in_microns:.2f} microns, 
        #        endpoints: {endpoints}, 
        #        branchpoints: {branchpoints},
        #        degrees: {list_of_degrees},
        #        ortho neighbors: {num_ortho_neighbors},
        #        diag neighbors: {num_diag_neighbors}"
        #     ''')
        
        # make a set of all pixels that make up the component (in tuples)
        coords = np.argwhere(cropped_component_mask > 0)
        root_hair_pixel_set = set(tuple(coordinate) for coordinate in coords)

        # make a kernel that holds the 8 neighbors of a pixel
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        # also make kernels that hold 2 orthogonal neighbors and 2 diagonal neighbors (forward directions only)
        ortho_neighbors = [(0, 1), (1, 0)]
        diag_neighbors = [(1, 1), (1, -1)]

        list_of_degrees = []
        num_ortho_neighbors = 0
        num_diag_neighbors = 0
        branch_coords = []
        # for each pixel in the component:
        for y, x in coords:
            num_neighbors = 0
            # check how many of its neighbors are also in the component by comparing 8-neighbor kernel
            for dy, dx in neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_neighbors += 1
            list_of_degrees.append(num_neighbors)
            if num_neighbors >= 3:
                branch_coords.append((y,x))
            # check how many orthogonal neighbors are in the component by comparing to ortho kernel
            for dy, dx in ortho_neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_ortho_neighbors += 1
            # check how many diagonal neighbors are in the component by comparing to diag kernel 
            for dy, dx in diag_neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_diag_neighbors += 1


            
        endpoints = list_of_degrees.count(1)
        branchpoints = sum(1 for x in list_of_degrees if x >= 3)
        # middle_points = list_of_degrees.count(2)

        # filter root hairs that aren't good for measuring 
        if endpoints == 1:
            continue
            
        # filter root hairs that have significant branching 
        valid_branches = True
        if endpoints > 2:
            # if num_neighbors >= 3:
            for y, x in branch_coords:
                length_per_branch = {}
                for dy, dx in neighbors: 
                    if (y + dy, x + dx) in root_hair_pixel_set:
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
                        
                short_branches = 0
                for length in length_per_branch.values():
                    if length < 5:
                        short_branches += 1
                if len(length_per_branch.keys()) == 4 and short_branches < 2:
                    valid_branches = False 
                    break
                if len(length_per_branch.keys()) == 3 and short_branches < 1:
                    valid_branches = False  
                    break
        
        if not valid_branches:
            continue 

        # calculate the length of the component using ortho and diag neighbors
        length = num_ortho_neighbors + (num_diag_neighbors * np.sqrt(2))
        length_in_microns = length * microscope_conversion_factor

        # filter root hairs that are too long/too short  
        if length_in_microns > upper or length_in_microns < lower: 
            continue

        # need to compare one_component_mask to contour
        coords_of_endpoints = [tuple(coords[i]) for i, d in enumerate(list_of_degrees) if d == 1]
        coords_of_endpoints = [(y + y_min, x + x_min) for y,x in coords_of_endpoints]
                
        # coords_of_contour = np.squeeze(contour[0])
        # coords_of_contour = set(map(tuple, coords_of_contour))
        coords_of_contour = contour[0].reshape(-1, 2)
        coords_of_contour = set(map(tuple, coords_of_contour.tolist()))

        endpoint_near_root = False
        for y,x in coords_of_endpoints:
            for n in range(1,8): # check if any of the pixels within a 20 pixel radius of the endpoint are in the contour
                if (x+n, y) in coords_of_contour or (x-n, y) in coords_of_contour:
                    endpoint_near_root = True
                    break
            if endpoint_near_root:
                break
        
        if not endpoint_near_root:
            continue 

        kernel = np.ones((4,4), np.uint8)
        thicker_mask = cv2.dilate(one_component_mask, kernel, iterations=1)


        valid_root_hair_masks.append({
            'id': i,
            'mask': one_component_mask,
            'thicker mask': thicker_mask,
            'length': length,
            'length in microns': length_in_microns,
        })

    # # testing
    # print(f"Number of filtered components: {filtered}")

    return valid_root_hair_masks, components_masks


def makeFinalMaskWithFinalRootHairs(image_grey, valid_root_hair_masks):
    ' This function creates an overlay of the filtered root hairs on top of the original grayscale image.'

    color_root = cv2.cvtColor(image_grey, cv2.COLOR_GRAY2RGB)
    overlay = color_root.copy()
    for i in range(0, len(valid_root_hair_masks)):
        valid_dict = valid_root_hair_masks[i]
        ind_mask = (valid_dict['mask'] * 255).astype(np.uint8)
        overlay[ind_mask == 255] = [0, 0 , 255]
    root_hair_overlay = cv2.addWeighted(color_root, 0.2, overlay, 0.8, 0)

    return root_hair_overlay