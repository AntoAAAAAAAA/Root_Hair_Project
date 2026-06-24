# ------ Main Root Mask ---------- 

def createBinaryMask(image):
    '''This function takes a grayscale image as input and processes it to create a binary mask of the main root. The mask is returned as a binary image where the main root is highlighted.'''
    
    _, mask = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY_INV)
    return mask

def adaptiveThreshold(image, block_size=35, C=8):
    '''This function applies adaptive thresholding to a grayscale image. The block size and constant C can be adjusted to improve the results. The function returns a binary image where the main root is highlighted.'''
    
    kernel4x4 = np.ones((4,4), np.uint8)
    adapt = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    # The last two numbers are the block size and the constant subtracted from the mean, respectively. You can adjust these parameters to see how they affect the result.
    adaptive_thresholded_image = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel4x4)
    return adaptive_thresholded_image

def createOpenedMaskWithRectKernel(mask, kernel_size=(2,10)):
    '''This function takes a binary mask as input and applies morphological opening to remove small noise and separate connected components. The kernel size can be adjusted to improve the results. The function returns a new binary mask where the main root is more distinct.'''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask

def createOpenedMaskwithElliptKernel(mask, kernel_size=(5,5)):
    '''This function takes a binary mask as input and applies morphological opening to remove small noise and separate connected components. The kernel size can be adjusted to improve the results. The function returns a new binary mask where the main root is more distinct.'''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask

def createBinaryHoleFilledMask(mask):
    '''This function takes a binary mask as input and fills in any holes using scipy.ndimage.binary_fill_holes. The function returns a new binary mask.'''
    
    mask_bool = mask > 0
    filled_bool = scipy.ndimage.binary_fill_holes(mask_bool)
    filled_holes = (filled_bool.astype(np.uint8)) * 255
    return filled_holes

def createDistanceTransformMask(mask, threshold=8):
    '''This function takes a binary mask as input and applies a distance transform to identify the core of the main root. The function returns a new binary mask where the core of the main root is highlighted.'''
    
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    core = (dist_map >= threshold).astype(np.uint8) * 255
    return core

def createDilatedMask(mask, core, ellipse_kernel_size=(2,10), iterations=2):
    '''This function takes a binary mask and a core mask as input, and applies dilation to the core mask to expand it. The dilation is limited by the inputted mask. The function returns a new binary mask where the core of the main root is expanded.'''

    expanded = core.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse_kernel_size)

    for _ in range(iterations):
        expanded = cv2.dilate(expanded, kernel, iterations=1)
        expanded = cv2.bitwise_and(expanded, mask)
    
    return expanded

def createConnectedComponentMask(mask):
    '''This function takes a binary mask as input and identifies the largest connected component, which is assumed to be the main root. It returns a new binary mask where only the largest connected component is highlighted.'''
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background label
    CC_mask = np.zeros_like(mask)
    CC_mask[labels == largest_label] = 255
    return CC_mask

def createMorphologicallyClosedMask(mask, x=7): 
    '''This function takes a binary mask as input and applies morphological closing to fill in any small holes or gaps in the mask. It returns a new binary mask where the main root is more solid and continuous.'''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x,x)) # Adjust the kernel size as needed (e.g., (5,5) or (7,7))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask


# ---- Root Hair Mask -------

def createNewRootHairMask(image_grey, main_root):
    '''This function creates a new mask for root hairs using adaptive thresholding.'''

    better_adapt = cv2.adaptiveThreshold(image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 8)
    root_hair_mask = better_adapt.copy()
    expanded_root_mask = cv2.dilate(main_root, np.ones((7,7), np.uint8), iterations=1)
    root_hair_mask[expanded_root_mask > 0] = 0

    return root_hair_mask

def addMainRootToSkeletonizedHairs(skeletonized_hairs, contour):
    'This function adds the main root boundary countour to skeletonized hairs to make a version with contours.'

    skeletonized_hairs_with_contours = skeletonized_hairs.copy()
    skeletonized_hairs_with_contours = cv2.drawContours(skeletonized_hairs_with_contours, contour, -1, 255, 1)

    return skeletonized_hairs_with_contours

# This version has the testing code in it 
def makeValidRootHairAnalysis(skeletonized_hairs, contour, microscope_conversion_factor, upper, lower):
    ''''This function takes each component of the skeletonized hair mask and filters valid root hairs. '
    The function chooses root hairs based on connectivity to main root, length parameters, the validity of endpoints, 
    and lack of branching, and measures them. 
    '''

    skeletonized_hairs_binary = (skeletonized_hairs > 0).astype(np.uint8)
    components_masks = sk.measure.label(skeletonized_hairs_binary, connectivity=2)
    num_objects = components_masks.max()
    print(f"Number of connected components: {num_objects}")

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
        # branchpoints = sum(1 for x in list_of_degrees if x >= 3)
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

        # need to compare root_hair_mask to contour
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
        thicker_mask = cv2.dilate(root_hair_mask, kernel, iterations=1)


        valid_root_hair_masks.append({
            'id': i,
            'mask': root_hair_mask,
            'thicker mask': thicker_mask,
            'length': length,
            'length in microns': length_in_microns,
        })

    # # testing
    # print(f"Number of filtered components: {filtered}")

    return valid_root_hair_masks, components_masks