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
