import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy

def make_grayscale_image(image_path):
    '''This function reads an image from a path and converts it to grayscale.'''

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image_gray

def create_binary_mask(image):
    '''This function takes a grayscale image as input and processes it to create a binary mask of the main root. The mask is returned as a binary image where the main root is highlighted.'''
    
    _, mask = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY_INV)
    return mask

def adaptive_threshold(image, block_size=35, C=8):
    '''This function applies adaptive thresholding to a grayscale image. The block size and constant C can be adjusted to improve the results. The function returns a binary image where the main root is highlighted.'''
    
    kernel4x4 = np.ones((4,4), np.uint8)
    adapt = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    # The last two numbers are the block size and the constant subtracted from the mean, respectively. You can adjust these parameters to see how they affect the result.
    adaptive_thresholded_image = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel4x4)
    return adaptive_thresholded_image

def create_opened_mask_with_rectangle_kernel(mask, kernel_size=(2,10)):
    '''This function takes a binary mask as input and applies morphological opening to remove small noise and separate connected components. The kernel size can be adjusted to improve the results. The function returns a new binary mask where the main root is more distinct.'''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask

def create_opened_mask_with_elliptical_kernel(mask, kernel_size=(5,5)):
    '''This function takes a binary mask as input and applies morphological opening to remove small noise and separate connected components. The kernel size can be adjusted to improve the results. The function returns a new binary mask where the main root is more distinct.'''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask

def create_binary_hole_filled_mask(mask):
    '''This function takes a binary mask as input and fills in any holes using scipy.ndimage.binary_fill_holes. The function returns a new binary mask.'''
    
    mask_bool = mask > 0
    filled_bool = scipy.ndimage.binary_fill_holes(mask_bool)
    filled_holes = (filled_bool.astype(np.uint8)) * 255
    return filled_holes

def create_distance_transform_mask(mask, threshold=8):
    '''This function takes a binary mask as input and applies a distance transform to identify the core of the main root. The function returns a new binary mask where the core of the main root is highlighted.'''
    
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    core = (dist_map >= threshold).astype(np.uint8) * 255
    return core

def create_dilated_mask(mask, core, ellipse_kernel_size=(2,10), iterations=2):
    '''This function takes a binary mask and a core mask as input, and applies dilation to the core mask to expand it. The dilation is limited by the inputted mask. The function returns a new binary mask where the core of the main root is expanded.'''

    expanded = core.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse_kernel_size)

    for _ in range(iterations):
        expanded = cv2.dilate(expanded, kernel, iterations=1)
        expanded = cv2.bitwise_and(expanded, mask)
    
    return expanded

def create_connected_component_mask(mask):
    '''This function takes a binary mask as input and identifies the largest connected component, which is assumed to be the main root. It returns a new binary mask where only the largest connected component is highlighted.'''
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background label
    CC_mask = np.zeros_like(mask)
    CC_mask[labels == largest_label] = 255
    return CC_mask

def create_morphologically_closed_mask(mask, x=7):
    '''This function takes a binary mask as input and applies morphological closing to fill in any small holes or gaps in the mask. It returns a new binary mask where the main root is more solid and continuous.'''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x,x)) # Adjust the kernel size as needed (e.g., (5,5) or (7,7))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask

def create_contours_and_fill(mask):
    '''This function takes a binary mask as input, finds the primary contour of the main root, and fills it in to create a solid mask. It returns a new binary mask where the main root is fully filled in.'''
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_closed_contour = np.zeros_like(mask)
    cv2.drawContours(mask_closed_contour, contours, -1, 255, -1)
    return mask_closed_contour

def overlay_mask_on_image(image_gray, final_mask):
    '''This function takes the original grayscale image and the final mask, and creates an overlay where the mask area is highlighted in red.'''
    
    # Convert grayscale image to RGB
    img_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    # Create an overlay where the mask area is highlighted in red
    overlay = img_rgb.copy()
    overlay[final_mask == 255] = [255, 0, 0]  # Red color for the mask area

    alpha = 0.4  # Transparency factor
    final_overlay = cv2.addWeighted(img_rgb, alpha, overlay, 1 - alpha, 0)

    return final_overlay
