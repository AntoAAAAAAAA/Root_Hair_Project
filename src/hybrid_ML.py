import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from ultralytics import SAM

def hybrid_main(image, image_gray):
    '''This function creates the main root mask using SAM and then creates the root hair mask by subtracting the SAM mask from a thresholded mask.'''
    
    model = SAM('sam2_l.pt')

    _, thresh_mask = cv2.threshold(image_gray, 130, 255, cv2.THRESH_BINARY_INV)
    # _, otsu_mask = cv2.threshold( image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    print("Thresholding complete")
    num_labels, labels, stats, centroids =cv2.connectedComponentsWithStats(thresh_mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    binary_mask = (labels == largest_label).astype(np.uint8) * 255
    

    # dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    # Imma try putting the otsu_mask into distanceTransform instead of the binary
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    core = (dist_map >= 5).astype(np.uint8) * 255
    print("Distance transform complete")

    # new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,10)) #vert kernel
    # new_opened = cv2.morphologyEx(core, cv2.MORPH_OPEN, new_kernel)

    pts = cv2.findNonZero(core)
    x, y, w, h = cv2.boundingRect(pts) 
    # (x,y) is top left. +x is going right, +y is going down
    # coordinates are (x,y), (x+h,y), (x,y+w), (x+w,y+h)
    # bounded_rect = cv2.rectangle(image_gray.copy(), (x, y), (x+w, y+h), (255,0,0), 2)
    print("Bounding rectangle complete")

    print("SAM prediction initialized...")
    results = model.predict(image, bboxes = [[x, y, x+w, y+h]])
    print("SAM prediction complete")

    # annotated = results[0].plot()


    # Creation of root hair mask
    sam_core = results[0].masks.data[0].cpu().numpy()
    sam_core_grayscale = (sam_core > 0).astype(np.uint8) * 255
    # sam_mask_grayscale_resized = cv2.resize(sam_core_grayscale, 
    #                                 (image_gray.shape[1], image_gray.shape[0]),
    #                                 interpolation=cv2.INTER_NEAREST
                                    # )
    
    # expanded_sam_grayscale = cv2.dilate(sam_core_grayscale, np.ones((3,3), np.uint8), iterations=1)
    
    # final_subtracted_mask = cv2.subtract(otsu_mask, sam_mask_grayscale_resized)
    final_subtracted_mask = thresh_mask.copy()
    final_subtracted_mask[sam_core_grayscale > 0] = 0
    
    return sam_core_grayscale, final_subtracted_mask, thresh_mask, binary_mask, core

def hybrid_main2(image, image_gray, model):
    '''This version uses SAM to subtract out the background. The resulting image 
    is then thresholded twice: 1) whole root w/ hairs and 2) core main root only. 
    The core threshold is used to create a bounding rectangle, which is then given to SAM-2 
    to find the main root with optimal accuracy. The resulting core main root is then 
    subtracted from the whole root threshold mask, leaving behind a mask that contains 
    only the root hairs.  

    Returns:
        tuple:
            sam_mask_grayscale (np.ndarray): Binary mask of the detected main root
                after SAM-assisted segmentation and grayscale processing.
            
            root_hair_mask (np.ndarray): Binary mask containing candidate root hair
                regions after subtraction of the main root.
    '''

    # Uses SAM-2, large model
    print('\n')
    print('[1/5] Running SAM full-image segmentation...')
    # Run model using bounding box that has 4 points of the image
    H, W = image.shape[:2]
    results = model.predict(image, bboxes = [[0, 0 , W, H]], verbose = False)
    print('     Segmentation complete')
    # Find the largest mask --> background 
    annotated = results[0].masks.data.cpu().numpy()
    areas = [mask.sum() for mask in annotated]
    largest_idx = np.argmax(areas)
    largest_mask = annotated[largest_idx]
    largest_mask = largest_mask.astype(np.uint8)

    # Turn the SAM background black, and keep the grayscale image of the root with its hairs 
    overlay = image_gray.copy()
    overlay[largest_mask > 0] = 255
    print('\n')
    print("[2/5] Thresholding Image...")
    # Thresholding of the overlay image and the grayscale image 
    _, main_thresh = cv2.threshold(overlay, 180, 255, cv2.THRESH_BINARY_INV)
    _, core_thresh = cv2.threshold(image_gray, 130, 255, cv2.THRESH_BINARY_INV)
    print("     Thresholding complete")


    # --- Main root logic done on core_thresh to extract the main root ----
    print('\n')
    print("[3/5] Begin finding main root...")
    # Find connected components from core 
    num_labels, labels, stats, centroids =cv2.connectedComponentsWithStats(core_thresh)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    binary_mask = (labels == largest_label).astype(np.uint8) * 255
    
    
    # Distance transform
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    core = (dist_map >= 5).astype(np.uint8) * 255   
    print("     Distance transform complete")
    
    # Create a bounding rectangle using the resulting core
    pts = cv2.findNonZero(core)
    x, y, w, h = cv2.boundingRect(pts) 
    # (x,y) is top left. +x is going right, +y is going down
    # coordinates are (x,y), (x+h,y), (x,y+w), (x+w,y+h)
    print("     Bounding rectangle complete")
    
    # Plug bounding rectangle into SAM and let it extract the main root with optimal accuracy
    print("     SAM prediction begin for main root...")
    results = model.predict(image, bboxes = [[x, y, x+w, y+h]], verbose = False)
    print("     SAM main root found!")
    
    # Take SAM mask results and extract the main root (largest one)
    all_masks = results[0].masks.data.cpu().numpy()
    areas = [mask.sum() for mask in all_masks]
    largest_idx = np.argmax(areas)
    sam_core = all_masks[largest_idx]
    sam_core_grayscale = (sam_core > 0).astype(np.uint8) * 255

    # Dilate the main root mask
    expanded_root = cv2.dilate(
    sam_core_grayscale,
    np.ones((3,3), np.uint8),
    iterations=1
    )

    # Subtract the dilated SAM main root from the original image. Left with mask of just root hairs (final_subtracted_mask)
    final_subtracted_mask = main_thresh.copy()
    final_subtracted_mask[expanded_root > 0] = 0
    
    return sam_core_grayscale, final_subtracted_mask