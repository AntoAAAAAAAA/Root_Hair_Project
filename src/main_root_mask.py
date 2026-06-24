import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
from ultralytics import SAM

def makeGrayscaleImage(image_path):
    '''This function reads an image from a path and converts it to grayscale.'''

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image_gray

def createContoursAndFill(mask):
    '''This function takes a binary mask of the main root as input, finds the primary contour of the main root, 
    and fills it in to create a solid mask. 
    It returns a new binary mask where the main root is fully filled in.'''
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #this got changed from cv2.CHAIN_APPROX_SIMPLE. Keeps every single coordinate
    mask_closed_contour = np.zeros_like(mask)
    cv2.drawContours(mask_closed_contour, contours, -1, 255, -1)
    return mask_closed_contour, contours

# def overlayMaskOnImage(image_gray, final_mask):
#     '''This function takes the original grayscale image and the final mask, and creates an overlay where the mask area is highlighted in red.'''
    
#     # Convert grayscale image to RGB
#     img_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

#     # Create an overlay where the mask area is highlighted in red
#     overlay = img_rgb.copy()
#     overlay[final_mask == 255] = [255, 0, 0]  # Red color for the mask area

#     alpha = 0.4  # Transparency factor
#     final_overlay = cv2.addWeighted(img_rgb, alpha, overlay, 1 - alpha, 0)

#     return final_overlay
