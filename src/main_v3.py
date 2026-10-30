import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import plotly as plot
import plotly.express as px
from ultralytics import SAM

from src.main_root_mask import *
from src.root_hairs_mask import *
from src.final_visualize import *
from src.ML import nnUnet

def main_v3(image_gray, microscope_conversion_factor, upper, lower, predictor):

    # Put image through ML model 
    segmentation = nnUnet(image_gray, predictor)

    #  Create new masks
    segmentation = np.squeeze(segmentation) # remove the 1 dimension (1, 768, 1024) --> (768, 1024)
    main_root = (segmentation == 1).astype(np.uint8) * 255
    root_hairs = (segmentation == 2).astype(np.uint8) * 255

    # skeletonize hairs and create contour of main_root 
    skeleton_hairs = skeletonizeRootHairMask(root_hairs)
    contours, _ = cv2.findContours(main_root, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter for valid root hairs 
    valid_root_hair_masks, _ = extractAndFilterRootHairs(skeleton_hairs, contours, microscope_conversion_factor, upper, lower)
    
    # Create final overlay and a list of individual root hairs that can be displayed 
    fig, traces = makeFinalPlotlyVisual(image_gray, valid_root_hair_masks)

    return {
        'fig': fig,
        'traces': traces
    }