import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import plotly as plot
import plotly.express as px
from ultralytics import SAM

from main_root_mask import *
from root_hairs_mask import *
from final_visualize import *
from hybrid_ML import *

# always include image_gray as an input for mains. Streamlit is weird with grayscale conversion
def main_v2(image, image_gray, microscope_conversion_factor, upper, lower):
    
    # Creation of main root mask 
    sam_mask_grayscale, root_hair_mask  = hybrid_main2(image, image_gray)

    # Creat root hair mask
    skeletonized_hairs = skeletonizeRootHairMask(root_hair_mask)
    mask_closed_contour, contours = createContoursAndFill(sam_mask_grayscale)
    valid_root_hair_masks, components_masks = makeValidRootHairMasks(skeletonized_hairs, contours, microscope_conversion_factor, upper, lower)


    fig = makeFinalPlotlyVisual(image_gray, valid_root_hair_masks)
    return fig, root_hair_mask, valid_root_hair_masks

if __name__ == "__main__":
    '''To find root_hair masks and final visualizations for images in a folder.'''
    # folder_path = '/Users/antoantony/Root_hair_test_stuff/root_hair_150/images/predict'
    # folder = Path(folder_path)
    # store_results = {}
    # idx = 1
    # for image_path in folder.glob('*.bmp'):
    #     print(f'Starting image {idx} of 16')
    #     image = cv2.imread(str(image_path))
    #     image_gray = makeGrayscaleImage(image_path)
    #     fig = main_v2(image, image_gray, microscope_conversion_factor=3.393626769, upper=100.0, lower=30.0)
    #     fig.show(renderer="browser")
    #     print(f'Finished image {idx} of 16')
    #     store_results[image_path] = fig
    #     idx += 1


    '''To find root_hair mask of one image with its path.'''
    image_path = '/Users/antoantony/Root_hair_test_stuff/root_hair_150/images/predict/147.bmp'
    image = cv2.imread(str(image_path))
    image_gray = makeGrayscaleImage(image_path)
    fig, root_hair_mask, valid_root_hair_masks = main_v2(image, image_gray, microscope_conversion_factor=3.393626769, 
                  upper=100.0, lower=1.0)
    fig.show(renderer='browser')
    
    figure, ax = plt.subplots(1,2, figsize=(20,10))
    ax[0].imshow(image_gray, cmap='gray')
    ax[0].set_title('Original Grayscale Image')
    ax[0].axis('off')
    ax[1].imshow(root_hair_mask, cmap='gray')
    ax[1].set_title('Root Hair Mask')
    ax[1].axis('off')
  
    plt.tight_layout()
    plt.show()