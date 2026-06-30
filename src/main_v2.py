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

# Note: Always include image_gray as an input for mains. Streamlit is weird with grayscale conversion, so this is necessary
def main_v2(image, image_gray, microscope_conversion_factor, upper, lower, model):
    '''
    Returns:
        tuple:
            fig (plotly.graph_objects.Figure): Interactive Plotly figure showing
                the analyzed root hairs overlaid on the grayscale image.

            root_hair_mask (np.ndarray): Binary mask containing the candidate root
                hair regions after subtraction of the main root.

            valid_root_hair_masks (list[np.ndarray]): List of binary masks
                corresponding to root hairs that passed the filtering criteria.
    '''
    print("\n" + "=" * 50)
    print("BEGIN ROOT HAIR ANALYSIS")
    print("=" * 50)

    # Find main root and create root hair mask 
    sam_mask_grayscale, root_hair_mask  = hybrid_main2(image, image_gray, model)

    # Analyze individual root hairs and filter valid ones
    print('\n')
    print("[4/5] Analyzing individual root hairs...")
    skeletonized_hairs = skeletonizeRootHairMask(root_hair_mask)
    mask_closed_contour, contours = createContoursAndFill(sam_mask_grayscale)
    valid_root_hair_masks, components_masks = makeValidRootHairAnalysis(skeletonized_hairs, contours, microscope_conversion_factor, upper, lower)

    # Create final visualization (interactive plotly)
    print('\n')
    print('[5/5] Creating final plotly figure...')
    fig, traces = makeFinalPlotlyVisual(image_gray, valid_root_hair_masks)

    print('\n')
    print('=' * 50)
    print('ANALYSIS COMPLETE')
    print('=' * 50)
    print('\n')

    return {
        "fig": fig,
        "traces": traces,
        "root_hair_mask": root_hair_mask,
        "valid_root_hair_masks": valid_root_hair_masks,
        'contours': contours
    }


if __name__ == "__main__":
    '''Testing: To find root_hair masks and final visualizations for images in a folder.'''
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


    '''Testing: To find root_hair mask of one image with its path.'''
    '''134-149'''
    image_path = '/Users/antoantony/Root_hair_hybrid_tests/root_hair_150/images/predict/147.bmp'
    image = cv2.imread(str(image_path))
    image_gray = makeGrayscaleImage(image_path)
    model = SAM('Root_Hair/sam2_l.pt') 

    results_dict = main_v2(image, image_gray, microscope_conversion_factor=3.393626769, 
                  upper=300.0, lower=10.0, model=model)
    fig = results_dict['fig']
    root_hair_mask = results_dict['root_hair_mask']
    valid_root_hair_masks = results_dict['valid_root_hair_masks']
    contours = results_dict['contours']

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