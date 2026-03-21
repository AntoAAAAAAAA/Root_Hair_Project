import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy

from main_root_mask import *
from root_hairs_mask import *

def main(image_path, microscope_conversion_factor):

    # Creation of main root mask 
    image_grey = makeGrayscaleImage(image_path)
    mask = createBinaryMask(image_grey)
    # 4x4 kernel with adaptive threshold (this actually seems to make it worse, so maybe ignore this part)
    # mask = adaptive_threshold(mask, block_size=35, C=8)
    CC_mask = createConnectedComponentMask(mask)
    closed_mask = createMorphologicallyClosedMask(CC_mask)
    final_mask, contour = createContoursAndFill(closed_mask) #this is mask_closed_contour
    final_overlay = overlayMaskOnImage(image_grey, final_mask)

    #Creation of root hair mask, filtering of valid root hairs, and measurement
    root_hair_mask = createNewRootHairMask(image_grey, final_mask)
    skeletonized_hairs = skeletonizeRootHairMask(root_hair_mask)
    skeletonized_hairs_with_contours = addMainRootToSkeletonizedHairs(skeletonized_hairs, contour)
    valid_root_hair_masks, components_masks = makeValidRootHairMasks(skeletonized_hairs, contour, microscope_conversion_factor)
    root_hair_overlay = makeFinalMaskWithFinalRootHairs(image_grey, valid_root_hair_masks)

    fig, ax = plt.subplots(3, 3, figsize=(15,8))
    ax[0, 0].imshow(image_grey, cmap='gray')
    ax[0, 0].set_title('Original Grayscale Image')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(final_mask, cmap='gray')
    ax[0, 1].set_title('Closed Contour Mask')
    ax[0, 1].axis('off')
    ax[0, 2].imshow(root_hair_mask, cmap='gray')
    ax[0, 2].set_title('Root Hair Mask (Better Adaptive Threshold - Closed Contour)')
    ax[0, 2].axis('off')
    ax[1, 0].imshow(final_overlay, cmap='nipy_spectral')
    ax[1, 0].set_title('Final_overlay')
    ax[1, 0].axis('off')
    ax[1, 1].imshow(skeletonized_hairs, cmap='gray')
    ax[1, 1].set_title('Skeletonized Root Hairs')
    ax[1, 1].axis('off')
    ax[1, 2].imshow(skeletonized_hairs_with_contours, cmap='gray')
    ax[1, 2].set_title('Skeletonized Root Hairs with Contours')
    ax[1, 2].axis('off')
    ax[2, 0].imshow(components_masks, cmap='nipy_spectral')
    ax[2, 0].set_title('Labeled Components')
    ax[2, 0].axis('off')
    ax[2, 1].imshow(root_hair_overlay, cmap='nipy_spectral')
    ax[2, 1].set_title('Final Labeled Components after filtering')
    ax[2, 1].axis('off')
    # fig.delaxes(ax[2,0])
    fig.delaxes(ax[2,2])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "/Users/antoantony/9-30/KO/KO 10 um T0/KO 10 um_1.bmp"
    microscope_conversion_factor = 3.93626769
    main(image_path, microscope_conversion_factor)