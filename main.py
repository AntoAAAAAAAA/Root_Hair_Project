import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy

from root_mask import *

def main():
    image_path = "/Users/antoantony/9-30/KO/KO 10 um T0/KO 10 um_4.bmp"
    image_gray = make_grayscale_image(image_path)
    mask = create_binary_mask(image_gray)
    # 4x4 kernel with adaptive threshold (this actually seems to make it worse, so maybe ignore this part)
    # mask = adaptive_threshold(mask, block_size=35, C=8)
    CC_mask = create_connected_component_mask(mask)
    closed_mask = create_morphologically_closed_mask(CC_mask)
    final_mask = create_contours_and_fill(closed_mask)
    final_overlay = overlay_mask_on_image(image_gray, final_mask)

    plt.imshow(final_overlay, cmap='gray')
    plt.axis('off')
    plt.title('Final Overlay of Mask on Original Image')
    plt.show()

if __name__ == "__main__":
    main()
