from hybrid_ML import *
from main_root_mask import *
from root_hairs_mask import *
from final_visualize import *

import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from ultralytics import SAM


image_path = '/Users/antoantony/Root_hair_test_stuff/root_hair_150/images/predict/134.bmp'
image = cv2.imread(str(image_path))
image_gray = makeGrayscaleImage(image_path)
sam_mask_grayscale, root_hair_mask, overlay, binary_mask, thresh, core = hybrid_main2(image, image_gray)
fig, ax = plt.subplots(2,3, figsize=(10,10))
images = [
    (thresh, "Thresholded Image"),
    (core, "Distance Transform Core"),
    (binary_mask, "Largest Connected Component"),
    (overlay, "SAM-Cleaned Overlay"),
    (sam_mask_grayscale, "Main Root SAM Mask"),
    (root_hair_mask, "Root Hair Mask"),
]
for i, (img, title) in enumerate(images):
    row = i // 3
    col = i % 3
    ax[row, col].imshow(img, cmap="gray")
    ax[row, col].set_title(title)
    ax[row, col].axis("off")
plt.tight_layout()
plt.show()

# sam_mask_grayscale, final_subtracted_mask, thresh_mask, binary_mask, core = hybrid_main(image, image_gray)
# fig, ax = plt.subplots(2,3, figsize=(10,10))

# images = [
#     (image_gray, "Original Grayscale"),
#     (thresh_mask, "Threshold Mask"),
#     (binary_mask, "Largest Connected Component"),
#     (core, "Distance Transform Core"),
#     (sam_mask_grayscale, "Main Root SAM Mask"),
#     (final_subtracted_mask, "Root Hair Mask"),
# ]

# for i, (img, title) in enumerate(images):
#     row = i // 3
#     col = i % 3

#     ax[row, col].imshow(img, cmap='gray')
#     ax[row, col].set_title(title)
#     ax[row, col].axis('off')

# plt.tight_layout()
# plt.show()