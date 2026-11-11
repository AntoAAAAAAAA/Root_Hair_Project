from pathlib import Path
import tempfile
import cv2
import numpy as np
import torch
import sys

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

def nnUnet(image_gray, predictor):
    '''Runs the input image from Streamlit into the nnUnet ML model'''

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)
        input_path = temp_dir / 'root_0000.png'

        # replace temporary .png with image_gray 
        cv2.imwrite(str(input_path), image_gray)

        # straight from nnUnetv2 example code 
        img, props = SimpleITKIO().read_images([str(input_path)])
        predictions = predictor.predict_single_npy_array(img, props, None, None, False)
        segmentation = predictions

        return segmentation

def get_torch_device():
    """Select the best available PyTorch device."""

    if sys.platform == "darwin" and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


### ----Testing--------
if __name__ == "__main__":
    image_path = '/Users/antoantony/Root_Hair_ML/Images/GrayImages/root_0198_0000.png'
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    predictor = initalize_predictor()
    segmentation = nnUnet(image_gray, predictor)