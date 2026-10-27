from pathlib import Path
import tempfile
import cv2
import numpy as np
import torch
import sys


def nnUnet(image_gray, predictor):
    '''Runs the input image from Streamlit into the nnUnet ML model'''

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)
        input_path = temp_dir / 'root_0000.png'

        cv2.imwrite(str(input_path), image_gray)

        predictions = predictor.predict_from_files(
            [[str(input_path)]],
            None,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
        segmentation = predictions[0]

        return np.asarray(segmentation)

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