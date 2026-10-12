from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from pathlib import Path
import torch
import os 
import sys
import tempfile
import cv2
import numpy as np


def initalize_predictor():
    '''Initializes the nnUnet predictor'''
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device("mps"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    def get_resource_path(relative_path):
        if getattr(sys, "frozen", False):
            base_path = Path(sys._MEIPASS)
        else:
            base_path = Path(__file__).resolve().parents[1]

        return base_path / relative_path


    MODEL_DIR = get_resource_path("Compressed_Unet")
    os.environ["nnUNet_extTrainer"] = str(MODEL_DIR)

    predictor.initialize_from_trained_model_folder(
    str(MODEL_DIR),
    use_folds=(0,),
    checkpoint_name="checkpoint_final.pth"
    )

    return predictor 

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

def parse_segmentation(segmentation):

    segmentation = np.squeeze(segmentation) # remove the 1 dimension 

    main_root = (segmentation == 1).astype(np.uint8) * 255
    root_hair = (segmentation == 2).astype(np.uint8) * 255

    cv2.imshow('Window', root_hair)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()     

    fig = None
    trace = None

    return fig, trace


### ----Testing--------
if __name__ == "__main__":
    image_path = '/Users/antoantony/Root_Hair_ML/Images/GrayImages/root_0198_0000.png'
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    predictor = initalize_predictor()
    segmentation = nnUnet(image_gray, predictor)
    fig, trace = parse_segmentation(segmentation)
