from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from pathlib import Path
import torch
import os 
import sys


predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=False,
    device=torch.device("mps"),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False,
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
    checkpoint_name="checkpoint_final.pth",
)


# input_images = "/Users/antoantony/Root_Hair_ML/Images/val_test"
# os.makedirs('/Users/antoantony/Root_Hair_ML/val_test_output/100_test', exist_ok=True)
# output_image = "/Users/antoantony/Root_Hair_ML/val_test_output/100_test"

# nnunet = predictor.predict_from_files(
#     input_images,
#     output_image,
#     save_probabilities=False,
#     overwrite=True,
#     num_processes_preprocessing=1,
#     num_processes_segmentation_export=1,
#     folder_with_segs_from_prev_stage=None,
#     num_parts=1,
#     part_id=0
# )