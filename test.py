from src.root_hairs_mask import makeValidRootHairAnalysis2

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import plotly as plot
import plotly.express as px
from ultralytics import SAM
import sys
from pathlib import Path

from src.main_root_mask import *
from src.root_hairs_mask import *
from src.final_visualize import *
from src.ML import nnUnet, get_torch_device
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def initalize_predictor():
    '''Initializes the nnUnet predictor'''

    device = get_torch_device()

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    def get_resource_path(relative_path):
        if getattr(sys, "frozen", False):
            base_path = Path(sys._MEIPASS)
        else:
            base_path = Path(__file__).resolve().parent

        return base_path / relative_path


    MODEL_DIR = get_resource_path("Compressed_Unet")
    os.environ["nnUNet_extTrainer"] = str(MODEL_DIR)

    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))

    predictor.initialize_from_trained_model_folder(
    str(MODEL_DIR),
    use_folds=(0,),
    checkpoint_name="checkpoint_final.pth"
    )

    return predictor 

if __name__ == '__main__':

    image_path = '/Users/antoantony/Root_Hair_ML/Images/KeptImages/008.png'
    microscope_conversion_factor = 3.34
    upper = 500
    lower = 3

    ###### main_v3 code #######
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    model = initalize_predictor()
    segmentation = nnUnet(image_gray, model)

    segmentation = np.squeeze(segmentation) # remove the 1 dimension (1, 768, 1024) --> (768, 1024)
    main_root = (segmentation == 1).astype(np.uint8) * 255
    root_hairs = (segmentation == 2).astype(np.uint8) * 255

    skeleton_hairs = skeletonizeRootHairMask(root_hairs)
    contours, _ = cv2.findContours(main_root, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ####### TESTING BLOCK ##########


    graph = makeValidRootHairAnalysis2(skeleton_hairs, contours, microscope_conversion_factor, upper, lower)
    # sknw.draw_graph(image_gray, graph)
    plt.figure(figsize=(12, 8))
    plt.imshow(root_hairs, cmap='gray')

    # Plot all edges
    for u, v, key, data in graph.edges(keys=True, data=True):
        ps = data["pts"]
        plt.plot(ps[:, 1], ps[:, 0], color="green", linewidth=1)

    # Plot all nodes
    ps = np.array([data["o"] for _, data in graph.nodes(data=True)])

    if len(ps) > 0:
        plt.plot(ps[:, 1], ps[:, 0], "r.", markersize=5)

    plt.title("Skeleton Graph Visualization")
    plt.savefig("skeleton_graph.png", dpi=200, bbox_inches="tight")
    plt.close()
    #################################