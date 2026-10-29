from src.root_hairs_mask import extractAndFilterRootHairs

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

    image_path = '/Users/antoantony/Root_Hair_ML/Images/KeptImages/015.png'
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

    valid_root_hair_masks, graph = extractAndFilterRootHairs(skeleton_hairs, contours, microscope_conversion_factor, upper, lower)

    ####################################

    # ####### INDIVIDUAL ROOT HAIR DEBUG VISUALIZATIONS #######

    # output_dir = (
    #     Path(__file__).resolve().parent
    #     / f"root_hair_debug_{Path(image_path).stem}"
    # )
    # output_dir.mkdir(parents=True, exist_ok=True)

    # background = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    # for i, hair in enumerate(valid_root_hair_masks, start=1):

    #     thick_mask = hair["thicker mask"] > 0

    #     # Create overlay for this individual hair only
    #     thick_overlay = background.copy()
    #     color = np.array([255, 60, 60], dtype=np.uint8)  # red-ish
    #     alpha = 0.75

    #     thick_overlay[thick_mask] = (
    #         (1 - alpha) * thick_overlay[thick_mask]
    #         + alpha * color
    #     ).astype(np.uint8)

    #     # Create figure
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    #     # 1. Entire root hair mask
    #     axes[0].imshow(root_hairs, cmap="gray")
    #     axes[0].set_title("Entire Root Hair Mask")

    #     # 2. Entire skeleton graph
    #     ax = axes[1]
    #     ax.imshow(image_gray, cmap="gray", alpha=0.35)

    #     # Draw all graph branches
    #     for u, v, key, data in graph.edges(keys=True, data=True):
    #         pts = data["pts"]
    #         ax.plot(
    #             pts[:, 1], pts[:, 0],
    #             color="cyan",
    #             linewidth=1.2
    #         )

    #     # Draw all graph nodes
    #     for node_id, data in graph.nodes(data=True):
    #         y, x = data["o"]

    #         if graph.degree(node_id) == 1:
    #             node_color = "lime"   # endpoints
    #         else:
    #             node_color = "red"    # degree >= 2 nodes / junction-type nodes

    #         ax.scatter(
    #             x, y,
    #             c=node_color,
    #             s=12,
    #             zorder=3
    #         )

    #     ax.set_title("Skeleton Graph\nGreen = Endpoints | Red = Nodes")

    #     # 3. Dilated overlay for selected root hair
    #     axes[2].imshow(thick_overlay)
    #     axes[2].set_title("Dilated Root Hair Overlay")

    #     for ax in axes:
    #         ax.axis("off")

    #     fig.suptitle(
    #         f"Root Hair {i} | ID: {hair['id']} | "
    #         f"Length: {hair['length in microns']:.2f} µm",
    #         fontsize=14
    #     )

    #     plt.tight_layout()

    #     fig.savefig(
    #         output_dir / f"root_hair_{i:03d}_id_{hair['id']}.png",
    #         dpi=150,
    #         bbox_inches="tight"
    #     )

    #     plt.close(fig)

    # print(
    #     f"Saved {len(valid_root_hair_masks)} individual root-hair "
    #     f"visualizations to: {output_dir}"
    # )

    #################################

    ####### COMBINED ROOT HAIR VISUALIZATION #######

    # Create colored mask of all reconstructed root hairs
    colored_mask = np.zeros((*root_hairs.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)

    for hair in valid_root_hair_masks:
        color = rng.integers(60, 256, size=3, dtype=np.uint8)
        colored_mask[hair["thicker mask"] > 0] = color

    # Overlay colored hairs onto original image
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    hair_pixels = np.any(colored_mask > 0, axis=2)

    overlay[hair_pixels] = (
        0.3 * overlay[hair_pixels]
        + 0.7 * colored_mask[hair_pixels]
    ).astype(np.uint8)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Original image
    axes[0, 0].imshow(image_gray, cmap="gray")
    axes[0, 0].set_title("Original Image")

    # 2. Complete skeleton graph
    ax = axes[0, 1]
    ax.imshow(image_gray, cmap="gray", alpha=0.35)

    for u, v, key, data in graph.edges(keys=True, data=True):
        pts = data["pts"]
        ax.plot(
            pts[:, 1], pts[:, 0],
            color="cyan",
            linewidth=1.2
        )

    for node_id, data in graph.nodes(data=True):
        y, x = data["o"]

        node_color = (
            "lime" if graph.degree(node_id) == 1
            else "red"
        )

        ax.scatter(
            x, y,
            c=node_color,
            s=12,
            zorder=3
        )

    ax.set_title("Skeleton Graph (Green = Endpoints, Red = Nodes)")

    # 3. Entire root-hair binary mask
    axes[1, 0].imshow(root_hairs, cmap="gray")
    axes[1, 0].set_title("Root Hair Segmentation Mask")

    # 4. Colored overlay of reconstructed hairs
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(
        f"Reconstructed Root Hairs (n={len(valid_root_hair_masks)})"
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        Path(__file__).resolve().parent
        / "root_hairs_vd2.png"
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved visualization to: {output_path}")

    ######################################################