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

    image_path = '/Users/antoantony/Root_Hair_ML/Images/KeptImages/064.png'
    microscope_conversion_factor = 3.34
    # upper = 500
    upper = 1000
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

    #############################################################################################

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

    ###############################################################################################

    ####### COMBINED ROOT HAIR VISUALIZATION #######

    # # Create colored mask of all reconstructed root hairs
    # colored_mask = np.zeros((*root_hairs.shape, 3), dtype=np.uint8)
    # rng = np.random.default_rng(42)

    # for hair in valid_root_hair_masks:
    #     color = rng.integers(60, 256, size=3, dtype=np.uint8)
    #     colored_mask[hair["thicker mask"] > 0] = color

    # # Overlay colored hairs onto original image
    # overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    # hair_pixels = np.any(colored_mask > 0, axis=2)

    # overlay[hair_pixels] = (
    #     0.3 * overlay[hair_pixels]
    #     + 0.7 * colored_mask[hair_pixels]
    # ).astype(np.uint8)

    # # Create 2x2 figure
    # fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # # 1. Original image
    # axes[0, 0].imshow(image_gray, cmap="gray")
    # axes[0, 0].set_title("Original Image")

    # # 2. Complete skeleton graph
    # ax = axes[0, 1]
    # ax.imshow(image_gray, cmap="gray", alpha=0.35)

    # for u, v, key, data in graph.edges(keys=True, data=True):
    #     pts = data["pts"]
    #     ax.plot(
    #         pts[:, 1], pts[:, 0],
    #         color="cyan",
    #         linewidth=1.2
    #     )

    # for node_id, data in graph.nodes(data=True):
    #     y, x = data["o"]

    #     node_color = (
    #         "lime" if graph.degree(node_id) == 1
    #         else "red"
    #     )

    #     ax.scatter(
    #         x, y,
    #         c=node_color,
    #         s=12,
    #         zorder=3
    #     )

    # ax.set_title("Skeleton Graph (Green = Endpoints, Red = Nodes)")

    # # 3. Entire root-hair binary mask
    # axes[1, 0].imshow(root_hairs, cmap="gray")
    # axes[1, 0].set_title("Root Hair Segmentation Mask")

    # # 4. Colored overlay of reconstructed hairs
    # axes[1, 1].imshow(overlay)
    # axes[1, 1].set_title(
    #     f"Reconstructed Root Hairs (n={len(valid_root_hair_masks)})"
    # )

    # for ax in axes.flat:
    #     ax.axis("off")

    # plt.tight_layout()

    # output_path = (
    #     Path(__file__).resolve().parent
    #     / "root_hairs_vd2.png"
    # )

    # plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    # plt.close(fig)

    # print(f"Saved visualization to: {output_path}")

    ############################################################################################################

    
    ####### PLOTLY GRAPH DEBUG VISUALIZATION #######

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    background = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    # Create a lighter background for the graph
    graph_background = (
        0.35 * background + 0.65 * 255
    ).astype(np.uint8)

    # Create colored overlay and retain each hair's color
    colored_mask = np.zeros((*root_hairs.shape, 3), dtype=np.uint8)
    hair_colors = {}
    rng = np.random.default_rng(42)

    for hair in valid_root_hair_masks:
        color = rng.integers(60, 256, size=3, dtype=np.uint8)
        hair_colors[hair["id"]] = color
        colored_mask[hair["thicker mask"] > 0] = color

    overlay = background.copy()
    hair_pixels = np.any(colored_mask > 0, axis=2)

    overlay[hair_pixels] = (
        0.3 * background[hair_pixels]
        + 0.7 * colored_mask[hair_pixels]
    ).astype(np.uint8)

    # Create 2x2 interactive figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Original Image",
            "Skeleton Graph — Hover for Node / Edge IDs",
            "Root Hair Segmentation Mask",
            "Reconstructed Hairs — Hover for Hair IDs"
        ),
        horizontal_spacing=0.04,
        vertical_spacing=0.08
    )

    # Background images
    images = [
        (background, 1, 1),
        (graph_background, 1, 2),
        (cv2.cvtColor(root_hairs, cv2.COLOR_GRAY2RGB), 2, 1),
        (overlay, 2, 2)
    ]

    for img, row, col in images:
        fig.add_trace(
            go.Image(z=img, hoverinfo="skip"),
            row=row,
            col=col
        )

    # -------- GRAPH: EDGES --------
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge_id = (min(u, v), max(u, v), key)
        pts = data["pts"]

        fig.add_trace(
            go.Scattergl(
                x=pts[:, 1],
                y=pts[:, 0],
                mode="lines+markers",
                line=dict(color="cyan", width=2),
                marker=dict(color="cyan", size=2),
                showlegend=False,
                hovertemplate=(
                    f"<b>Edge ID: {edge_id}</b>"
                    "<extra></extra>"
                )
            ),
            row=1,
            col=2
        )

    # -------- GRAPH: NODES --------
    node_x, node_y = [], []
    node_ids, node_degrees, node_colors = [], [], []

    for node_id, data in graph.nodes(data=True):
        y, x = data["o"]
        degree = graph.degree(node_id)

        node_x.append(x)
        node_y.append(y)
        node_ids.append(node_id)
        node_degrees.append(degree)

        if degree == 1:
            node_colors.append("lime")
        elif degree == 2:
            node_colors.append("orange")
        else:
            node_colors.append("red")

    node_info = np.column_stack((node_ids, node_degrees))

    fig.add_trace(
        go.Scattergl(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=8,
                color=node_colors,
                line=dict(color="black", width=0.5)
            ),
            customdata=node_info,
            showlegend=False,
            hovertemplate=(
                "<b>Node ID: %{customdata[0]}</b>"
                "<br>Degree: %{customdata[1]}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=2
    )

    # -------- OVERLAY: HAIR IDENTIFIERS --------
    for hair in valid_root_hair_masks:

        ys, xs = np.where(hair["mask"] > 0)
        if len(xs) == 0:
            continue

        # Place an identifying marker on an actual hair pixel
        midpoint = len(xs) // 2

        hair_id = hair["id"]
        color = hair_colors[hair_id]
        css_color = f"rgb({color[0]}, {color[1]}, {color[2]})"

        edge_ids = hair["edge_ids"]
        node_ids_for_hair = hair["node_ids"]

        fig.add_trace(
            go.Scatter(
                x=[xs[midpoint]],
                y=[ys[midpoint]],
                mode="markers",
                marker=dict(
                    size=10,
                    color=css_color,
                    line=dict(color="black", width=1)
                ),
                name=f"Hair {hair_id}",
                hovertemplate=(
                    f"<b>Hair ID: {hair_id}</b>"
                    f"<br>Length: {hair['length in microns']:.2f} µm"
                    f"<br>Node IDs: {node_ids_for_hair}"
                    f"<br>Edge IDs: {edge_ids}"
                    "<extra></extra>"
                )
            ),
            row=2,
            col=2
        )

    # -------- FIGURE SETTINGS --------
    fig.update_layout(
        width=1500,
        height=1100,
        title="Root Hair Reconstruction Debugging",
        hovermode="closest",
        legend=dict(
            title="Reconstructed Hair IDs",
            x=1.01,
            y=0.9,
            font=dict(size=10)
        ),
        margin=dict(l=20, r=210, t=90, b=20)
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")

    # output_path = (
    #     Path(__file__).resolve().parent
    #     / "root_hairs_interactive_debug.html"
    # )

    # fig.write_html(
    #     str(output_path),
    #     include_plotlyjs=True,
    #     config={"scrollZoom": True}
    # )

    # print(f"Interactive visualization saved to: {output_path}")

    # Optional: display it immediately
    fig.show()

    ############################################################################################################

    #### MORE TESTING FOR INDIVDUAL NODES #####

    def inspectNode(node_to_inspect):
        from itertools import combinations

        if node_to_inspect not in graph:
            print(f"Node {node_to_inspect} does not exist.")
            return

        degree = graph.degree(node_to_inspect)

        print(f"\nInspecting node {node_to_inspect}")
        print(f"Degree: {degree}")

        if "debug_vectors" not in graph.nodes[node_to_inspect]:
            print("No matching analysis for this node.")
            return

        vectors = graph.nodes[node_to_inspect]["debug_vectors"]
        pairings = graph.nodes[node_to_inspect]["debug_pairings"]

        results = []

        for edge_a, edge_b in combinations(vectors, 2):

            cosine = np.clip(
                np.dot(vectors[edge_a], vectors[edge_b]),
                -1.0,
                1.0
            )

            angle = np.degrees(np.arccos(cosine))
            selected = pairings.get(edge_a) == edge_b

            results.append((angle, edge_a, edge_b, selected))

        # Display highest-angle pairs first
        for angle, edge_a, edge_b, selected in sorted(
            results,
            reverse=True
        ):
            status = "MATCHED" if selected else ""

            print(
                f"{edge_a} <-> {edge_b}: "
                f"{angle:.1f} degrees {status}"
            )

    # inspectNode(93) ##### CHANGE NODE TO INSPECT