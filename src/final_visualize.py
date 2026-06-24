import plotly.express as px
import cv2
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


def makeFinalPlotlyVisual(image_gray, valid_root_hair_masks):
    '''This function creates a final plotly interactive figure for presentation.
    
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure containing the
        grayscale image and overlays for each valid root hair.

        traces: List of heatmaps, where each heatmap is a valid root hair

    '''
    
    fig = px.imshow(image_gray, binary_string=True)
    # Disable hover on the background trace
    fig.data[0].hovertemplate = None
    fig.data[0].hoverinfo = "skip"

    # fig.update_traces(hoverinfo='skip', selector=dict(type='image'))
    # fig.data[0].hoverinfo = 'skip'
    # fig.show()

    # overlay = color_root.copy()
    # for i in range(0, len(valid_root_hair_masks)):
    #     valid_dict = valid_root_hair_masks[i]
    #     ind_mask = (valid_dict['mask'] * 255).astype(np.uint8)
    #     overlay[ind_mask == 255] = [255, 0 , 0]

    traces = []
    for i in range(0, len(valid_root_hair_masks)):
        valid_dict = valid_root_hair_masks[i]
        ind_mask = (valid_dict['thicker mask']).astype(float)
        ind_mask[ind_mask == 0] = np.nan

        id = valid_dict['id']
        length = valid_dict['length in microns']

        # text = f"Component {id} \n Length: {length:.2f}"
        trace = go.Heatmap(
            z = ind_mask,
            #name=f"Component {id}",
            showscale = False,
            hoverongaps = False,
            hovertemplate = f"Length: {length:.2f} µm   "
        )
        traces.append(trace)


    fig.add_traces(traces)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # fig.show()
    return fig, traces

def makeVisualOfContours(image_gray, contours):
    blank = np.zeros_like(image_gray)

    cv2.drawContours(
        blank,
        contours,
        -1,
        255,
        1
    )

    # plt.imshow(blank, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return blank