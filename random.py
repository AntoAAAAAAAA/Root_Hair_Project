# INPUTS
# grayscale image
# foreground_mask          # root + hairs + maybe some noise
# main_root_mask          # your current filled main-root segmentation

# STEP 1: main root boundary
root_boundary = main_root_mask - erode(main_root_mask)

# STEP 2: isolate candidate hair material
hair_candidate_mask = foreground_mask.copy()
hair_candidate_mask[main_root_mask > 0] = 0

# STEP 3: connected components on candidate hair regions
components = connected_components(hair_candidate_mask)

accepted_hairs = []

# 4. For each component:
#    - extract component
#    - compute endpoints / branchpoints
#    - reject if branched/intersecting
#    - measure geodesic endpoint-to-endpoint length
#    - convert px -> um
#    - reject if outside lab range
#    - check distance to main-root boundary
#    - keep if attached or near-attached

for comp in components:
    # basic geometric filtering
    if area_too_small(comp):
        continue
    if too_round(comp):
        continue
    if too_thick(comp):
        continue

    # must touch or nearly touch root boundary
    if not touches_or_near_boundary(comp, root_boundary):
        continue

    # local ROI crop
    roi = crop_component_with_padding(comp)

    # skeletonize only the component ROI
    skel = skeletonize(roi)

    # prune spurs / tiny branches
    skel = prune_skeleton(skel)

    # identify endpoints
    endpoints = find_skeleton_endpoints(skel)
    if len(endpoints) < 2:
        continue

    # choose base endpoint = one closest to main root boundary
    base = endpoint_closest_to_boundary(endpoints, root_boundary)

    # choose tip endpoint = farthest geodesic endpoint from base
    tip = farthest_endpoint_along_skeleton(skel, base)

    # measure length along skeleton path
    length = geodesic_length(skel, base, tip)

    accepted_hairs.append({
        "component": comp,
        "skeleton": skel,
        "base": base,
        "tip": tip,
        "length": length
    })


### ChatGPT generated idea for geodesic length without using convolution 
import numpy as np

def analyze_component(mask):
    """
    mask: binary 2D array with one connected component only (0/1)
    returns a dict with endpoints, branchpoints, n_ortho, n_diag, total_edges, length_pixels
    """

    coords = np.argwhere(mask > 0)
    pixel_set = set(map(tuple, coords))
    # pixel_set = {tuple(x) for x in coords}

    # 8-neighbor offsets
    neighbors8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    # Only forward directions so each edge is counted once
    ortho_dirs = [(0, 1), (1, 0)]
    diag_dirs  = [(1, 1), (1, -1)]

    degrees = []
    n_ortho = 0
    n_diag = 0

    for r, c in coords:
        degree = 0

        # degree count
        for dr, dc in neighbors8:
            if (r + dr, c + dc) in pixel_set:
                degree += 1
        degrees.append(degree)

        # count orthogonal edges once
        for dr, dc in ortho_dirs:
            if (r + dr, c + dc) in pixel_set:
                n_ortho += 1

        # count diagonal edges once
        for dr, dc in diag_dirs:
            if (r + dr, c + dc) in pixel_set:
                n_diag += 1

    degrees = np.array(degrees)
    endpoints = int(np.sum(degrees == 1))
    branchpoints = int(np.sum(degrees > 2))
    total_edges = n_ortho + n_diag
    length_pixels = n_ortho + n_diag * np.sqrt(2)

    return {
        "num_pixels": int(len(coords)),
        "endpoints": endpoints,
        "branchpoints": branchpoints,
        "n_ortho": int(n_ortho),
        "n_diag": int(n_diag),
        "total_edges": int(total_edges),
        "length_pixels": float(length_pixels),
        "degrees": degrees
    }