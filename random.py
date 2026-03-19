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