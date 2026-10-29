import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import scipy as scipy
import os


def skeletonizeRootHairMask(root_hair_mask):
    'This function skeletonizes the individual root hairs found in a mask that contains only root hairs.'

    skeletonized_hairs = sk.morphology.skeletonize(root_hair_mask // 255) # Turn image binary (0's and 1's)
    skeletonized_hairs = (skeletonized_hairs * 255).astype(np.uint8) # convert back to 0 and 255

    return skeletonized_hairs

def findBranchLength(y, x, neighbors, length_per_branch, root_hair_pixel_set, 
                     uniquePixelsTraversed, branchID, current_length): 
    '''A recursive function used to find the lengths of branches starting from a branchpoint (a pixel with more than 2 neighbors in it's 8-pixel radius).'''     
    
    # Add current coordinates to storage so we know that we have already traversed this pixel
    uniquePixelsTraversed.add((y,x))
    
    # Find the number of neighbors surrounding a pixel
    num_neighbors = 0
    for dy, dx in neighbors:
        if (y+dy, x+dx) in root_hair_pixel_set:
            num_neighbors += 1
    
    # If we the pixel only has 1 neighbor, we know we are at the end of a branch
    if num_neighbors == 1: 
        return current_length
    
    # If there are >1 neighbors, we recursively run the function on surrounding pixels 
    for dy, dx in neighbors:
        if (y+dy, x+dx) in root_hair_pixel_set and (y+dy, x+dx) not in uniquePixelsTraversed:
            return findBranchLength(y+dy, x+dx, neighbors, length_per_branch, 
                                    root_hair_pixel_set, uniquePixelsTraversed, branchID, 
                                    current_length+1)
    # Assuming none of the surrounding pixels meet our criteria, we can end the function 
    return current_length               

def makeValidRootHairAnalysis(skeletonized_hairs, contours, microscope_conversion_factor, upper, lower):
    '''This function takes each component of the skeletonized root hair mask and filters valid root hairs. '
    The function chooses root hairs based on connectivity to main root, length parameters, the validity of endpoints, 
    and lack of branching. It also measures them. Primarily used for main_v2 workflow. 

    Returns:
        tuple:
            valid_root_hair_masks (list[dict]): A list of dictionaries, each representing
                a valid root hair that passed the connectivity, branching, length, and
                main-root proximity filters. Each dictionary contains:

                - 'id' (int): Unique connected-component label identifying the root hair.
                - 'mask' (np.ndarray): Full-size binary mask of the individual root hair,
                where 1 represents root-hair pixels and 0 represents background.
                - 'thicker mask' (np.ndarray): Dilated version of the root-hair mask,
                used for visualization purposes.
                - 'cropped_component_mask' (np.ndarray): Binary mask cropped around
                the root hair with a 3-pixel margin, bounded by image dimensions.
                - 'length' (float): Estimated root-hair skeleton length in pixels,
                calculated using orthogonal and diagonal pixel connections.
                - 'length in microns' (float): Estimated root-hair length in microns,
                calculated using the microscope conversion factor.

            components_masks (np.ndarray): Full-size integer-labeled array produced
                by connected-component analysis of the skeletonized root-hair mask.
                Background pixels are labeled 0, while each connected component
                receives a unique positive integer label. Includes both accepted
                and rejected root-hair candidates.

    '''

    skeletonized_hairs_binary = (skeletonized_hairs > 0).astype(np.uint8)
    components_masks = sk.measure.label(skeletonized_hairs_binary, connectivity=2)
    num_objects = components_masks.max()
    print(f"     Number of components found: {num_objects}")

    valid_root_hair_masks = []

    # filtered = 0 # for testing

    # Create set of coordinates that make up contour of main root 
    all_contours = np.vstack(contours)
    coords_of_contours = all_contours.reshape(-1, 2)
    coords_of_contours = set(map(tuple, coords_of_contours.tolist()))

    for i in range(1, num_objects + 1):

        # choose a single
        root_hair_mask = (components_masks == i).astype(np.uint8)

        # Check that root hair mask has pixels 
        if root_hair_mask.sum() == 0:
            continue

        # Crop mask to zoom in on the individual pixels that make up the root hair 
        y, x = np.where(root_hair_mask > 0)
        adjustment = 3
        h , w = root_hair_mask.shape
        y_min = max(0, y.min() - adjustment)
        y_max = min(h-1, y.max() + adjustment)
        x_min = max(0, x.min() - adjustment)
        x_max = min(w-1, x.max() + adjustment)
        cropped_component_mask = root_hair_mask[y_min:y_max+1, x_min:x_max+1]

        # Create a set of every pixel that makes up the root hair 
        coords = np.argwhere(cropped_component_mask > 0)
        root_hair_pixel_set = set(tuple(coordinate) for coordinate in coords)

        # Make kernels that represent neighboring pixels 
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        ortho_neighbors = [(0, 1), (1, 0)]
        diag_neighbors = [(1, 1), (1, -1)]

        # --- Identify all neighbors, orthogonal neighbors, and diagonal neighbors -----
        list_of_degrees = []
        num_ortho_neighbors = 0
        num_diag_neighbors = 0
        branch_coords = []

        # Loop through each pixel in the root hair 
        for y, x in coords:
            num_neighbors = 0

            # Check number of neighbors using 8-neighbor kernel
            for dy, dx in neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_neighbors += 1
            list_of_degrees.append(num_neighbors)
            if num_neighbors >= 3:
                branch_coords.append((y,x))

            # Check number of orthogonal neighbors using orthogonal kernel
            for dy, dx in ortho_neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_ortho_neighbors += 1

            # Check number of diagonal neighbors using diagonal kernel
            for dy, dx in diag_neighbors:
                if (y + dy, x + dx) in root_hair_pixel_set:
                    num_diag_neighbors += 1

        # Find # of endpoints
        endpoints = list_of_degrees.count(1)

        # Exclude root hairs that only have 1 endpoint  
        if endpoints == 1:
            continue
            
        # # Filter root hairs that have branches that are significantly long  
        # valid_branches = True
        # if endpoints > 2:
        #     for y, x in branch_coords:
        #         length_per_branch = {}
        #         for dy, dx in neighbors: 
        #             if (y + dy, x + dx) in root_hair_pixel_set:
        #                 # Traverse pixels that make up root hair and recursively find how long the branches are
        #                 uniquePixelsTraversed = set()
        #                 uniquePixelsTraversed.add((y,x))
        #                 branchID = (y+dy, x+dx)
        #                 length_per_branch[branchID]= findBranchLength(y+dy, 
        #                                                               x+dx, 
        #                                                               neighbors,
        #                                                               length_per_branch, 
        #                                                               root_hair_pixel_set, 
        #                                                               uniquePixelsTraversed, 
        #                                                               branchID, 
        #                                                               current_length = 0)

        #         # Find how many short branches there are   
        #         short_branches = 0
        #         for length in length_per_branch.values():
        #             if length < 5:
        #                 short_branches += 1
                
        #         # Filter out root hairs that have significant branching 
        #         if len(length_per_branch.keys()) == 4 and short_branches < 2:
        #             valid_branches = False 
        #             break
        #         if len(length_per_branch.keys()) == 3 and short_branches < 1:
        #             valid_branches = False  
        #             break
    
        # if not valid_branches:
        #     continue 

        # Calculate the length of the root hair using ortho and diag neighbors
        length = num_ortho_neighbors + (num_diag_neighbors * np.sqrt(2))
        length_in_microns = length * microscope_conversion_factor

        # Exclude root hairs that are too long/too short  
        if length_in_microns > upper or length_in_microns < lower: 
            continue
        
        # # Check if root hair is within a reasonable distance to the main root 
        # coords_of_endpoints = [tuple(coords[i]) for i, d in enumerate(list_of_degrees) if d == 1]
        # coords_of_endpoints = [(y + y_min, x + x_min) for y,x in coords_of_endpoints]
        # endpoint_near_root = False
        # for y,x in coords_of_endpoints:
        #     for n in range(1,8): # check if any of the pixels within a 8 pixel radius of the endpoint are in the contours
        #         if (x+n, y) in coords_of_contours or (x-n, y) in coords_of_contours:
        #             endpoint_near_root = True
        #             break
        #     if endpoint_near_root:
        #         break
        
        # if not endpoint_near_root:
        #     continue 
        
        # Create a dilated/thicker mask of the root hair (for visualization purposes)
        kernel = np.ones((4,4), np.uint8)
        thicker_mask = cv2.dilate(root_hair_mask, kernel, iterations=1)


        valid_root_hair_masks.append({
            'id': i,
            'mask': root_hair_mask,
            'thicker mask': thicker_mask,
            'cropped_component_mask': cropped_component_mask,
            'length': length,
            'length in microns': length_in_microns,
        })

        # # testing
        # for mask_dict in valid_root_hair_masks:
        #     cropped_component_mask = mask_dict['cropped_component_mask']
        #     i = mask_dict['id']

        #     debugFolder = 'debug_images'
        #     os.makedirs(debugFolder, exist_ok= True)
        #     plt.figure(figsize=(5,5))
        #     plt.imshow(cropped_component_mask, cmap='gray')
        #     plt.title(f"Component {i}")
        #     plt.axis('off')
        #     plt.savefig(f'{debugFolder}/component_{i}.png', bbox_inches='tight')
        #     plt.close()

        

    print("     Root hair analysis complete")
    return valid_root_hair_masks, components_masks




from itertools import combinations
import networkx as nx
def pairBranches(vectors):
    ''' Receives vectors from a single node and calculates valid pairings across a junction 
    using maximum-weight matching. 
    
    Returns:
        dict:
            pairings: node_id: {branch1:branch2, branch2:branch1, branch5:None }'''
            
            
    matching_graph = nx.Graph()

    # Each original skeleton edge becomes a node
    matching_graph.add_nodes_from(vectors)

    for edge_a, edge_b in combinations(vectors, 2):

        vector_a = vectors[edge_a]
        vector_b = vectors[edge_b]

        cosine = np.clip(
            np.dot(vector_a, vector_b),
            -1.0,
            1.0
        )

        angle = np.degrees(np.arccos(cosine))

        # Higher angles are better for outward vectors
        matching_graph.add_edge(
            edge_a,
            edge_b,
            weight=float(angle)
        )

    matching = nx.max_weight_matching(
        matching_graph,
        maxcardinality=True
    )
    # matching = nx.maximal_matching(matching_graph)

    # Bidirectional lookup dictionary
    pairings = {edge: None for edge in vectors}

    for edge_a, edge_b in matching:
        pairings[edge_a] = edge_b
        pairings[edge_b] = edge_a

    return pairings

import sknw
def extractAndFilterRootHairs(skeletonized_hairs, contours, microscope_conversion_factor, upper, lower):
    '''Breaks skeleton mask of root hairs into individual root hair masks. Primarily used
    for main_v3 workflow.
    
    Returns:
        tuple:
            valid_root_hair_masks (list[dict]): A list of dictionaries, each representing
                a valid root hair that passed the connectivity, branching, length, and
                main-root proximity filters. Each dictionary contains:

                - 'id' (int): Unique connected-component label identifying the root hair.
                - 'mask' (np.ndarray): Full-size binary mask of the individual root hair,
                where 1 represents root-hair pixels and 0 represents background.
                - 'thicker mask' (np.ndarray): Dilated version of the root-hair mask,
                used for visualization purposes.
                - 'length' (float): Estimated root-hair skeleton length in pixels,
                calculated using orthogonal and diagonal pixel connections.
                - 'length in microns' (float): Estimated root-hair length in microns,
                calculated using the microscope conversion factor.
    '''

    # initiate final dictionary that will be what is returned 
    valid_root_hair_masks = []

    # Pull coordinates of main contour 
    all_contours = np.vstack(contours)
    coords_of_contours = all_contours.reshape(-1, 2)
    coords_of_contours = set(map(tuple, coords_of_contours.tolist()))

    # turn root hairs into graph w/ junctions, branches, and endpoints
    graph = sknw.build_sknw(skeletonized_hairs, multi=True, iso=False, ring=False, full=True)

    # pull coordinates for junctions (nodes), branches, and endpoints
    branch_coords = {(min(u, v), max(u, v), key): data['pts'] for u, v, key, data in graph.edges(keys=True, data=True)}
    endpoints_array = {node_id: graph.nodes[node_id]['o'] for node_id, degree in graph.degree() if degree == 1}
    node_arrays = {node_id: graph.nodes[node_id]['o'] for node_id, degree in graph.degree() if degree >= 2} 


    ### Create a dictionary for each junction that contains pairs of branches for traversal
    
    # initiate dict 
    junction_pairings = {}

    # Find endpoints that are nearest to the main root
    endpoints_array_near_root = {}
    for endpoint_id, endpoint_coord in endpoints_array.items():
        for n in range(1,4): # check if pixels within a 4 pixel range left and right of endpoint are in contours
            y,x = endpoint_coord
            if (x+n, y) in coords_of_contours or (x-n, y) in coords_of_contours:
                endpoints_array_near_root[endpoint_id] = endpoint_coord

    # iterate through each node:
    for node_id, node_coord in node_arrays.items(): # node: int, node_coord: (y, x)
        pass
        # create vectors for each branch coming out of node (pair these up)
        vectors = {} # neighbor_id : vector 
        for u, v, key, data in graph.edges(node_id, keys=True, data=True):

            edge_id = (min(u, v), max(u, v), key) # unique identifier of an edge 
            pts = data['pts'] #pts: 2D numpy array

            # Determine which end of the branch is near the junction
            dist_to_start = np.linalg.norm(pts[0] - node_coord)
            dist_to_end = np.linalg.norm(pts[-1] - node_coord)

            if dist_to_start < dist_to_end:
                next_pt = pts[min(2, len(pts)-1)] #next_pt: 1D numpy array [y, x]
            else:
                next_pt = pts[max(0, len(pts)-3)]

            vector = next_pt - node_coord # vector: 1D numpy array

            norm = np.linalg.norm(vector)

            # Try to find another pixel with a nonzero displacement
            if norm == 0:
                distances = np.linalg.norm(pts - node_coord, axis=1)
                valid_indices = np.flatnonzero(distances > 0)

                if len(valid_indices) > 0:
                    # Choose the nearest noncoincident point
                    next_pt = pts[valid_indices[np.argmin(distances[valid_indices])]]
                    vector = next_pt - node_coord
                    norm = np.linalg.norm(vector)

            # Fallback: use node on other part of branch 
            if norm == 0:
                other_node_id = v if node_id == u else u
                vector = graph.nodes[other_node_id]['o'] - node_coord
                norm = np.linalg.norm(vector)

            if norm == 0: # a vector of 0 can generate when the node = next_pt 
                continue 

            vectors[edge_id] = vector / norm

        # output a dict that contains the proper pairings (pair up deadend branches with 'None' or something else)
        pairings = pairBranches(vectors)

        # create an identifier to the dictionary, and add the dict of pairings as the value
        junction_pairings[node_id] = pairings # node_id: {branch1:branch2, branch2:branch1, branch3:branch4, branch4:branch3, branch5:None }

    ### Traverse the graph using nodes

    # initiate a list of valid roots ('valid_roots')
    valid_roots = []

    # make a set that tracks all the unique edges that are traversed
    completed_paths = set()

    # iterate through a list of endpoints that are near the main root:
    for idx, (endpoint_id, endpoint_coord) in enumerate(endpoints_array_near_root.items()):

        # initiate a list for creation of individual root ('individual_root') using id's
        individual_root = []
        # initiate set() for 'single root' 
        visited_edges = set()
        visited_nodes = set()

        # add endpoint to 'individual root' and set
        individual_root.append(endpoint_id)
        visited_nodes.add(endpoint_id)

        # find connecting branch and add 
        u, v, key = next(iter(graph.edges(endpoint_id, keys=True))) # by same logic as remark above, there is 1 branch_id
        first_branch_id = (min(u, v), max(u, v), key)

        # find connecting junction 
        first_node_id = v if endpoint_id == u else u

        # add branch id to set and individual root
        individual_root.append(first_branch_id)
        visited_edges.add(first_branch_id)

        # find node and add 
        individual_root.append(first_node_id)
        visited_nodes.add(first_node_id)

        ### start a loop to iterate through successive nodes and branches  
        incoming_branch_id = first_branch_id
        current_node_id = first_node_id
        while True:

            # B/c endpoints aren't in junction_pairings, check if loop reaches an endpoint first
            if graph.degree(current_node_id) == 1:
                break
            # also check that the node is in junction pairings, otherwise the preceding dictionary lookup won't work
            if current_node_id not in junction_pairings:
                break

            # look up junction, use branch pairing to find opposite branch
            opposite_branch_id = junction_pairings[current_node_id].get(incoming_branch_id)
            if opposite_branch_id == None:
                break # Stop loop once a branch terminates at a junction 

            # check that opposite branch isn't in set()
                # if not then add to individual root hair list and set()
            if opposite_branch_id in visited_edges:
                break # no duplicates
                 
            # find opposite junction and add 
            opposite_node_id = None
            u, v, key = opposite_branch_id # u and v are two nodes at ends of a branch 
            opposite_node_id = v if current_node_id == u else u

            # check that we aren't traversing a node already traversed 
            if opposite_node_id in visited_nodes:
                break

            # add edges and nodes to set and individual root
            visited_edges.add(opposite_branch_id)
            individual_root.append(opposite_branch_id)
            visited_nodes.add(opposite_node_id)
            individual_root.append(opposite_node_id)

            # update incoming branch and current node for next iteration 
            incoming_branch_id = opposite_branch_id
            current_node_id = opposite_node_id

        # use completed_paths to make sure this path has not already been traversed (no duplicate roots)
        list_edge_ids = individual_root[1::2]
        forward = tuple(list_edge_ids) 
        backward = tuple(reversed(list_edge_ids))
        root_hair_signature = min(forward, backward)

        if root_hair_signature in completed_paths:
            continue
        completed_paths.add(root_hair_signature)
        # append the individual root hair list to list for final root hairs if not a duplicate 
        valid_roots.append(individual_root)


        ### Create root hair mask (type: np.uint8) from individual root
        # make blank mask 
        grid = np.zeros_like(skeletonized_hairs)

        # condense individual root into list of coordinates 
        coords_list = []
        for item in individual_root:
            
            coords = None
            if type(item) == int: # an int means its a node_id
                coords = graph.nodes[item]['o'] # 1D numpy array
                coords = coords[np.newaxis, :] # turn into 2D numpy array
                coords_list.append(coords)
            if type(item) == tuple: # a tuple means its a branch_id
                coords = branch_coords[item] # 2D numpy array 
                coords_list.append(coords)

        coords_list = np.rint(np.vstack(coords_list)).astype(int) # remove 2nd dimension and rounded to nearest int

        ### Filters for good root hairs 
        # Calculate the length of the root hair and exclude root hairs that are too long/too short  
        length_in_pixels = sum(graph[u][v][key]["weight"] for u, v, key in list_edge_ids)
        length_in_microns = length_in_pixels * microscope_conversion_factor
        if length_in_microns > upper or length_in_microns < lower: 
            continue

        # # Check if root hair is within a reasonable distance to the main root 
        # # Find coordinates of 2 endpoints nodes of the individual root 
        # start_node_id = individual_root[0]
        # end_node_id = individual_root[-1]

        # start_node = graph.nodes[start_node_id]['o']
        # end_node = graph.nodes[end_node_id]['o']
        # coords_of_endpoints = np.rint(np.vstack((start_node, end_node))).astype(int) # combine coordinates and round in case node has floats 

        # # find proximity of endpoints to main root using main root contour coordinates 
        # endpoint_near_root = False
        # for y,x in coords_of_endpoints:
        #     for n in range(1,8): # check if any of the pixels within a 8 pixel radius of the endpoint are in the contours
        #         if (x+n, y) in coords_of_contours or (x-n, y) in coords_of_contours:
        #             endpoint_near_root = True
        #             break
        #     if endpoint_near_root:
        #         break
        
        # if not endpoint_near_root:
        #     continue 

        # Paint blank mask with coordinates 
        grid[coords_list[:, 0], coords_list[:, 1]] = 1
        root_hair_mask = grid
        
        # Create a dilated/thicker mask of the root hair (for visualization purposes)
        kernel = np.ones((4,4), np.uint8)
        thicker_mask = cv2.dilate(root_hair_mask, kernel, iterations=1)

        ### append dictionary to final list that contains final information for each individual root 
        valid_root_hair_masks.append({
            'id': idx,
            'mask': root_hair_mask,
            'thicker mask': thicker_mask,
            # 'cropped_component_mask': cropped_component_mask,
            'length': length_in_pixels,
            'length in microns': length_in_microns,
        })
    
    return valid_root_hair_masks, graph


def makeFinalMaskWithFinalRootHairs(image_grey, valid_root_hair_masks):
    '(Used primarily in testing) This function creates an overlay of the filtered root hairs on top of the original grayscale image.'

    
    color_root = cv2.cvtColor(image_grey, cv2.COLOR_GRAY2RGB)
    overlay = color_root.copy()
    for i in range(0, len(valid_root_hair_masks)):
        valid_dict = valid_root_hair_masks[i]
        ind_mask = (valid_dict['thicker mask'] * 255).astype(np.uint8)
        overlay[ind_mask == 255] = [0, 0 , 255]
    root_hair_overlay = cv2.addWeighted(color_root, 0.2, overlay, 0.8, 0)

    return root_hair_overlay