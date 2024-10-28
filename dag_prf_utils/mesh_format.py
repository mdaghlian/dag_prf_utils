import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
from scipy.spatial import ConvexHull
opj = os.path.join

from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *

def dag_find_isolated_vx(mesh_info, roi_bool):
    ''' Find isolated vertices in (connected to no other faces) '''
    assert 'i' in mesh_info.keys(), 'mesh_info should have i,j,k'
    roi_idx = np.where(roi_bool)[0]
    # Which Vx are connected to at least 2 others..
    in_face_x = {} 
    for face_x in ['i', 'j', 'k']:
        in_face_x[face_x] = np.isin(mesh_info[face_x], roi_idx) * 1.0
    faces_with_1_vx_only = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) == 1
    faces_with_more_than_1_vx = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) > 1
    vx_in_faces_with_1_vx_only = np.unique(mesh_info['faces'][faces_with_1_vx_only,:].flatten())
    vx_in_faces_with_more_than_1_vx = np.unique(mesh_info['faces'][faces_with_more_than_1_vx,:].flatten())
    # Only those in the ROI
    vx_in_faces_with_1_vx_only = np.intersect1d(vx_in_faces_with_1_vx_only, roi_idx)
    vx_in_faces_with_more_than_1_vx = np.intersect1d(vx_in_faces_with_more_than_1_vx, roi_idx)

    isolated_vx = np.setdiff1d(vx_in_faces_with_1_vx_only, vx_in_faces_with_more_than_1_vx)
    return isolated_vx

def dag_find_border_vx(mesh_info, roi_bool, return_type='bool'):
    '''
    Find those vx which are on a border... 
    '''
    assert 'i' in mesh_info.keys(), 'mesh_info should have i,j,k'
    roi_idx = np.where(roi_bool)[0] # Which vx inside ROI
    # Which faces have only 2 vx inside ROI?
    in_face_x = {} 
    for face_x in ['i', 'j', 'k']:
        in_face_x[face_x] = np.isin(mesh_info[face_x], roi_idx) * 1.0
    border_faces = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) >0
    border_faces &= (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) <= 2
    border_vx = []
    for face_x in ['i', 'j', 'k']:                    
        border_vx.append(
            mesh_info[face_x][(border_faces * in_face_x[face_x])==1]
        )
    border_vx = np.concatenate(border_vx)
    border_vx = np.unique(border_vx) # Unique

    if return_type=='bool':
        border_vx_out = np.zeros_like(roi_bool)
        border_vx_out[border_vx] = True
    elif return_type=='idx':
        border_vx_out = border_vx
    elif return_type=='coord':
        border_vx_out = [
            mesh_info['x'][border_vx], 
            mesh_info['y'][border_vx], 
            mesh_info['z'][border_vx],                    
            ]
        
    return border_vx_out
    

def dag_find_border_vx_in_order(mesh_info, roi_bool, return_coords=False):
    '''dag_find_border_vx_in_order
    Find the border vertices in order to draw a closed loop    
    '''
    assert 'i' in mesh_info.keys(), 'mesh_info should have i,j,k'
    outer_edge_list = dag_get_roi_border_edge(mesh_info, roi_bool)    
    border_vx = dag_order_edges(outer_edge_list)
    if not return_coords:
        return border_vx
    # border_vx = sum(border_vx, []) # flatten list
    border_vx_coords = []
    for i_vx in border_vx:
        border_vx_coords.append([
            mesh_info['x'][i_vx],
            mesh_info['y'][i_vx],
            mesh_info['z'][i_vx],
        ])

    return border_vx,border_vx_coords


def dag_get_roi_border_edge(mesh_info, roi_bool):
    '''
    Find those vx which are on a border... 
    '''
    assert 'i' in mesh_info.keys(), 'mesh_info should have i,j,k'
    roi_idx = np.where(roi_bool)[0]
    in_face_x = {}
    for face_x in ['i', 'j', 'k']:
        in_face_x[face_x] = np.isin(mesh_info[face_x], roi_idx) * 1.0
    f_w_outer_edge = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) == 2
    f_w_outer_edge = np.where(f_w_outer_edge)[0]
    ij_faces_match = (in_face_x['i'][f_w_outer_edge] + in_face_x['j'][f_w_outer_edge])==2
    jk_faces_match = (in_face_x['j'][f_w_outer_edge] + in_face_x['k'][f_w_outer_edge])==2
    ki_faces_match = (in_face_x['k'][f_w_outer_edge] + in_face_x['i'][f_w_outer_edge])==2

    ij_faces_match = f_w_outer_edge[ij_faces_match]
    jk_faces_match = f_w_outer_edge[jk_faces_match]
    ki_faces_match = f_w_outer_edge[ki_faces_match]
    
    outer_edge_list = []
    # ij
    outer_edge_list.append(
        np.vstack([mesh_info['i'][ij_faces_match],mesh_info['j'][ij_faces_match]]),
    )
    # jk
    outer_edge_list.append(
        np.vstack([mesh_info['j'][jk_faces_match],mesh_info['k'][jk_faces_match]]),
    )
    # ki     
    outer_edge_list.append(
        np.vstack([mesh_info['k'][ki_faces_match],mesh_info['i'][ki_faces_match]]),
    )    
    # for face in ij_faces_match:
    #     outer_edge_list.append([
    #         mesh_info['i'][face]
    #     ])
    outer_edge_list = np.hstack(outer_edge_list).T
    return outer_edge_list

def dag_order_edges(edges):
    '''Order the edges to form a closed loop'''
    unique_vx = list(np.unique(edges.flatten()))
    # print(unique_vx[0])
    # Step 1: Create an adjacency list
    adjacency_list = {}
    for i_edge in range(edges.shape[0]):
        u, v = edges[i_edge,0],edges[i_edge,1]
        if u not in adjacency_list:
            adjacency_list[u] = []
        if v not in adjacency_list:
            adjacency_list[v] = []
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)

    # Step 2: Choose a starting vertex
    start_vertex = list(adjacency_list.keys())[0]

    # There may be more than one closed path... (e.g., V2v V2d)
    set_unordered = set(unique_vx)
    ordered_list_multi = []
    set_ordered = set(sum(ordered_list_multi, [])) # flatten list and make it a set
    missing_vx = list(set_unordered - set_ordered)
    while len(missing_vx)!=0:
        start_vertex = missing_vx[0]
        ordered_list = dag_traverse_graph(start_vertex, adjacency_list)
        ordered_list_multi.append(ordered_list)
        set_ordered = set(sum(ordered_list_multi, [])) # flatten list and make it a set
        missing_vx = list(set_unordered - set_ordered)

    # OPTION: Convert ordered list to edge list
    # ordered_edges = [(ordered_list[i], ordered_list[(i + 1) % len(ordered_list)]) for i in range(len(ordered_list))]


    return ordered_list_multi#np.array(ordered_list)


def dag_traverse_graph(start_vertex, adjacency_list):
    '''Traverse a graph using 
    i.e. we have a list of edges, and we want to find the order of the vertices
    Useful for drawing an ROI

    start_vertex: int       - the starting vertex
    adjacency_list: dict    - the adjacency list of the graph   
                            - keys are vertices, values are the neighbours                             
    '''    
    # Build an ordered list of vertices, going through the graph
    ordered_list = []
    # Stack - to keep track of the vertices to visit
    stack = [start_vertex]
    # Visited - to keep track of the vertices we have visited
    visited = set()

    while stack:
        # Pop the last vertex from the stack
        current_vertex = stack.pop()
        # If we have not visited this vertex before...
        if current_vertex not in visited:
            # Mark as visited
            visited.add(current_vertex)
            # Add to ordered list
            ordered_list.append(current_vertex)
            # Visit neighbors in reverse order to maintain loop direction
            # (i.e., we want to go clockwise around the ROI)
            # this is added to the stack
            stack.extend(reversed(adjacency_list[current_vertex]))
            
    return ordered_list

def dag_mesh_interpolate(coords1, coords2, interp):
    '''Interpolate coordinate from 1 to 2 in step interp
    So we can inflate the mesh    
    '''
    coords_interp = ((1-interp) * coords1) + (interp * coords2)
    return coords_interp

def dag_mesh_slice(mesh_info, **kwargs):
    '''Slice the mesh along a plane'''
    vx_to_remove = np.zeros_like(mesh_info['x'], dtype=bool)
    for b in ['min', 'max']:
        for c in ['x', 'y', 'z']:
            this_bc = kwargs.get(f'{b}_{c}', None)
            if this_bc is None:
                continue
            if b=='min':
                vx_to_remove |= mesh_info[c]<this_bc
            elif b=='max':
                vx_to_remove |= mesh_info[c]>this_bc
    if vx_to_remove.sum()==0:
        print('No vx to mask')
        return mesh_info
    elif vx_to_remove.all():
        print('Removing everything...')
        return None
    else:
        print(f'{vx_to_remove.sum()} to remove')

    # Create a mapping from old vx to new ones
    
    old_vx_idx = np.arange(mesh_info['x'].shape[0])
    old_vx_idx = old_vx_idx[~vx_to_remove]
    new_vx_idx = np.arange(old_vx_idx.shape[0])
    vx_map = dict(zip(old_vx_idx, new_vx_idx))

    
    # face mask 
    face_to_remove = np.zeros_like(mesh_info['i'], dtype=bool)
    vx_to_remove_idx = np.where(vx_to_remove)[0]
    for c in ['i', 'j', 'k']:
        face_to_remove |= np.isin(mesh_info[c], vx_to_remove_idx)        

    new_mesh_info = {}
    # Sort out vertices
    for c in mesh_info.keys():
        if c in ['i', 'j', 'k']:
            continue
        new_mesh_info[c] = mesh_info[c][~vx_to_remove].copy()

    # Sort out faces 
    for c in ['i', 'j', 'k']:
        # 1 remove faces 
        face_w_old_idx = mesh_info[c][~face_to_remove].copy()
        # 2 fix the ids.. 
        new_mesh_info[c] = np.array([vx_map[old_idx] for old_idx in face_w_old_idx])

    return new_mesh_info

# ************************************************************************
# MESSING AROUND - WITH FLATTENING
# def dag_sph2flat(coords):
#     '''https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion'''
#     # First move to 0,0,0...
#     sph_centre = np.mean(coords, axis=0)
#     coords -= sph_centre

#     # # UV mapping
#     # # https://en.wikipedia.org/wiki/UV_mapping#Finding_UV_on_a_sphere    
#     # abs_coords = coords
#     # u = 0.5 + (np.arctan2(abs_coords[:,2], abs_coords[:,0]) / np.pi*2) # z,x
#     # print(np.isnan(u).mean())
#     # v = 0.5 + (np.arcsin(abs_coords[:,1]) / np.pi)  # y
#     # print(np.arcsin(abs_coords[:,1]))
#     # print(np.isnan(v).mean())
#     # print(np.isnan(u).sum())

#     # print(np.where(np.isnan(v)))


#     x,y,z = coords[:,0], coords[:,1], coords[:,2]
#     XsqPlusYsq = x**2 + y**2
#     r = np.sqrt(XsqPlusYsq + z**2)               # r
#     elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
#     az = np.arctan2(y,x)                           # phi
#     # plt.figure()
#     # plt.scatter(r, az,c=x)
#     # bb
#     # plt.figure()
#     # plt.scatter(u, v,c=x)
#     # plt.figure()
#     # # plt.scatter(, elev,c=x)    
#     # bloop
#     # return az,elev
#     return x,y,z

def dag_dilate_and_drop(mesh_info, vx_bool_start, **kwargs):
    '''Dilate boolean array on mesh until it is contiguous
    Or until max_drop is reached (i.e., number of isolated vx is satisfactory)    
    '''
    max_drop = kwargs.get('max_drop', 10)
    drop_isolated = kwargs.get('drop_isolated', True)

    vx_bool = vx_bool_start.copy()
    keep_going = True
    n_steps = 0     
    while keep_going & (n_steps<100):
        # Find any isolated vx?
        if drop_isolated:
            isolated_vx = dag_find_isolated_vx(roi_bool=vx_bool, mesh_info=mesh_info)
            vx_bool[isolated_vx] = False
        # Is it contiguous?
        vx_border = dag_find_border_vx_in_order(roi_bool=vx_bool, mesh_info=mesh_info)
        if len(vx_border)==1:
            # Ok how many are we dropping?
            keep_going = False
            break
        
        # Lets remove the little loops...
        vx_to_drop = []
        for i_vx in vx_border:
            if len(i_vx)<max_drop:
                # Dropping the vx in this 
                vx_to_drop.extend(i_vx)
        if len(vx_to_drop)==0:            
            # Try again, but lets dilate the selection
            vx_bool = dag_mesh_morph(mesh_info=mesh_info, vx_bool=vx_bool, morph=1)
        else:
            # Try again with removeing small borders
            vx_bool[vx_to_drop] = False
        
        n_steps += 1            
        # Every 10 steps print out the number of vx
        if n_steps%10==0:
            print(f'Vx included: {vx_bool.mean()*100:.2f}%')
    return vx_bool





def dag_sph2flat(coords, **kwargs):
    '''Flatten a sphere to 2D
    This is a probably a bad way to flatten the cortex
    You should probably do proper surface cuts etc...
    But this is a quick and dirty way to do it. And may even be ok when you do if for ROIs...
    It is a work in progress, which may improve overtime...
    
    TODO: 
    * option to define the centre... 
    * better way to do the projection?
    https://en.wikipedia.org/wiki/Map_projection
    
    
    # Adjust longitudes based on the new center longitude
    lon -= center_lon
    
    # Ensure lon is within the range [-pi, pi]
    lon = (lon + np.pi) % (2 * np.pi) - np.pi    
    '''
    # First move to 0,0,0...
    coords -= coords.mean(axis=0)
    # Sanity check -> is distance to 0 should be the same for all points    
    d20 = np.sqrt(np.sum(coords**2, axis=1))
    atol = 10
    if not np.allclose(d20, d20[0], atol=atol):
        print(f'Warning: Not all points are equidistant from 0,0,0 atol={atol}')

    # Now flatten to 2D using longitude and latitude
    x,y,z = coords[:,0], coords[:,1], coords[:,2]
    lat= np.arctan2(z, np.sqrt(x**2 + y**2)) * 2 # 
    lon = np.arctan2(y, x)
    # print(f'Lat: {lat.min()} {lat.max()}')
    # print(f'Lon: {lon.min()} {lon.max()}')


    # Adjust longitudes based on the new center longitude
    centre_bool = kwargs.get('centre_bool', None)
    if centre_bool is not None:
        centre_lat = lat[centre_bool].mean()
        centre_lon = lon[centre_bool].mean()    
        print('centering!')
        lat -= centre_lat
        lat = (lat + np.pi) % (2 * np.pi) - np.pi
        lon -= centre_lon
        lon = (lon + np.pi) % (2 * np.pi) - np.pi

    return lon, lat

def dag_igl_flatten(mesh_info, **kwargs):
    '''Flatten a sphere to 2D
    This is a probably a bad way to flatten the cortex
    You should probably do proper surface cuts etc...
    But this is a quick and dirty way to do it. And may even be ok when you do if for ROIs...
    It is a work in progress, which may improve overtime...
    
    TODO: 
    * option to define the centre... 
    * better way to do the projection?
    https://en.wikipedia.org/wiki/Map_projection
    
    
    # Adjust longitudes based on the new center longitude
    lon -= center_lon
    
    # Ensure lon is within the range [-pi, pi]
    lon = (lon + np.pi) % (2 * np.pi) - np.pi    
    '''
    centre_bool = kwargs.pop('centre_bool', 'bleep') # np.ones_like(mesh_info['x'], dtype=bool))
    roi_bool = centre_bool.copy()
    successful_flatten = False
    # morph = kwargs.pop('morph', 0)
    import contextlib
    n_steps = 0
    total_morph = 0
    while (not successful_flatten) & (n_steps<100):        
        submesh_info = dag_submesh_from_mesh(mesh_info, submesh_bool=roi_bool, check_contiguous=False, **kwargs)
        if submesh_info is None:
            # Not contiguous
            roi_bool = dag_mesh_morph(mesh_info=mesh_info, vx_bool=roi_bool, morph=1)
            total_morph += 1
            n_steps += 1
            continue

        obj_str = dag_obj_write(submesh_info)
        # Write to file
        obj_file = '/tmp/tmp.obj'
        dag_str2file(filename=obj_file, txt=obj_str)
        # https://github.com/libigl/libigl-python-bindings/blob/main/tutorial/tutorials.ipynb
        import igl
        v, f = igl.read_triangle_mesh(obj_file)
        os.remove(obj_file)    
        
        # Find the open boundary
        bnd = igl.boundary_loop(f)

        # Map the boundary to a circle, preserving edge proportions
        bnd_uv = igl.map_vertices_to_circle(v, bnd)
        # Harmonic parametrization for the internal vertices
        uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
        arap = igl.ARAP(v, f, 2, np.zeros(0))
        uva = arap.solve(np.zeros((0, 0)), uv)
        # Check for any nans
        nans_present = (np.isnan(uva)).sum()
        print(f'Nans present: {nans_present.mean()}')
        nans_present = nans_present>0
        # Combine both outputs to check for "error"
        # combined_output = stdout_output + stderr_output
        if (0.1>uva.max()) or (100000<uva.max()) or nans_present: #'error' in output.lower():            
            print(f'Failed to flatten, retrying with morph {total_morph}')
            roi_bool = dag_mesh_morph(mesh_info=mesh_info, vx_bool=roi_bool, morph=1)
            total_morph += 1
        else:
            successful_flatten = True   
            # Lets do a quick check, we want roughly the same orientation
            # If not, then we need to flip the orientation
            # [1] x 
            # corr_x = dag_get_corr(v[:,0], uva[:,0])
            # corr_y = dag_get_corr(v[:,1], uva[:,1])
            # if corr_x<0:
            #     uva[:,0] *= -1
            # if corr_y<0:
            #     uva[:,1] *= -1
            # print(f'Corr x: {corr_x:.2f} y: {corr_y:.2f}')
            # bloop
            print(f'Successful flatten with morph={total_morph}')
            print('Here it looks like this...')
            fig, axs = plt.subplots(1,2)
            axs[0].scatter(uva[:,0], uva[:,1], c=v[:,0])
            axs[0].set_title('Old x color')
            axs[1].scatter(uva[:,0], uva[:,1], c=v[:,1])
            axs[1].set_title('Old y color')
            plt.show()
        
        n_steps += 1            
            
    
    p1 = np.zeros_like(mesh_info['x'])
    p2 = np.zeros_like(mesh_info['y'])
    # check for nans
    assert not np.isnan(uva).any(), 'Nans present in uva'
    p1[submesh_info['vx_idx']] = uva[:,0]
    p2[submesh_info['vx_idx']] = uva[:,1]
    # face_to_cut = np.ones_like(mesh_info['i'], dtype=bool)
    # face_to_cut[submesh_info['face_idx']] = False
    vx_to_include = np.zeros_like(mesh_info['x'], dtype=bool)
    vx_to_include[submesh_info['vx_idx']] = True
    f_to_include = np.zeros_like(mesh_info['i'], dtype=bool)
    f_to_include[submesh_info['face_idx']] = True
    return p1, p2 , vx_to_include, f_to_include

import copy
def dag_flatten(mesh_info, **kwargs):
    '''Take the spherical coordinates
    This is a bad way to flatten the sphere - you should probably do proper surface cuts etc...    
    flatten them to 2D (just polar)
    '''
    method = kwargs.get('method', 'latlon')
    vx_to_include = kwargs.get('vx_to_include', np.ones_like(mesh_info['x'], dtype=bool))
    f_to_include = kwargs.get('f_to_include', np.ones_like(mesh_info['i'], dtype=bool))
    z = kwargs.get('z', 0)
    flat_info = {}
    if method=='latlon':
        p1, p2 = dag_sph2flat(mesh_info['coords'], **kwargs)
    elif method=='igl':
        p1, p2, vx_to_include_IGL, face_to_include_IGL = dag_igl_flatten(mesh_info, **kwargs)
        vx_to_include = vx_to_include_IGL
        f_to_include = face_to_include_IGL


    # find relative scale...
    flat_info['x'] = p1 - p1.mean()# Demean
    flat_info['y'] = p2 - p2.mean()# Demean
    flat_info['z'] = np.ones_like(flat_info['x']) * z

    # FACES TO REMOVE [1] - missing vx    
    vx_not_included = np.where(~vx_to_include)[0]
    f_with_missing_vx = np.isin(mesh_info['i'], vx_not_included) + \
        np.isin(mesh_info['j'], vx_not_included) + \
        np.isin(mesh_info['k'], vx_not_included)    
    
    # [2] - faces with long edges
    # Find the mean length of an edge 
    face_lengths = []
    for i_f in range(mesh_info['i'].shape[0]):
        ei2j = np.sqrt(
            (flat_info['x'][mesh_info['i'][i_f]] - flat_info['x'][mesh_info['j'][i_f]])**2 +
            (flat_info['y'][mesh_info['i'][i_f]] - flat_info['y'][mesh_info['j'][i_f]])**2
        )
        ei2k = np.sqrt(
            (flat_info['x'][mesh_info['i'][i_f]] - flat_info['x'][mesh_info['k'][i_f]])**2 +
            (flat_info['y'][mesh_info['i'][i_f]] - flat_info['y'][mesh_info['k'][i_f]])**2
        )
        ej2k = np.sqrt(
            (flat_info['x'][mesh_info['j'][i_f]] - flat_info['x'][mesh_info['k'][i_f]])**2 +
            (flat_info['y'][mesh_info['j'][i_f]] - flat_info['y'][mesh_info['k'][i_f]])**2
        )
        face_lengths.append(ei2j+ei2k+ej2k)
    face_lengths = np.array(face_lengths)
    m_face_lengths = face_lengths.mean()
    std_face_lengths = face_lengths.std()
    # Find the faces with edges > 4*std
    f_w_long_edges = face_lengths > m_face_lengths + 4*std_face_lengths

    # If these are in the faces to include then remove them...
    f_to_include[f_with_missing_vx] = False
    f_to_include[f_w_long_edges] = False
    print(f'Faces with missing vx: {f_with_missing_vx.sum()}')
    print(f'Faces with long edges: {f_w_long_edges.sum()}')   
    print(f_to_include.mean()) 

    # Keep only the specified faces 
    flat_info['faces']  = mesh_info['faces'][f_to_include,:]
    flat_info['i']      = mesh_info['i'][f_to_include]
    flat_info['j']      = mesh_info['j'][f_to_include]
    flat_info['k']      = mesh_info['k'][f_to_include]

    pts = np.vstack([flat_info['x'],flat_info['y'], flat_info['z']]).T    
    # pts[cut_bool] = 0 # Move pts to cut to 0
    polys = flat_info['faces']
    return pts, polys, vx_to_include

def dag_is_contiguous(mesh_info, vx_bool):
    '''Check if the mesh is contiguous'''
    vx_border = dag_find_border_vx_in_order(roi_bool=vx_bool, mesh_info=mesh_info)
    return vx_border


def dag_submesh_from_mesh(mesh_info, submesh_bool, **kwargs):
    '''Create a submesh from a mesh
    '''
    check_contiguous = kwargs.get('check_contiguous', 'message')
    check_missing_vx = kwargs.get('check_missing_vx', True)
    check_unique_faces = kwargs.get('check_unique_faces', True)
    morph = kwargs.get('morph', 0)
    # Check is contiguous?
    if check_contiguous:
        vx_border = dag_find_border_vx_in_order(roi_bool=submesh_bool, mesh_info=mesh_info)
        if len(vx_border)!=1:
            print('Submesh is not contiguous')
            return None
        
    if morph!=0:
        submesh_bool = dag_mesh_morph(mesh_info=mesh_info, vx_bool=submesh_bool, morph=morph)

    submesh = {
        'full_mesh': copy.deepcopy(mesh_info),
        'vx_idx': np.where(submesh_bool)[0],
        'x': mesh_info['x'][submesh_bool],
        'y': mesh_info['y'][submesh_bool],
        'z': mesh_info['z'][submesh_bool],        
        'coords': mesh_info['coords'][submesh_bool,:],        
    }
    # Other keys...
    for k in mesh_info.keys():
        if k not in submesh.keys():
            submesh[k] = submesh['full_mesh'][k]
    # Translate old to new index
    new_idx = np.arange(submesh_bool.shape[0])
    old_idx = np.where(submesh_bool)[0]
    # Create a mapping from old vx to new ones
    submesh['idx_map'] = dict(zip(old_idx, new_idx))
    # Now sort out the faces
    # Check if any of the faces are in the submesh
    submesh['face_idx'] = np.where(np.isin(mesh_info['i'], submesh['vx_idx']) & \
        np.isin(mesh_info['j'], submesh['vx_idx']) & \
        np.isin(mesh_info['k'], submesh['vx_idx']))[0]
    
    # Translate old to new index
    for c in ['i', 'j', 'k']:
        old_c = mesh_info[c][submesh['face_idx']]
        submesh[c] = np.array([submesh['idx_map'][old_idx] for old_idx in old_c])
    # Faces 
    submesh['faces'] = np.array([submesh['i'], submesh['j'], submesh['k']]).T
    # Check for non-unique faces 
    if check_unique_faces:
        ordered_faces = np.sort(submesh['faces'], axis=1)
        ordered_face_str = [f'{f[0]}_{f[1]}_{f[2]}' for f in ordered_faces]
        unique_faces = np.unique(ordered_face_str)
        if len(unique_faces)!=len(ordered_face_str):
            print(f'Non-unique faces!!!')

    if check_missing_vx:
        # Remove missing vx
        ids = np.arange(submesh['x'].shape[0])
        ids_in_faces = submesh['faces'].flatten()
        missing_vx = np.setdiff1d(ids, ids_in_faces)
        if missing_vx.shape[0]>0:
            print(f'Missing {missing_vx.shape[0]} vx')
            print(f'If you want to be more inclusive try again with morph>0')
        # Check contiguous again

        # if remove_missing_vx:            
        #     print('re running with missing vx removed')
        #     submesh_bool[missing_vx] = False
        #     submesh = dag_submesh_from_mesh(mesh_info=mesh_info, submesh_bool=submesh_bool, **kwargs)
    return submesh

def dag_mesh_morph(mesh_info, vx_bool, morph=1):
    ''' Dilate/erode the mesh from the border by buffer_n
    Dilate  (+ve) - 1 find faces in the border and add the neighbours
    Erode   (-ve) - 1 find faces in the border and remove the neighbours
    '''
    vx_bool_loop = vx_bool.copy()
    while np.abs(morph)>0:
        if morph<0:
            vx_border = dag_find_border_vx(roi_bool=vx_bool_loop, mesh_info=mesh_info)
            # Remove the vx in the border
            vx_bool_loop[vx_border] = False
            morph += 1
        elif morph>0:
            # Find the border
            vx_idx = np.where(vx_bool_loop)[0] # Which vx inside ROI
            # Which faces have only 2 vx inside ROI?
            in_face_x = {} 
            for face_x in ['i', 'j', 'k']:
                in_face_x[face_x] = np.isin(mesh_info[face_x], vx_idx) * 1.0
            border_faces = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) >0
            border_faces &= (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) <= 2
            # Find the vx in the border faces, not in the ROI, and add them
            vx_border_idx = np.unique(mesh_info['faces'][border_faces,:].flatten())
            vx_bool_loop[vx_border_idx] = True
            morph -= 1            

    return vx_bool_loop



def dag_cut_box(mesh_info, **kwargs):
    '''Find vx to cut for a box

    '''
    border_buffer = kwargs.get('border_buffer', 10) # % of the max distance

    borders = {}
    borders['x_min'] = kwargs.get('x_min', None)
    borders['x_max'] = kwargs.get('x_max', None)
    borders['y_min'] = kwargs.get('y_min', None)
    borders['y_max'] = kwargs.get('y_max', None)
    borders['z_min'] = kwargs.get('z_min', None)
    borders['z_max'] = kwargs.get('z_max', None)
    vx_bool = kwargs.get('vx_bool', None)
    if vx_bool is not None:
        for b in ['x', 'y', 'z']:
            borders[f'{b}_min'] = mesh_info[b][vx_bool].min()
            borders[f'{b}_max'] = mesh_info[b][vx_bool].max()
    # Now find the vx to include 
    vx_to_include = np.ones_like(mesh_info['x'], dtype=bool)
    # find the biggest distance
    abs_border_buffer = 0
    for b in ['x', 'y', 'z']:
        abs_border_buffer = max(
            abs_border_buffer, 
            (mesh_info[b].max() - mesh_info[b].min()) * border_buffer / 100)

    for b in ['x', 'y', 'z']:
        if borders[f'{b}_min'] is not None:
            vx_to_include &= mesh_info[b]>borders[f'{b}_min'] - abs_border_buffer
        if borders[f'{b}_max'] is not None:
            vx_to_include &= mesh_info[b]<borders[f'{b}_max'] + abs_border_buffer

    return vx_to_include

# ************************************************************************
def dag_plotly_eye(el, az, zoom):
    # x = zoom*np.cos(np.radians(el))*np.cos(np.radians(az))
    # y = zoom*np.cos(np.radians(el))*np.sin(np.radians(az))
    # z = zoom*np.sin(np.radians(el))

    x = zoom*np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az))
    y = zoom*np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az))
    z = zoom*np.sin(np.deg2rad(el))    
    # fig.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=z)))
    # return dict(
    #     eye=dict(x=x, y=y, z=z),        
    #     )
    return x,y,z

# ************************************************************************
def dag_ply_write(mesh_info, diplay_rgb=None, hemi=None, values=None, incl_rgb=True, x_offset=None):
    n_vx = mesh_info['x'].shape[0]
    n_f = mesh_info['i'].shape[0]
    if not isinstance(values, np.ndarray):
        values = np.ones(n_vx)
    # Create the ply string -> following this format
    ply_str  = f'ply\n'
    ply_str += f'format ascii 1.0\n'
    ply_str += f'element vertex {n_vx}\n'
    ply_str += f'property float x\n'
    ply_str += f'property float y\n'
    ply_str += f'property float z\n'
    if incl_rgb:
        ply_str += f'property uchar red\n'
        ply_str += f'property uchar green\n'
        ply_str += f'property uchar blue\n'
    ply_str += f'property float quality\n'
    ply_str += f'element face {n_f}\n'
    ply_str += f'property list uchar int vertex_index\n'
    ply_str += f'end_header\n'
    
    if x_offset is None:
        if hemi==None:
            x_offset = 0
        elif 'lh' in hemi:
            x_offset = -50
        elif 'rh' in hemi:
            x_offset = 50    
    for i_vx in range(n_vx):
        ply_str += f'{float(mesh_info["x"][i_vx])+x_offset:.6f} ' 
        ply_str += f'{float(mesh_info["y"][i_vx]):.6f} ' 
        ply_str += f'{float(mesh_info["z"][i_vx]):.6f} ' 
        if incl_rgb:
            ply_str += f' {diplay_rgb[i_vx,0]} {diplay_rgb[i_vx,1]} {diplay_rgb[i_vx,2]} '

        ply_str += f'{values[i_vx]:.3f}\n'
    
    for i_f in range(n_f):
        ply_str += f'3 {int(mesh_info["i"][i_f])} {int(mesh_info["j"][i_f])} {int(mesh_info["k"][i_f])} \n'

    return ply_str

def dag_get_rgb_str(rgb_vals):
    '''
    dag_srf_to_ply
    Convert srf file to .ply
    
    '''
    n_vx = rgb_vals.shape[0]
    # Also creating an rgb str...
    rgb_str = ''    
    for v_idx in range(n_vx):
        rgb_str += f'{rgb_vals[v_idx][0]},{rgb_vals[v_idx][1]},{rgb_vals[v_idx][2]}\n'
    return rgb_str    

def dag_obj_write(mesh_info, **kwargs):
    '''
    dag_obj_write
    Convert mesh_info to .obj file
    '''
    x_offset = kwargs.get('x_offset', 0)
    n_vx = mesh_info['x'].shape[0]
    n_f = mesh_info['i'].shape[0]
    # Create the obj string -> following this format
    obj_str  = f'# OBJ file\n'
    obj_str += f'# Vertices: {n_vx}\n'
    obj_str += f'# Faces: {n_f}\n'
    for i_vx in range(n_vx):
        obj_str += f'v {float(mesh_info["x"][i_vx])+x_offset:.6f} ' 
        obj_str += f'{float(mesh_info["y"][i_vx]):.6f} ' 
        obj_str += f'{float(mesh_info["z"][i_vx]):.6f}\n'

    for i_f in range(n_f):
        obj_str += f'f {int(mesh_info["i"][i_f])+1} {int(mesh_info["j"][i_f])+1} {int(mesh_info["k"][i_f])+1}\n'
    
    return obj_str


def dag_fs_write(mesh_info, output_file, **kwargs):

    pts = mesh_info['coords'].copy() 
    # pts[:,0] -= 100 
    # pts[:,1] -= 100
    # pts[:,2] += 100

    # V2 
    # pts[:,[0,1]] = pts[:,[1,0]]
    # V3 
    # pts[:,[0,2]] = pts[:,[2,0]]
    # V4
    # pts[:,0] += 25

    # Switch x and y
    # pts[:,[0,1]] = pts[:,[1,0]]
    polys = mesh_info['faces'].copy()
    comment = kwargs.get('comment', '')
    print(mesh_info.keys())
    # Step 2: Open the file in binary write mode
    with open(output_file, 'wb') as fp:
        # Write the magic number header for .surf format
        fp.write(b'\xFF\xFF\xFE')  # Magic number for FreeSurfer binary .surf file
        fp.write((comment+'\n\n').encode())
        fp.write(struct.pack('>2I', len(pts), len(polys)))
        fp.write(pts.astype(np.float32).byteswap().tostring())
        fp.write(polys.astype(np.uint32).byteswap().tostring())
        fp.write(b'\n')
        if mesh_info['volume_info']!={}:
            fp.write(dag_serialize_volume_info(mesh_info['volume_info']))

    print(f"Surface file '{output_file}'")

# ************************************************************************
def dag_vtk_to_ply(vtk_file):
    '''
    dag_vtk_to_ply
    Convert .vtk file to .ply
    
    '''
    
    with open(vtk_file) as f:
        vtk_lines = f.readlines()
    # Find number of vertices & faces
    for i, line in enumerate(vtk_lines):
        if 'POINTS' in line:
            # n_vx is the only integer on this line
            n_vx = int(line.split(' ')[1])
            point_line = i
        if 'POLYGONS' in line:
            # n_f is the only integer on this line
            n_f = int(line.split(' ')[1])
            poly_line = i

    # Create the ply string -> following this format
    ply_str  = f'ply\n'
    ply_str += f'format ascii 1.0\n'
    ply_str += f'element vertex {n_vx}\n'
    ply_str += f'property float x\n'
    ply_str += f'property float y\n'
    ply_str += f'property float z\n'
    # ply_str += f'property float quality\n'
    ply_str += f'element face {n_f}\n'
    ply_str += f'property list uchar int vertex_index\n'
    ply_str += f'end_header\n'

    # Now add vertex coordinates (from points_line+1 to points_line+n_vx)
    for i in range(point_line+1, point_line+n_vx+1):
        ply_str += vtk_lines[i]
    
    # Now add the faces
    for i in range(poly_line+1, poly_line+n_f+1):
        ply_str += vtk_lines[i]

    # save the ply file
    ply_file = vtk_file.replace('.vtk', '.ply')
    dag_str2file(filename=ply_file, txt=ply_str)




