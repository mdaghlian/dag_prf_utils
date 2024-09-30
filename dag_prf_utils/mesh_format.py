import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
from scipy.spatial import ConvexHull
opj = os.path.join

from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *

def dag_find_border_vx(roi_bool, mesh_info, return_type='bool'):
    '''
    Find those vx which are on a border... 
    '''
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
    

def dag_find_border_vx_in_order(roi_bool, mesh_info, return_coords=False):
    '''dag_find_border_vx_in_order
    Find the border vertices in order to draw a closed loop    
    '''
    outer_edge_list = dag_get_roi_border_edge(roi_bool, mesh_info)    
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


def dag_get_roi_border_edge(roi_bool, mesh_info):
    '''
    Find those vx which are on a border... 
    '''
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



import copy
def dag_latlon_flatten(sphere_mesh_info, **kwargs):
    '''Take the spherical coordinates
    This is a bad way to flatten the sphere - you should probably do proper surface cuts etc...    
    flatten them to 2D (just polar)
    '''
    z = kwargs.get('z', 0)
    latlon_flat = {}
    p1, p2 = dag_sph2flat(sphere_mesh_info['coords'], **kwargs)
    # find relative scale...
    mag = sphere_mesh_info['coords'].max() / p1.max() 
    latlon_flat['x'] = p1 * mag
    latlon_flat['y'] = p2 * mag
    latlon_flat['z'] = np.ones_like(latlon_flat['x']) * z
    
    # Cut faces with any of the "cut_bool" vertices in them
    cut_bool = kwargs.get('cut_bool', None)
    if cut_bool is not None:
        # Find any faces with vertices in the cut
        cut_vx = np.where(cut_bool)[0]
        cut_faces = np.isin(sphere_mesh_info['i'], cut_vx) + np.isin(sphere_mesh_info['j'], cut_vx) + np.isin(sphere_mesh_info['k'], cut_vx)
        cut_faces = cut_faces>0
        print(f'Cutting {cut_faces.sum()} faces out of {cut_faces.shape[0]}')
    else:
        cut_faces = np.zeros(sphere_mesh_info['i'].shape[0], dtype=bool)
    
    
    # Find the mean length of an edge 
    face_lengths = []
    for i_f in range(sphere_mesh_info['i'].shape[0]):
        ei2j = np.sqrt(
            (latlon_flat['x'][sphere_mesh_info['i'][i_f]] - latlon_flat['x'][sphere_mesh_info['j'][i_f]])**2 +
            (latlon_flat['y'][sphere_mesh_info['i'][i_f]] - latlon_flat['y'][sphere_mesh_info['j'][i_f]])**2
        )
        ei2k = np.sqrt(
            (latlon_flat['x'][sphere_mesh_info['i'][i_f]] - latlon_flat['x'][sphere_mesh_info['k'][i_f]])**2 +
            (latlon_flat['y'][sphere_mesh_info['i'][i_f]] - latlon_flat['y'][sphere_mesh_info['k'][i_f]])**2
        )
        ej2k = np.sqrt(
            (latlon_flat['x'][sphere_mesh_info['j'][i_f]] - latlon_flat['x'][sphere_mesh_info['k'][i_f]])**2 +
            (latlon_flat['y'][sphere_mesh_info['j'][i_f]] - latlon_flat['y'][sphere_mesh_info['k'][i_f]])**2
        )
        face_lengths.append(ei2j+ei2k+ej2k)
    face_lengths = np.array(face_lengths)
    m_face_lengths = face_lengths.mean()
    std_face_lengths = face_lengths.std()
    # Find the faces with edges > 4*std
    f_w_long_edges = face_lengths > m_face_lengths + 4*std_face_lengths
    
    cut_faces |= f_w_long_edges
    print(f'Faces with long edges: {f_w_long_edges.sum()}')    

    latlon_flat['faces']  = sphere_mesh_info['faces'][~cut_faces,:]
    latlon_flat['i']      = sphere_mesh_info['i'][~cut_faces]
    latlon_flat['j']      = sphere_mesh_info['j'][~cut_faces]
    latlon_flat['k']      = sphere_mesh_info['k'][~cut_faces]

    pts = np.vstack([latlon_flat['x'],latlon_flat['y'], latlon_flat['z']]).T    
    pts[cut_bool] = 0 # Move pts to cut to 0
    polys = latlon_flat['faces']
    return pts, polys


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




