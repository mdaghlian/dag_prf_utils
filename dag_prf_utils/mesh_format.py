import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
from scipy.spatial import ConvexHull
opj = os.path.join

from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *

def dag_sph2flat(coords):
    '''https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion'''
    # First move to 0,0,0...
    sph_centre = np.mean(coords, axis=0)
    coords -= sph_centre

    # # UV mapping
    # # https://en.wikipedia.org/wiki/UV_mapping#Finding_UV_on_a_sphere    
    # abs_coords = coords
    # u = 0.5 + (np.arctan2(abs_coords[:,2], abs_coords[:,0]) / np.pi*2) # z,x
    # print(np.isnan(u).mean())
    # v = 0.5 + (np.arcsin(abs_coords[:,1]) / np.pi)  # y
    # print(np.arcsin(abs_coords[:,1]))
    # print(np.isnan(v).mean())
    # print(np.isnan(u).sum())

    # print(np.where(np.isnan(v)))


    x,y,z = coords[:,0], coords[:,1], coords[:,2]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    # plt.figure()
    # plt.scatter(r, az,c=x)
    # bb
    # plt.figure()
    # plt.scatter(u, v,c=x)
    # plt.figure()
    # # plt.scatter(, elev,c=x)    
    # bloop
    return az,elev
import copy
def dag_fake_flatten(sphere_mesh_info):
    '''Take the spherical coordinates
    > flatten them to 2D (just polar)
    > find the enclosing edges convex hull 
    > remove the faces with 3 vx in the hull    
    '''
    fake_flat = {}
    p1, p2 = dag_sph2flat(sphere_mesh_info['coords'])
    # find relative scale...
    mag = sphere_mesh_info['coords'].max() / p1.max() 
    fake_flat['x'] = p1 * mag
    fake_flat['y'] = p2 * mag
    fake_flat['z'] = np.zeros_like(p1)    
    pts = np.vstack([fake_flat['x'],fake_flat['y'], fake_flat['z']]).T
    outer_vx = ConvexHull(pts[:,:2]).vertices
    plt.figure()
    plt.plot(
        pts[:,0], pts[:,1], c=sphere_mesh_info['x']
    )
    bloop
    in_face_x = {} 
    for face_x in ['i', 'j', 'k']:
        in_face_x[face_x] = np.isin(sphere_mesh_info[face_x], outer_vx) * 1.0    
    
    # remove faces with 2+ vx in the cut
    f_w_23cutvx = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) >= 2
    fake_flat['faces']  = sphere_mesh_info['faces'][~f_w_23cutvx,:]
    fake_flat['i']      = sphere_mesh_info['i'][~f_w_23cutvx]
    fake_flat['j']      = sphere_mesh_info['j'][~f_w_23cutvx]
    fake_flat['k']      = sphere_mesh_info['k'][~f_w_23cutvx]


    # # Now find those outer vertices which appear in 2 faces
    # flat_faces = fake_flat['faces'].ravel()
    # vx_counts = np.sum(flat_faces[:, None]==outer_vx, axis=0)
    # print(vx_counts)

    # vx_in2face = np.where(vx_counts==2)[0]
    # faces2remove = []
    # for i_outvx in vx_in2face:
    #     this_vx = outer_vx[i_outvx]
    #     # find it ...

    #     faces2remove.append(
    #         np.where(np.sum(fake_flat['faces'] == this_vx, axis=1) > 0)[0][0]
    #     )
    # # Remove the face
    # if faces2remove is not []:
    #     fake_flat['faces'] = np.delete(fake_flat['faces'], faces2remove, axis=0)
    #     fake_flat['i'] = np.delete(fake_flat['i'], faces2remove)
    #     fake_flat['j'] = np.delete(fake_flat['j'], faces2remove)
    #     fake_flat['k'] = np.delete(fake_flat['k'], faces2remove)


    # Now find those outer vertices which appear in 2 faces
    flat_faces = fake_flat['faces'].ravel()
    vx_counts = np.sum(flat_faces[:, None]==outer_vx, axis=0)

    polys = fake_flat['faces']    
    return pts, polys

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
    outer_edge_list = dag_get_roi_border_edge(roi_bool, mesh_info)
    border_vx = dag_order_edges(outer_edge_list)
    if not return_coords:
        return border_vx
    border_vx = sum(border_vx, []) # flatten list
    border_vx_coords = [
        mesh_info['x'][border_vx], 
        mesh_info['y'][border_vx], 
        mesh_info['z'][border_vx],                    
    ]
    return border_vx_coords


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
    unique_vx = list(np.unique(edges.flatten()))

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
    # Step 3 and 4: Traverse the graph and build the ordered list
    ordered_list = []
    stack = [start_vertex]
    visited = set()

    while stack:
        current_vertex = stack.pop()
        if current_vertex not in visited:
            visited.add(current_vertex)
            ordered_list.append(current_vertex)

            # Visit neighbors in reverse order to maintain loop direction
            stack.extend(reversed(adjacency_list[current_vertex]))
    return ordered_list

# ************************************************************************

def dag_mesh_slice(mesh_info, **kwargs):
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
def dag_ply_write(mesh_info, diplay_rgb, hemi=None, values=None, incl_rgb=True, x_offset=None):
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