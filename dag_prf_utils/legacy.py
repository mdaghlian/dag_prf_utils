



# ***********************************************************************************************************************
# ***********************************************************************************************************************

# from dag_prf_utils import *
# def dag_get(sub, fs_dir, roi, ply_file=None):
#     # [1] Get roi idx: 
#     roi_idx = dag_load_roi(sub, roi, fs_dir)
#     roi_idc = np.where(roi_idx)[0]
#     dag_parse_surf(opj(fs_dir, sub, ))
    
#     bound_vx = []

#     return vx_bound
# def obj_to_ply(obj_file, rgb_vals):
#     with open(obj_file) as f:
#         obj_lines = f.readlines()
#     with open(obj_file) as f:    
#         obj_str = f.read()
#     n_vx = obj_str.count('v') # Number of vertices
#     n_f = obj_str.count('f')  # Number of faces 
#     # Create the ply string -> following this format
#     ply_str  = f'ply\n'
#     ply_str += f'format ascii 1.0\n'
#     ply_str += f'element vertex {n_vx}\n'
#     ply_str += f'property float x\n'
#     ply_str += f'property float y\n'
#     ply_str += f'property float z\n'
#     ply_str += f'property uchar red\n'
#     ply_str += f'property uchar green\n'
#     ply_str += f'property uchar blue\n'
#     ply_str += f'element face {n_f}\n'
#     ply_str += f'property list uchar int vertex_index\n'
#     ply_str += f'end_header\n'


#     # Cycle through the lines of the obj file and add vx + coords + rgb
#     v_idx = 0 # Keep count of vertices     
#     for i in range(len(obj_lines)):
#         if obj_lines[i][0]=='v':
#             split_coord = obj_lines[i][2:-1].split(' ')
#             # for some reason in .ply files the first coordinates valence is flipped (-1 to 1) 
#             # also the order is 0,2,1 from obj...
#             ply_str += f'{float(split_coord[0]*-1):.6f} '  # *-1
#             ply_str += f'{float(split_coord[2]):.6f} '
#             ply_str += f'{float(split_coord[1]):.6f} '
#             # Now add the rgb values. as integers between 0 and 255
#             ply_str += f' {int(rgb_vals[v_idx][0]*255)} {int(rgb_vals[v_idx][1]*255)} {int(rgb_vals[v_idx][2]*255)}\n'
            
#             v_idx += 1 # next vertex
        
#         elif obj_lines[i][0]=='f':
#             # After we finished all the vertices, we need to define the faces
#             # -> these are triangles (hence 3 at the beginning of each line)
#             # -> the index of the three vx is given
#             # For some reason the index is 1 less in .ply files vs .obj files
#             # ... i guess like the difference between matlab and python
#             ply_str += '3 ' 
#             split_idx = obj_lines[i][2::].split(' ')
#             ply_str += f'{int(split_idx[0])-1} '
#             ply_str += f'{int(split_idx[1])-1} '
#             ply_str += f'{int(split_idx[2])-1} '
#             ply_str += '\n'
#     return ply_str

