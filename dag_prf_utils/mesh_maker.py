import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
import linescanning.utils as lsutils
opj = os.path.join

# New way of viewing subject surfaces without using pycortex..
# Stages:
# [1] Specify the mesh you want to use
# -> e.g "pial", or "inflated"
# [2] Convert from freesurfer file to asc
# -> mris_convert  asc 
# -> rename .asc as .srf file
# [3] Convert from .srf to .obj (using brainder scripts)
# [4] Use .obj file and some values specified by user (e.g., eccentricity) to write a .ply file
# [5] Load the .ply file in meshlab


# Programs files:
prog_folder = os.environ.get('PATH_HOME')
# Brainder script files, needed to convert from .srf to .obj
brainder_script = opj(prog_folder, 'brainder_script') 
srf2obj_path = opj(brainder_script,'srf2obj')

mesh_lab_init = opj(prog_folder,'MeshLab2022.02-linux', 'usr', 'bin', 'meshlab')
blender_init = opj(prog_folder,'blender-3.4.1-linux-x64', 'blender')

deriv_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'#os.environ.get('DIR_DATA_DERIV')
mesh_dir = opj(deriv_dir, 'mesh_files')
if not os.path.exists(mesh_dir):
    os.mkdir(mesh_dir)

def dag_srf_to_ply(srf_file, rgb_vals, hemi=None, values=None):
    if not isinstance(values, np.ndarray):
        values = np.ones(rgb_vals.shape[0])
    with open(srf_file) as f:
        srf_lines = f.readlines()
    n_vx, n_f = srf_lines[1].split(' ')
    n_vx, n_f = int(n_vx), int(n_f)    
    # Create the ply string -> following this format
    ply_str  = f'ply\n'
    ply_str += f'format ascii 1.0\n'
    ply_str += f'element vertex {n_vx}\n'
    ply_str += f'property float x\n'
    ply_str += f'property float y\n'
    ply_str += f'property float z\n'
    ply_str += f'property uchar red\n'
    ply_str += f'property uchar green\n'
    ply_str += f'property uchar blue\n'
    ply_str += f'property float quality\n'
    ply_str += f'element face {n_f}\n'
    ply_str += f'property list uchar int vertex_index\n'
    ply_str += f'end_header\n'

    if hemi==None:
        x_offset = 0
    elif 'lh' in hemi:
        x_offset = -50
    elif 'rh' in hemi:
        x_offset = 50
    # Cycle through the lines of the obj file and add vx + coords + rgb
    v_idx = 0 # Keep count of vertices     
    for i in range(2,len(srf_lines)):
        # If there is a '.' in the line then it is a vertex
        if '.' in srf_lines[i]:
            split_coord = srf_lines[i][:-2:].split(' ')                        
            coord_count = 0
            for coord in split_coord:
                if ('.' in coord) & (coord_count==0): # Add x_offset
                    ply_str += f'{float(coord)+x_offset:.6f} ' 
                    coord_count += 1
                elif '.' in coord:
                    ply_str += f'{float(coord):.6f} ' 
                    coord_count += 1                    
            
            # Now add the value of the parameters...
            # Now add the rgb values. as integers between 0 and 255
            ply_str += f' {int(rgb_vals[v_idx][0]*255)} {int(rgb_vals[v_idx][1]*255)} {int(rgb_vals[v_idx][2]*255)} '
            
            ply_str += f'{values[v_idx]:.3f}\n'
            v_idx += 1 # next vertex
        
        else:
            # After we finished all the vertices, we need to define the faces
            # -> these are triangles (hence 3 at the beginning of each line)
            # -> the index of the three vx is given
            # For some reason the index is 1 less in .ply files vs .obj files
            # ... i guess like the difference between matlab and python
            ply_str += '3 ' 
            split_idx = srf_lines[i][:-1:].split(' ')
            ply_str += f'{int(split_idx[0])} '
            ply_str += f'{int(split_idx[1])} '
            ply_str += f'{int(split_idx[2])} '
            ply_str += '\n'
    return ply_str



def dag_fs_to_ply(sub, data, fs_dir, mesh_name='inflated', out_dir=None, under_surf='curv', **kwargs):
    '''
    fs_to_ply:
        Create surface files for a subject, and a specific parameter.
        
    Arguments:
        sub             str             e.g. 'sub-01': Name of subject in freesurfer file
        data            np.ndarray      What are we plotting on the surface? 1D array, same length as the number of vertices in subject surface.
        fs_dir          str             Location of the Freesurfer folder
        mesh_name      str              What kind of surface are we plotting on? e.g., pial, inflated...
                                                            Default: inflated
        under_surf      str             What is going underneath the data (e.g., what is the background)?
                                        default is curv. Could also be thick, (maybe smoothwm) 
        out_dir         str             Where to put the mesh files which are made
    **kwargs:
        data_mask       bool array      Mask to hide certain values (e.g., where rsquared is not a good fit)
        data_alpha      np.ndarray      Alpha values for plotting. Where this is specified the undersurf is used instead
        surf_name       str             Name of your surface e.g., 'polar', 'rsq'
                                        *subject name is added to front of surf_name

        ow              bool            Overwrite? If surface with same name already exists, do you want to overwrite it?
                                        Default True
        *** COLOR
        cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                            Default: viridis
        vmin            float           Minimum value for colormap
                                                            Default: 10th percentile in data
        vmax            float           Max value for colormap
                                                            Default: 90th percentile in data
                                                                    
        open_mlab       bool
        # ? possible
        *** CAMERA
        do_scrn_shot    bool            Take screenshots?   Default: True
        do_col_bar      bool            Show color bar?                                             
        azimuth         float           camera angle(0-360) Default: 0
        zoom            float           camera zoom         Default: 1.00
        elevation       float           camera angle(0-360) Default: 0
        roll            float           camera angle(0-360) Default: 0
        ***
    '''
    open_mlab = kwargs.get('open_mlab', True)
    # Get path to subjects surface file
    path_to_sub_surf = opj(fs_dir, sub, 'surf')
    # Check name for surface:
    surf_name = kwargs.get('surf_name', None)
    if surf_name==None:
        print('surf_name not specified, using sub+date')
        surf_name = sub + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M')
    else:
        surf_name = sub + '_' + surf_name + '_' + mesh_name
    # If out_dir not specified, make a folder in mesh_dir
    if out_dir==None:
        out_dir = opj(mesh_dir, surf_name)
            
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    overwrite = kwargs.get('ow', True)
    print(f'File to be named: {surf_name}')        
    if (os.path.exists(opj(out_dir, f'lh.{surf_name}'))) & (not overwrite) :
        print(f'{surf_name} already exists for {sub}, not overwriting surf files...')
        return

    if (os.path.exists(opj(path_to_sub_surf, f'lh.{surf_name}'))): 
        print(f'Overwriting: {surf_name} for {sub}')
    else:
        print(f'Writing: {surf_name} for {sub}')

    # load the undersurf file values, & get number of vx in each hemisphere
    n_hemi_vx = []
    us_values = []
    for ih in ['lh.', 'rh.']:
        with open(opj(path_to_sub_surf,f'{ih}{under_surf}'), 'rb') as h_us:
            h_us.seek(15)
            this_us_vals = np.fromstring(h_us.read(), dtype='>f4').byteswap().newbyteorder()
            us_values.append(this_us_vals)
            n_hemi_vx.append(this_us_vals.shape[0])    
    n_vx = np.sum(n_hemi_vx)
    us_values = np.concatenate(us_values)    
    # Load mask for data to be plotted on surface
    data_mask = kwargs.get('data_mask', np.ones(n_vx, dtype=bool))
    data_alpha = kwargs.get('data_alpha', np.ones(n_vx))
    data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0
    if not isinstance(data, np.ndarray):
        print(f'Just creating {under_surf} file..')
        surf_name = under_surf
        data = np.zeros(n_vx)
        data_alpha = np.zeros(n_vx)        
    
    # Load colormap properties: (cmap, vmin, vmax)
    cmap = kwargs.get('cmap', 'viridis')    
    vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
    vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))


    # Create rgb values mapping from data to cmap
    data_cmap = mpl.cm.__dict__[cmap] 
    data_norm = mpl.colors.Normalize()
    data_norm.vmin = vmin
    data_norm.vmax = vmax
    data_col = data_cmap(data_norm(data))

    # Create rgb values mapping from under_surf to grey cmap
    us_cmap = mpl.cm.__dict__['Greys'] # Always grey underneath
    us_norm = mpl.colors.Normalize()
    if under_surf=='curv':
        us_norm.vmin = -1 # Always -1,1 range...
        us_norm.vmax = 1  
    elif under_surf=='thickness':        
        us_norm.vmin = 0 # Always -1,1 range...
        us_norm.vmax = 5          
    us_col = us_cmap(us_norm(us_values))


    display_rgb = (data_col * data_alpha[...,np.newaxis]) + \
        (us_col * (1-data_alpha[...,np.newaxis]))
    # Save the mesh files first as .asc, then .srf, then .obj
    # Then save them as .ply files, with the display rgb data for each voxel
    ply_file_2open = []
    for ih in ['lh.', 'rh.']:
        mesh_name_file = opj(path_to_sub_surf, f'{ih}{mesh_name}')
        asc_surf_file = opj(out_dir,f'{ih}{surf_name}.asc')
        srf_surf_file = opj(out_dir,f'{ih}{surf_name}.srf')
        # obj_surf_file = opj(out_dir,f'{ih}{surf_name}.obj')
        ply_surf_file = opj(out_dir,f'{ih}{surf_name}.ply')
        ply_file_2open.append(ply_surf_file)
        # [1] Make asc file using freesurfer mris_convert command:
        os.system(f'mris_convert {mesh_name_file} {asc_surf_file}')
        # [2] Rename .asc as .srf file to avoid ambiguity (using "brainders" conversion tool)
        os.system(f'cp {asc_surf_file} {srf_surf_file}')
        # [3] Use brainder script to create .obj file        
        # os.system(f'{srf2obj_path} {srf_surf_file} > {obj_surf_file}')
        # [4] Use my script to write a ply file...
        if ih=='lh.':
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[:n_hemi_vx[0],:]) # lh
            ply_str = dag_srf_to_ply(srf_surf_file, display_rgb[:n_hemi_vx[0],:], hemi=ih, values=data) # lh
        else:
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[n_hemi_vx[0]:,:]) # rh
            ply_str = dag_srf_to_ply(srf_surf_file, display_rgb[n_hemi_vx[0]:,:],hemi=ih, values=data) # rh
        # Now save the ply file
        ply_file_2write = open(ply_surf_file, "w")
        ply_file_2write.write(ply_str)
        ply_file_2write.close()       
        
    # Now use meshlab and "import mesh" to view the surfaces
    if open_mlab:
        # mlab_cmd = f'{mesh_lab_init} {ply_file_2open[0]} {ply_file_2open[1]}'
        mlab_cmd = f'{mesh_lab_init} {ply_file_2open[0]} {ply_file_2open[1]}'
        os.system(mlab_cmd)



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

