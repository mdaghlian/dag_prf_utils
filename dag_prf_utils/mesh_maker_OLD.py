import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
opj = os.path.join
# try:
#     from nibabel.freesurfer.io import write_morph_data
# except ImportError:
#     raise ImportError('Error importing nibabel... Not a problem unless you want to use FSMaker')
from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *


path_to_utils = os.path.abspath(os.path.dirname(__file__))

# Functions for messing around with meshes that are not freesurfer related...

def dag_fs_to_ply(sub, data, fs_dir=os.environ['SUBJECTS_DIR'], mesh_name='inflated', out_dir=None, under_surf='curv', **kwargs):
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
                                                                
        return_ply_file bool            Return the ply files which have been made
        
    '''
    return_ply_file = kwargs.get('return_ply_file', False)    
    # Get path to subjects surface file
    path_to_sub_surf = opj(fs_dir, sub, 'surf')
    # Check name for surface:
    surf_name = kwargs.get('surf_name', None)
    if surf_name==None:
        print('surf_name not specified, using sub+date')
        surf_name = sub + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M')
    else:
        surf_name = sub + '_' + surf_name + '_' + mesh_name
            
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
    vmin = kwargs.get('vmin', np.nanmin(data[data_mask]))
    vmax = kwargs.get('vmin', np.nanmax(data[data_mask]))

    # vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
    # vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))


    # Create rgb values mapping from data to cmap
    data_cmap = dag_get_cmap(cmap)
    # data_cmap = mpl.cm.__dict__[cmap] 
    data_norm = mpl.colors.Normalize()
    data_norm.vmin = vmin
    data_norm.vmax = vmax
    data_col = data_cmap(data_norm(data))
    
    # CHANGE FOR NAN
    # data[~data_mask] = 0

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
        # 
        obj_surf_file = opj(out_dir,f'{ih}{surf_name}.obj')    
        rgb_surf_file = opj(out_dir,f'{ih}{surf_name}_rgb.csv')    
        #
        ply_surf_file = opj(out_dir,f'{ih}{surf_name}.ply')
        ply_file_2open.append(ply_surf_file)
        # [1] Make asc file using freesurfer mris_convert command:
        os.system(f'mris_convert {mesh_name_file} {asc_surf_file}')
        # [2] Rename .asc as .srf file to avoid ambiguity (using "brainders" conversion tool)
        os.system(f'mv {asc_surf_file} {srf_surf_file}')
        
        # *** EXTRA BITS... ****
        # ***> keeping the option because maybe some people like .obj files?
        # [*] Use brainder script to create .obj file    
        # os.system(f'{srf2obj_path} {srf_surf_file} > {obj_surf_file}')
        # ^^^  ^^^

        # [4] Use my script to write a ply file...
        if ih=='lh.':
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[:n_hemi_vx[0],:]) # lh
            ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[:n_hemi_vx[0],:], hemi=ih, values=data, **kwargs) # lh
        else:
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[n_hemi_vx[0]:,:]) # rh
            ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[n_hemi_vx[0]:,:],hemi=ih, values=data, **kwargs) # rh
        # Now save the ply file
        ply_file_2write = open(ply_surf_file, "w")
        ply_file_2write.write(ply_str)
        ply_file_2write.close()       

        # Remove the srf file
        os.system(f'rm {srf_surf_file}')

        # # Now save the rgb csv file
        # rgb_file_2write = open(rgb_surf_file, "w")
        # rgb_file_2write.write(rgb_str)
        # rgb_file_2write.close()       
        
    # Return list of .ply files to open...
    if return_ply_file:
        return ply_file_2open

def dag_srf_to_ply(srf_file, rgb_vals=None, hemi=None, values=None, incl_rgb=True, **kwargs):
    '''
    dag_srf_to_ply
    Convert srf file to .ply
    
    '''
    x_offset = kwargs.get('x_offset', None)
    
    if not isinstance(values, np.ndarray):
        values = np.ones(rgb_vals.shape[0])
    with open(srf_file) as f:
        srf_lines = f.readlines()
    n_vx, n_f = srf_lines[1].split(' ')
    n_vx, n_f = int(n_vx), int(n_f)
    # Also creating an rgb str...
    rgb_str = ''    
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
            if incl_rgb:
                ply_str += f' {int(rgb_vals[v_idx][0]*255)} {int(rgb_vals[v_idx][1]*255)} {int(rgb_vals[v_idx][2]*255)} '
                # ply_str += f' {rgb_vals[v_idx][0]} {rgb_vals[v_idx][1]} {rgb_vals[v_idx][2]} '

            # RGB str
            rgb_str += f'{int(rgb_vals[v_idx][0]*255)},{int(rgb_vals[v_idx][1]*255)},{int(rgb_vals[v_idx][2]*255)}\n'
            
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
    return ply_str, rgb_str

def dag_srf_to_ply_basic(srf_file, hemi=None):
    '''dag_srf_to_ply_basic
    Convert srf file to .ply (not including values, or rgb stuff)
    
    '''
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
            ply_str += f'{0:.3f}\n'
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

def dag_calculate_rgb_vals(data, **kwargs):
    '''
    dag_calculate_rgb_vals:
        Create an array of RGB values to plot on the surface

        
    Arguments:
        data            np.ndarray      What are we plotting on the surface? 1D array, same length as the number of vertices in subject surface.

    **kwargs:
        under_surf      np.ndarray      What is underlay of the data? i.e., going underneath the data (e.g., what is the background)?
                                        could be the curvature...
        data_mask       bool array      Mask to hide certain values (e.g., where rsquared is not a good fit)
        data_alpha      np.ndarray      Alpha values for plotting. Where this is specified the undersurf is used instead

        *** COLOR
        cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                            Default: viridis
        vmin            float           Minimum value for colormap
                                                            Default: 10th percentile in data
        vmax            float           Max value for colormap
                                                            Default: 90th percentile in data
                                                                
    '''
    under_surf = kwargs.get('under_surf', np.zeros((data.shape[0], 4)))
    data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
    data_alpha = kwargs.get('data_alpha', np.ones_like(data, dtype=float))
    data_alpha[~data_mask] = 0 # Make values to be masked have alpha = 0

    # Load colormap properties: (cmap, vmin, vmax)
    cmap = kwargs.get('cmap', 'viridis')    
    vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
    vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))
    
    # Create rgb values mapping from data to cmap
    data_cmap = dag_get_cmap(cmap)
    # data_cmap = mpl.cm.__dict__[cmap] 
    data_norm = mpl.colors.Normalize()
    data_norm.vmin = vmin
    data_norm.vmax = vmax
    data_col = data_cmap(data_norm(data))

    # Create a color bar...
    plt.close('all')
    plt.colorbar(mpl.cm.ScalarMappable(norm=data_norm, cmap=data_cmap))
    data_col_bar = plt.gcf()
    rgb_vals = (data_col * data_alpha[...,np.newaxis]) + \
        (under_surf * (1-data_alpha[...,np.newaxis]))
    
    return rgb_vals, data_col_bar

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


# ******
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
