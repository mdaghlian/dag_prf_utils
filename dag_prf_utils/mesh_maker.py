import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
opj = os.path.join
from dag_prf_utils.utils import *

'''
Experimental way to view MRI surface data (without pycortex; e.g., to view retinotopic maps)
> why do this? 
Pycortex is very powerful, but also quite complex. The source code is difficult to follow, and when it doesn't work; it is difficult to find out why. The idea here is to have a simple script which allows you to plot data on the cortical surface quickly. It should also allow you to specify you're own custom color maps. It (hopefully) allows you to view you're surface in a 3D software package of your choice. Here I am using meshlab. 

You can install Meshlab, and specify the path to run the function. 

What it does: 
[1] For a subject, take a freesurfer surface (e.g., pial, or inflated), convert it into a "mesh file" format which can be easily read by standard 3D rendering software (e.g.,".ply", using meshlab)

[2] Render some anatomical properties of this data on the surface (e.g., the curvature, or sulcal depth) 

[3] Plot arbitrary data on the surface (e.g., retinotopic stuff, like polar angle)
> this can be any values (of a length which matches the number of vertices), specified by the user
> you can specify any matplotlib colormap
> and you can specify the alpha, allowing it to nicely blend with the anatomical data (e.g. the curvature)

This is all saved in a .ply file, and can be viewed using meshlab
> you can also click on individual vertices inside meshlab, to get there position, index, and values (of the data you specified)

[1] Specify freesurfer directory, subject, and surface type
> "/my_project/derivatives/freesurfer/"
> "sub-01"
> "pial" (or could be "inflated")
This gives us the location of the freesurfer file which has the coordinates of every vertex in the mesh, as well as which vertices go together to form the face. This freesurfer file is currently in a binarised format - which takes lower memory, but cannot be read as text. 
[2] Use freesurfer function "mris_convert


# Experimental way to view surfaces (without pycortex)
# Stages:
# [1] Specify the mesh, created by freesurfer you want to use
# -> e.g "pial", or "inflated" 
# [2] Convert from freesurfer file to asc
# -> mris_convert  asc 
# -> rename .asc as .srf file
# [3] Convert from .srf to .ply (using brainder scripts)
# [4] Load the .ply file in meshlab
'''



# Programs files:
prog_folder = os.environ.get('PATH_HOME')
# Brainder script files, needed to convert from .srf to .obj
brainder_script = opj(prog_folder, 'brainder_script') 
srf2obj_path = opj(brainder_script,'srf2obj')

mesh_lab_init = opj(prog_folder,'MeshLab2022.02-linux', 'usr', 'bin', 'meshlab')
blender_init = opj(prog_folder,'blender-3.4.1-linux-x64', 'blender')

from nibabel.freesurfer.io import read_morph_data, write_morph_data
class FSMaker(object):
    '''Used to make a freesurfer file, and view a surface in freesurfer. 
    One of many options for surface plotting. 
    Will create a curv file in subjects freesurfer dir, and load it a specific colormap 
    saved as the relevant command
    '''
    def __init__(self, sub, fs_dir):
        
        self.sub = sub        
        self.fs_dir = fs_dir        # Where the freesurfer files are        
        self.sub_surf_dir = opj(fs_dir, sub, 'surf')
        self.custom_surf_dir = opj(self.sub_surf_dir, 'custom')
        if not os.path.exists(self.custom_surf_dir):
            os.mkdir(self.custom_surf_dir)        
        n_vx = dag_load_nverts(self.sub, self.fs_dir)
        self.n_vx = {'lh':n_vx[0], 'rh':n_vx[1]}
        self.overlay_str = {}
        self.open_surf_cmds = {}

    def add_surface(self, data, surf_name, **kwargs):
        '''
        See dag_calculate_rgb_vals...
        data            np.ndarray      What are we plotting...
        surf_name       str             what are we calling the file

        '''

        data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')    
        vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
        vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))
        cmap_nsteps = kwargs.get('cmap_nsteps', 10)

        data_masked = np.zeros_like(data, dtype=float)
        data_masked[data_mask] = data[data_mask]
        exclude_min_val = vmin - 1
        data_masked[~data_mask] = exclude_min_val

        # SAVE masked data AS A CURVE FILE
        lh_masked_param = data_masked[:self.n_vx['lh']]
        rh_masked_param = data_masked[self.n_vx['lh']:]

        # now save results as a curve file
        print(f'Saving {surf_name} in {self.custom_surf_dir}')

        write_morph_data(opj(self.custom_surf_dir, f'lh.{surf_name}'),lh_masked_param)
        write_morph_data(opj(self.custom_surf_dir, f'rh.{surf_name}'),rh_masked_param)        
        
        # Make custom overlay:
        # value - rgb triple...
        fv_param_steps = np.linspace(vmin, vmax, cmap_nsteps)
        fv_color_steps = np.linspace(0,1, cmap_nsteps)
        fv_cmap = mpl.cm.__dict__[cmap]
        
        ## make colorbar - uncomment to save a png of the color bar...
        # cb_cmap = mpl.cm.__dict__[cmap] 
        # cb_norm = mpl.colors.Normalize()
        # cb_norm.vmin = vmin
        # cb_norm.vmax = vmax
        # plt.close('all')
        # plt.colorbar(mpl.cm.ScalarMappable(norm=cb_norm, cmap=cb_cmap))
        # col_bar = plt.gcf()
        # col_bar.savefig(opj(self.sub_surf_dir, f'lh.{surf_name}_colorbar.png'))

        overlay_custom_str = 'overlay_custom='
        for i, fv_param in enumerate(fv_param_steps):
            this_col_triple = fv_cmap(fv_color_steps[i])
            this_str = f'{float(fv_param):.2f},{int(this_col_triple[0]*255)},{int(this_col_triple[1]*255)},{int(this_col_triple[2]*255)},'

            # print(this_str)
            overlay_custom_str += this_str    
        
        print('Custom overlay string saved here: (self.overlay_str[surf_name])')
        self.overlay_str[surf_name] = overlay_custom_str
    
    def open_fs_surface(self, surf_name, mesh='inflated'):
        # surf name - which surface to load...
        # mesh -> loading inflated? pial? etc.
        os.chdir(self.sub_surf_dir) # move to freeview dir        
        fview_cmd = self.save_fs_cmd(surf_name=surf_name, mesh=mesh)
        os.system(fview_cmd)        

    def save_fs_cmd(self, surf_name, mesh='inflated'):
        lh_surf_path = opj(self.custom_surf_dir, f'lh.{surf_name}')
        rf_surf_path = opj(self.custom_surf_dir, f'rh.{surf_name}')

        fview_cmd = f'''freeview -f lh.{mesh}:overlay={lh_surf_path}:{self.overlay_str[surf_name]} rh.{mesh}:overlay={rf_surf_path}:{self.overlay_str[surf_name]}'''
        dag_str2file(filename=opj(self.custom_surf_dir, f'{surf_name}_cmd.txt'),txt=fview_cmd)
        return fview_cmd

def dag_mlab_open(ply_file_list):
    if not isinstance(ply_file_list, list):
        ply_file_list = [ply_file_list]

    mlab_cmd = f'{mesh_lab_init} '
    for i in ply_file_list:
        mlab_cmd += f'{i} '
    
    os.system(mlab_cmd)

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
                                                                
        open_mlab       bool            Open meshlab at the end...
        return_ply_file bool            Return the ply files which have been made

        
        TODO: open meshlab to a specific angle...
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
    vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
    vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))


    # Create rgb values mapping from data to cmap
    data_cmap = mpl.cm.__dict__[cmap] 
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
        os.system(f'cp {asc_surf_file} {srf_surf_file}')
        
        # *** EXTRA BITS... ****
        # ***> keeping the option because maybe some people like .obj files?
        # [*] Use brainder script to create .obj file    
        os.system(f'{srf2obj_path} {srf_surf_file} > {obj_surf_file}')
        # ^^^  ^^^

        # [4] Use my script to write a ply file...
        if ih=='lh.':
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[:n_hemi_vx[0],:]) # lh
            ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[:n_hemi_vx[0],:], hemi=ih, values=data) # lh
        else:
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[n_hemi_vx[0]:,:]) # rh
            ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[n_hemi_vx[0]:,:],hemi=ih, values=data) # rh
        # Now save the ply file
        ply_file_2write = open(ply_surf_file, "w")
        ply_file_2write.write(ply_str)
        ply_file_2write.close()       

        # Now save the rgb csv file
        rgb_file_2write = open(rgb_surf_file, "w")
        rgb_file_2write.write(rgb_str)
        rgb_file_2write.close()       
        
    # Now use meshlab and "import mesh" to view the surfaces
    if open_mlab:        
        mlab_cmd = f'{mesh_lab_init} {ply_file_2open[0]} {ply_file_2open[1]}'
        os.system(mlab_cmd)

    # Return list of .ply files to open...
    if return_ply_file:
        return ply_file_2open

def dag_srf_to_ply(srf_file, rgb_vals=None, hemi=None, values=None, incl_rgb=False):
    '''
    dag_srf_to_ply
    Convert srf file to .ply
    
    '''
    
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
    data_cmap = mpl.cm.__dict__[cmap] 
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

