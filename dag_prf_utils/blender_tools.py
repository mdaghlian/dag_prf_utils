import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
opj = os.path.join

from .mesh_maker import *
'''
Experimental way to view MRI surface data (without pycortex; e.g., to view retinotopic maps)
> why do this? 
Pycortex is very powerful, but also quite complex. The source code is difficult to follow, and when it doesn't work; it is difficult to find out why. The idea here is to have a simple script which allows you to plot data on the cortical surface quickly. It should also allow you to specify you're own custom color maps. It (hopefully) allows you to view you're surface in a 3D software package of your choice. Here I am using meshlab. 
You can install Blender, and specify the path to run the function. 
'''

# Programs files:
prog_folder = os.environ.get('PATH_HOME')
# Brainder script files, needed to convert from .srf to .obj
brainder_script = opj(prog_folder, 'brainder_script') 
srf2obj_path = opj(brainder_script,'srf2obj')

mesh_lab_init = opj(prog_folder,'MeshLab2022.02-linux', 'usr', 'bin', 'meshlab')
blender_init = opj(prog_folder,'blender-3.4.1-linux-x64', 'blender')


def dag_fs_to_obj_and_rgb(sub, fs_dir,data=None, mesh_name='inflated', out_dir=None, under_surf='curv', **kwargs):
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

        *** COLOR
        cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                            Default: viridis
        vmin            float           Minimum value for colormap
                                                            Default: 10th percentile in data
        vmax            float           Max value for colormap
                                                            Default: 90th percentile in data
                                                                
        return_ply_file bool            Return the ply files which have been made

        
    '''
    save_ply = kwargs.get("save_ply", True)
    save_rgb = kwargs.get("save_rgb", True)
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
        save_rgb = False        
    
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
    
    # Write the script that we will use to load things in blender
    script_file = opj(out_dir, 'eg_script.py') # where the script is going to go...
    if not os.path.exists(script_file):
        with open('./blender_eg_script.py', 'r') as file:
            main_blender_script = file.read()        
        main_blender_script = f'mesh_dir = {out_dir} \n{main_blender_script}'
        script_file_2write = open(script_file, "w")
        script_file_2write.write(main_blender_script)
        script_file_2write.close()               


    # Save the mesh files first as .asc, then .srf, then .obj
    # Then save them as .ply files, with the display rgb data for each voxel

    for ih in ['lh.', 'rh.']:
        mesh_name_file = opj(path_to_sub_surf, f'{ih}{mesh_name}')
        asc_surf_file = opj(out_dir,f'{ih}{surf_name}.asc')
        srf_surf_file = opj(out_dir,f'{ih}{surf_name}.srf')
        ply_surf_file = opj(out_dir,f'{ih}{surf_name}.ply')
        obj_surf_file = opj(out_dir,f'{ih}{surf_name}.obj')    
        rgb_surf_file = opj(out_dir,f'{ih}{surf_name}_rgb.csv')    

        if save_ply:
            # [1] Make asc file using freesurfer mris_convert command:
            os.system(f'mris_convert {mesh_name_file} {asc_surf_file}')
            # [2] Rename .asc as .srf file to avoid ambiguity (using "brainders" conversion tool)
            os.system(f'cp {asc_surf_file} {srf_surf_file}')        
            # [*] Use brainder script to create .obj file    
            os.system(f'{srf2obj_path} {srf_surf_file} > {obj_surf_file}')

            # [4] Use my script to write a ply file...
            if ih=='lh.':
                # ply_str = obj_to_ply(obj_surf_file, display_rgb[:n_hemi_vx[0],:]) # lh
                ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[:n_hemi_vx[0],:], hemi=ih, values=data, incl_rgb=False) # lh
            else:
                # ply_str = obj_to_ply(obj_surf_file, display_rgb[n_hemi_vx[0]:,:]) # rh
                ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[n_hemi_vx[0]:,:],hemi=ih, values=data, incl_rgb=False) # rh
            # Now save the ply file
            ply_file_2write = open(ply_surf_file, "w")
            ply_file_2write.write(ply_str)
            ply_file_2write.close()

            # Remove unwanted files & clean up:
            for i_file in [asc_surf_file, srf_surf_file, obj_surf_file]:
                if os.path.exists(i_file):
                    os.system(f'rm {i_file}')
            # Now save the rgb csv file
            if save_rgb:                
                rgb_file_2write = open(rgb_surf_file, "w")
                rgb_file_2write.write(rgb_str)
                rgb_file_2write.close()               
        
        elif save_rgb:
            if ih=='lh.':
                rgb_str = dag_get_rgb_str(rgb_vals=display_rgb[:n_hemi_vx[0],:])
            else:
                rgb_str = dag_get_rgb_str(rgb_vals=display_rgb[n_hemi_vx[0]:,:])
            rgb_file_2write = open(rgb_surf_file, "w")
            rgb_file_2write.write(rgb_str)
            rgb_file_2write.close()               
