import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
opj = os.path.join

from .mesh_maker import *

# Programs files:
prog_folder = os.environ.get('PATH_HOME')
blender_init = opj(prog_folder,'blender-3.5.0-linux-x64', 'blender')
print(f'If you want to use blender check - is this the path to it? {blender_init}')
os.environ['FS_LICENSE'] = '/data1/projects/dumoulinlab/Lab_members/Marcus/programs/linescanning/misc/license.txt'
class BlendMaker(object):
    '''Used to make a blender file for a single subject

    
    '''
    def __init__(self, sub, fs_dir, out_dir):
        self.sub = sub
        
        self.fs_dir = fs_dir        # Where the freesurfer files are        
        self.sub_surf_dir = opj(fs_dir, sub, 'surf')
        self.out_dir = out_dir      # Where are we putting the files...
        if not os.path.exists(self.out_dir):
            print(f'Making {self.out_dir}')
            os.mkdir(self.out_dir)
        self.blender_script = opj(out_dir, 'blender_script.py')
        self.under_surf_rgb = {
            'curv'      :[], 
            'thickness' :[]} # RGB values to go under the surface
        self.n_vx = {'lh':None, 'rh':None}
        # BASIC SETUP -> APPLY FOR ALL SUBJECTS...
        # [1] Make mesh files for the pial, and the inflated hemispheres
        for i_mesh in ['pial', 'inflated']:
            for i_hemi in ['lh', 'rh']:
                mesh_name_file = opj(self.sub_surf_dir, f'{i_hemi}.{i_mesh}')
                asc_surf_file = opj(self.out_dir,f'{i_hemi}.{i_mesh}.asc')
                srf_surf_file = opj(self.out_dir,f'{i_hemi}.{i_mesh}.srf')
                ply_surf_file = opj(self.out_dir,f'{i_hemi}.{i_mesh}.ply')

                if os.path.exists(ply_surf_file):
                    print(f'Already exists: {ply_surf_file}')
                    continue
                # [*] Make asc file using freesurfer mris_convert command:
                os.system(f'mris_convert {mesh_name_file} {asc_surf_file}')
                # [*] Rename .asc as .srf file to avoid ambiguity...
                os.system(f'mv {asc_surf_file} {srf_surf_file}')        
                # [*] Create .ply file, using my script
                ply_str = dag_srf_to_ply_basic(srf_file=srf_surf_file, hemi=i_hemi)
                # Now save the ply file
                ply_file_2write = open(ply_surf_file, "w")
                ply_file_2write.write(ply_str)
                ply_file_2write.close()
                # For cleanness remove the .srf file too...
                os.system(f'rm {srf_surf_file}')
        
        # [2] Make rgb files for curvature and depth (under_surfs / us)
        # [-> curvature]
        for us_name in ['curv', 'thickness']:
            for i_hemi in ['lh', 'rh']:
                with open(opj(self.sub_surf_dir,f'{i_hemi}.{us_name}'), 'rb') as h_us:
                    h_us.seek(15)
                    us_vals = np.fromstring(h_us.read(), dtype='>f4').byteswap().newbyteorder()
                if us_name=='curv':
                    vmin,vmax = -1,1
                elif us_name=='thickness':
                    vmin,vmax = 0,5
                rgb_vals, data_col_bar = dag_calculate_rgb_vals(data=us_vals, cmap='Greys', vmin=vmin, vmax=vmax)
                data_col_bar.savefig(opj(self.out_dir, f'{us_name}_rgb.png'))

                rgb_str = dag_get_rgb_str(rgb_vals=rgb_vals)
                
                # SAVE useful info in object
                # -> THE RGB VALUES FOR UNDERSURFACE
                self.under_surf_rgb[us_name].append(np.copy(rgb_vals))
                # -> number of voxels
                self.n_vx[i_hemi] = len(us_vals)
                # Save as files
                rgb_us_file = opj(self.out_dir,f'{i_hemi}.{us_name}_rgb.csv')
                rgb_file_2write = open(rgb_us_file, "w")
                rgb_file_2write.write(rgb_str)
                rgb_file_2write.close()            

        # [3] Write the script used to call blender
        blender_script_str = "mesh_dir = '" + self.out_dir + "'\n"
        blender_script_str += blender_eg_script
        
        bscript_file_2write = open(self.blender_script, "w")
        bscript_file_2write.write(blender_script_str)
        bscript_file_2write.close()           
    def launch_blender(self):
        os.system(f'{blender_init} --python {self.blender_script}')

    def add_cmap(self, data, surf_name, us_name='curv', **kwargs):
        '''
        See dag_calculate_rgb_vals...
        data            np.ndarray      What are we plotting...
        surf_name       str             what are we calling the file
        us_name         str             What goes underneath (if using alpha values)
        '''
        data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
        data_alpha = kwargs.get('data_alpha', np.ones_like(data, dtype=float))
        data_alpha[~data_mask] = 0 # Make values to be masked have alpha = 0
        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')    
        vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
        vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))
        for i,i_hemi in enumerate(['lh','rh']):
            if i_hemi=='lh':
                this_data = data[:self.n_vx['lh']]
                this_data_mask = data_mask[:self.n_vx['lh']]
                this_data_alpha = data_alpha[:self.n_vx['lh']]

            elif i_hemi=='rh':
                this_data = data[self.n_vx['lh']:]
                this_data_mask = data_mask[self.n_vx['lh']:]
                this_data_alpha = data_alpha[self.n_vx['lh']:]

            this_under_surf = self.under_surf_rgb[us_name][i]
            rgb_vals, data_col_bar = dag_calculate_rgb_vals(
                data=this_data, 
                under_surf=this_under_surf, 
                data_mask=this_data_mask,
                data_alpha=this_data_alpha,
                cmap=cmap,vmin=vmin,vmax=vmax)
            data_col_bar.savefig(opj(self.out_dir, f'{surf_name}_rgb.png'))
            rgb_str = dag_get_rgb_str(rgb_vals=rgb_vals)
            rgb_file = opj(self.out_dir,f'{i_hemi}.{surf_name}_rgb.csv')
            rgb_file_2write = open(rgb_file, "w")
            rgb_file_2write.write(rgb_str)
            rgb_file_2write.close()            

        

def dag_fs_to_ply_and_rgb(sub, fs_dir,data=None, mesh_name='inflated', out_dir=None, under_surf='curv', **kwargs):
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
        surf_name = sub + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + under_surf + '_' + mesh_name
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
        surf_name = sub + '_' + under_surf + '_' + mesh_name
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
        # with open('./blender_eg_script.py', 'r') as file:
        #     main_blender_script = file.read()        
        main_blender_script = f'mesh_dir = {out_dir} \n{blender_eg_script}'
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
        rgb_surf_file = opj(out_dir,f'{ih}{surf_name}_rgb.csv')    

        if save_ply:
            # [1] Make asc file using freesurfer mris_convert command:
            os.system(f'mris_convert {mesh_name_file} {asc_surf_file}')
            # [2] Rename .asc as .srf file to avoid ambiguity (using "brainders" conversion tool)
            os.system(f'cp {asc_surf_file} {srf_surf_file}')        

            # [4] Use my script to write a ply file...
            if ih=='lh.':
                ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[:n_hemi_vx[0],:], hemi=ih, values=data, incl_rgb=False) # lh
            else:
                ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[n_hemi_vx[0]:,:],hemi=ih, values=data, incl_rgb=False) # rh
            # Now save the ply file
            ply_file_2write = open(ply_surf_file, "w")
            ply_file_2write.write(ply_str)
            ply_file_2write.close()

            # Remove unwanted files & clean up:
            for i_file in [asc_surf_file, srf_surf_file]:
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

blender_eg_script = '''
import csv
import os 
opj = os.path.join
import bpy
import bmesh
import numpy as np
from numpy import genfromtxt

# start in object mode & clear any startup clutter...
objects = [o for o in bpy.data.objects]
for obj in objects:
    bpy.data.objects.remove(obj)

meshes = [o for o in bpy.data.meshes]
for m in meshes:
    bpy.data.meshes.remove(m)
mats = [o for o in bpy.data.materials]
for m in mats:
    bpy.data.materials.remove(m)    

# Now load the 2 hemispheres, and create a slider to shrink and inflate (from pial to inflated)    
def add_mesh_slider(obj1, obj2):
    mesh_data1 = obj1.data
    mesh_data2 = obj2.data
    
    shape_key_basis = obj1.shape_key_add(name='Basis', from_mix=False)
    shape_key = obj1.shape_key_add(name='Interpolated', from_mix=False)

    verts2 = [v for v in mesh_data2.vertices]
    coords2 = np.array([v.co for v in verts2])

    for j,v in enumerate(verts2):
        shape_key.data[j].co = coords2[j]
    
    mesh_data1.update()
    bpy.data.objects.remove(obj2)

for i_hemi in ['lh', 'rh']:
    for i_mesh in ['pial', 'inflated']:
        bpy.ops.import_mesh.ply(filepath=opj(mesh_dir, f'{i_hemi}.{i_mesh}.ply'))        
    obj1 = bpy.data.objects[f'{i_hemi}.pial']
    obj2 = bpy.data.objects[f'{i_hemi}.inflated']
    add_mesh_slider(obj1, obj2)
    # Rename .pial to just hemi...
    obj1.name = i_hemi
    bpy.data.meshes[f'{i_hemi}.pial'].name = i_hemi

# Now add the rgb colors to the meshes    
# -> Get list of rgb files to add
file_list = os.listdir(mesh_dir)
rgb_files = {'lh':[], 'rh':[]}
for i in file_list:
    if 'lh' in i:
        hemi = 'lh'
    elif 'rh' in i:
        hemi = 'rh'
    else:
        hemi = None
    if 'rgb.csv' in i:
        rgb_files[hemi].append(i)
for ih in ['lh', 'rh']:
    rgb_files[ih].sort()
        
for ih,hemi in enumerate(['lh', 'rh']):
    # bpy.data.objects[ih].name = hemi
    # bpy.data.meshes[ih].name = hemi
    # Loop through and add color layers
    for i1, i_col in enumerate(rgb_files[hemi]):
        print('loading')
        print(i_col)
        # [1] Load in rgb color data for this map
        rgb_data = genfromtxt(opj(mesh_dir,i_col), delimiter=',')
        if rgb_data.max()>1:
            rgb_data = rgb_data/255
        r,g,b = rgb_data[:,0]*1, rgb_data[:,1], rgb_data[:,2]

        # Add new color layer
        color_layer = bpy.data.meshes[hemi].vertex_colors.new(name=i_col)
        i = 0
        for poly in bpy.data.meshes[hemi].polygons:
            for idx in poly.loop_indices:
                this_vx = bpy.data.meshes[hemi].loops[idx].vertex_index
                color_layer.data[i].color = (r[this_vx], g[this_vx], b[this_vx], 0.5)
                i += 1

# Update the viewport to show the new shading
for area in bpy.data.screens['Layout'].areas:    
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'SOLID'
                space.shading.color_type = 'VERTEX'


bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

'''


# ********************************* BLENDER TOOLS ****************************************
# THESE ARE ONLY GOING TO WORK *INSIDE* BLENDER...
bp_script_add_animation='''
# ADD ANIMATION TO VERTEX COLOR LAYERS...
obj = bpy.data.objects['lh']
total_n_frames = 20
total_n_layers = len(obj.data.vertex_colors)
layer_names = list(obj.data.vertex_colors.keys())
def set_vcols(frame):
        
    layer_to_plot = int(total_n_layers * frame / total_n_frames)            

    if layer_to_plot>total_n_layers:
        layer_to_plot = total_n_layers
    bpy.data.meshes["lh"].attributes.active_color_index = layer_to_plot#layer_names[layer_to_plot]
    bpy.data.meshes["rh"].attributes.active_color_index = layer_to_plot#layer_names[layer_to_plot]

def my_handler(scene):
    frame = scene.frame_current
    set_vcols(frame)

bpy.app.handlers.frame_change_pre.append(my_handler) 
'''

