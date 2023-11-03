import numpy as np  
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
opj = os.path.join

from .mesh_maker import *
from .fs_tools import *

blender_init = os.environ['BLENDER']


class BlendMaker(object):
    '''Used to make a blender file for a single subject

    
    '''
    def __init__(self, sub, fs_dir, out_dir, **kwargs):

        self.sub = sub        
        self.fs_dir = fs_dir        # Where the freesurfer files are        
        self.sub_surf_dir = opj(fs_dir, sub, 'surf')
        self.out_dir = out_dir      # Where are we putting the files...
        time_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.blender_file_name = opj(self.out_dir, f'{self.sub}_{time_now}.blend')
        self.surf_names = [] # List of surf names...
        self.roi_names = []
        # ** OPTIONAL **
        self.ow = kwargs.get('ow', False)
        if os.path.exists(self.out_dir) and self.ow:
            print('Overwriting existing file')
            os.system(f'rm -rf {self.out_dir}')            
        if not os.path.exists(self.out_dir):
            print(f'Making {self.out_dir}')
            os.mkdir(self.out_dir)
        self.blender_script = opj(out_dir, 'blender_script.py')
        self.under_surf_rgb = {
            'curv'      :{'lh':[], 'rh':[]}, 
            'thickness' :{'lh':[], 'rh':[]}} # RGB values to go under the surface
        n_verts = dag_load_nverts(self.sub, self.fs_dir)
        self.n_vx = {'lh':n_verts[0], 'rh':n_verts[1]}
        # BASIC SETUP -> APPLY FOR ALL SUBJECTS...
        # [1] Make mesh files for the pial, and the inflated hemispheres
        mesh_list = ['pial', 'inflated', 'sphere']
        for i_mesh in mesh_list:
            for i_hemi in ['lh', 'rh']:
                mesh_name_file = opj(self.sub_surf_dir, f'{i_hemi}.{i_mesh}')
                asc_surf_file = opj(self.out_dir,f'{i_hemi}.{i_mesh}.asc')
                srf_surf_file = opj(self.out_dir,f'{i_hemi}.{i_mesh}.srf')
                ply_surf_file = opj(self.out_dir,f'{i_hemi}.{i_mesh}.ply')

                if os.path.exists(ply_surf_file) and self.ow:
                    print('Overwriting: {ply_surf_file}')
                elif os.path.exists(ply_surf_file) and not self.ow:
                    print(f'Already exists: {ply_surf_file}, and not overwriting')                
                    continue
                # [*] Make asc file using freesurfer mris_convert command:
                os.system(f'mris_convert {mesh_name_file} {asc_surf_file}')
                # [*] Rename .asc as .srf file to avoid ambiguity...
                os.system(f'mv {asc_surf_file} {srf_surf_file}')        
                # [*] Create .ply file, using my script
                ply_str = dag_srf_to_ply_basic(srf_file=srf_surf_file, hemi=i_hemi)
                # Now save the ply file
                dag_str2file(filename=ply_surf_file, txt=ply_str)
                # For cleanness remove the .srf file too...
                os.system(f'rm {srf_surf_file}')
        
        # [2] Make rgb files for curvature and depth (under_surfs / us)
        # [-> curvature]
        for us_name in ['curv', 'thickness']:
            self.surf_names.append(us_name)
            for i_hemi in ['lh', 'rh']:
                rgb_us_file = opj(self.out_dir,f'{i_hemi}.{us_name}_rgb.csv')
                if os.path.exists(rgb_us_file) and self.ow:
                    print(f'Overwriting: {rgb_us_file}')
                elif os.path.exists(rgb_us_file) and not self.ow:
                    print(f'Already exists: {rgb_us_file}, and not overwriting')
                    # Load
                    rgb_vals = np.ones((self.n_vx[i_hemi], 4)) # need to be padded so for alpha values
                    rgb_vals[:,:3] = genfromtxt(rgb_us_file, delimiter=',')
                    self.under_surf_rgb[us_name][i_hemi] = np.copy(rgb_vals)
                    continue
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
                self.under_surf_rgb[us_name][i_hemi] = rgb_vals
                # Save as files
                dag_str2file(filename=rgb_us_file, txt=rgb_str)

    def add_cmap(self, data, surf_name, us_name='curv', **kwargs):
        '''
        See dag_calculate_rgb_vals...
        data            np.ndarray      What are we plotting...
        surf_name       str             what are we calling the file
        us_name         str             What goes underneath (if using alpha values)
        '''
        ow = kwargs.get('ow', self.ow) # Overwrite?
        self.surf_names.append(surf_name)
        data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
        data_alpha = kwargs.get('data_alpha', np.ones_like(data, dtype=float))
        data_alpha[~data_mask] = 0 # Make values to be masked have alpha = 0
        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')    
        vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
        vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))
        for i,i_hemi in enumerate(['lh','rh']):
            rgb_file = opj(self.out_dir,f'{i_hemi}.{surf_name}_rgb.csv')
            if os.path.exists(rgb_file) and ow:
                print(f'Overwriting: {rgb_file}')
            elif os.path.exists(rgb_file) and not ow:
                print(f'Already exists: {rgb_file}, and not overwriting')                
                continue

            if i_hemi=='lh':
                this_data = data[:self.n_vx['lh']]
                this_data_mask = data_mask[:self.n_vx['lh']]
                this_data_alpha = data_alpha[:self.n_vx['lh']]

            elif i_hemi=='rh':
                this_data = data[self.n_vx['lh']:]
                this_data_mask = data_mask[self.n_vx['lh']:]
                this_data_alpha = data_alpha[self.n_vx['lh']:]

            this_under_surf = self.under_surf_rgb[us_name][i_hemi]
            rgb_vals, data_col_bar = dag_calculate_rgb_vals(
                data=this_data, 
                under_surf=this_under_surf, 
                data_mask=this_data_mask,
                data_alpha=this_data_alpha,
                cmap=cmap,vmin=vmin,vmax=vmax)
            data_col_bar.savefig(opj(self.out_dir, f'{surf_name}_rgb.png'))
            rgb_str = dag_get_rgb_str(rgb_vals=rgb_vals)
            
            dag_str2file(filename=rgb_file, txt=rgb_str)
    
    def add_roi(self, roi):
        '''
        Save a numpy array of 
        '''
        self.roi_names.append(roi)
        roi_LR = dag_load_roi(self.sub, roi, self.fs_dir, split_LR=True)
        for i_hemi in ['lh', 'rh']:
            roi_file = opj(self.out_dir,f'{i_hemi}.{roi}_roi.npy')
            np.save(roi_file, roi_LR[i_hemi])

    def launch_blender(self, **kwargs):
        '''
        By default loads *everything* (pial, inflated, sphere) and all colormaps in the file
        
        mesh_list       which meshes to load 
        load_all_surf   loads all surfaces in the file (ignores surf_list)
        surf_list       which surfaces to load
        hemi_list       which hemispheres to load
        '''
        mesh_list = kwargs.get('mesh_list', ['pial', 'inflated', 'sphere'])
        if not isinstance(mesh_list, list):
            mesh_list = [mesh_list]
        
        load_all_surf = kwargs.get('load_all_surf', False)
        surf_list = kwargs.get('surf_list', self.surf_names)
        if not isinstance(surf_list, list):
            surf_list = [surf_list]        

        load_all_roi = kwargs.get('load_all_roi', False)
        roi_list = kwargs.get('roi_list', self.roi_names)
        if not isinstance(roi_list, list):
            roi_list = [roi_list]   

        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]        

        save_blender = kwargs.get('save_blender', False)
        close_blender = kwargs.get('save_blender', False)

        # [3] Write the script used to call blender
        blender_script_str = ''
        blender_script_str += bscript_start        
        #
        blender_script_str += "mesh_dir = '" + self.out_dir + "'\n"        
        blender_script_str += "blender_filename = '" + self.blender_file_name + "'\n"
        blender_script_str += f"mesh_list = {mesh_list}\n"
        blender_script_str += f"hemi_list = {hemi_list}\n"
        blender_script_str += bscript_load_mesh
        # 
        blender_script_str += f"load_all_surf = {str(load_all_surf)}\n"
        blender_script_str += f"surf_list = {surf_list}\n"        
        blender_script_str += bscript_load_rgb
        #
        blender_script_str += f"load_all_roi = {str(load_all_roi)}\n"
        blender_script_str += f"roi_list = {roi_list}\n"        
        blender_script_str += bscript_load_roi        
        #
        blender_script_str += bscript_end
        #
        if save_blender:
            blender_script_str += bscript_save
        if close_blender:
            blender_script_str += bscript_close

        
        if os.path.exists(self.blender_script):
            os.system(f'rm {self.blender_script}')
        
        dag_str2file(filename=self.blender_script, txt=blender_script_str)

        os.system(f'{blender_init} --python {self.blender_script}')



bscript_start = '''
### ALWAYS DO THIS AT THE START
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

# Function to create sliders from one mesh to another... (useful later on)
def add_mesh_slider(obj1, obj2, slider_name, add_basis=True):
    mesh_data1 = obj1.data
    mesh_data2 = obj2.data
    if add_basis:
        shape_key_basis = obj1.shape_key_add(name='Basis', from_mix=False)
    shape_key = obj1.shape_key_add(name=slider_name, from_mix=False)

    verts2 = [v for v in mesh_data2.vertices]
    coords2 = np.array([v.co for v in verts2])

    for j,v in enumerate(verts2):
        shape_key.data[j].co = coords2[j]
    
    mesh_data1.update()
    bpy.data.objects.remove(obj2)

# NEXT REMEMBER TO ADD: mesh_dir (where are we working with blender)    
# hemi_list (which hemispheres to load)
# mesh_list (which meshes to load)
'''

bscript_load_mesh = '''
### Load meshes & slider
for i_hemi in hemi_list:
    for i_mesh in mesh_list:
        bpy.ops.import_mesh.ply(filepath=opj(mesh_dir, f'{i_hemi}.{i_mesh}.ply'))        
    
    obj1 = bpy.data.objects[f'{i_hemi}.{mesh_list[0]}']
    if len(mesh_list)>1: # Add sliders if more than one mesh...
        for i_slider in range(1, len(mesh_list)):            
            obj2 = bpy.data.objects[f'{i_hemi}.{mesh_list[i_slider]}']
            add_basis = i_slider==1
            add_mesh_slider(obj1, obj2, slider_name=mesh_list[i_slider], add_basis=add_basis)

    # Rename obj1 to just hemi...
    obj1.name = i_hemi
    bpy.data.meshes[f'{i_hemi}.{mesh_list[0]}'].name = i_hemi
'''

bscript_load_rgb = '''
# Before this - have you specified load_all_surf (all rgb files in folder)
# Or surf_list (only load these specified surfaces)
rgb_files = {'lh':[], 'rh':[]}
rgb_col_bars = []
if load_all_surf:
    file_list = os.listdir(mesh_dir)
    for i in file_list:
        if 'lh' in i:
            hemi = 'lh'
        elif 'rh' in i:
            hemi = 'rh'
        else:
            hemi = None
        if 'rgb.csv' in i:
            rgb_files[hemi].append(i)
        if 'rgb.png' in i:
            rgb_col_bars.append(i)
    rgb_col_bars.sort()
    for ih in hemi_list:
        rgb_files[ih].sort()
else: # else, load specified surf..
    for hemi in hemi_list:
        for i_surf in surf_list:
            rgb_files[hemi].append(f'{hemi}.{i_surf}_rgb.csv')
        rgb_files[hemi].sort()
    for i_surf in surf_list:
        rgb_col_bars.append(f'{i_surf}_rgb.png')
    
for ih,hemi in enumerate(hemi_list):
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
'''

bscript_load_roi = '''
# Before this - have you specified load_all_roi (all roi files in folder)
# Or roi_list (only load these specified rois)
roi_files = {'lh':[], 'rh':[]}
if load_all_surf:
    file_list = os.listdir(mesh_dir)
    for i in file_list:
        if 'lh' in i:
            hemi = 'lh'
        elif 'rh' in i:
            hemi = 'rh'
        else:
            hemi = None
        if 'roi.npy' in i:
            print(i)
            roi_files[hemi].append(i)
    for ih in hemi_list:
        roi_files[ih].sort()
else: # else, load specified surf..
    for hemi in hemi_list:
        for i_roi in roi_list:
            roi_files[hemi].append(f'{hemi}.{i_roi}_roi.npy')
        roi_files[hemi].sort()

for ih,hemi in enumerate(hemi_list):
    # Loop through and add roi
    for i1, i_roi in enumerate(roi_files[hemi]):
        print('loading')
        print(i_roi)
        # [1] Load in roi for this map
        this_roi = np.where(np.load(opj(mesh_dir,i_roi)))[0].tolist()
        print(type(this_roi))
        print(type(this_roi[0]))
        this_n_vx = len(this_roi)
        # Add new roi
        this_group = bpy.data.objects[hemi].vertex_groups.new(name=i_roi)
        # this_group = bpy.data.meshes[hemi].vertex_groups.new(name=i_roi)
        # Enter object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        this_group.add(this_roi, 1.0, 'REPLACE')
'''

bscript_end = '''
### ALWAYS DO THIS AT THE END
# Update the viewport to show the new shading
for area in bpy.data.screens['Layout'].areas:    
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'SOLID'
                space.shading.color_type = 'VERTEX'


bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

'''

bscript_save = '''
bpy.ops.wm.save_as_mainfile(filepath=blender_filename)
'''

bscript_close = '''
bpy.ops.wm.quit_blender()
'''













# ************************************************************************************************************************************
# ************************************************************************************************************************************
# ************************************************************************************************************************************
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
def add_mesh_slider(obj1, obj2, slider_name, add_basis=True):
    mesh_data1 = obj1.data
    mesh_data2 = obj2.data
    if add_basis:
        shape_key_basis = obj1.shape_key_add(name='Basis', from_mix=False)
    shape_key = obj1.shape_key_add(name=slider_name, from_mix=False)

    verts2 = [v for v in mesh_data2.vertices]
    coords2 = np.array([v.co for v in verts2])

    for j,v in enumerate(verts2):
        shape_key.data[j].co = coords2[j]
    
    mesh_data1.update()
    bpy.data.objects.remove(obj2)

if do_sphere:
    mesh_list = ['pial', 'inflated', 'sphere']  
else:
    mesh_list = ['pial', 'inflated']  
for i_hemi in ['lh', 'rh']:
    for i_mesh in mesh_list:
        bpy.ops.import_mesh.ply(filepath=opj(mesh_dir, f'{i_hemi}.{i_mesh}.ply'))        
    obj1 = bpy.data.objects[f'{i_hemi}.pial']
    obj2 = bpy.data.objects[f'{i_hemi}.inflated']
    add_mesh_slider(obj1, obj2, slider_name='inflated')
    if do_sphere:
        obj2 = bpy.data.objects[f'{i_hemi}.sphere']
        add_mesh_slider(obj1, obj2, slider_name='sphere', add_basis=False)        
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

