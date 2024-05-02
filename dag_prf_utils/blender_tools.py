import numpy as np  
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
opj = os.path.join

from dag_prf_utils.mesh_maker import *
from dag_prf_utils.mesh_format import *
# from dag_prf_utils.fs_tools import *
try:
    blender_init = os.environ['BLENDER']
except:
    blender_init = 'blender '


class BlendMaker(GenMeshMaker):
    '''Used to make a blender file for a single subject
    
    '''
    def __init__(self, sub, fs_dir, output_dir, **kwargs):
        super().__init__(sub, fs_dir, output_dir)
        time_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.blender_file_name = opj(self.output_dir, f'{self.sub}_{time_now}.blend')
        self.roi_names = []
        self.surf_names = []
        # ** OPTIONAL **
        self.ow = kwargs.get('ow', False)
        if os.path.exists(self.output_dir) and self.ow:
            print('Overwriting existing file')
            os.system(f'rm -rf {self.output_dir}')            
        if not os.path.exists(self.output_dir):
            print(f'Making {self.output_dir}')
            os.mkdir(self.output_dir)
        self.blender_script = opj(output_dir, 'blender_script.py')
        # BASIC SETUP -> APPLY FOR ALL SUBJECTS...
        # [1] Make mesh files for the pial, and the inflated hemispheres
        for mesh in ['pial', 'inflated', 'sphere']:
            self.add_ply_surface(
                surf_name=mesh, mesh_name=mesh, ow=self.ow,
                incl_rgb=False, incl_values=False, 
            )
        # [2] Make rgb files for curvature 
        # [-> curvature]
        self.us_cols_split = {}
        for us in self.us_cols.keys():
            self.us_cols_split[us] = {}                
            self.us_cols_split[us]['lh'] = self.us_cols[us][:self.n_vx['lh'],:]
            self.us_cols_split[us]['rh'] = self.us_cols[us][self.n_vx['lh']:,:]
        
        for us_name in ['curv']:
            self.surf_names.append(us_name)
            for i_hemi in ['lh', 'rh']:
                rgb_us_file = opj(self.output_dir,f'{i_hemi}.{us_name}_rgb.csv')
                if os.path.exists(rgb_us_file) and self.ow:
                    print(f'Overwriting: {rgb_us_file}')
                elif os.path.exists(rgb_us_file) and not self.ow:
                    print(f'Already exists: {rgb_us_file}, and not overwriting')
                    continue
                
                rgb_str = dag_get_rgb_str(self.us_cols_split[us_name][i_hemi])
                # Save as files
                dag_str2file(filename=rgb_us_file, txt=rgb_str)

    def add_blender_cmap(self, data, surf_name, **kwargs):
        '''
        See dag_calculate_rgb_vals...
        data            np.ndarray      What are we plotting...
        surf_name       str             what are we calling the file
        us_name         str             What goes underneath (if using alpha values)
        '''
        ow = kwargs.get('ow', self.ow) # Overwrite?
        self.surf_names.append(surf_name)
        display_rgb, cmap_dict = self.return_display_rgb(data, return_cmap_dict=True, split_hemi=True, **kwargs)
        # Save the colormap
        fig = dag_cmap_plotter(title=surf_name, return_fig=True, **cmap_dict)
        fig.savefig(opj(self.output_dir, f'{surf_name}_rgb.png'))
        for hemi in ['lh','rh']:
            rgb_file = opj(self.output_dir,f'{hemi}.{surf_name}_rgb.csv')
            if os.path.exists(rgb_file) and ow:
                print(f'Overwriting: {rgb_file}')
            elif os.path.exists(rgb_file) and not ow:
                print(f'Already exists: {rgb_file}, and not overwriting')                
                continue
            
            rgb_str = dag_get_rgb_str(display_rgb[hemi])
            # Save as files
            dag_str2file(filename=rgb_file, txt=rgb_str)


    def launch_blender(self, **kwargs):
        '''
        By default loads *everything* (pial, inflated, sphere) and all colormaps in the file
        
        mesh_list       which meshes to load 
        load_all_surf   loads all surfaces in the file (ignores surf_list)
        surf_list       which surfaces to load
        hemi_list       which hemispheres to load
        run_blender         
        '''
        mesh_list = kwargs.get('mesh_list', ['pial', 'inflated', 'sphere'])
        if not isinstance(mesh_list, list):
            mesh_list = [mesh_list]
        
        load_all_surf = kwargs.get('load_all_surf', False)
        surf_list = kwargs.get('surf_list', self.surf_names)
        if not isinstance(surf_list, list):
            surf_list = [surf_list]        

        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]        

        run_blender = kwargs.get('run_blender', False)
        save_blender = kwargs.get('save_blender', False)
        close_blender = kwargs.get('save_blender', False)

        # [3] Write the script used to call blender
        blender_script_str = ''
        blender_script_str += bscript_start        
        #
        blender_script_str += "mesh_dir = '" + self.output_dir + "'\n"        
        blender_script_str += "blender_filename = '" + self.blender_file_name + "'\n"
        blender_script_str += f"mesh_list = {mesh_list}\n"
        blender_script_str += f"hemi_list = {hemi_list}\n"
        blender_script_str += bscript_load_mesh
        # 
        blender_script_str += f"load_all_surf = {str(load_all_surf)}\n"
        blender_script_str += f"surf_list = {surf_list}\n"        
        blender_script_str += bscript_load_rgb
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
        if run_blender:
            os.system(f'{blender_init} --python {self.blender_script}')



# ************************************************************************************************************
# ************************************************************************************************************

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
#rgb_col_bars = []
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
    #     if 'rgb.png' in i:
    #         rgb_col_bars.append(i)
    # rgb_col_bars.sort()
    for ih in hemi_list:
        rgb_files[ih].sort()
else: # else, load specified surf..
    for hemi in hemi_list:
        for i_surf in surf_list:
            rgb_files[hemi].append(f'{hemi}.{i_surf}_rgb.csv')
        rgb_files[hemi].sort()
    # for i_surf in surf_list:
    #     rgb_col_bars.append(f'{i_surf}_rgb.png')
    
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
                color_layer.data[i].color = (r[this_vx], g[this_vx], b[this_vx], 1)
                i += 1
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

