import numpy as np  
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import copy
import pickle
import os
opj = os.path.join

from dag_prf_utils.mesh_maker import *
from dag_prf_utils.mesh_format import *

# ************************** SPECIFY BLENDER PATH HERE **************************
# Check for command:
blender_cmd = subprocess.getstatusoutput(f"command -v blender")[1]
if blender_cmd == '':
    print('could not find blender command, specify it in the blender_tools.py file')
    blender_cmd = 'blender ' # specify path to blender here

class BlendMaker(GenMeshMaker):
    '''Used to make a blender file for a single subject
    
    '''
    def __init__(self, sub, fs_dir, output_dir, **kwargs):
        super().__init__(sub, fs_dir, output_dir)
        time_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.blender_file_name = opj(self.output_dir, f'{self.sub}_{time_now}.blend')
        self.blend_vx_col = {}        
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
        # [1] Save the meshes
        mesh_dict = copy.deepcopy(self.mesh_info)
        # Save as a pickle
        with open(opj(self.output_dir, 'mesh_info.pkl'), 'wb') as f:
            pickle.dump(mesh_dict, f)
        
        # [2] Make rgb files for curvature 
        for us in ['curv']:
            self.blend_vx_col[us] = {}
            self.blend_vx_col[us]['lh'] = self.us_cols[us][:self.n_vx['lh'],:]
            self.blend_vx_col[us]['rh'] = self.us_cols[us][self.n_vx['lh']:,:]
        

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
        plt.close(fig)
        # Save the rgb values
        self.blend_vx_col[surf_name] = copy.deepcopy(display_rgb)



    def launch_blender(self, **kwargs):
        '''
        By default loads *everything* (pial, inflated, sphere) and all colormaps in the file
        
        mesh_list       which meshes to load 
        load_all_surf   loads all surfaces in the file (ignores surf_list)
        surf_list       which surfaces to load
        hemi_list       which hemispheres to load
        run_blender         
        '''
        # Save the colour data
        with open(opj(self.output_dir, 'blend_vx_col.pkl'), 'wb') as f:
            pickle.dump(self.blend_vx_col, f)

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
        blender_script_str += "mesh_dir = '.' \n"   
        blender_script_str += "blender_filename = '" + self.blender_file_name + "'\n"
        blender_script_str += bscript_load_mesh
        # 
        blender_script_str += bscript_load_rgb
        if os.path.exists(opj(self.output_dir, 'movie.npz')):
            blender_script_str += bscript_movie
        if os.path.exists(opj(self.output_dir, 'xy.npy')):
            blender_script_str += bscript_uv_map
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
import pickle
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


'''

bscript_load_mesh = '''
# Load the pickle file with the mesh info
import pickle
with open(opj(mesh_dir, 'mesh_info.pkl'), 'rb') as f:
    mesh_info = pickle.load(f)
# Load the pickle file with the color info
with open(opj(mesh_dir, 'blend_vx_col.pkl'), 'rb') as f:
    blend_vx_col = pickle.load(f)

# mesh_list = list(mesh_info.keys())
mesh_list = ['pial', 'inflated',] # 'sphere']

hemi_list = list(mesh_info[mesh_list[0]].keys())
# hemi_list = ['lh'] 

n_vx = {}
for i_hemi in hemi_list:
    n_vx[i_hemi] = len(mesh_info[mesh_list[0]][i_hemi]['coords'])

### Load meshes & slider
for i_hemi in hemi_list:
    for i_mesh in mesh_list:
        # Create a new mesh and object
        mesh_data = bpy.data.meshes.new(f"{i_hemi}.{i_mesh}")
        mesh_data.from_pydata(
            mesh_info[i_mesh][i_hemi]['coords'], 
            [], 
            mesh_info[i_mesh][i_hemi]['faces']
        )
        mesh_data.update()

        mesh_object = bpy.data.objects.new(f"{i_hemi}.{i_mesh}", mesh_data)
        bpy.context.collection.objects.link(mesh_object)        
    
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
# Now add the rgb colors to the meshes
for ih,hemi in enumerate(hemi_list):
    # bpy.data.objects[ih].name = hemi
    # bpy.data.meshes[ih].name = hemi
    # Loop through and add color layers
    for i1, i_col in enumerate(blend_vx_col.keys()):
        print('loading')
        print(i_col)
        # [1] Load in rgb color data for this map
        rgb_data = blend_vx_col[i_col][hemi].copy()
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

bscript_movie = '''
### ADD ANIMATION 
# Load colours
full_movie_cols = np.load(opj(mesh_dir, 'movie.npz'))['movie']
num_timepoints = full_movie_cols.shape[1]
fps = 2 # frames per second
colors = full_movie_cols.copy()
movie_cols = {}
for i_hemi in hemi_list:
    if i_hemi == 'lh':
        movie_cols[i_hemi] = full_movie_cols[:n_vx['lh'],:,:]
    else:
        movie_cols[i_hemi] = full_movie_cols[n_vx['lh']:,:,:]

# Now add the movie rgb colors to the meshes
for ih,hemi in enumerate(hemi_list):
    # bpy.data.objects[ih].name = hemi
    # bpy.data.meshes[ih].name = hemi
    # Loop through and add color layers
    # How many layers are there?
    n_layers = len(bpy.data.meshes[hemi].vertex_colors)
    for i1 in range(num_timepoints):        
        rgb_data = movie_cols[hemi][:,i1,:].copy()
        if rgb_data.max()>1:
            rgb_data = rgb_data/255
        r,g,b = rgb_data[:,0]*1, rgb_data[:,1], rgb_data[:,2]

        # Add new color layer
        color_layer = bpy.data.meshes[hemi].vertex_colors.new(name=f'movie_{i1:03}')
        
        # layer_names.append(f'movie_{i1:03}')
        i = 0
        for poly in bpy.data.meshes[hemi].polygons:
            for idx in poly.loop_indices:
                this_vx = bpy.data.meshes[hemi].loops[idx].vertex_index
                color_layer.data[i].color = (r[this_vx], g[this_vx], b[this_vx], 1)
                i += 1
        
def set_vcols(frame):
    layer_to_plot = frame 
    if layer_to_plot > num_timepoints:
        layer_to_plot = num_timepoints-1
    for i_hemi in hemi_list:
        bpy.data.meshes[i_hemi].attributes.active_color_index = layer_to_plot + n_layers
def my_handler(scene):
    frame = scene.frame_current
    set_vcols(frame)
bpy.app.handlers.frame_change_pre.append(my_handler)
'''

bscript_uv_map = '''
### ADD UV MAP
full_uv_coords = np.load(opj(mesh_dir, 'face_xy.npy'))
uv_coords = {}
for i_hemi in hemi_list:
    if i_hemi == 'lh':
        uv_coords[i_hemi] = full_uv_coords[:n_vx['lh'],:]
    else:
        uv_coords[i_hemi] = full_uv_coords[n_vx['lh']:,:]

# Now add the uv coords to the meshes
for ih,hemi in enumerate(hemi_list):
    mesh = bpy.data.meshes[hemi]
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if not bm.loops.layers.uv:
        bm.loops.layers.uv.new(name='UVMap')
    uv_layer = bm.loops.layers.uv['UVMap']
    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].uv = (uv_coords[hemi][loop.vert.index,0], uv_coords[hemi][loop.vert.index,1])
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


