import csv
import os 
opj = os.path.join
import bpy
import bmesh
import numpy as np
from numpy import genfromtxt


# start in object mode
# Clear all objects...
objects = [o for o in bpy.data.objects]
for obj in objects:
    bpy.data.objects.remove(obj)

meshes = [o for o in bpy.data.meshes]
for m in meshes:
    bpy.data.meshes.remove(m)
mats = [o for o in bpy.data.materials]
for m in mats:
    bpy.data.materials.remove(m)    

## ***
file_list = os.listdir(mesh_dir)

obj_files = {'lh':[], 'rh':[]}
ply_files = {'lh':[], 'rh':[]}
rgb_files = {'lh':[], 'rh':[]}

for i in file_list:
    if 'lh' in i:
        hemi = 'lh'
    elif 'rh' in i:
        hemi = 'rh'
    
    if '.obj' in i:
        obj_files[hemi] = i
    elif '.ply' in i:
        ply_files[hemi] = i

    elif 'rgb.csv' in i:
        rgb_files[hemi].append(i)

for ih,hemi in enumerate(['lh', 'rh']):
    # Load mesh
    # bpy.ops.import_scene.obj(filepath=opj(mesh_dir,obj_files[hemi]))
    bpy.ops.import_mesh.ply(filepath=opj(mesh_dir,ply_files[hemi]))    
    # Name mesh as hemi
    bpy.data.objects[ih].name = hemi
    bpy.data.meshes[ih].name = hemi
    # Loop through and add color layers
    for i1, i_col in enumerate(rgb_files[hemi]):
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

# ADD ANIMATION
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