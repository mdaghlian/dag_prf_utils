import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
import copy
from scipy.spatial import ConvexHull
import subprocess
opj = os.path.join

from io import BytesIO
import base64
from dag_prf_utils.utils import *
from dag_prf_utils.mesh_format import *
from dag_prf_utils.mesh_maker import *

import io
import base64
import matplotlib.image as mpimg
from PIL import Image

path_to_utils = os.path.abspath(opj(os.path.dirname(__file__), os.pardir))

import pickle


class MeshJS(GenMeshMaker):
    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], output_dir=[], **kwargs):
        '''
        Make a super cool interactive viewer for surface        
                
        '''
        super().__init__(sub, fs_dir, output_dir, **kwargs)
        # Load the html file and the js file
        self.html_file = opj(path_to_utils, 'test_plotlyjs', 'index.html')
        self.html_str_template = open(self.html_file, 'r').read()        
        self.html_str = open(self.html_file, 'r').read()
        self.js_file = opj(path_to_utils, 'test_plotlyjs', 'script.js')
        self.js_str_template = open(self.js_file, 'r').read()
        self.js_str = open(self.js_file, 'r').read()
        self.vx_col = {}
    
    def reset_html(self):
        self.html_str = copy(self.html_str_template)
        self.js_str = copy(self.js_str_template)


    def add_mesh_to_js(self, hemi):
        # [1] Starting mesh
        this_mesh = copy(self.mesh_info['inflated'][hemi])
        
        for key in ['x', 'y', 'z', 'i', 'j', 'k']:
            this_str = '[' + ','.join([str(x) for x in this_mesh[key]]) + ']'
            self.js_str = self.js_str.replace(f'swap{key}', this_str)
        # # For inflation 
        # for mesh in ['pial', 'inflated', 'sphere']:
        #     for key in ['x', 'y', 'z', 'i', 'j', 'k']:
        #         this_str = '[' + ', '.join([str(x) for x in this_mesh[key]]) + ']'
        #         self.js_str = self.js_str.replace(f'here{key}', this_str)
            
    def add_vx_col(self, vx_col_name, data, **kwargs):
        '''Add a surface
        Properties of the surface:
        - data: array (len vertices). Used to make the colormap
        - data_mask: array (len vertices). Used to mask based on threshold
        - cmap_name : string, name of colormap
        - vmin : float vmin for cmap
        - vmax : float vmax for cmap        
        - rgb_direct # ignore everything just put in the RGB values
        '''                
        disp_rgb = self.return_display_rgb(data, split_hemi=True, **kwargs)
        hemi = 'lh'
        rgb_str = f'"{vx_col_name}": [ '
        for i in range(len(disp_rgb[hemi])):
            rgb_str += f'[{disp_rgb[hemi][i][0]}, {disp_rgb[hemi][i][1]}, {disp_rgb[hemi][i][2]}],'
        rgb_str = rgb_str[:-1] + '],'
        print(rgb_str[:200])
        self.js_str = self.js_str.replace('swapcol', rgb_str)   

    def write_output(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Write the html file
        html_file = opj(self.output_dir, 'main.html')
        with open(html_file, 'w') as f:
            f.write(self.html_str)
        
        # replace colorscheme token
        self.js_str = self.js_str.replace('swapcol', '')
        with open(opj(self.output_dir, 'script.js'), 'w') as f:
            f.write(self.js_str)
        print(f'HTML file written to {html_file}')


