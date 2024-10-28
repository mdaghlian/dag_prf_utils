import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
from scipy.spatial import ConvexHull
import subprocess
opj = os.path.join

from io import BytesIO
import base64
from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *
from dag_prf_utils.mesh_format import *
from dag_prf_utils.fs_tools import *
try:
    import plotly.graph_objects as go
except:
    print('No plotly')

try: 
    from dash import Dash, dcc, html, Input, Output, State
    import dash
    from flask import Flask, send_file,render_template_string, send_from_directory
except:
    print('no dash..')

path_to_utils = os.path.abspath(os.path.dirname(__file__))

class GenMeshMaker(FSMaker):
    '''Build on top of FSMaker. Extend to other methods
    sub         subject name in FS dir
    fs_dir      Freesurfer files 
    output_dir  Where to put files we make
    do_fs_setup Option to run without freesurfer
    '''
    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], output_dir=[], **kwargs):
        super().__init__(sub, fs_dir)
        # self.output_dir = os.path.abspath(output_dir)
        self.output_dir = output_dir
        if isinstance(output_dir, str):
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)                  
        
        # [1] Load mesh info
        # PULL FROM FREESURFER
        self.do_offsets = kwargs.get('do_offsets', True)
        self.mesh_info = {}        
        for mesh_name in ['inflated', 'sphere', 'pial', ]:
            self.mesh_info[mesh_name] = self._return_fs_mesh_coords_both_hemis(mesh=mesh_name)
        # [2] Load under surf values
        self.us_values = {}
        self.us_cols = {}
        for us in ['curv', 'thickness']:
            self.us_values[us] = self._return_fs_curv_vals_both_hemis(curv_name=us, return_type='concat',)            
            if us=='curv':
                vmin,vmax=-1,2
                self.us_values[us] = (self.us_values[us]<0)*1.0
            elif us=='thickness':
                vmin,vmax=0,5
            self.us_cols[us] = self._data_to_rgb(
                data=self.us_values[us], 
                vmin=vmin, vmax=vmax, cmap='Greys_r',
            )
        # ... other
        self.ply_files = {}

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region Color functions 
    """
    Functions for displaying color on the surface
    return_display_rgb      Return the rgb values for the surface
    _data_to_rgb            Simple mapping from values to colors
    _combine2maps           Combine two maps (e.g., data and undersurface)
    get_us_cols             Get the undersurface colors
    """

    def return_display_rgb(self, data=None, **kwargs):
        '''Per vertex rgb values
            data            np.ndarray      What are we plotting on the surface? 1D array, same length as the number of vertices in subject surface.
            under_surf      str             What is going underneath the data (e.g., what is the background)?
                                            default is curv. Could also be thick, (maybe smoothwm) 
        **kwargs:
            data_mask       bool array      Mask to hide certain values (e.g., where rsquared is not a good fit)
            data_alpha      np.ndarray      Alpha values for plotting. Where this is specified the undersurf is used instead
            *** COLOR
            cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                                Default: viridis
            vmin            float           Minimum value for colormap
                                                                Default: min in data
            vmax            float           Max value for colormap
                                                                Default: max in data
            return_cmap_dict bool                                                                 
        '''
        split_hemi = kwargs.get('split_hemi', False)
        unit_rgb = kwargs.get('unit_rgb', True)
        return_cmap_dict = kwargs.get('return_cmap_dict', False)
        under_surf = kwargs.get('under_surf', 'curv')
        # Load mask for data to be plotted on surface
        data_mask = kwargs.get('data_mask', np.ones(self.total_n_vx, dtype=bool))
        data_alpha = kwargs.get('data_alpha', np.ones(self.total_n_vx))
        clear_lower = kwargs.get('clear_lower', False)
        clear_upper = kwargs.get('clear_upper', False)
        data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0
        
        # ROI arguments
        roi_list = kwargs.get('roi_list', None)
        if roi_list is not None:
            if not isinstance(roi_list, list):
                roi_list = [roi_list]
        roi_fill = kwargs.get('roi_fill', False)
        roi_col = kwargs.get('roi_col', [1,0,0,1])

        if not isinstance(data, (np.ndarray, list, pd.DataFrame)):
            print(f'Just using undersurface file..')
            data = np.zeros(self.total_n_vx)
            data_alpha = np.zeros(self.total_n_vx)        
        
        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')    
        try:
            vmin = kwargs.get('vmin', np.nanmin(data[data_mask]))
            vmax = kwargs.get('vmax', np.nanmax(data[data_mask]))
        except:
            vmin = kwargs.get('vmin', np.nanmin(data))
            vmax = kwargs.get('vmax', np.nanmax(data))
        # By default use vmin as mask...
        if clear_lower:
            data_alpha[data<vmin] = 0
        if clear_upper:
            data_alpha[data>vmax] = 0
                
        # Create rgb values mapping from data to cmap
        data_col1 = self._data_to_rgb(
            data=data, cmap=cmap, vmin=vmin, vmax=vmax
        )        
        data_col2 = self.get_us_cols(under_surf)
        display_rgb = self._combine2maps(
            data_col1=data_col1, 
            data_col2=data_col2,
            data_alpha=data_alpha)

        # Add ROIs
        if roi_list is not None:
            full_roi_bool = np.zeros(self.total_n_vx, dtype=bool)
            for roi in roi_list:
                if roi_fill:
                    this_roi_bool = self._return_roi_bool_both_hemis(
                        roi_name=roi, return_type='concat', 
                    )
                else:
                    this_roi_bool = self._return_roi_bool_border_both_hemis(
                        roi_name=roi, return_type='concat', 
                    )
                full_roi_bool[this_roi_bool] += True
            display_rgb[full_roi_bool, :] = roi_col


        if not unit_rgb:
            display_rgb = (display_rgb * 255) # .astype(np.uint8)
        display_rgb_dict = {
            'lh' : display_rgb[:self.n_vx['lh'],:],
            'rh' : display_rgb[self.n_vx['lh']:,:],
        }

        if split_hemi:
            display_rgb = display_rgb_dict
        cmap_dict = {
            'cmap' : cmap, 
            'vmin' : vmin, 
            'vmax' : vmax,             
        }

        if return_cmap_dict:
            return display_rgb, cmap_dict
        else:
            return display_rgb    
    
    def _data_to_rgb(self, data, cmap='viridis', vmin=None, vmax=None):
        '''
        Simple mapping from values to colors
        '''                        
        # Create rgb values mapping from data to cmap
        try: 
            data_cmap = dag_get_cmap(cmap)  
        except:
            data_cmap = dag_cmap_from_str(cmap)
        # data_cmap = dag_get_cmap(cmap)
        data_norm = mpl.colors.Normalize()
        data_norm.vmin = vmin
        data_norm.vmax = vmax
        data_col = data_cmap(data_norm(data))
        return data_col
    
    def _combine2maps(self, data_col1, data_col2, data_alpha, **kwargs):
        display_rgb = (data_col1 * data_alpha[...,np.newaxis]) + \
            (data_col2 * (1-data_alpha[...,np.newaxis]))
        return display_rgb

    def get_us_cols(self, key, **kwargs):
        if key not in self.us_cols.keys():
            data = self.get_us_values(key)
            self.us_cols[key] = self._data_to_rgb(
                data, **kwargs
            )
        return self.us_cols[key]

    #endregion

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region MESH COORDS 

    def _return_fs_mesh_coords(self, mesh, hemi):
        '''Return dict with the vx and faces of mesh specified
        '''
        try:
            this_mesh_info = dag_read_fs_mesh(opj(self.sub_surf_dir, f'{hemi}.{mesh}'))
        except:
            this_mesh_info = dag_read_fs_mesh(opj(self.sub_surf_dir, f'{hemi}.{mesh}.T1'))

        # Put it into x,y,z, i,j,k format. Plus add offset to x
        mesh_info = {}                                    
        mesh_info = copy(this_mesh_info)
        mesh_info['x']=this_mesh_info['coords'][:,0]
        if self.do_offsets:
            print(f'Adding offset to mesh...')
            if 'sphere' in mesh:
                mesh_info['x'] += 100 if hemi=='rh' else -100
            elif 'inflated' in mesh:
                mesh_info['x'] += 50 if hemi=='rh' else -50
            elif 'pial' in mesh:
                mesh_info['x'] += 25 if hemi=='rh' else -25
        mesh_info['y']=this_mesh_info['coords'][:,1]
        mesh_info['z']=this_mesh_info['coords'][:,2]
        mesh_info['i']=this_mesh_info['faces'][:,0]
        mesh_info['j']=this_mesh_info['faces'][:,1]
        mesh_info['k']=this_mesh_info['faces'][:,2]
        # mesh_info['coords'] = this_mesh_info['coords']
        # mesh_info['faces'] = this_mesh_info['faces']


        return mesh_info
    
    def _return_fs_mesh_coords_both_hemis(self, mesh, return_type='dict'):
        mesh_info_D = {}
        mesh_info_L = []
        for hemi in ['lh', 'rh']:
            mesh_info_D[hemi] = self._return_fs_mesh_coords(mesh, hemi)
            mesh_info_L.append(mesh_info_D[hemi])
        if return_type=='dict':
            mesh_info = mesh_info_D
        elif return_type=='list':
            mesh_info = mesh_info_L
        return mesh_info
    
    def get_mesh_info(self, key):
        if key not in self.mesh_info.keys():
            self.mesh_info[key] = self._return_fs_mesh_coords_both_hemis(mesh=key)
        return self.mesh_info[key]
    #endregion MESH COORDS
    

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region CURV VALS 
    def _return_fs_curv_vals(self, curv_name, hemi='', **kwargs):
        '''Finds a curv file in fs folder. Returns the specified values

        curv_name       Name of curv file (can include hemi)
        hemi            hemisphere

        '''
        include = kwargs.pop('include', [])
        custom_only = kwargs.pop('custom_only', False)
        kwargs['return_msg'] = None # Don't return an error...
        kwargs['recursive'] = True
        if curv_name in ['curv', 'thickness']:
            curv_file = opj(self.sub_surf_dir,f'{hemi}.{curv_name}')
        elif custom_only: # Only look in custom folder
            curv_file = dag_find_file_in_folder(
                filt = [curv_name, *include, hemi],
                path=self.custom_surf_dir,
                **kwargs
            )
        else:
            curv_file = dag_find_file_in_folder(
                filt = [curv_name, *include, hemi],
                path=self.sub_surf_dir,
                **kwargs
            )
        if isinstance(curv_file, list):
            print(curv_file)
        curv_vals = dag_read_fs_curv_file(curv_file)
        return curv_vals
    
    def _return_fs_curv_vals_both_hemis(self, curv_name, return_type='dict', **kwargs):
        curv_val_D = {}
        curv_val_L = []
        for hemi in ['lh', 'rh']:
            curv_val_D[hemi] = self._return_fs_curv_vals(curv_name, hemi, **kwargs)
            curv_val_L.append(curv_val_D[hemi])
        if return_type=='dict':
            curv_val = curv_val_D
        elif return_type=='list':
            curv_val = curv_val_L
        elif return_type=='concat':
            curv_val = np.concatenate(curv_val_L)
        return curv_val
    
    def get_us_values(self, key):
        if key not in self.us_values.keys():
            self.us_values[key] = self._return_fs_curv_vals_both_hemis(
                curv_name=key, return_type='concat')
        return self.us_values[key]
    
    def get_us_values_per_hemi(self, hemi, key):
        us_vals = self.get_us_values(key).copy()
        if hemi=='lh':
            us_vals = us_vals[:self.n_vx['lh']]
        elif hemi=='rh':
            us_vals = us_vals[self.n_vx['lh']:]
        return us_vals  

    #endregion CURV VALS
    
    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region ROI FUNCTIONS
    def _return_roi_bool(self, roi_name, hemi='', **kwargs):
        roi_bool = dag_load_roi(self.sub, roi_name, fs_dir=self.fs_dir, split_LR=True)[hemi]
        return roi_bool
    def _return_roi_bool_both_hemis(self, roi_name, return_type='dict', **kwargs):
        roi_bool_D = {}
        roi_bool_L = []
        for hemi in ['lh', 'rh']:
            roi_bool_D[hemi] = self._return_roi_bool(roi_name, hemi, **kwargs)
            roi_bool_L.append(roi_bool_D[hemi])
        if return_type=='dict':
            roi_bool = roi_bool_D
        elif return_type=='list':
            roi_bool = roi_bool_L
        elif return_type=='concat':
            roi_bool = np.concatenate(roi_bool_L)
        return roi_bool
    
    # -> borders.
    def _return_roi_bool_border(self, roi_name, hemi, **kwargs):
        roi_bool = self._return_roi_bool(roi_name, hemi, **kwargs)
        roi_bool_border = dag_find_border_vx(roi_bool, self.mesh_info['inflated'][hemi], return_type='bool')
        return roi_bool_border    

    def _return_roi_bool_border_both_hemis(self, roi_name, return_type='dict', **kwargs):
        roi_bool_D = {}
        roi_bool_L = []
        for hemi in ['lh', 'rh']:
            roi_bool_D[hemi] = self._return_roi_bool_border(roi_name, hemi, **kwargs)
            roi_bool_L.append(roi_bool_D[hemi])
        if return_type=='dict':
            roi_bool = roi_bool_D
        elif return_type=='list':
            roi_bool = roi_bool_L
        elif return_type=='concat':
            roi_bool = np.concatenate(roi_bool_L)
        return roi_bool
    
    # -> borders in order
    def _return_roi_borders_in_order(self, roi_list, **kwargs):
        '''Return the border vertices in order for each ROI in roi_list
        TODO: make generic for boolean
        '''
        if not isinstance(roi_list, list):
            roi_list = [roi_list]
        mesh_name = kwargs.get('mesh_name', 'inflated')
        combine_matches = kwargs.get('combine_matches', False)
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        roi_obj = []
        roi_count = -1
        for i_roi,roi in enumerate(roi_list):                
            roi_bool = dag_load_roi(self.sub, roi, fs_dir=self.fs_dir, split_LR=True, combine_matches=combine_matches, recursive_search=True)
            if 'lh' not in roi_bool.keys():
                # We found extra matches!!!
                print(f'Found extra matches for {roi}')
                extra_roi_list = list(roi_bool.keys())
            else:
                extra_roi_list = [roi]
                roi_bool = {roi : roi_bool}
            for i_roi_extra, roi_extra in enumerate(extra_roi_list):
                roi_count += 1
                for ih,hemi in enumerate(hemi_list):
                    if roi_bool[roi_extra][hemi].sum()==0:
                        continue
                    border_vx_list, border_coords_list = dag_find_border_vx_in_order(
                        roi_bool=roi_bool[roi_extra][hemi], 
                        mesh_info=self.mesh_info[mesh_name][hemi], 
                        return_coords=True,
                        )

                    for ibvx,border_vx in enumerate(border_vx_list):
                        first_instance = (ih==0) & (ibvx==0) # Only show legend for first instance
                        this_roi_dict = {
                            'hemi' : hemi,
                            'roi' : roi_extra,
                            'border_vx' : border_vx,
                            'border_coords' : border_coords_list[ibvx],
                            'first_instance' : first_instance,
                        }
                        roi_obj.append(this_roi_dict)          
        return roi_obj
    
    #endregion ROI FUNCTIONS

    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region PLY FUNCTIONS
    def add_ply_surface(self, data=None, surf_name=None, **kwargs):
        '''
        Arguments:
        Specify surf_name to save
        '''
        overwrite = kwargs.get('ow', True)
        mesh_name = kwargs.get('mesh_name', 'inflated')
        incl_rgb = kwargs.get('incl_rgb', True)
        if data is None:
            incl_rgb = False
        print(f'File to be named: {surf_name}')        
        if (os.path.exists(opj(self.output_dir, f'lh.{surf_name}'))) & (not overwrite) :
            print(f'{surf_name} already exists for {self.sub}, not overwriting surf files...')
            return

        if (os.path.exists(opj(self.output_dir, f'lh.{surf_name}'))): 
            print(f'Overwriting: {surf_name} for {self.sub}')
        else:
            print(f'Writing: {surf_name} for {self.sub}')

        # Load mask for data to be plotted on surface
        if incl_rgb: 
            display_rgb = self.return_display_rgb(data=data, split_hemi=True, unit_rgb=False, **kwargs)        
        else:
            display_rgb = {'lh':None, 'rh':None}

        # Save the mesh files first as .asc, then .srf, then .obj
        # Then save them as .ply files, with the display rgb data for each voxel
        ply_file_2open = []
        for hemi in ['lh', 'rh']:
            ply_str = dag_ply_write(
                mesh_info   = self.mesh_info[mesh_name][hemi], 
                diplay_rgb  = display_rgb[hemi], 
                hemi        = hemi, 
                values      = data, 
                incl_rgb    = incl_rgb,
                )
            ply_surf_file = opj(self.output_dir,f'{hemi}.{surf_name}.ply')
            with open(ply_surf_file, 'w') as file:
                file.write(ply_str)   
            ply_file_2open.append(ply_surf_file)
        # Return list of .ply files to open...
        self.ply_files[surf_name] = [
            ply_file_2open
        ]
    def open_ply_mlab(self):
        # mlab_cmd = 'meshlab '
        mlab_cmd = '/data1/projects/dumoulinlab/Lab_members/Marcus/programs/MeshLab2022.02-linux/AppRun '
        for key in self.ply_files.keys():            
            mlab_cmd += f' lh.{key}.ply'
            mlab_cmd += f' rh.{key}.ply'
        # os.chdir(self.output_dir)
        # os.system(mlab_cmd)
        subprocess.run(mlab_cmd, shell=True, cwd=self.output_dir)
    
    #endregion PLY FUNCTIONS