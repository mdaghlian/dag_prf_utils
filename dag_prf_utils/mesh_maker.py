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
try: 
    import cortex
    import cortex.freesurfer
    from cortex.formats import read_gii
    have_cortex=True
except:
    have_cortex=False
    print('No cortex')
have_cortex = False

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
        if not isinstance(data, (np.ndarray, list, pd.DataFrame, pd.Series)):
            print(f'Just using undersurface file..')
            data = np.zeros(self.total_n_vx)
            kwargs['data_alpha'] = np.zeros(self.total_n_vx)  
        
        split_hemi = kwargs.get('split_hemi', False)
        unit_rgb = kwargs.get('unit_rgb', True)
        return_cmap_dict = kwargs.get('return_cmap_dict', False)
        under_surf = kwargs.get('under_surf', 'curv')
        # Load mask for data to be plotted on surface
        data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
        data_sub_mask = kwargs.get('data_sub_mask', None)        
        data_alpha = kwargs.get('data_alpha', np.ones_like(data))
        clear_lower = kwargs.get('clear_lower', False)
        clear_upper = kwargs.get('clear_upper', False)
        
        if data_sub_mask is not None:
            d_full = np.zeros(self.total_n_vx)
            d_full[data_sub_mask] = data
            data = d_full
            #
            dm_full = np.zeros(self.total_n_vx, dtype=bool)
            dm_full[data_sub_mask] = data_mask
            data_mask = dm_full
            #
            da_full = np.zeros(self.total_n_vx)
            da_full[data_sub_mask] = data_alpha
            data_alpha = da_full

            

        data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0
        
        # ROI arguments
        roi_list = kwargs.get('roi_list', None)
        if roi_list is not None:
            if not isinstance(roi_list, list):
                roi_list = [roi_list]
        roi_fill = kwargs.get('roi_fill', False)
        roi_col = kwargs.get('roi_col', [1,0,0,1])
        
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
        if (key not in self.mesh_info.keys()) or (self.mesh_info[key] is {}):
            print('bloop')
            self.mesh_info[key] = self._return_fs_mesh_coords_both_hemis(mesh=key)
        return self.mesh_info[key]
    
    def idx2bool(self,idx):
        b_out = np.zeros(self.total_n_vx, dtype=bool)
        b_out[idx] = True
        return b_out
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
    #region FLAT FUNCTIONS        
    def add_flat_to_mesh_info(self, **kwargs):
        '''
        Add the flatmap to the mesh_info
        '''
        flat_name = kwargs.get('flat_name', 'flat')
        flat_pts_hemi = kwargs.get('flat_pts', None)
        flat_polys_hemi = kwargs.get('flat_polys', None)        
        self.mesh_info[flat_name] = {}
        for hemi in ['lh','rh']:            
            flat_pts = flat_pts_hemi[hemi]
            flat_polys = flat_polys_hemi[hemi]
            self.mesh_info[flat_name][hemi] = {}
            self.mesh_info[flat_name][hemi]['coords'] = flat_pts
            self.mesh_info[flat_name][hemi]['faces'] = flat_polys
            self.mesh_info[flat_name][hemi]['x'] = flat_pts[:,0]
            self.mesh_info[flat_name][hemi]['y'] = flat_pts[:,1]
            self.mesh_info[flat_name][hemi]['z'] = flat_pts[:,2]
            self.mesh_info[flat_name][hemi]['i'] = flat_polys[:,0]
            self.mesh_info[flat_name][hemi]['j'] = flat_polys[:,1]
            self.mesh_info[flat_name][hemi]['k'] = flat_polys[:,2]    

    def make_flat_map(self, centre_bool=None, **kwargs):
        '''
        Pycotex uses flatmaps for a bunch of things
        But if you can't be bothered to do it properly, and just want
        to display freesurfer ROIs in pycortex you can do this

        Custom method to flatten (not using mris_flatten)
        * option 1: use latitude and longitude
        * option 2: do some clever UV mapping with igl code...

        TODO: remove cut from Y 
        '''

        if not os.path.exists(self.custom_surf_dir):
            os.makedirs(self.custom_surf_dir)
        method = kwargs.pop('method', 'latlon')
        morph = kwargs.pop('morph', 0) # How much to dilate or erode the mask (if doing igl)
        hemi_project = kwargs.get('hemi_project', 'sphere')
        flat_name = kwargs.get('flat_name', 'flat')
        centre_roi = kwargs.get('centre_roi', None)
        if centre_bool is None:
            centre_bool = np.ones_like(self.total_n_vx, dtype=bool)
        centre_bool_hemi = {
            'lh': centre_bool[:self.n_vx['lh']],
            'rh': centre_bool[self.n_vx['lh']:]
        }
        vx_to_include = kwargs.pop('vx_to_include', centre_bool)        
        vx_to_include = {
            'lh': vx_to_include[:self.n_vx['lh']],
            'rh': vx_to_include[self.n_vx['lh']:]
        }

        cut_box = kwargs.get('cut_box', False)                        
        
        hemi_pts = {}
        hemi_polys = {}
        pts_combined = []
        polys_combined = []
        # Where to put flatmatp in z plane..
        new_z = np.mean(np.hstack(
            [self.mesh_info['inflated']['lh']['z'],self.mesh_info['inflated']['rh']['z']]
            ))
        infl_x = np.hstack(
            [self.mesh_info['inflated']['lh']['x'],self.mesh_info['inflated']['rh']['x']]
            )

        for hemi in ['lh','rh']:
            hemi_kwargs = kwargs.copy()
            hemi_kwargs['z'] = new_z
            # hemi_kwargs['morph'] = morph
            if centre_roi is not None:
                # Load the ROI bool for this hemisphere
                centre_bool_hemi[hemi] |= self._return_roi_bool_both_hemis(centre_roi, **kwargs)[hemi]
                print(centre_bool)
            # Cut a box around them?            
            if cut_box:
                hemi_kwargs['vx_to_include'] = dag_cut_box(
                    mesh_info=self.mesh_info['inflated'][hemi],
                    vx_bool=centre_bool_hemi[hemi],
                )
            else:
                hemi_kwargs['vx_to_include'] = vx_to_include[hemi]
            hemi_kwargs['vx_to_include'] = dag_mesh_morph(
                mesh_info=self.mesh_info['inflated'][hemi], 
                vx_bool=hemi_kwargs['vx_to_include'], 
                morph=morph)
            hemi_kwargs['centre_bool'] = centre_bool_hemi[hemi]
            pts,polys,_ = dag_flatten(
                mesh_info=self.mesh_info[hemi_project][hemi], 
                method=method,
                **hemi_kwargs)        
            flat = pts
            # do some cleaning...

            # Demean everything
            # Disconnected points 
            connected_pts = np.zeros(len(pts), dtype=bool)
            connected_pts[np.unique(polys)] = True
            flat[connected_pts] -= flat[connected_pts].mean(axis=0)
            scale_x = (infl_x.max() - infl_x.min()) / (flat[:,0].max() - flat[:,0].min())
            flat *= scale_x*3 # Meh seems nice enough
            # if hemi == 'rh':
            #     # Flip x and y,
            #     # pts[:,0] = -pts[:,0]
            #     pts[:,1] = -pts[:,1]                 
            if hemi == 'lh':
                max_x_lh = flat[:,0].max()
            else:
                flat[:,0] += max_x_lh - flat[:,0].min() + .1 * (max_x_lh - flat[:,0].min())

            hemi_pts[hemi] = flat.copy()
            hemi_polys[hemi] = polys.copy()
            pts_combined.append(flat)
            if hemi == 'rh':
                polys += len(hemi_pts['lh'])
            polys_combined.append(polys)

        pts_combined = np.vstack(pts_combined)
        polys_combined = np.vstack(polys_combined)
        self.add_flat_to_mesh_info(flat_name=flat_name, flat_pts=hemi_pts, flat_polys=hemi_polys)


    def flat_mpl(self, **kwargs):
        '''Plot using matplotlib 
        '''
        data=kwargs.pop('data', None)
        surf_name = kwargs.pop('surf_name', 'data')
        rot_angles = kwargs.pop('rot_angles', None)
        roi_list = kwargs.pop('roi_list', [])
        if not isinstance(roi_list, list):
            roi_list = [roi_list]
        flat_name = kwargs.pop('flat_name', 'flat')
        try:
            self.get_mesh_info(flat_name)
        except:
            self.add_flat_to_mesh_info(flat_name=flat_name)
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
        else:
            fig = ax.get_figure()

        disp_rgb, cmap_dict = self.return_display_rgb(
            data=data, split_hemi=True, return_cmap_dict=True, **kwargs
        )
        self.get_mesh_info(flat_name)
        mpts = {
            'lh': self.mesh_info[flat_name]['lh']['coords'],
            'rh': self.mesh_info[flat_name]['rh']['coords'],
        }
        mpolys = {
            'lh': self.mesh_info[flat_name]['lh']['faces'],
            'rh': self.mesh_info[flat_name]['rh']['faces'],
        }


        for hemi in hemi_list: 
            if rot_angles is not None:
                mpts[hemi] = dag_coord_rot(mpts[hemi], rot_angles)         
            if np.isnan(mpts[hemi][0][0]):
                continue
            triang = mpl.tri.Triangulation(
                mpts[hemi][:,0],
                mpts[hemi][:,1],
                triangles=mpolys[hemi],
            )
            # Plot the triangulated data using tripcolor
            cmap = mpl.colors.ListedColormap(disp_rgb[hemi])
            c = np.arange(len(disp_rgb[hemi]))
            ax.tripcolor(
                triang,
                c,
                cmap=cmap,
                shading='gouraud',  # Smooth interpolation between vertices
            )
        for roi in roi_list:
            roi_obj = self._return_roi_borders_in_order(roi)
            for roi_dict in roi_obj:
                hemi = roi_dict['hemi']
                if hemi not in hemi_list:
                    continue
                x = mpts[hemi][roi_dict['border_vx'],0]
                y = mpts[hemi][roi_dict['border_vx'],1]
                ax.plot(
                    x,y, 
                    # roi_dict['border_coords'][:,0],
                    # roi_dict['border_coords'][:,1],
                    '.',
                    color='k',
                    linewidth=2, markersize=.8,
                    label=roi_dict['roi'] if roi_dict['first_instance'] else None,
                )
        # Add color bar
        norm = mpl.colors.Normalize(vmin=cmap_dict['vmin'], vmax=cmap_dict['vmax'])
        cmap = dag_get_cmap(cmap_dict['cmap'])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='horizontal', label=surf_name)
        ax.axis('off')
        ax.set_aspect('equal')
        return cmap_dict
    
    def reload_flat(self, flat_name):
        '''Reload the flatmap
        '''
        self.mesh_info[flat_name] = {}
        self.add_flat_to_mesh_info(flat_name=flat_name)

    #endregion FLAT FUNCTIONS







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

