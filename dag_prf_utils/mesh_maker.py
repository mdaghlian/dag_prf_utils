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
                os.mkdir(self.output_dir)                  
        
        # [1] Load mesh info
        # PULL FROM FREESURFER
        self.mesh_info = {}        
        for mesh_name in ['inflated', 'sphere', 'pial']:
            self.mesh_info[mesh_name] = self._return_fs_mesh_coords_both_hemis(mesh=mesh_name)
        # [2] Load under surf values
        self.us_values = {}
        self.us_cols = {}
        for us in ['curv', 'thickness']:
            self.us_values[us] = self._return_fs_curv_vals_both_hemis(curv_name=us, return_type='concat',)            
            if us=='curv':
                vmin,vmax=-1,1
            elif us=='thickness':
                vmin,vmax=0,5
            self.us_cols[us] = self._data_to_rgb(
                data=self.us_values[us], 
                vmin=vmin, vmax=vmax, cmap='Greys_r',
            )
        # ... other
        self.ply_files = {}
    # *****************************************************************
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
        vmin = kwargs.get('vmin', np.nanmin(data[data_mask]))
        vmax = kwargs.get('vmax', np.nanmax(data[data_mask]))
        # By default use vmin as mask...
        data_alpha[data<vmin] = 0
                
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
            display_rgb = (display_rgb * 255).astype(np.uint8)
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

    # *****************************************************************
    # COLOR FUNCTIONS
    def _data_to_rgb(self, data, cmap='viridis', vmin=None, vmax=None):
        '''
        Simple mapping from values to colors
        '''                        
        # Create rgb values mapping from data to cmap
        data_cmap = dag_get_cmap(cmap)
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



    # *****************************************************************
    # MESH COORDS
    def _return_fs_mesh_coords(self, mesh, hemi):
        '''Return dict with the vx and faces of mesh specified
        '''
        this_mesh_info = dag_read_fs_mesh(opj(self.sub_surf_dir, f'{hemi}.{mesh}'))
        # Put it into x,y,z, i,j,k format. Plus add offset to x
        mesh_info = {}                                    
        mesh_info['x']=this_mesh_info['coords'][:,0]
        mesh_info['x'] += 50 if hemi=='rh' else -50
        mesh_info['y']=this_mesh_info['coords'][:,1]
        mesh_info['z']=this_mesh_info['coords'][:,2]
        mesh_info['i']=this_mesh_info['faces'][:,0]
        mesh_info['j']=this_mesh_info['faces'][:,1]
        mesh_info['k']=this_mesh_info['faces'][:,2]
        mesh_info['coords'] = this_mesh_info['coords']
        mesh_info['faces'] = this_mesh_info['faces']

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
    
    # *****************************************************************
    # CURV VALS
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

    # *****************************************************************
    # ROI BOOL
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
    
    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    # .ply code: make .ply files. Can be used with meshlab or blender
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


    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    # plotly code        
    def plotly_return_mesh_dict(self, data, **kwargs):
        '''Return a dict with mesh info [x,y,z,i,j,k,intensity,vertexcolor]
        '''
        do_intensity = kwargs.get('do_intensity', False)
        do_vertexcolor = kwargs.get('do_vertexcolor', True)
        mesh_name = kwargs.get('mesh_name', 'inflated')
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        

        disp_rgb = self.return_display_rgb(
            data=data, split_hemi=True, **kwargs
        )
        if data is None:
            data = np.zeros(self.total_n_vx)


        data_mask = kwargs.get('data_mask', np.ones(self.total_n_vx, dtype=bool))
        data_masked = data.copy()
        data_masked[~data_mask] = np.nan
        data_4_dict = {
            'lh' : data_masked[:self.n_vx['lh']],
            'rh' : data_masked[self.n_vx['lh']:],
        }
        
        # Save the mesh files first as .asc, then .srf, then .obj
        # Then save them as .ply files, with the display rgb data for each voxel
        mesh_dict = {}
        for hemi in hemi_list:
            mesh_dict[hemi] = {}
            
            mesh_dict[hemi]['x']=self.mesh_info[mesh_name][hemi]['x'].copy()
            mesh_dict[hemi]['y']=self.mesh_info[mesh_name][hemi]['y'].copy()
            mesh_dict[hemi]['z']=self.mesh_info[mesh_name][hemi]['z'].copy()
            mesh_dict[hemi]['i']=self.mesh_info[mesh_name][hemi]['i'].copy()
            mesh_dict[hemi]['j']=self.mesh_info[mesh_name][hemi]['j'].copy()
            mesh_dict[hemi]['k']=self.mesh_info[mesh_name][hemi]['k'].copy()            
            if do_vertexcolor:
                mesh_dict[hemi]['vertexcolor']=disp_rgb[hemi]
            if do_intensity:
                mesh_dict[hemi]['intensity'] = data_4_dict[hemi]
            mesh_dict[hemi] = dag_mesh_slice(mesh_dict[hemi], **kwargs)
        return mesh_dict

    def add_plotly_surface(self, data=None, **kwargs):
        '''
        Create a plotly surface plot 
        Arguments:
            See add_ply_surface, same principle...

        '''        
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        return_mesh_obj = kwargs.get('return_mesh_obj', False)
        roi_list = kwargs.pop('ply_roi_list', [])        

        mesh_dict = self.plotly_return_mesh_dict(data, **kwargs)
        mesh3d_obj = []        
        for hemi in hemi_list:
            this_mesh3d = go.Mesh3d(
                **mesh_dict[hemi],
                name=hemi,
                # showscale=True
                )
            mesh3d_obj.append(this_mesh3d)
        if len(roi_list)>0:
            roi_obj = self.plotly_return_roi_obj(roi_list=roi_list, **kwargs)
            mesh3d_obj += roi_obj

        if return_mesh_obj:
            return mesh3d_obj
        ply_axis_dict = dict(
            showgrid=False, 
            showticklabels=False, 
            title='',
            showbackground=False,
        )
        fig = go.Figure(
            data=mesh3d_obj,
            layout=go.Layout(
                scene=dict(
                    xaxis=ply_axis_dict,
                    yaxis=ply_axis_dict,
                    zaxis=ply_axis_dict,
                    # bgcolor='rgba(0,0,0,0)'  # Set background color to transparent
                    
                )
            ),            
        )
        return fig
    

    def plotly_return_roi_obj(self, roi_list, **kwargs):
        '''
        Return a plotly object for a given roi
        '''
        if not isinstance(roi_list, list):
            roi_list = [roi_list]
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        mesh_name = kwargs.get('mesh_name', 'inflated')
        # marker_kwargs = kwargs.get()
        roi_cols = dag_get_col_vals(
            np.arange(len(roi_list)),
            vmin = -1, vmax=7, cmap='jet'
        )
        roi_obj = []
        for hemi in hemi_list:
            for i_roi,roi in enumerate(roi_list):
                # Load roi index:
                roi_bool = dag_load_roi(self.sub, roi, fs_dir=self.fs_dir, split_LR=True)[hemi]

                border_vx_list = dag_find_border_vx_in_order(
                    roi_bool=roi_bool, 
                    mesh_info=self.mesh_info[mesh_name][hemi], 
                    return_coords=False,
                    )
                # If more than one closed path (e.g., v2v and v2d)...
                for border_vx in border_vx_list:
                    # Create a the line object for the border
                    border_line = go.Scatter3d(
                        x=self.mesh_info[mesh_name][hemi]['x'][border_vx],
                        y=self.mesh_info[mesh_name][hemi]['y'][border_vx],
                        z=self.mesh_info[mesh_name][hemi]['z'][border_vx],
                        mode='lines',
                        name=roi,
                        marker=dict(
                            size=10,
                            color=roi_cols[i_roi],
                        ),
                        line=dict(
                            color=roi_cols[i_roi],
                            width=10, 
                        ),
                        opacity=1,
                    )
                    roi_obj.append(border_line)
        return roi_obj


    # WEB SECTION: usef for easy switching b/w surface overlays in html
    def web_get_ready(self, **kwargs):
        '''
        Prep everything
        > self.web_hemi_list    which hemis being plot
        > self.web_mesh         used for plotting on 
        > self.web_vxcol        store the vx colors
        > self.web_vx_col_list  list of overlay names 
        '''
        # how many hemis? 
        self.web_hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(self.web_hemi_list, list):
            self.web_hemi_list = [self.web_hemi_list]
        kwargs['hemi_list'] = self.web_hemi_list

        self.web_mesh = self.add_plotly_surface(
            return_mesh_obj=True,
            **kwargs)
        
        self.web_vxcol = {
            'lh' : {},
            'rh' : {}, 
        }
        self.web_vxcol_list = []

    def web_add_vx_col(self, vx_col_name,  **kwargs):
        '''
        Add an instance of vx_col... (i.e., eccentricity)
        '''
        kwargs['hemi_list'] = self.web_hemi_list
        roi_list = kwargs.pop('ply_roi_list', None)
        this_mesh = self.plotly_return_mesh_dict(**kwargs)
        for ih in this_mesh.keys():
            self.web_vxcol[ih][vx_col_name] = this_mesh[ih]['vertexcolor']
        if roi_list is not None:
            if not isinstance(roi_list, list):
                roi_list =  [roi_list]

            self.web_mesh += self.plotly_return_roi_obj(roi_list)
        self.web_vxcol_list.append(vx_col_name)

    def web_launch(self):
        '''
        Return a figure! 
        '''
        fig = go.Figure()
        for web_mesh in self.web_mesh:
            fig.add_trace(web_mesh)
        # fig.add_trace(self.web_mesh[0])
        # fig.add_trace(self.web_mesh[1])

        # Create buttons list:
        button_info_for_list = []
        for vx_col_name in self.web_vxcol_list:
            this_col_list = []
            for hemi in self.web_hemi_list:
                this_col_list.append(
                    self.web_vxcol[hemi][vx_col_name]
                )
            
            this_button_entry = dict(
                args=[{"vertexcolor": this_col_list}],
                label=vx_col_name,
                method="restyle"
            )
            button_info_for_list.append(this_button_entry)

        # Update plot sizing
        fig.update_layout(
            # width=800,
            # height=900,
            autosize=True,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
        )

        # Update 3D scene options
        fig.update_scenes(
            # aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode="manual"
        )

        # Add dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=button_info_for_list,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )

        # Add annotation
        fig.update_layout(
            annotations=[
                dict(text="Surface color scale:", showarrow=False,
                    x=0, y=1.08, yref="paper", align="left")
            ]
        )        
        return fig


    # **************************************
    # **************************************
    # **************************************
    # **************************************
    def web_launch2(self):
        '''
        Return a Dash app! 
        '''
        app = dash.Dash(__name__)
        self.create_figure()
        app.layout = html.Div([
            dcc.Graph(id='mesh-plot', figure=self.dash_fig),
            dcc.Dropdown(
                id='vertex-color-dropdown',
                options=[{'label': col_name, 'value': col_name} for col_name in self.web_vxcol_list],
                value=self.web_vxcol_list[0]
            ),                                      # Dropdown menu - change the surface colour
            html.Div(id='vertex-index-output'),     # Print which vertex you have clicked on 
            html.Div(id='mpl-figure-output'),       # Plot the output of the figure (based on click)

        ])
        @app.callback(
            Output('mesh-plot', 'figure'),
            [Input('vertex-color-dropdown', 'value')]
        )
        def update_figure(selected_color):
            if selected_color is None:
                raise dash.exceptions.PreventUpdate            
            self.update_figure_with_color(selected_color)
            return self.dash_fig

        @app.callback(
            Output('vertex-index-output', 'children'),
            [Input('mesh-plot', 'clickData')]
        )
        def display_click_data(clickData):
            if clickData is not None:
                point_index = clickData['points'][0]['pointNumber']
                mesh_index = clickData['points'][0]['curveNumber']
                hemi_name = self.web_hemi_list[mesh_index]                
                return f'Clicked hemi: {hemi_name}, Vertex Index: {point_index}'

        @app.callback(
            Output('mpl-figure-output', 'children'),
            [Input('mesh-plot', 'clickData')]
        )
        def display_mpl_figure(clickData):
            if clickData is not None:
                point_index = clickData['points'][0]['pointNumber']
                mesh_index = clickData['points'][0]['curveNumber']
                hemi_name = self.web_hemi_list[mesh_index]                
                if hemi_name == 'rh':
                    point_index += self.n_vx['lh']
                
                mpl_fig = self.mpl_fig_maker(idx=point_index, return_fig=True)
                # Scale the matplotlib figure to a sensible size. Keeping the aspect
                # mpl_fig.set_tight_layout(True)
                mpl_fig_path = opj(self.output_dir, 'mpl_fig.svg')
                mpl_fig.savefig(mpl_fig_path)
                # Convert the Matplotlib figure to an image
                with open(mpl_fig_path, 'r') as f:
                    svg_content = f.read()                
                svg_data_uri = 'data:image/svg+xml;base64,' + base64.b64encode(svg_content.encode()).decode()
                # Return the image tag embedding the SVG. Make it a sensible size
                svg4html = html.Img(
                    src=svg_data_uri,
                    id='mpl-figure-image',
                    style={'width': '100%', 'height': 'auto'},
                    )
                return svg4html
        app.scripts.config.serve_locally = True
        app.css.config.serve_locally = True


        # # Create the folder to store Dash app files
        # self.app_folder = opj(self.output_dir, 'dash_app')
        # if not os.path.exists(self.app_folder):
        #     os.makedirs(self.app_folder)

        # # Save the Dash app as HTML file
        # app_index_file = os.path.join(self.app_folder, 'index.html')
        # with open(app_index_file, 'w') as f:
        #     f.write(app.index())

        # default values
        # app.config.assets_folder = opj(self.app_folder,'assets')     # The path to the assets folder.
        # app.config.include_asset_files = True   # Include the files in the asset folder
        # app.config.assets_external_path = ""    # The external prefix if serve_locally == False
        # app.config.assets_url_path = opj(self.app_folder,'assets')     # The path to the assets folder.
        # # Run the Flask server to serve the Dash app
        # server = Flask(__name__)

        # @server.route('/')
        # def serve_index():
        #     return send_from_directory(self.app_folder, 'index.html')

        # @server.route('/<path:path>')
        # def serve_static(path):
        #     return send_from_directory(self.app_folder, path)


        return app

    def create_figure(self):
        self.dash_fig = go.Figure()
        for web_mesh in self.web_mesh:
            self.dash_fig.add_trace(web_mesh)
        self.camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )  # Default camera position
        # Update plot sizing
        self.dash_fig.update_layout(
            autosize=True,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
            scene_camera=self.camera,
            uirevision='constant'  # Preserve camera settings
        )

        # Update 3D scene options
        self.dash_fig.update_scenes(
            aspectmode="manual"
        )



    def update_figure_with_color(self, selected_color):
        this_col_list = []
        for hemi in self.web_hemi_list:
            this_col_list.append(
                self.web_vxcol[hemi][selected_color]
            )
        # Update facecolor for each mesh trace
        for i in range(len(self.web_mesh)):
            self.dash_fig.data[i].update(vertexcolor=this_col_list[i])
