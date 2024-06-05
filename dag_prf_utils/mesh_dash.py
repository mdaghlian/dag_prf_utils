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
from dag_prf_utils.mesh_format import *
from dag_prf_utils.mesh_maker import *
try:
    import plotly.graph_objects as go
except:
    print('No plotly')

try: 
    from dash import Dash, dcc, html, Input, Output, State
    import dash
    # import dash_bootstrap_components as dbc
    from flask import Flask, send_file,render_template_string, send_from_directory
except:
    print('no dash..')


import io
import base64
import matplotlib.image as mpimg
from PIL import Image

path_to_utils = os.path.abspath(os.path.dirname(__file__))

import pickle
def dag_mesh_pickle(mesh_dash, **kwargs):
    # Path to the pickle file
    file_name = kwargs.get('file_name', 'mesh_dash.pickle')
    pickle_file_path = opj(mesh_dash.output_dir, file_name)
    print(f'pickling mesh_dash object to : {pickle_file_path}')
    if os.path.exists(pickle_file_path):
        os.remove(pickle_file_path)
    # Writing the variable to the pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(mesh_dash, f,) 
        #**kwargs)

class MeshDash(GenMeshMaker):
    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], output_dir=[], **kwargs):
        '''Meshes are the plotly stuff'''
        super().__init__(sub, fs_dir, output_dir, **kwargs)

    #region PLOTLY FUNCTIONS
    def plotly_return_mesh_dict(self, data, **kwargs):
        '''Return a dict with mesh info [x,y,z,i,j,k,intensity,vertexcolor]
        '''
        return_cmap_dict = kwargs.pop('return_cmap_dict', False)
        do_intensity = kwargs.get('do_intensity', False)
        do_vertexcolor = kwargs.get('do_vertexcolor', True)
        mesh_name = kwargs.get('mesh_name', 'inflated')
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        
        disp_rgb, cmap_dict = self.return_display_rgb(
            data=data, split_hemi=True, return_cmap_dict=True, **kwargs
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
        if return_cmap_dict:
            return mesh_dict, cmap_dict
        else:
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
                showlegend=True,
                # hoverinfo='skip',                
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
                    xaxis_visible=False, yaxis_visible=False,zaxis_visible=False
                    # bgcolor='rgba(0,0,0,0)'  # Set background color to transparent
                    
                ),
            # legend=dict(
            #         yanchor="bottom",
            #         y=0.01,
            #         xanchor="left",
            #         x=0.01
            #     )                
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
                if roi_bool.sum()==0:
                    continue
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
    #endregion PLOTLY FUNCTIONS


    # *****************************************************************
    # *****************************************************************
    # *****************************************************************
    #region DASH FUNCTIONS    

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
        # Move to dash launch...
        # self.web_mesh = self.add_plotly_surface(
        #     return_mesh_obj=True,
        #     **kwargs)        
        self.hemi_count = len(self.web_hemi_list)
        self.roi_obj = []
        self.roi_list = []
        self.web_vxcol = {}
        self.web_vxcol_list = []
        self.web_inflated = kwargs.get('inflate_type', 'inflated')

    
    def web_add_vx_col(self, vx_col_name, data, **kwargs):
        '''Add a surface
        Properties of the surface:
        - data: array (len vertices). Used to make the colormap
        - data4mask: array (len vertices). Used to mask based on threshold
        - rsq_thresh: threshold applied to data4mask 
        - cmap_name : string, name of colormap
        - vmin : float vmin for cmap
        - vmax : float vmax for cmap        
        - rgb_direct # ignore everything just put in the RGB values
        '''        
        rgb_direct = kwargs.get('rgb_direct', False)
        data4mask = kwargs.get('data4mask', np.ones_like(data))
        
        # c_ for properties we want to change at 
        c_rsq_thresh = kwargs.get('rsq_thresh', 0)
        c_data_mask = data4mask>c_rsq_thresh
        c_vmin = kwargs.get('vmin', np.nanmin(data[c_data_mask]))
        c_vmax = kwargs.get('vmax', np.nanmax(data[c_data_mask]))

        c_cmap = kwargs.get('cmap', 'viridis')
        
        self.web_vxcol[vx_col_name] = {}            
        self.web_vxcol[vx_col_name]['data']         = data
        self.web_vxcol[vx_col_name]['data4mask']    = data4mask
        # Properties we will allow to change by clicker
        self.web_vxcol[vx_col_name]['c_rsq_thresh'] = c_rsq_thresh
        self.web_vxcol[vx_col_name]['c_vmin'] = c_vmin
        self.web_vxcol[vx_col_name]['c_vmax'] = c_vmax
        self.web_vxcol[vx_col_name]['c_cmap'] = c_cmap
        # RGB direct
        self.web_vxcol[vx_col_name]['rgb_direct'] = rgb_direct # To be assigned
        self.web_vxcol_list.append(vx_col_name)

    def web_add_roi(self, roi_list, **kwargs):
        '''
        Add a roi to the plot
        '''
        if not isinstance(roi_list, list):
            roi_list = [roi_list]
        self.roi_list = roi_list
        self.roi_info = []            
        hemi_list = kwargs.get('hemi_list', self.web_hemi_list)
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        mesh_name = kwargs.get('mesh_name', 'inflated')
        combine_matches = kwargs.pop('combine_matches', False)
        # marker_kwargs = kwargs.get()
        roi_cols = dag_get_col_vals(
            np.arange(len(roi_list)),
            vmin = -1, vmax=7, cmap='jet'
        )
        self.roi_obj = []
        for ih,hemi in enumerate(hemi_list):
            for i_roi,roi in enumerate(roi_list):
                # Load roi index:
                roi_bool = dag_load_roi(self.sub, roi, fs_dir=self.fs_dir, split_LR=True, combine_matches=combine_matches)[hemi]                
                if roi_bool.sum()==0:
                    continue
                border_vx_list = dag_find_border_vx_in_order(
                    roi_bool=roi_bool, 
                    mesh_info=self.mesh_info[mesh_name][hemi], 
                    return_coords=False,
                    )
                # If more than one closed path (e.g., v2v and v2d)...
                this_roi_col = '#{:02x}{:02x}{:02x}'.format(*(roi_cols[i_roi]*255).astype(int))

                for ibvx,border_vx in enumerate(border_vx_list):
                    first_instance = (ih==0) & (ibvx==0) # Only show legend for first instance
                    this_roi_dict = {
                        'hemi' : hemi,
                        'roi' : roi,
                        'border_vx' : border_vx,
                        'roi_col' : this_roi_col,
                        'border_line' : None,
                        'data_id' : None,
                        'first_instance' : first_instance,
                    }
                    self.roi_obj.append(this_roi_dict)        
    
    def web_remake_roi(self):
        mesh_name = 'inflated'
        # marker_kwargs = kwargs.get()
        for i_sub_roi,v_sub_roi in enumerate(self.roi_obj):
            hemi = v_sub_roi['hemi']
            roi = v_sub_roi['roi']
            border_vx = v_sub_roi['border_vx']
            this_roi_col = v_sub_roi['roi_col']
            first_instance = v_sub_roi['first_instance']
            # Create a the line object for the border
            border_line = go.Scatter3d(
                x=self.mesh_info[mesh_name][hemi]['x'][border_vx],
                y=self.mesh_info[mesh_name][hemi]['y'][border_vx],
                z=self.mesh_info[mesh_name][hemi]['z'][border_vx],
                mode='lines',
                name=roi,
                marker=dict(
                    size=10,
                    color=this_roi_col, ),
                line=dict(
                    color=this_roi_col, 
                    width=10, ),
                opacity=1,
                showlegend=True if first_instance else False, 
            )
            self.roi_obj[i_sub_roi]['border_line'] = border_line    
    
    def get_web_vx_col_info(
            self, vx_col_name, 
            rsq_thresh=None,
            cmap=None, 
            vmin=None, 
            vmax=None,):
        '''For a given vx_col_name. Update the c_ attributes
        Returns RGB per vertex, and a  
        Return 
        '''    
        assert vx_col_name in self.web_vxcol_list
        
        c_update = {
            'c_rsq_thresh' : rsq_thresh,
            'c_cmap' : cmap,
            'c_vmin' : vmin,
            'c_vmax' : vmax,            
        }
        for c in c_update.keys():
            if c_update[c] is not None:
                self.web_vxcol[vx_col_name][c] = c_update[c]
            
        # RETURN RGB & CMAP INFO
        make_rgb_time = time.time()
        if self.web_vxcol[vx_col_name]['rgb_direct']:            
            disp_rgb_not_split = self._combine2maps(
                data_col1=self.web_vxcol[vx_col_name]['data'], 
                data_col2=self.get_us_cols('curv'),
                data_alpha=(self.web_vxcol[vx_col_name]['data4mask']>self.web_vxcol[vx_col_name]['c_rsq_thresh'])*1.0,
            )
            disp_rgb = {
                'lh' : disp_rgb_not_split[:self.n_vx['lh'],:],
                'rh' : disp_rgb_not_split[self.n_vx['lh']:,:],
            }            
            cmap_fig = plt.figure()
        else:
            disp_rgb, cmap_dict = self.return_display_rgb(
                return_cmap_dict=True, unit_rgb=True, split_hemi=True, 
                data=self.web_vxcol[vx_col_name]['data'],
                data_mask=self.web_vxcol[vx_col_name]['data4mask']>self.web_vxcol[vx_col_name]['c_rsq_thresh'],
                cmap = self.web_vxcol[vx_col_name]['c_cmap'],
                vmin = self.web_vxcol[vx_col_name]['c_vmin'],
                vmax = self.web_vxcol[vx_col_name]['c_vmax'],
                )
            # Save CMAP to svg
            cmap_fig = dag_cmap_plotter(
                cmap=cmap_dict['cmap'], 
                vmin=cmap_dict['vmin'], 
                vmax=cmap_dict['vmax'], 
                title=str(vx_col_name), 
                return_fig=True, )
        print(f'Make RGB time = {time.time() - make_rgb_time}')
        
        cmap_fig.tight_layout()
        cmap_fig.canvas.draw()
        # cmap_path = opj(self.output_dir, f'cmap_{vx_col_name}.svg')
        # cmap_fig.savefig(cmap_path)        
        # with open(cmap_path, 'r') as f:
        #     svg_content = f.read()    
        # svg_data_uri = 'data:image/svg+xml;base64,' + base64.b64encode(svg_content.encode()).decode()
        # # Return the image tag embedding the SVG. Make it a sensible size
        # svg4html = html.Img(
        #     src=svg_data_uri,
        #     id=f'colbar-{vx_col_name}',
        #     style={'width': '100%', 'height': 'auto'},
        #     )            
        img_4html = self.web_return_embedded_img(
            fig=cmap_fig, 
            id=f'colbar',
            className='colbar',
            )
        return disp_rgb, html.Div(img_4html) # , className='colbar')
    
    # ***** DASH *****
    def create_figure(self):
        self.dash_fig = go.Figure()
        web_mesh = self.add_plotly_surface(return_mesh_obj=True)
        for ih,hemi in enumerate(self.web_hemi_list):                        
            self.dash_fig.add_trace(web_mesh[ih])
        if self.roi_list != []:
            self.web_remake_roi() 
            for iroi,vroi in enumerate(self.roi_obj):
                self.dash_fig.add_trace(vroi['border_line'])
                self.roi_obj[iroi]['data_id'] = iroi + self.hemi_count

        self.camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )  # Default camera position
        self.dash_fig.update_layout(legend=dict(
            yanchor="top",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor= "#141414",            

        ))        
        # Update plot sizing
        self.dash_fig.update_layout(
            autosize=True,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_dark",
            scene_camera=self.camera,
            uirevision='constant',  # Preserve camera settings
        )

        # Update 3D scene options
        self.dash_fig.update_scenes(
            aspectmode="manual",
            xaxis_visible=False, yaxis_visible=False,zaxis_visible=False,
        )

    def web_launch_with_dash(self, **kwargs):
        '''
        Return a Dash app! 

        TODO:
        > colorbar
        > update zoom after panning?
        > ROIS
        > Add clicker position
        > hemi on & hemi off...
        '''
        assets_type = kwargs.get('assets_type', 'boring')
        assets_type = 'mesh_dash_assets_boring' if assets_type=='boring' else 'mesh_dash_assets'
        app = dash.Dash(
            __name__,
            assets_folder=opj(os.path.dirname(__file__),assets_type)
            )
        self.image_type = kwargs.get('image_type', 'png')                
        # plt.style.use('dark_background')
        self.create_figure()
        init_vx_col = self.web_vxcol_list[0]
        init_vmin = self.web_vxcol[init_vx_col]['c_vmin']
        init_vmax = self.web_vxcol[init_vx_col]['c_vmax']
        init_rsq_thresh = self.web_vxcol[init_vx_col]['c_rsq_thresh']
        init_cmap = self.web_vxcol[init_vx_col]['c_cmap']

        num_input_args = dict(type='number', n_submit=0, debounce=True)
        if self.roi_list==[]:
            roi_html = html.Div(id='rois-dropdown')
        else:
            checkbox_options = [{'label' : roi,  'value' : roi} for roi in self.roi_list]
            roi_html = html.Div([
                dcc.Dropdown(
                    id='rois-dropdown',
                    options=checkbox_options,
                    value=[],
                    multi=True,
                    )
            ], className='column2')

        app.layout = html.Div([
            # html.H1('awesome brain viewer'),
            # COLUMN 1
            html.Div([
                # Radius
                html.Label('radius', className='label'),
                dcc.Input(id='radius', value=2,  **num_input_args),
                # Inflate 
                html.Label('inflate', className='label'),
                dcc.Input(id='inflate', value=1,  **num_input_args),
            ], className='column'),
            # COLUMN 2
            html.Div([
                # COLOR STUFF
                html.Label('vmin', className='label'),
                dcc.Input(id='vmin', value=init_vmin,  **num_input_args),
                html.Label('vmax', className='label'),
                dcc.Input(id='vmax', value=init_vmax,  **num_input_args),
            ], className='column'),
            # COLUMN 3
            html.Div([            
                html.Label('cmap', className='label'),
                dcc.Input(id='cmap',  type='string', value=init_cmap, n_submit=0, debounce=True),
                html.Label('rsq_thresh', className='label'),
                dcc.Input(id='rsq_thresh', value=init_rsq_thresh,  **num_input_args),
            ], className='column'),

            html.Hr(),            
            roi_html,  # ROI HTML
            html.Div([
                dcc.Dropdown(
                    id='vertex-color-dropdown',
                    options=[{'label': col_name, 'value': col_name} for col_name in self.web_vxcol_list],
                    value=self.web_vxcol_list[0],
                ),  # Dropdown menu - change the surface colour
            ], className='column2'),
            # MESH PLOT
            html.Div(id='color-chain'),
            html.Hr(), 
            dcc.Graph(id='mesh-plot', figure=self.dash_fig),
            html.Div(
                id='colbar-div',
                style={'maxWidth': '400px', 'width': '100%', 'overflowX': 'auto', 'overflowY': 'auto'},
            ),  # Plot the colorbar
            # Update on / off
            dcc.Checklist(id='vxtoggle',
                options=[{'label': 'Plot vx?', 'value': 'on'}],
                value=[]
            ),
            html.Div(id='vxtoggle-hidden'),
            html.Div(id='vertex-index-output'),  # Print which vertex you have clicked on
            html.Div(
                id='mpl-figure-output',                
                # style={'maxWidth': '800px', 'width': '100%', 'overflowX': 'auto', 'overflowY': 'auto'},
            ),  # Plot the output of the figure (based on click)
            
            # Add histogram
            dcc.Checklist(id='hist-toggle',
                options=[{'label': 'histogram', 'value': 'on'}],
                value=[]
            ),
            html.Div(
                id='hist-output',                
                # style={'maxWidth': '800px', 'width': '100%', 'overflowX': 'auto', 'overflowY': 'auto'},
            ),  # Plot the output of the figure (based on click)                        
            html.Div(id='camera-position-output'),

        ], className='container')  # Add a container class to center the content

        # SAVE CAMERA POSITION AFTER DRAGGING...
        @app.callback(
            Output("camera-position-output", "children"),
            Input("mesh-plot", "relayoutData")
        )
        def show_data(relayoutData):
            if relayoutData is None:
                raise dash.exceptions.PreventUpdate
            if 'scene.camera' in relayoutData:
                # Update self.camera with the current camera position
                self.camera = relayoutData['scene.camera']
                # TODO : update zoom value here
                raise dash.exceptions.PreventUpdate
            else:
                raise dash.exceptions.PreventUpdate

        # RADIUS
        @app.callback(
            Output('mesh-plot', 'figure', allow_duplicate=True),
            Input('radius', 'value'),
            prevent_initial_call=True
        )
        def update_figure_radius(radius):
            # INFLATE
            print('RADIUS CALLBACK')
            # CHECK FOR RADIUS CHANGE
            if (radius is not None) & (radius != 0):
                # Update camera radius (current camera)
                radius = float(radius)
                radius_now = np.sqrt(self.camera['eye']['x']**2 + self.camera['eye']['y']**2 + self.camera['eye']['z']**2)
                scale = radius / radius_now
                self.camera['eye']['x'] *= scale
                self.camera['eye']['y'] *= scale
                self.camera['eye']['z'] *= scale
                # Update layout with new camera settings
                self.dash_fig.update_layout(scene_camera=self.camera)
            else:
                raise dash.exceptions.PreventUpdate 
            return self.dash_fig
        # INFLATE 
        @app.callback(
            Output('mesh-plot', 'figure'),
            Input('inflate', 'value')
        )
        def update_figure_inflate(inflate):
            # INFLATE
            print('INFLATE CALLBACK')
            if inflate is not None:            
                self.update_figure_inflate(inflate)
            else:
                raise dash.exceptions.PreventUpdate 
            return self.dash_fig

        # COLOR CALLBACKS     
        # -> prevent callbacks to callbacks
        self.do_col_updates = {
            'vmin' : True,
            'vmax' : True,
            'cmap' : True,
            'rsq_thresh' : True,
        }   
        _,self.current_col_bar = self.get_web_vx_col_info(self.web_vxcol_list[0], rsq_thresh=0)
        self.current_col_args = {
            'vx_col' : self.web_vxcol_list[0],
            'c_vmin' : self.web_vxcol[self.web_vxcol_list[0]]['c_vmin'],
            'c_vmax' : self.web_vxcol[self.web_vxcol_list[0]]['c_vmax'],
            'c_cmap' : self.web_vxcol[self.web_vxcol_list[0]]['c_cmap'],
            'c_rsq_thresh' : self.web_vxcol[self.web_vxcol_list[0]]['c_rsq_thresh'],            
        }
        
        # COLOR CHAIN
        @app.callback(
            [
                Output('mesh-plot', 'figure', allow_duplicate=True),
                Output('colbar-div', 'children'),
                Output('hist-output', 'children'),
            ],
            Input('color-chain', 'children'),
            prevent_initial_call='initial_duplicate'
        )
        def color_chain(color_chain):
            print('COLOR CHAIN')
            self.update_hist()        
            return self.dash_fig, self.current_col_bar, self.current_hist
        
        # COLOR [1] DROPDOWN
        @app.callback(
            [
                Output('color-chain', 'children', allow_duplicate=True),
                Output('vmin', 'value'),# 
                Output('vmax', 'value'),# 
                Output('cmap', 'value'),# 
                Output('rsq_thresh', 'value'),# 
            ],
            Input('vertex-color-dropdown', 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_col_dropdown(selected_color):
            print('COL DROPDOWN CALLBACK')
            if selected_color is not None:
                # CHECK FOR CHANGE IN COLOR
                if selected_color!=self.current_col_args['vx_col']:
                    # Update colors
                    disp_rgb, cmap_path = self.get_web_vx_col_info(
                        vx_col_name=selected_color,         
                    )
                    self.update_figure_with_color(disp_rgb)  
                    self.current_col_bar = cmap_path        
                    self.current_col_args['vx_col'] = selected_color
                    for key in self.web_vxcol[selected_color].keys():
                        self.current_col_args[key] = self.web_vxcol[selected_color][key]
                    for key in self.do_col_updates.keys():
                        self.do_col_updates[key] = False
                    return html.Div(), self.current_col_args['c_vmin'],self.current_col_args['c_vmax'],self.current_col_args['c_cmap'],self.current_col_args['c_rsq_thresh']
            raise dash.exceptions.PreventUpdate                
        
        # COLOR [2] vmin
        @app.callback(
            Output('color-chain', 'children', allow_duplicate=True),
            Input('vmin', 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_col_vmin(vmin):
            if not self.do_col_updates['vmin']:
                self.do_col_updates['vmin'] = True
                raise dash.exceptions.PreventUpdate                
            
            print('VMIN CALLBACK')
            if vmin is not None:
                # CHECK FOR CHANGE IN COLOR
                if vmin != self.current_col_args['c_vmin']:
                    selected_color = self.current_col_args['vx_col']
                    self.web_vxcol[selected_color]['c_vmin'] = vmin
                    self.current_col_args['c_vmin'] = vmin
                    disp_rgb, cmap_path = self.get_web_vx_col_info(
                        vx_col_name=selected_color,          
                    )                    
                    self.update_figure_with_color(disp_rgb)  
                    self.current_col_bar = cmap_path        
                    
                    return html.Div()
            raise dash.exceptions.PreventUpdate                

        # COLOR [3] vmax
        @app.callback(
            Output('color-chain', 'children', allow_duplicate=True),
            Input('vmax', 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_col_vmax(vmax):
            if not self.do_col_updates['vmax']:
                self.do_col_updates['vmax'] = True
                raise dash.exceptions.PreventUpdate                

            print('VMAX CALLBACK')
            if vmax is not None:
                # CHECK FOR CHANGE IN COLOR
                if vmax != self.current_col_args['c_vmax']:
                    selected_color = self.current_col_args['vx_col']
                    self.web_vxcol[selected_color]['c_vmax'] = vmax
                    self.current_col_args['c_vmax'] = vmax
                    disp_rgb, cmap_path = self.get_web_vx_col_info(
                        vx_col_name=selected_color,          
                    )                    
                    self.update_figure_with_color(disp_rgb)  
                    self.current_col_bar = cmap_path        
                    
                    return html.Div()
            raise dash.exceptions.PreventUpdate                

        # COLOR [4] cmap
        @app.callback(
            Output('color-chain', 'children', allow_duplicate=True),
            Input('cmap', 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_col_cmap(cmap):
            if not self.do_col_updates['cmap']:
                print('CMAP PREVENT')
                self.do_col_updates['cmap'] = True
                raise dash.exceptions.PreventUpdate                

            print('CMAP CALLBACK')
            if cmap is not None:
                # CHECK FOR CHANGE IN COLOR
                if cmap != self.current_col_args['c_cmap']:
                    selected_color = self.current_col_args['vx_col']
                    self.web_vxcol[selected_color]['c_cmap'] = cmap
                    self.current_col_args['c_cmap'] = cmap
                    disp_rgb, cmap_path = self.get_web_vx_col_info(
                        vx_col_name=selected_color,          
                    )                    
                    self.update_figure_with_color(disp_rgb)  
                    self.current_col_bar = cmap_path        
                    
                    return html.Div()
            raise dash.exceptions.PreventUpdate   

        # COLOR [5] rsq_thresh
        @app.callback(
            Output('color-chain', 'children', allow_duplicate=True),
            Input('rsq_thresh', 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_col_rsq_thresh(rsq_thresh):
            if not self.do_col_updates['rsq_thresh']:
                self.do_col_updates['rsq_thresh'] = True
                raise dash.exceptions.PreventUpdate                

            print('RSQ THRESH CALLBACK')
            if rsq_thresh is not None:
                # CHECK FOR CHANGE IN COLOR
                if rsq_thresh != self.current_col_args['c_rsq_thresh']:
                    selected_color = self.current_col_args['vx_col']
                    self.web_vxcol[selected_color]['c_rsq_thresh'] = rsq_thresh
                    self.current_col_args['c_rsq_thresh'] = rsq_thresh
                    disp_rgb, _ = self.get_web_vx_col_info(
                        vx_col_name=selected_color,          
                    )                    
                    self.update_figure_with_color(disp_rgb)  
                    
                    return html.Div()
            raise dash.exceptions.PreventUpdate
        # # Histogram?
        self.hist_on = False
        self.current_hist = None
        @app.callback(
            Output('color-chain', 'children'),
            Input('hist-toggle', 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_output(value):
            if 'on' in value:
                self.hist_on = True
                self.update_hist()
                return html.Div()
            else:
                self.hist_on = False            
                raise dash.exceptions.PreventUpdate


        # UPDATE VERTEX PLOTS ON CLICK?
        self.vx_toggle_on = False
        @app.callback(
            [Output('vxtoggle-hidden', 'children'),],
            [Input('vxtoggle', 'value')]
        )
        def update_output(value):
            if 'on' in value:
                self.vx_toggle_on = True
            else:
                self.vx_toggle_on = False
            raise dash.exceptions.PreventUpdate


        # CLICKER FUNCTION (DISPLAYS MATPLOTLIB FIGURE, IF DEFINED)
        self.last_clicktime = time.time()
        @app.callback(
            [
                Output('mpl-figure-output', 'children'),
                Output('vertex-index-output', 'children'),
            ],
            [Input('mesh-plot', 'clickData')]
        )
        def display_mpl_figure(clickData):
            if not self.vx_toggle_on:
                print('Toggled OFF')
                raise dash.exceptions.PreventUpdate 
            now = time.time()
            if (now - self.last_clicktime)<5:
                print('Clicked too soon...')
                raise dash.exceptions.PreventUpdate 
            else:
                self.last_clicktime = now
            print('CLICK CALLBACK')
            if clickData is not None:
                right_now = time.time()
                point_index = clickData['points'][0]['pointNumber']
                mesh_index = clickData['points'][0]['curveNumber']
                hemi_name = self.web_hemi_list[mesh_index]
                print(self.dash_fig.data[mesh_index]['vertexcolor'][point_index,:])                                
                if hemi_name == 'rh':
                    full_point_index = self.n_vx['lh']+point_index
                else:
                    full_point_index = point_index
                click_str = f'Clicked hemi: {hemi_name}, vx id: {point_index}, full_id {full_point_index}'
                mpl_figs = self.web_return_mpl_figs(full_point_index), click_str                
                self.last_clicktime = time.time()
                finished_now = time.time()
                print(f'time = {finished_now - right_now}')
                return mpl_figs
        

        # ROI call back
        @app.callback(
            Output('mesh-plot', 'figure', allow_duplicate=True),
            Input('rois-dropdown', 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def roi_update(roi_dropdown):
            print('ROI CALLBACK')
            for roi_lines in self.roi_obj:
                if roi_lines['roi'] in roi_dropdown:
                    # set visible:
                    self.dash_fig.data[roi_lines['data_id']].update(visible=True)
                else:
                    self.dash_fig.data[roi_lines['data_id']].update(visible=False)
            return self.dash_fig
        




        # Help to speed things up?
        app.scripts.config.serve_locally = True
        app.css.config.serve_locally = True
        return app


    def update_hist(self):
        if self.hist_on:
            vx_col_name = self.current_col_args['vx_col']
            data=self.web_vxcol[vx_col_name]['data']
            data_mask=self.web_vxcol[vx_col_name]['data4mask']>self.web_vxcol[vx_col_name]['c_rsq_thresh'],                
            data = data[data_mask]            
            this_hist, ax = plt.subplots(1, figsize=(5,2))            
            bins = np.linspace(
                self.current_col_args['c_vmin'],
                self.current_col_args['c_vmax'],
                100, 
            )
            edges, bin, patches = ax.hist(data, bins)
            bin_cols = dag_get_col_vals(
                col_vals=bin,
                cmap=self.current_col_args['c_cmap'],
                vmin=self.current_col_args['c_vmin'],
                vmax=self.current_col_args['c_vmax'],
            )
            for j, p in enumerate(patches):
                # Set face color
                p.set_facecolor(bin_cols[j])                 
            ax.set_title(f'Histogram of {vx_col_name}')            
            self.current_hist = self.web_return_embedded_img(
                fig=this_hist,
                id='hist-figure',
                className='hist-figure',
            )
        else:
            self.current_hist = html.Div()

    def update_figure_with_color(self, disp_rgb):
        vx_update_time = time.time()
        # Update vertexcolor for each mesh trace        
        for i,ih in enumerate(self.web_hemi_list):
            self.dash_fig.data[i].update(vertexcolor=disp_rgb[ih])        
        print(f'Vertex update time = {time.time() - vx_update_time}')


    def update_figure_inflate(self, inflate):
        inflate_time = time.time()
        new_vx_coords = {}
        for hemi in self.web_hemi_list:
            # INTERPOLATE
            new_vx_coords[hemi] = dag_mesh_interpolate(
                coords1=self.mesh_info['pial'][hemi]['coords'],
                coords2=self.mesh_info[self.web_inflated][hemi]['coords'],
                interp=inflate,
                )
        # Update the vertex coordinates for each webmesh
        for i_hemi,v_hemi in enumerate(self.web_hemi_list):
            self.dash_fig.data[i_hemi].update(
                x=new_vx_coords[v_hemi][:,0],
                y=new_vx_coords[v_hemi][:,1],
                z=new_vx_coords[v_hemi][:,2],
                )
        # Update the ROI coordinates
        for i_roi,v_roi in enumerate(self.roi_obj):
            this_hemi = v_roi['hemi']
            this_border_vx = v_roi['border_vx']
            self.dash_fig.data[self.roi_obj[i_roi]['data_id']].update(
                x=new_vx_coords[this_hemi][this_border_vx,0],
                y=new_vx_coords[this_hemi][this_border_vx,1],
                z=new_vx_coords[this_hemi][this_border_vx,2],
                )
        print(f'Inflate time = {time.time() - inflate_time}')

        


    def web_return_mpl_figs(self, idx):
        '''
        Run through the mpl figure plotters...
        '''
        if not hasattr(self, 'mpl_fig_makers'):
            print('No mpl fig makers')
            return       
        fig_time = time.time()
        figs = []
        for key in self.mpl_fig_makers.keys():
            this_fig = self.mpl_fig_makers[key]['func'](
                idx, **self.mpl_fig_makers[key]['kwargs'])
            this_fig.suptitle = f'{key} - {idx} - {this_fig._suptitle}'
            this_fig.tight_layout()
            this_fig.canvas.draw()
            # get title 
            # Save the Matplotlib figure as an SVG, making sure nothing is cut off
            img_4html = self.web_return_embedded_img(
                fig=this_fig, 
                id=f'mpl-figure',
                className='mpl-figure',
            )
            #     style={'width': '100%', 'height': 'auto'},
            figs.append(img_4html)
        print(f'Fig time = {time.time() - fig_time}')
        return html.Div(figs, ) # className='mpl-figure')


    def web_add_mpl_fig_maker(self, mpl_func, mpl_key='mpl1', mpl_kwargs={}):
        '''
        Add a function to make a matplotlib figure
        '''
        if not hasattr(self, 'mpl_fig_makers'):
            self.mpl_fig_makers = {}
        self.mpl_fig_makers[mpl_key] = {}
        self.mpl_fig_makers[mpl_key]['func'] = mpl_func
        self.mpl_fig_makers[mpl_key]['kwargs'] = mpl_kwargs
        
    def web_return_embedded_img(self, fig, id, className):
        '''
        Create embedding for image...
        '''
        if self.image_type == 'png':
            # Save the Matplotlib figure as a PNG image
            image_format = 'png'
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format=image_format)
            image_buffer.seek(0)
        elif self.image_type == 'svg':
            # Save the Matplotlib figure as an SVG image
            image_format = 'svg'
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format=image_format)
            image_buffer.seek(0)
        else:
            raise ValueError("Unsupported image_type. Use 'png' or 'svg'.")
        
        # Convert the image buffer to base64-encoded string
        image_data_uri = f"data:image/{self.image_type};base64," + base64.b64encode(image_buffer.getvalue()).decode()
        
        # Embed the image in HTML
        image4html = html.Img(
            src=image_data_uri,
            id=id, 
            className=className
            # style={'width': '100%', 'height': 'auto'},
        )
        return image4html        
    #endregion DASH FUNCTIONS


    #startregion HTML functions
    def html_get_ready(self, **kwargs):
        '''For making a single html file
        Not messing with dash
        '''
        self.html_hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        self.html_mesh_name = kwargs.get('mesh_name', 'inflated')
        if not isinstance(self.html_hemi_list, list):
            self.html_hemi_list = [self.html_hemi_list]
        self.html_multi = {}
        self.html_cmaps = {}
    
    def html_add_surf(self, data, surf_name, **kwargs):
        '''Add surfaces...
        '''
        # Update kwargs
        kwargs['hemi_list'] = kwargs.get('hemi_list', self.html_hemi_list)
        kwargs['mesh_name'] = kwargs.get('mesh_name', self.html_mesh_name)
        self.html_multi[surf_name] = self.add_plotly_surface(
            data=data, 
            return_mesh_obj=True,
            **kwargs
        )
        _,cmap_dict = self.return_display_rgb(
                return_cmap_dict=True, unit_rgb=True, 
                data=data,
                **kwargs
                )
        this_cmap = dag_cmap_plotter(
                cmap=cmap_dict['cmap'], 
                vmin=cmap_dict['vmin'], 
                vmax=cmap_dict['vmax'], 
                title=surf_name, 
                return_fig=True, )
        this_cmap.tight_layout()
        image_buffer = io.BytesIO()
        this_cmap.savefig(image_buffer, format='png')
        image_buffer.seek(0)
        self.html_cmaps[surf_name] = "data:image/png;base64," +  base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        # self.html_cmaps[surf_name] = Image.open(image_buffer)        
    

    def html_make(self, **kwargs):
        
        do_multi = kwargs.get('do_multi', True)
        mesh_list = list(self.html_multi.keys())
        n_plots = len(mesh_list)
        # If do multi: subfigures 
        if do_multi:
            from plotly.subplots import make_subplots
            n_cols = int(np.ceil(np.sqrt(n_plots)))
            n_rows = int(np.ceil(n_plots / n_cols))
            fig = make_subplots(
                rows=n_rows, 
                cols=n_cols, specs=[[{'type': 'surface'}]*n_cols]*n_rows, )  
            i_plot = 0
            
            for i_row in np.arange(1, n_rows+1):
                for i_col in np.arange(1, n_cols+1):
                    key = mesh_list[i_plot]
                    for this_mesh in self.html_multi[key]:
                        fig.append_trace(this_mesh, row=i_row, col=i_col)

                    # Now add the matplotlib figure self.html_cmaps[key] to the same subplot
                    # Calculate size of colorbar relative to subplot dimensions
                    colorbar_size = min(0.2, 0.5 / max(n_rows, n_cols))                    
                    fig.add_layout_image(
                            dict(
                                source=self.html_cmaps[key],
                                xref=f"x{i_col}", yref=f"y{i_row}",
                                x=0.5, y=-0.2,  # x is centered, y is below the plot
                                xanchor="center", yanchor="top",
                                sizex=1, sizey=colorbar_size,
                                sizing="contain",
                                opacity=0.5,
                                layer="above"

                            ),
                    )
                        
                    # fig.update_scenes(
                    #     aspectmode="manual",
                    #     xaxis_visible=False, yaxis_visible=False,zaxis_visible=False,
                    #     row=i_row, col=i_col,
                    # )
                    # fig.update_yaxes(showgrid=False, row=i_row, col=i_col)
                    i_plot += 1     

            fig.update_layout(
                # width=800,
                # height=900,
                autosize=True,
                margin=dict(t=0, b=0, l=0, r=0),
                template="plotly_white",
            )                    
        else:

            fig = go.Figure()
            for key in self.html_multi.keys():
                for this_mesh in self.html_multi[key]:
                    fig.add_trace(this_mesh)
                # Now add the matplotlib figure self.html_cmaps[key] to the same subplot

            # Create buttons list:
            button_info_for_list = []
            for key in self.html_multi.keys():
                this_col_list = []
                for this_mesh in self.html_multi[key]:
                    this_col_list.append(this_mesh)
                
                this_button_entry = dict(
                    args=[{"vxcol": this_col_list}],
                    label=key,
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


            
    

    #endregion HTML functions