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
    from flask import Flask, send_file,render_template_string, send_from_directory
except:
    print('no dash..')

path_to_utils = os.path.abspath(os.path.dirname(__file__))

import pickle
def dag_mesh_pickle(mesh_dash, **kwargs):
    # Path to the pickle file
    pickle_file_path = opj(mesh_dash.output_dir, 'mesh_dash.pickle')
    print(f'pickling mesh_dash object to : {pickle_file_path}')
    if os.path.exists(pickle_file_path):
        os.remove(pickle_file_path)
    # Writing the variable to the pickle file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(mesh_dash, f, **kwargs)

class MeshDash(GenMeshMaker):
    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], output_dir=[], **kwargs):
        '''Meshes are the plotly stuff'''
        super().__init__(sub, fs_dir, output_dir, **kwargs)

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

        self.web_mesh = self.add_plotly_surface(
            return_mesh_obj=True,
            **kwargs)        
        self.web_vxcol = {}
        self.web_vxcol_list = []
    
    def web_add_vx_col(self, vx_col_name, data, **kwargs):
        '''Add a surface
        Properties of the surface:
        - data: array (len vertices). Used to make the colormap
        - data4mask: array (len vertices). Used to mask based on threshold
        - rsq_thresh: threshold applied to data4mask 
        - cmap_name : string, name of colormap
        - vmin : float vmin for cmap
        - vmax : float vmax for cmap        
        '''        
        
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
        # self.web_vxcol[vx_col_name]['c_rgb'] = [] # To be assigned
        self.web_vxcol_list.append(vx_col_name)
        
    
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
        disp_rgb, cmap_dict = self.return_display_rgb(
            return_cmap_dict=True, unit_rgb=False, split_hemi=True, 
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
        
        cmap_fig.tight_layout()
        cmap_fig.canvas.draw()
        cmap_path = opj(self.output_dir, f'cmap_{vx_col_name}.svg')
        cmap_fig.savefig(cmap_path)        
        with open(cmap_path, 'r') as f:
            svg_content = f.read()    
        # print(svg_content)
        svg_data_uri = 'data:image/svg+xml;base64,' + base64.b64encode(svg_content.encode()).decode()
        # Return the image tag embedding the SVG. Make it a sensible size
        svg4html = html.Img(
            src=svg_data_uri,
            id=f'colbar-{vx_col_name}',
            style={'width': '100%', 'height': 'auto'},
            )            
        
        return disp_rgb, html.Div(svg4html)
    
    # ***** DASH *****

    def web_launch_with_dash(self):
        '''
        Return a Dash app! 

        TODO:
        > colorbar
        > update zoom after panning?
        > ROIS
        > Add clicker position
        > hemi on & hemi off...
        '''
        app = dash.Dash(__name__)        
        self.create_figure()
        init_vx_col = self.web_vxcol_list[0]
        init_vmin = self.web_vxcol[init_vx_col]['c_vmin']
        init_vmax = self.web_vxcol[init_vx_col]['c_vmax']
        init_rsq_thresh = self.web_vxcol[init_vx_col]['c_rsq_thresh']
        init_cmap = self.web_vxcol[init_vx_col]['c_cmap']

        app_layout_number_args = dict(type='number', n_submit=0, debounce=True)
        app.layout = html.Div([
            # Add camera controls here
            html.Label('radius'),
            dcc.Input(id='radius', value=2, **app_layout_number_args),            
            # Inflate 
            html.Label('inflate'),
            dcc.Input(id='inflate', value=1, **app_layout_number_args),            
            
            
            # COLOR STUFF
            html.Label('vmin'),
            dcc.Input(id='vmin', value=init_vmin, **app_layout_number_args),                        
            html.Label('vmax'),
            dcc.Input(id='vmax', value=init_vmax, **app_layout_number_args),                        
            html.Label('rsq_thresh'),
            dcc.Input(id='rsq_thresh', value=init_rsq_thresh, **app_layout_number_args),                        
            html.Label('cmap'),
            dcc.Input(id='cmap', type='string', value=init_cmap, n_submit=0, debounce=True),                        
            #
            dcc.Dropdown(
                id='vertex-color-dropdown',
                options=[{'label': col_name, 'value': col_name} for col_name in self.web_vxcol_list],
                value=self.web_vxcol_list[0]
            ),                                      # Dropdown menu - change the surface colour
            dcc.Graph(id='mesh-plot', figure=self.dash_fig),
            html.Div(
                id='colbar',
                style={'overflow-x': 'auto', 'overflow-y': 'auto'},
                ),       # Plot the colorbar
            html.Div(id='vertex-index-output'),     # Print which vertex you have clicked on 
            html.Div(
                id='mpl-figure-output',
                style={'overflow-x': 'auto', 'overflow-y': 'auto'},
                ),       # Plot the output of the figure (based on click)
            html.Div(id='camera-position-output'),

        ], style={'width': '100%', 'height': '100vh'})

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

        _,self.current_col_bar = self.get_web_vx_col_info(self.web_vxcol_list[0], rsq_thresh=0)
        self.current_col_args = {
            'vx_col' : self.web_vxcol_list[0],
            'c_vmin' : self.web_vxcol[self.web_vxcol_list[0]]['c_vmin'],
            'c_vmax' : self.web_vxcol[self.web_vxcol_list[0]]['c_vmax'],
            'c_cmap' : self.web_vxcol[self.web_vxcol_list[0]]['c_cmap'],
            'c_rsq_thresh' : self.web_vxcol[self.web_vxcol_list[0]]['c_rsq_thresh'],            
        }

        @app.callback(
            [
                Output('mesh-plot', 'figure'),
                Output('colbar', 'children'),
                # UPDATE THE VALUES...
                Output('vmin', 'value'),# 
                Output('vmax', 'value'),# 
                Output('cmap', 'value'),# 
                Output('rsq_thresh', 'value'),# 

            ],
            [
                Input('radius', 'value'),               # RADIUS CONTROL
                Input('inflate', 'value'),              # INFLATE CONTROL
                Input('vertex-color-dropdown', 'value'),# COLOR DROP DOWN 
                #
                Input('vmin', 'value'),# 
                Input('vmax', 'value'),# 
                Input('cmap', 'value'),# 
                Input('rsq_thresh', 'value'),# 
            ]
        )
        def update_figure(
            radius, 
            inflate, 
            selected_color,
            vmin,
            vmax,
            cmap,
            rsq_thresh,            
            ):
            # ALL NONE? 
            arg_list = [radius, inflate, selected_color,vmin,vmax,cmap,rsq_thresh,]
            if all(arg is None for arg in arg_list):
                raise dash.exceptions.PreventUpdate                
            # for i in arg_list:
            #     print(i)
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

            elif vmin != self.current_col_args['c_vmin']:
                selected_color = self.current_col_args['vx_col']
                self.web_vxcol[selected_color]['c_vmin'] = vmin
                self.current_col_args['c_vmin'] = vmin
                disp_rgb, cmap_path = self.get_web_vx_col_info(
                    vx_col_name=selected_color,          
                )                    
                self.update_figure_with_color(disp_rgb)  
                self.current_col_bar = cmap_path        

            elif vmax != self.current_col_args['c_vmax']:
                selected_color = self.current_col_args['vx_col']
                self.web_vxcol[selected_color]['c_vmax'] = vmax
                self.current_col_args['c_vmax'] = vmax
                disp_rgb, cmap_path = self.get_web_vx_col_info(
                    vx_col_name=selected_color,          
                )                    
                self.update_figure_with_color(disp_rgb)  
                self.current_col_bar = cmap_path                        


            elif cmap != self.current_col_args['c_cmap']:
                selected_color = self.current_col_args['vx_col']
                self.web_vxcol[selected_color]['c_cmap'] = cmap
                self.current_col_args['c_cmap'] = cmap
                disp_rgb, cmap_path = self.get_web_vx_col_info(
                    vx_col_name=selected_color,          
                )                    
                self.update_figure_with_color(disp_rgb)  
                self.current_col_bar = cmap_path  

            elif rsq_thresh != self.current_col_args['c_rsq_thresh']:
                selected_color = self.current_col_args['vx_col']
                self.web_vxcol[selected_color]['c_rsq_thresh'] = rsq_thresh
                self.current_col_args['c_rsq_thresh'] = rsq_thresh
                disp_rgb, cmap_path = self.get_web_vx_col_info(
                    vx_col_name=selected_color,          
                )                    
                self.update_figure_with_color(disp_rgb)  
                self.current_col_bar = cmap_path  
            else:
                print('No color changes')
            
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
                # print(self.camera)

            # INFLATE
            if inflate is not None:            
                self.update_figure_inflate(inflate)


            return self.dash_fig, self.current_col_bar, self.current_col_args['c_vmin'],self.current_col_args['c_vmax'],self.current_col_args['c_cmap'],self.current_col_args['c_rsq_thresh']

        self.last_click = time.time()
        # CLICKER FUNCTION (PRINTS VERTEX INDEX)
        @app.callback(
            Output('vertex-index-output', 'children'),
            [Input('mesh-plot', 'clickData')]
        )
        def display_click_data(clickData):
            now = time.time()
            if (now - self.last_clicktime)<0.5:
                raise dash.exceptions.PreventUpdate      
            else: 
                self.last_clicktime = now

            if clickData is not None:
                point_index = clickData['points'][0]['pointNumber']
                mesh_index = clickData['points'][0]['curveNumber']
                hemi_name = self.web_hemi_list[mesh_index]                
                return f'Clicked hemi: {hemi_name}, Vertex Index: {point_index}'
        
        # CLICKER FUNCTION (DISPLAYS MATPLOTLIB FIGURE, IF DEFINED)
        @app.callback(
            Output('mpl-figure-output', 'children'),
            [Input('mesh-plot', 'clickData')]
        )
        def display_mpl_figure(clickData):
            now = time.time()
            if (now - self.last_clicktime)<0.5:
                raise dash.exceptions.PreventUpdate      
            else: 
                self.last_clicktime = now            
            if clickData is not None:
                point_index = clickData['points'][0]['pointNumber']
                mesh_index = clickData['points'][0]['curveNumber']
                hemi_name = self.web_hemi_list[mesh_index]                                
                if hemi_name == 'rh':
                    point_index += self.n_vx['lh']
                mpl_figs = self.web_return_mpl_figs(point_index)                

                return mpl_figs
        app.scripts.config.serve_locally = True
        app.css.config.serve_locally = True
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
        self.dash_fig.update_layout(legend=dict(
            yanchor="top",
            y=0.01,
            xanchor="left",
            x=0.01
        ))        
        # Update plot sizing
        self.dash_fig.update_layout(
            autosize=True,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white",
            scene_camera=self.camera,
            uirevision='constant',  # Preserve camera settings
        )

        # Update 3D scene options
        self.dash_fig.update_scenes(
            aspectmode="manual"
        )

    def update_figure_with_color(self, disp_rgb):
        # Update vertexcolor for each mesh trace
        for i,ih in enumerate(self.web_hemi_list):
            self.dash_fig.data[i].update(vertexcolor=disp_rgb[ih])        

    def update_figure_inflate(self, inflate):
        this_vx_coord = []
        for hemi in self.web_hemi_list:
            # INTERPOLATE
            this_vx_coord.append(
                dag_mesh_interpolate(
                    coords1=self.mesh_info['pial'][hemi]['coords'],
                    coords2=self.mesh_info['inflated'][hemi]['coords'],
                    interp=inflate,
                )
            )
        # Update facecolor for each mesh trace
        for i in range(len(self.web_mesh)):
            self.dash_fig.data[i].update(
                x=this_vx_coord[i][:,0],
                y=this_vx_coord[i][:,1],
                z=this_vx_coord[i][:,2],
                )



        


    def web_return_mpl_figs(self, idx):
        '''
        Run through the mpl figure plotters...
        '''
        if not hasattr(self, 'mpl_fig_makers'):
            print('No mpl fig makers')
            return       
        figs = []
        for key in self.mpl_fig_makers.keys():
            this_fig = self.mpl_fig_makers[key]['func'](
                idx, **self.mpl_fig_makers[key]['kwargs'])
            this_fig.suptitle = f'{key} - {idx} - {this_fig._suptitle}'
            this_fig.tight_layout()
            this_fig.canvas.draw()
            # get title 
            # Save the Matplotlib figure as an SVG, making sure nothing is cut off
            mpl_fig_path = opj(self.output_dir, f'mpl_fig_{key}.svg')
            this_fig.savefig(mpl_fig_path)
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
            figs.append(svg4html)
        return html.Div(figs)


    def web_add_mpl_fig_maker(self, mpl_func, mpl_key, mpl_kwargs={}):
        '''
        Add a function to make a matplotlib figure
        '''
        if not hasattr(self, 'mpl_fig_makers'):
            self.mpl_fig_makers = {}
        self.mpl_fig_makers[mpl_key] = {}
        self.mpl_fig_makers[mpl_key]['func'] = mpl_func
        self.mpl_fig_makers[mpl_key]['kwargs'] = mpl_kwargs
        
    #endregion DASH FUNCTIONS