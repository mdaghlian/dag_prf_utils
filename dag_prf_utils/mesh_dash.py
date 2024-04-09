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



class MeshShow():
    def __init__(self):
        '''Meshes are the plotly stuff'''
        self.web_mesh = [] # list of go.Mesh3d
        self.web_vxcol = []  # list of go.Mesh3d
        self.web_val1 = []
        self.web_val2
    
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
            ),
            html.Div(id='vertex-index-output')            
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
