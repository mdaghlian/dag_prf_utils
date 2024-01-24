import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
from scipy.spatial import ConvexHull
opj = os.path.join

from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *
from dag_prf_utils.mesh_format import *

import plotly.graph_objects as go
try: 
    from dash import Dash, dcc, html, Input, Output, State
    import dash
except:
    print('no dash..')

path_to_utils = os.path.abspath(os.path.dirname(__file__))

class GenMeshMaker(object):
    '''Used to make .ply files 
    One of many options for surface plotting. 
    '''
    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], output_dir=[]):
        
        self.sub = sub        
        self.fs_dir = fs_dir        # Where the freesurfer files are        
        self.sub_surf_dir = opj(fs_dir, sub, 'surf')
        self.sub_label_dir = opj(fs_dir, sub, 'label')
        self.output_dir = output_dir
        if isinstance(output_dir, str):
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)        
        n_vx = dag_load_nverts(self.sub, self.fs_dir)
        self.n_vx = {'lh':n_vx[0], 'rh':n_vx[1]}
        self.total_n_vx = sum(n_vx)
        self.ply_files = {}
        # Load the 'under surface' (i.e., the curvature)
        self.us_values = []
        for hemi in ['lh', 'rh']:
            with open(opj(self.sub_surf_dir,f'{hemi}.curv'), 'rb') as h_us:
                h_us.seek(15)
                this_us_vals = np.fromstring(h_us.read(), dtype='>f4').byteswap().newbyteorder()
                self.us_values.append(this_us_vals)
        # us cmap
        self.us_values = np.concatenate(self.us_values)    
        self.us_cmap = mpl.cm.__dict__['Greys'] # Always grey underneath
        self.us_norm = mpl.colors.Normalize()
        self.us_norm.vmin = -1 # Always -1,1 range...
        self.us_norm.vmax = 1  
        self.us_col = self.us_cmap(self.us_norm(self.us_values))
        # Load inflated surface
        self.mesh_info = {}
        mesh_list = ['inflated', 'sphere', 'pial', ] # 'white', 'orig']
        for mesh_name in mesh_list:
            self.mesh_info[mesh_name] = {}
            for hemi in ['lh', 'rh']:
                this_mesh_info = dag_read_fs_mesh(opj(self.sub_surf_dir, f'{hemi}.{mesh_name}'))
                # Put it into x,y,z, i,j,k format. Plus add offset to x
                self.mesh_info[mesh_name][hemi] = {}                                    
                self.mesh_info[mesh_name][hemi]['x']=this_mesh_info['coords'][:,0]
                self.mesh_info[mesh_name][hemi]['x'] += 50 if hemi=='rh' else -50
                self.mesh_info[mesh_name][hemi]['y']=this_mesh_info['coords'][:,1]
                self.mesh_info[mesh_name][hemi]['z']=this_mesh_info['coords'][:,2]
                self.mesh_info[mesh_name][hemi]['i']=this_mesh_info['faces'][:,0]
                self.mesh_info[mesh_name][hemi]['j']=this_mesh_info['faces'][:,1]
                self.mesh_info[mesh_name][hemi]['k']=this_mesh_info['faces'][:,2]


    def add_ply_surface(self, data, surf_name, **kwargs):
        '''
        Arguments:
            data            np.ndarray      What are we plotting on the surface? 1D array, same length as the number of vertices in subject surface.
            mesh_name      str              What kind of surface are we plotting on? e.g., pial, inflated...
                                                                Default: inflated
            under_surf      str             What is going underneath the data (e.g., what is the background)?
                                            default is curv. Could also be thick, (maybe smoothwm) 
        **kwargs:
            data_mask       bool array      Mask to hide certain values (e.g., where rsquared is not a good fit)
            data_alpha      np.ndarray      Alpha values for plotting. Where this is specified the undersurf is used instead
            surf_name       str             Name of your surface e.g., 'polar', 'rsq'
                                            *subject name is added to front of surf_name

            ow              bool            Overwrite? If surface with same name already exists, do you want to overwrite it?
                                            Default True
            *** COLOR
            cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                                Default: viridis
            vmin            float           Minimum value for colormap
                                                                Default: 10th percentile in data
            vmax            float           Max value for colormap
                                                                Default: 90th percentile in data
                                                                    
            
        '''
        overwrite = kwargs.get('ow', True)
        mesh_name = kwargs.get('mesh_name', 'inflated')
        print(f'File to be named: {surf_name}')        
        if (os.path.exists(opj(self.output_dir, f'lh.{surf_name}'))) & (not overwrite) :
            print(f'{surf_name} already exists for {self.sub}, not overwriting surf files...')
            return

        if (os.path.exists(opj(self.output_dir, f'lh.{surf_name}'))): 
            print(f'Overwriting: {surf_name} for {self.sub}')
        else:
            print(f'Writing: {surf_name} for {self.sub}')

        # Load mask for data to be plotted on surface
        data_mask = kwargs.get('data_mask', np.ones(self.total_n_vx, dtype=bool))
        data_alpha = kwargs.get('data_alpha', np.ones(self.total_n_vx))
        data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0
        if not isinstance(data, np.ndarray):
            print(f'Just creating curv file..')
            surf_name = 'curv'
            data = np.zeros(self.total_n_vx)
            data_alpha = np.zeros(self.total_n_vx)        
        
        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')    
        vmin = kwargs.get('vmin', np.nanmin(data[data_mask]))
        vmax = kwargs.get('vmax', np.nanmax(data[data_mask]))

        # Create rgb values mapping from data to cmap
        data_cmap = dag_get_cmap(cmap)
        data_norm = mpl.colors.Normalize()
        data_norm.vmin = vmin
        data_norm.vmax = vmax
        data_col = data_cmap(data_norm(data))


        display_rgb = (data_col * data_alpha[...,np.newaxis]) + \
            (self.us_col * (1-data_alpha[...,np.newaxis]))
        # Save the mesh files first as .asc, then .srf, then .obj
        # Then save them as .ply files, with the display rgb data for each voxel
        ply_file_2open = []
        for hemi in ['lh.', 'rh.']:
            mesh_name_file = opj(self.sub_surf_dir, f'{hemi}{mesh_name}')
            asc_surf_file = opj(self.output_dir,f'{hemi}{surf_name}.asc')
            srf_surf_file = opj(self.output_dir,f'{hemi}{surf_name}.srf')
            # 
            obj_surf_file = opj(self.output_dir,f'{hemi}{surf_name}.obj')    
            rgb_surf_file = opj(self.output_dir,f'{hemi}{surf_name}_rgb.csv')    
            #
            ply_surf_file = opj(self.output_dir,f'{hemi}{surf_name}.ply')
            ply_file_2open.append(ply_surf_file)
            # [1] Make asc file using freesurfer mris_convert command:
            os.system(f'mris_convert {mesh_name_file} {asc_surf_file}')
            # [2] Rename .asc as .srf file to avoid ambiguity (using "brainders" conversion tool)
            os.system(f'mv {asc_surf_file} {srf_surf_file}')
            
            # *** EXTRA BITS... ****
            # ***> keeping the option because maybe some people like .obj files?
            # [*] Use brainder script to create .obj file    
            # os.system(f'{srf2obj_path} {srf_surf_file} > {obj_surf_file}')
            # ^^^  ^^^

            # [4] Use my script to write a ply file...
            if hemi=='lh.':
                ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[:self.n_vx['lh'],:], hemi=hemi, values=data, **kwargs) # lh
            else:
                ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[self.n_vx['lh']:,:],hemi=hemi, values=data, **kwargs) # rh
            # Now save the ply file
            ply_file_2write = open(ply_surf_file, "w")
            ply_file_2write.write(ply_str)
            ply_file_2write.close()       

            # Remove the srf file
            os.system(f'rm {srf_surf_file}')

            # # Now save the rgb csv file
            # rgb_file_2write = open(rgb_surf_file, "w")
            # rgb_file_2write.write(rgb_str)
            # rgb_file_2write.close()       
            
        # Return list of .ply files to open...
        self.ply_files[surf_name] = [
            ply_file_2open
        ]

    def ply_add_roi(self, roi_list, **kwargs):
        '''
        Return a plotly object for a given roi
        '''
        roi_list_excl = kwargs.get('roi_list_excl', None)
        lr_roi_list = dag_get_lr_roi_list(self.sub_label_dir, roi_list, roi_list_excl)        
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])        
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]            
        mesh_name = kwargs.get('mesh_name', 'inflated')
        # marker_kwargs = kwargs.get()
        # roi_cols = dag_get_col_vals(
        #     np.arange(len(roi_list)),
        #     vmin = -1, vmax=7, cmap='jet'
        # )
        # roi_col = 'w'
        roi_outline = {}
        roi_outline['lh'] = np.zeros(self.n_vx['lh'])
        roi_outline['rh'] = np.zeros(self.n_vx['rh'])
        for hemi in hemi_list:
            for i_roi,roi in enumerate(lr_roi_list[hemi]):
                # Load roi index:

                roi_bool = dag_load_roi(self.sub, roi, fs_dir=self.fs_dir, split_LR=True)[hemi]
                border_vx_list = dag_get_roi_border_vx_in_order(
                    roi_bool=roi_bool, 
                    mesh_info=self.mesh_info[mesh_name][hemi], 
                    return_coords=False,
                    )
                
                # If more than one closed path (e.g., v2v and v2d)...
                for border_vx in border_vx_list:
                    roi_outline[hemi][border_vx] += 1
        roi_outline_full = np.hstack([roi_outline['lh'], roi_outline['rh']])
        self.add_ply_surface(
            surf_name='rois_outline',
            data=roi_outline_full!=0,
            data_mask=roi_outline_full!=0,
            data_cmap='greys',
            **kwargs,
        )

        return 

    def open_surfaces_mlab(self):
        os.chdir(self.output_dir)
        # mlab_cmd = 'meshlab '
        mlab_cmd = '/data1/projects/dumoulinlab/Lab_members/Marcus/programs/MeshLab2022.02-linux/AppRun '
        for key in self.ply_files.keys():            
            mlab_cmd += f' lh.{key}.ply'
            mlab_cmd += f' rh.{key}.ply'

        os.system(mlab_cmd)


    def plotly_surface(self, data=None, **kwargs):
        '''
        Create a plotly surface plot 
        Arguments:
            See add_ply_surface, same principle...

        '''        
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        return_mesh_obj = kwargs.get('return_mesh_obj', False)
        roi_list = kwargs.get('roi_list', [])
        if data is None: #not isinstance(data, np.ndarray):
            print(f'Just creating curv file..')
            data = np.zeros(self.total_n_vx)            

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
            roi_obj = self.plotly_return_roi_obj(**kwargs)
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
    
    def plotly_return_mesh_dict(self, data, **kwargs):
        do_intensity = kwargs.get('do_intensity', False)
        do_vertexcolor = kwargs.get('do_vertexcolor', True)
        mesh_name = kwargs.get('mesh_name', 'inflated')
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        
        # Load mask for data to be plotted on surface
        data_mask = kwargs.get('data_mask', np.ones(self.total_n_vx, dtype=bool))
        data_alpha = kwargs.get('data_alpha', np.ones(self.total_n_vx))
        data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0
        if not isinstance(data, np.ndarray):
            print(f'Just creating curv file..')
            # surf_name = 'curv'
            data = np.zeros(self.total_n_vx)
            data_alpha = np.zeros(self.total_n_vx)        
        
        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', 'viridis')    
        vmin = kwargs.get('vmin', np.nanmin(data[data_mask]))
        vmax = kwargs.get('vmax', np.nanmax(data[data_mask]))

        # Create rgb values mapping from data to cmap
        data_cmap = dag_get_cmap(cmap)
        data_norm = mpl.colors.Normalize()
        data_norm.vmin = vmin
        data_norm.vmax = vmax
        data_col = data_cmap(data_norm(data))


        display_rgb = (data_col * data_alpha[...,np.newaxis]) + \
            (self.us_col * (1-data_alpha[...,np.newaxis]))        
        
        disp_rgb = {
            'lh' : display_rgb[:self.n_vx['lh'],:],
            'rh' : display_rgb[self.n_vx['lh']:,:],
        }

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

                border_vx_list = dag_get_roi_border_vx_in_order(
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
        kwargs['hemi_list'] = self.web_hemi_list

        self.web_mesh = self.plotly_surface(
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
        this_mesh = self.plotly_return_mesh_dict(**kwargs)
        for ih in this_mesh.keys():
            self.web_vxcol[ih][vx_col_name] = this_mesh[ih]['vertexcolor']
        self.web_vxcol_list.append(vx_col_name)

    def web_launch(self):
        '''
        Return a figure! 
        '''
        fig = go.Figure()
        fig.add_trace(self.web_mesh[0])
        fig.add_trace(self.web_mesh[1])

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

    def web_launch_with_dash(self, fig):
        # Create a Dash app
        app = Dash(__name__)

        # Define the layout of the app
        app.layout = html.Div([
            dcc.Graph(id='graph', figure=fig), #go.Figure(mesh)),
            html.Label('eye-x'),
            dcc.Input(id='eye-x', type='number', value=0),
            html.Label('eye-y'),
            dcc.Input(id='eye-y', type='number', value=1),
            html.Label('eye-z'),
            dcc.Input(id='eye-z', type='number', value=1)    
        ])

        # Set initial values for eye position
        initial_eye_x, initial_eye_y, initial_eye_z = 0, 1, 1

        fig.update_layout(
            # scene=dict(aspectmode="cube"), 
            scene_camera=dict(eye=dict(x=initial_eye_x, y=initial_eye_y, z=initial_eye_z))
            )

        # Define a callback to update the 3D scene when the camera position changes
        @app.callback(
            [
                Output('eye-x', 'value'),
                Output('eye-y', 'value'),
                Output('eye-z', 'value')
            ],
            [
                Input('graph', 'relayoutData')
            ],
            [
                State('eye-x', 'value'),
                State('eye-y', 'value'),
                State('eye-z', 'value')
            ]            
        )

        def update_input_fields(relayoutData, eye_x, eye_y, eye_z):
            ctx = dash.callback_context
            if not ctx.triggered_id:
                input_id = 'No Input'
            else:
                input_id = ctx.triggered_id
            print(f'Triggered by {input_id}')
            
            if 'scene.camera' in relayoutData:
                camera_data = relayoutData['scene.camera']
                new_eye_x, new_eye_y, new_eye_z = camera_data['eye']['x'], camera_data['eye']['y'], camera_data['eye']['z']
                
                # Check if the changes are significant
                if (new_eye_x, new_eye_y, new_eye_z) != (eye_x, eye_y, eye_z):
                    return new_eye_x, new_eye_y, new_eye_z
            # If the keys are not present or the changes are not significant, return the current values
            return eye_x, eye_y, eye_z        
        # def update_input_fields(relayoutData):
        #     if 'scene.camera' in relayoutData:
        #         camera_data = relayoutData['scene.camera']
        #         eye_x, eye_y, eye_z = camera_data['eye']['x'], camera_data['eye']['y'], camera_data['eye']['z']
        #         print(camera_data)
        #         # el = 
        #         return eye_x, eye_y, eye_z
        #     else:
        #         # If the keys are not present, return the initial values
        #         return initial_eye_x, initial_eye_y, initial_eye_z

        # Define a callback to update the 3D scene when the input values change
        @app.callback(
            Output('graph', 'figure'),
            [
                Input('eye-x', 'value'),
                Input('eye-y', 'value'),
                Input('eye-z', 'value'),
            ]
        )
        def update_eye(eye_x, eye_y, eye_z):
            # eye_x,eye_y,eye_z = dag_plotly_eye(eye_x, eye_y, eye_z)
            fig.update_layout(
                # scene=dict(aspectmode="cube"),
                scene_camera=dict(eye=dict(x=eye_x, y=eye_y, z=eye_z))
            )
            return fig

        # Run the app
        app.run_server(
            host='0.0.0.0',
            port=8082,
            open_browser=True,
        )

    # def add_pyctx_surface()


def dag_plotly_eye(el, az, zoom):
    # x = zoom*np.cos(np.radians(el))*np.cos(np.radians(az))
    # y = zoom*np.cos(np.radians(el))*np.sin(np.radians(az))
    # z = zoom*np.sin(np.radians(el))

    x = zoom*np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az))
    y = zoom*np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az))
    z = zoom*np.sin(np.deg2rad(el))    
    return x,y,z
    # fig.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=z)))
    # return dict(
    #     eye=dict(x=x, y=y, z=z),        
    #     )

def dag_get_roi_border_vx(roi_bool, mesh_info, return_coords=False):
    '''
    Find those vx which are on a border... 
    '''
    roi_idx = np.where(roi_bool)[0]
    in_face_x = {}
    for face_x in ['i', 'j', 'k']:
        in_face_x[face_x] = np.isin(mesh_info[face_x], roi_idx) * 1.0
    border_faces = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) >0
    border_faces &= (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) <= 2
    border_vx = []
    for face_x in ['i', 'j', 'k']:                    
        border_vx.append(
            mesh_info[face_x][(border_faces * in_face_x[face_x])==1]
        )
    border_vx = np.concatenate(border_vx)
    border_vx = np.unique(border_vx)
    if not return_coords:
        return border_vx
    border_vx_coords = [
        mesh_info['x'][border_vx], 
        mesh_info['y'][border_vx], 
        mesh_info['z'][border_vx],                    
    ]
    return border_vx_coords

def dag_get_roi_border_vx_in_order(roi_bool, mesh_info, return_coords=False):
    outer_edge_list = dag_get_roi_border_edge(roi_bool, mesh_info)
    border_vx = dag_order_edges(outer_edge_list)
    if not return_coords:
        return border_vx
    border_vx = sum(border_vx, []) # flatten list
    border_vx_coords = [
        mesh_info['x'][border_vx], 
        mesh_info['y'][border_vx], 
        mesh_info['z'][border_vx],                    
    ]
    return border_vx_coords



def dag_get_roi_border_edge(roi_bool, mesh_info):
    '''
    Find those vx which are on a border... 
    '''
    roi_idx = np.where(roi_bool)[0]
    in_face_x = {}
    for face_x in ['i', 'j', 'k']:
        in_face_x[face_x] = np.isin(mesh_info[face_x], roi_idx) * 1.0
    f_w_outer_edge = (in_face_x['i'] + in_face_x['j'] + in_face_x['k']) == 2
    f_w_outer_edge = np.where(f_w_outer_edge)[0]
    ij_faces_match = (in_face_x['i'][f_w_outer_edge] + in_face_x['j'][f_w_outer_edge])==2
    jk_faces_match = (in_face_x['j'][f_w_outer_edge] + in_face_x['k'][f_w_outer_edge])==2
    ki_faces_match = (in_face_x['k'][f_w_outer_edge] + in_face_x['i'][f_w_outer_edge])==2

    ij_faces_match = f_w_outer_edge[ij_faces_match]
    jk_faces_match = f_w_outer_edge[jk_faces_match]
    ki_faces_match = f_w_outer_edge[ki_faces_match]
    
    outer_edge_list = []
    # ij
    outer_edge_list.append(
        np.vstack([mesh_info['i'][ij_faces_match],mesh_info['j'][ij_faces_match]]),
    )
    # jk
    outer_edge_list.append(
        np.vstack([mesh_info['j'][jk_faces_match],mesh_info['k'][jk_faces_match]]),
    )
    # ki     
    outer_edge_list.append(
        np.vstack([mesh_info['k'][ki_faces_match],mesh_info['i'][ki_faces_match]]),
    )    
    # for face in ij_faces_match:
    #     outer_edge_list.append([
    #         mesh_info['i'][face]
    #     ])
    outer_edge_list = np.hstack(outer_edge_list).T
    return outer_edge_list

def dag_order_edges(edges):
    unique_vx = list(np.unique(edges.flatten()))

    # Step 1: Create an adjacency list
    adjacency_list = {}
    for i_edge in range(edges.shape[0]):
        u, v = edges[i_edge,0],edges[i_edge,1]
        if u not in adjacency_list:
            adjacency_list[u] = []
        if v not in adjacency_list:
            adjacency_list[v] = []
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)

    # Step 2: Choose a starting vertex
    start_vertex = list(adjacency_list.keys())[0]

    # There may be more than one closed path... (e.g., V2v V2d)
    set_unordered = set(unique_vx)
    ordered_list_multi = []
    set_ordered = set(sum(ordered_list_multi, [])) # flatten list and make it a set
    missing_vx = list(set_unordered - set_ordered)
    while len(missing_vx)!=0:
        start_vertex = missing_vx[0]
        ordered_list = dag_traverse_graph(start_vertex, adjacency_list)
        ordered_list_multi.append(ordered_list)
        set_ordered = set(sum(ordered_list_multi, [])) # flatten list and make it a set
        missing_vx = list(set_unordered - set_ordered)

    # OPTION: Convert ordered list to edge list
    # ordered_edges = [(ordered_list[i], ordered_list[(i + 1) % len(ordered_list)]) for i in range(len(ordered_list))]


    return ordered_list_multi#np.array(ordered_list)


def dag_traverse_graph(start_vertex, adjacency_list):
    # Step 3 and 4: Traverse the graph and build the ordered list
    ordered_list = []
    stack = [start_vertex]
    visited = set()

    while stack:
        current_vertex = stack.pop()
        if current_vertex not in visited:
            visited.add(current_vertex)
            ordered_list.append(current_vertex)

            # Visit neighbors in reverse order to maintain loop direction
            stack.extend(reversed(adjacency_list[current_vertex]))
    return ordered_list
# ************************************************************************
def dag_3dcart2pol(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    
    return r, theta, phi

def dag_sort_coords(x,y,z):
    mx,my,mz = x.mean(),y.mean(),z.mean()
    r, theta, phi = dag_3dcart2pol(x-mx, y-my, z-mz)
    # set origin at centre 
    sort_idx = np.argsort(theta)
    return sort_idx # x[sort_idx],y[sort_idx], z[sort_idx]
    

def dag_mesh_slice(mesh_info, **kwargs):
    vx_to_remove = np.zeros_like(mesh_info['x'], dtype=bool)
    for b in ['min', 'max']:
        for c in ['x', 'y', 'z']:
            this_bc = kwargs.get(f'{b}_{c}', None)
            if this_bc is None:
                continue
            if b=='min':
                vx_to_remove |= mesh_info[c]<this_bc
            elif b=='max':
                vx_to_remove |= mesh_info[c]>this_bc
    if vx_to_remove.sum()==0:
        print('No vx to mask')
        return mesh_info
    elif vx_to_remove.all():
        print('Removing everything...')
        return None
    else:
        print(f'{vx_to_remove.sum()} to remove')

    # Create a mapping from old vx to new ones
    
    old_vx_idx = np.arange(mesh_info['x'].shape[0])
    old_vx_idx = old_vx_idx[~vx_to_remove]
    new_vx_idx = np.arange(old_vx_idx.shape[0])
    vx_map = dict(zip(old_vx_idx, new_vx_idx))

    
    # face mask 
    face_to_remove = np.zeros_like(mesh_info['i'], dtype=bool)
    vx_to_remove_idx = np.where(vx_to_remove)[0]
    for c in ['i', 'j', 'k']:
        face_to_remove |= np.isin(mesh_info[c], vx_to_remove_idx)        

    new_mesh_info = {}
    # Sort out vertices
    for c in mesh_info.keys():
        if c in ['i', 'j', 'k']:
            continue
        new_mesh_info[c] = mesh_info[c][~vx_to_remove].copy()

    # Sort out faces 
    for c in ['i', 'j', 'k']:
        # 1 remove faces 
        face_w_old_idx = mesh_info[c][~face_to_remove].copy()
        # 2 fix the ids.. 
        new_mesh_info[c] = np.array([vx_map[old_idx] for old_idx in face_w_old_idx])

    return new_mesh_info



            
def dag_find_vx_border_on_sphere(vx_idx, sphere_mesh_info):
    ''' Take advantage of the fact that the sphere is a convex hull'''
    vertices = np.column_stack((sphere_mesh_info['x'], sphere_mesh_info['y'], sphere_mesh_info['z']))

    # Extract the vertices in vx_list
    clump_vertices = vertices[vx_idx, :]

    # Compute the convex hull of the clump vertices
    hull = ConvexHull(clump_vertices)

    # Get the indices of the vertices on the convex hull
    outer_vertex_indices = np.unique(hull.vertices)
    return outer_vertex_indices




def dag_get_lr_roi_list(sub_label_dir, roi_list, roi_list_excl):
    '''
    Sort out the list of rois... per hemi
    Include make it capable of dealing with missing rois
    And fining matching ones
    '''
    sorted_roi_list = {
        'lh':[],
        'rh':[],
    }
    if not isinstance(roi_list, list):
        roi_list = [roi_list]
    for roi_name in roi_list:
        for hemi in ['lh', 'rh']:
            this_roi_path = dag_find_file_in_folder(
                filt=[roi_name, hemi],
                path=sub_label_dir,
                recursive=True,
                exclude=['._', '.thresh'] + list(roi_list_excl),
                return_msg=None,
                )
            if this_roi_path is not None:
                if isinstance(this_roi_path, list):
                    sorted_roi_list[hemi] += this_roi_path
                else:
                    sorted_roi_list[hemi].append(this_roi_path)

    return sorted_roi_list
