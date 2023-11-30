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
            out_dir         str             Where to put the mesh files which are made
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


    def open_surfaces_mlab(self):
        os.chdir(self.output_dir)
        mlab_cmd = 'meshlab '
        for key in self.ply_files.keys():
            mlab_cmd += f' lh.{key}.ply'
            mlab_cmd += f' rh.{key}.ply'
        print(mlab_cmd)
        os.system(mlab_cmd)


    def plotly_surface(self, data, **kwargs):
        '''
        Create a plotly surface plot 
        Arguments:
            See add_ply_surface, same principle...

        '''        
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        return_mesh_obj = kwargs.get('return_mesh_obj', False)
        # roi_list = kwargs.get('roi_list', [])
        if not isinstance(data, np.ndarray):
            print(f'Just creating curv file..')
            data = np.zeros(self.total_n_vx)            

        mesh_dict = self.plotly_return_mesh_dict(data, **kwargs)
        # Save the mesh files first as .asc, then .srf, then .obj
        # Then save them as .ply files, with the display rgb data for each voxel
        mesh3d_obj = []        
        for hemi in hemi_list:
            this_mesh3d = go.Mesh3d(
                **mesh_dict[hemi],
                # name='y',
                # showscale=True
                )
            mesh3d_obj.append(this_mesh3d)
        # if len(roi_list)>0:
        #     roi_obj = self.plotly_return_roi_obj(**kwargs)
        #     mesh3d_obj += roi_obj

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
            mesh_dict[hemi]['vertexcolor']=disp_rgb[hemi]
    
        return mesh_dict

    # def plotly_return_roi_obj(self, roi_list, **kwargs):
    #     '''
    #     Return a plotly object for a given roi
    #     '''
    #     if not isinstance(roi_list, list):
    #         roi_list = [roi_list]
    #     hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
    #     if not isinstance(hemi_list, list):
    #         hemi_list = [hemi_list]
    #     mesh_name = kwargs.get('mesh_name', 'inflated')

    #     roi_obj = []
    #     for hemi in hemi_list:
    #         for roi in roi_list:
    #             # Load roi index:
    #             roi_idx = dag_load_roi(
    #                 self.sub, 
    #                 roi=roi, 
    #                 fs_dir=self.fs_dir,
    #                 split_LR=True)[hemi]
    #             roi_idx = np.where(roi_idx)[0]
    #             # Find the border vertices
    #             border_vx_idx = dag_find_vx_border_on_sphere(roi_idx, self.mesh_info['sphere'][hemi])
    #             border_vx = np.column_stack((
    #                 self.mesh_info[mesh_name][hemi]['x'][border_vx_idx], 
    #                 self.mesh_info[mesh_name][hemi]['y'][border_vx_idx], 
    #                 self.mesh_info[mesh_name][hemi]['z'][border_vx_idx]))

    #             border_vx = np.column_stack((
    #                 self.mesh_info[mesh_name][hemi]['x'][roi_idx], 
    #                 self.mesh_info[mesh_name][hemi]['y'][roi_idx], 
    #                 self.mesh_info[mesh_name][hemi]['z'][roi_idx]))
                
    #             # print(border_vx)
    #             # return border_vx                
    #             # roi_coords = self.inflated[hemi]['coords'][roi_idx,:]
    #             # # Create a convex hull, to find the border
    #             # cvx_hull = ConvexHull(roi_coords)
    #             # # Get the border vertices
    #             # border_vx = roi_coords[cvx_hull.vertices,:]
    #             # Create a the line object for the border
    #             border_line = go.Scatter3d(
    #                 x=border_vx[:,0],
    #                 y=border_vx[:,1],
    #                 z=border_vx[:,2],
    #                 # mode=,
    #                 name=roi,
    #                 # marker=dict(
    #                 #     size=1,
    #                 #     color='red',
    #                 #     alpha=0.5,
    #                 # ),
    #                 # line=dict(
    #                 #     color='red',
    #                 #     width=2
    #                 # )
    #             )
    #             roi_obj.append(border_line)
    #     return roi_obj


# def dag_find_vx_border_on_sphere(vx_idx, sphere_mesh_info):
#     ''' Take advantage of the fact that the sphere is a convex hull'''
#     vertices = np.column_stack((sphere_mesh_info['x'], sphere_mesh_info['y'], sphere_mesh_info['z']))

#     # Extract the vertices in vx_list
#     clump_vertices = vertices[vx_idx, :]

#     # Compute the convex hull of the clump vertices
#     hull = ConvexHull(clump_vertices)

#     # Get the indices of the vertices on the convex hull
#     outer_vertex_indices = np.unique(hull.vertices)
#     return outer_vertex_indices


def dag_plotly_eye(el, az, zoom):
    x = zoom*np.cos(el)*np.cos(az)
    y = zoom*np.cos(el)*np.sin(az)
    z = zoom*np.sin(el)

    # fig.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=z)))
    return dict(eye=dict(x=x, y=y, z=z))



