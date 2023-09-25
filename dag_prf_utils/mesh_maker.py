import numpy as np  
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os
opj = os.path.join
# try:
#     from nibabel.freesurfer.io import write_morph_data
# except ImportError:
#     raise ImportError('Error importing nibabel... Not a problem unless you want to use FSMaker')
from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *


path_to_utils = os.path.abspath(os.path.dirname(__file__))

class FSMaker(object):
    '''Used to make a freesurfer file, and view a surface in freesurfer. 
    One of many options for surface plotting. 
    Will create a curv file in subjects freesurfer dir, and load it a specific colormap 
    saved as the relevant command
    '''
    def __init__(self, sub, fs_dir):
        
        self.sub = sub        
        self.fs_dir = fs_dir        # Where the freesurfer files are        
        self.sub_surf_dir = opj(fs_dir, sub, 'surf')
        self.sub_label_dir = opj(fs_dir, sub, 'label')
        self.custom_surf_dir = opj(self.sub_surf_dir, 'custom')
        if not os.path.exists(self.custom_surf_dir):
            os.mkdir(self.custom_surf_dir)        
        n_vx = dag_load_nverts(self.sub, self.fs_dir)
        self.n_vx = {'lh':n_vx[0], 'rh':n_vx[1]}
        self.overlay_str = {}
        self.open_surf_cmds = {}
        self.surf_list = []

    def add_surface(self, data, surf_name, **kwargs):
        '''
        See dag_calculate_rgb_vals...
        data            np.ndarray      What are we plotting...
        surf_name       str             what are we calling the file

        '''

        data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
        # Load colormap properties: (cmap, vmin, vmax)
        vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
        # Get the overlay custom str and overlay to save...
        overlay_custom_str, overlay_to_save = dag_make_overlay_str(masked_data=data[data_mask], **kwargs)
        
        data_masked = np.zeros_like(data, dtype=float)
        data_masked[data_mask] = data[data_mask]
        exclude_min_val = vmin - 1
        data_masked[~data_mask] = exclude_min_val

        # SAVE masked data AS A CURVE FILE
        lh_masked_param = data_masked[:self.n_vx['lh']]
        rh_masked_param = data_masked[self.n_vx['lh']:]

        # now save results as a curve file
        print(f'Saving {surf_name} in {self.custom_surf_dir}')

        n_faces = dag_load_nfaces(self.sub, self.fs_dir)
        dag_write_curv(
            fn=opj(self.custom_surf_dir, f'lh.{surf_name}'), 
            curv=lh_masked_param, 
            fnum=n_faces[0])
        dag_write_curv(
            fn=opj(self.custom_surf_dir, f'rh.{surf_name}'), 
            curv=rh_masked_param, 
            fnum=n_faces[1])        
        # write_morph_data(opj(self.custom_surf_dir, f'lh.{surf_name}'),lh_masked_param)
        # write_morph_data(opj(self.custom_surf_dir, f'rh.{surf_name}'),rh_masked_param)        
        
        dag_str2file(filename=opj(self.custom_surf_dir, f'{surf_name}_overlay'),txt=overlay_to_save)
        self.overlay_str[surf_name] = overlay_custom_str
        self.surf_list.append(surf_name)
    def open_fs_surface(self, surf_name=None, **kwargs):
        # surf name - which surface to load...
        
        os.chdir(self.sub_surf_dir) # move to freeview dir        
        fs_cmd = self.write_fs_cmd(surf_name=surf_name, **kwargs)
        # self.save_fs_cmd(surf_name, **kwargs)
        os.system(fs_cmd)        

    def save_fs_cmd(self, surf_name=None, **kwargs):
        cmd_name = kwargs.get('cmd_name', f'{surf_name}_cmd.txt')
        print(f'Custom overlay string saved here: ({opj(self.custom_surf_dir, cmd_name)})')
        fs_cmd = self.write_fs_cmd(surf_name=surf_name, **kwargs)
        dag_str2file(filename=opj(self.custom_surf_dir, cmd_name),txt=fs_cmd)
        
    def write_fs_cmd(self, surf_name=None, **kwargs):
        '''
        Write the bash command to open the specific surface with the overlay

        **kwargs 
        surf_name       which surface(s) to open (of the custom ones you have made)
        mesh_list       which mesh(es) to plot the surface info on (e.g., inflated, pial...)
        hemi_list       which hemispheres to load
        roi_list        which roi outlines to load
        roi_mask        mask by roi?

        -> Screen shot stuff
        do_scrn_shot    bool            take a screenshot of the surface when it is loaded?
        azimuth         float           camera angle(0-360) Default: 0
        zoom            float           camera zoom         Default: 1.00
        elevation       float           camera angle(0-360) Default: 0
        roll            float           camera angle(0-360) Default: 0        
        '''
        mesh_list = kwargs.get('mesh_list', ['inflated'])
        hemi_list = kwargs.get('hemi_list', ['lh', 'rh'])
        roi_list = kwargs.get('roi_list',None)
        roi_col_spec = kwargs.get('roi_col_spec', None)
        roi_mask = kwargs.get('roi_mask', None)
        # *** CAMERA ANGLE ***
        cam_azimuth     = kwargs.get('azimuth', 0)
        cam_zoom        = kwargs.get('zoom', 1)
        cam_elevation   = kwargs.get('elevation', 0)
        cam_roll        = kwargs.get('roll', 0)
        do_scrn_shot    = kwargs.get('do_scrn_shot', False)
        scr_shot_file   = kwargs.get('scr_shot_file', None)
        # *** COLOR BAR ***
        do_col_bar  = kwargs.get('do_col_bar', True)

        do_surf = True
        if surf_name is None:
            do_surf = False

        if not isinstance(mesh_list, list):
            mesh_list = [mesh_list]
        if not isinstance(hemi_list, list):
            hemi_list = [hemi_list]
        if not isinstance(surf_name, list):
            surf_name = [surf_name]        
        if (roi_list is not None) and (not isinstance(roi_list, list)):
            roi_list = [roi_list]

        if do_scrn_shot:         
            if scr_shot_file is None:
                # Not specified -save in custom surf dir
                scr_shot_file = opj(self.custom_surf_dir, f'{surf_name[0]}_az{cam_azimuth}_z{cam_zoom}_e{cam_elevation}_r{cam_roll}')
            if os.path.isdir(scr_shot_file):
                # Folder specified, add the name...
                scr_shot_flag = f"--ss {opj(scr_shot_file, surf_name[0])}"
            else:
                # Specific file specified
                scr_shot_flag = f"--ss {scr_shot_file}"
        else:
            scr_shot_flag = ""


        if do_col_bar:
            col_bar_flag = '--colorscale'
        else:
            col_bar_flag = ''

        fs_cmd = f'freeview -f '
        for mesh in mesh_list:
            for this_hemi in hemi_list:
                fs_cmd += f' {this_hemi}.{mesh}'
                if roi_list is not None:
                    for i_roi, roi in enumerate(roi_list):
                        if roi_col_spec is None:
                            roi_col = dag_get_col_vals(i_roi, 'jet', 0, len(roi_list))
                            roi_col = f'{int(roi_col[0]*255)},{int(roi_col[1]*255)},{int(roi_col[2]*255)}'
                        else:
                            roi_col = roi_col_spec
                        this_roi_path = self.get_roi_file(roi, this_hemi)
                        fs_cmd += f':label={this_roi_path}:label_outline=True:label_visible=True:label_color={roi_col}' # false...
                if do_surf:
                    for this_surf_name in surf_name:
                        # this_surf_path = opj(self.custom_surf_dir, f'{this_hemi}.{this_surf_name}')                
                        this_surf_path = self.get_surf_path(this_hemi=this_hemi, this_surf_name=this_surf_name)
                        this_overlay_str = self.get_overlay_str(this_surf_name, **kwargs)
                        fs_cmd += f':overlay={this_surf_path}:{this_overlay_str}'                        
                        if roi_mask is not None:
                            this_roi_path = self.get_roi_file(roi, this_hemi)
                            fs_cmd += f':overlay_mask={this_roi_path}'
        fs_cmd +=  f' --camera Azimuth {cam_azimuth} Zoom {cam_zoom} Elevation {cam_elevation} Roll {cam_roll} '
        fs_cmd += f'{col_bar_flag} {scr_shot_flag}'
        return fs_cmd 

    def get_roi_file(self, roi_name, hemi):
        roi = dag_find_file_in_folder(
            filt=[roi_name, hemi],
            path=self.sub_label_dir,
            recursive=True,
            exclude=['._', '.thresh']
            )
        if isinstance(roi, list):
            roi = roi[0]
        roi_path = opj(self.sub_label_dir, roi)
        return roi_path
    def get_surf_path(self, this_hemi, this_surf_name):
        # [1] Check if it exists in the custom surf dir
        this_surf_path = opj(self.custom_surf_dir, f'{this_hemi}.{this_surf_name}')
        if os.path.exists(this_surf_path):
            pass
        else: 
            # Now we need to look a bit deeper
            this_surf_path = dag_find_file_in_folder(
                filt=[this_hemi, f'.{this_surf_name}'],
                exclude=['pial'],
                path=self.sub_surf_dir,
                recursive=True,
                return_msg=None,
            )

        return this_surf_path
    
    def get_overlay_str(self, surf_name, overlay_cmap=None, **kwargs):
        if overlay_cmap is not None:
            overlay_str, _ = dag_make_overlay_str(cmap=overlay_cmap, **kwargs)
            return overlay_str
        if surf_name in self.overlay_str.keys():
            overlay_str = self.overlay_str[surf_name]
            return overlay_str

        # Not found in struct: check the custom surf dir...
        overlay_str = ':overlay_custom='            
        print(f'{surf_name} not in dict')
        print(f'Checking custom surf dir')
        overlay_str_file = dag_find_file_in_folder(
            filt=[surf_name, 'overlay'],
            path=self.sub_surf_dir,
            recursive=True,
            return_msg=None,
        )

        if overlay_str_file is None:
            overlay_str = ''#'greyscale :colormap=grayscale' #  grayscale/lut/heat/jet/gecolor/nih/pet/binary
        elif isinstance(overlay_str_file, list):
            overlay_str += overlay_str_file[0]

        return overlay_str
    
def dag_write_curv(fn, curv, fnum):
    ''' Adapted from https://github.com/simnibs/simnibs
    
    Writes a freesurfer .curv file

    Parameters
    ------------
    fn: str
        File name to be written
    curv: ndaray
        Data array to be written
    fnum: int
        Number of faces in the mesh
    '''
    def write_3byte_integer(f, n):
        b1 = struct.pack('B', (n >> 16) & 255)
        b2 = struct.pack('B', (n >> 8) & 255)
        b3 = struct.pack('B', (n & 255))
        f.write(b1)
        f.write(b2)
        f.write(b3)


    NEW_VERSION_MAGIC_NUMBER = 16777215
    vnum = len(curv)
    with open(fn, 'wb') as f:
        write_3byte_integer(f, NEW_VERSION_MAGIC_NUMBER)
        f.write(struct.pack(">i", int(vnum)))
        f.write(struct.pack('>i', int(fnum)))
        f.write(struct.pack('>i', 1))
        f.write(curv.astype('>f').tobytes())

def dag_fs_to_ply(sub, data, fs_dir, mesh_name='inflated', out_dir=None, under_surf='curv', **kwargs):
    '''
    fs_to_ply:
        Create surface files for a subject, and a specific parameter.                        
        
    Arguments:
        sub             str             e.g. 'sub-01': Name of subject in freesurfer file
        data            np.ndarray      What are we plotting on the surface? 1D array, same length as the number of vertices in subject surface.
        fs_dir          str             Location of the Freesurfer folder
        mesh_name      str              What kind of surface are we plotting on? e.g., pial, inflated...
                                                            Default: inflated
        under_surf      str             What is going underneath the data (e.g., what is the background)?
                                        default is curv. Could also be thick, (maybe smoothwm) 
        out_dir         str             Where to put the mesh files which are made
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
                                                                
        return_ply_file bool            Return the ply files which have been made
        
    '''
    return_ply_file = kwargs.get('return_ply_file', False)    
    # Get path to subjects surface file
    path_to_sub_surf = opj(fs_dir, sub, 'surf')
    # Check name for surface:
    surf_name = kwargs.get('surf_name', None)
    if surf_name==None:
        print('surf_name not specified, using sub+date')
        surf_name = sub + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M')
    else:
        surf_name = sub + '_' + surf_name + '_' + mesh_name
            
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    overwrite = kwargs.get('ow', True)
    print(f'File to be named: {surf_name}')        
    if (os.path.exists(opj(out_dir, f'lh.{surf_name}'))) & (not overwrite) :
        print(f'{surf_name} already exists for {sub}, not overwriting surf files...')
        return

    if (os.path.exists(opj(path_to_sub_surf, f'lh.{surf_name}'))): 
        print(f'Overwriting: {surf_name} for {sub}')
    else:
        print(f'Writing: {surf_name} for {sub}')

    # load the undersurf file values, & get number of vx in each hemisphere
    n_hemi_vx = []
    us_values = []
    for ih in ['lh.', 'rh.']:
        with open(opj(path_to_sub_surf,f'{ih}{under_surf}'), 'rb') as h_us:
            h_us.seek(15)
            this_us_vals = np.fromstring(h_us.read(), dtype='>f4').byteswap().newbyteorder()
            us_values.append(this_us_vals)
            n_hemi_vx.append(this_us_vals.shape[0])    
    n_vx = np.sum(n_hemi_vx)
    us_values = np.concatenate(us_values)    
    # Load mask for data to be plotted on surface
    data_mask = kwargs.get('data_mask', np.ones(n_vx, dtype=bool))
    data_alpha = kwargs.get('data_alpha', np.ones(n_vx))
    data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0
    if not isinstance(data, np.ndarray):
        print(f'Just creating {under_surf} file..')
        surf_name = under_surf
        data = np.zeros(n_vx)
        data_alpha = np.zeros(n_vx)        
    
    # Load colormap properties: (cmap, vmin, vmax)
    cmap = kwargs.get('cmap', 'viridis')    
    vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
    vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))


    # Create rgb values mapping from data to cmap
    data_cmap = dag_get_cmap(cmap)
    # data_cmap = mpl.cm.__dict__[cmap] 
    data_norm = mpl.colors.Normalize()
    data_norm.vmin = vmin
    data_norm.vmax = vmax
    data_col = data_cmap(data_norm(data))
    
    # CHANGE FOR NAN
    # data[~data_mask] = 0

    # Create rgb values mapping from under_surf to grey cmap
    us_cmap = mpl.cm.__dict__['Greys'] # Always grey underneath
    us_norm = mpl.colors.Normalize()
    if under_surf=='curv':
        us_norm.vmin = -1 # Always -1,1 range...
        us_norm.vmax = 1  
    elif under_surf=='thickness':        
        us_norm.vmin = 0 # Always -1,1 range...
        us_norm.vmax = 5          
    us_col = us_cmap(us_norm(us_values))


    display_rgb = (data_col * data_alpha[...,np.newaxis]) + \
        (us_col * (1-data_alpha[...,np.newaxis]))
    # Save the mesh files first as .asc, then .srf, then .obj
    # Then save them as .ply files, with the display rgb data for each voxel
    ply_file_2open = []
    for ih in ['lh.', 'rh.']:
        mesh_name_file = opj(path_to_sub_surf, f'{ih}{mesh_name}')
        asc_surf_file = opj(out_dir,f'{ih}{surf_name}.asc')
        srf_surf_file = opj(out_dir,f'{ih}{surf_name}.srf')
        # 
        obj_surf_file = opj(out_dir,f'{ih}{surf_name}.obj')    
        rgb_surf_file = opj(out_dir,f'{ih}{surf_name}_rgb.csv')    
        #
        ply_surf_file = opj(out_dir,f'{ih}{surf_name}.ply')
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
        if ih=='lh.':
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[:n_hemi_vx[0],:]) # lh
            ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[:n_hemi_vx[0],:], hemi=ih, values=data, **kwargs) # lh
        else:
            # ply_str = obj_to_ply(obj_surf_file, display_rgb[n_hemi_vx[0]:,:]) # rh
            ply_str, rgb_str = dag_srf_to_ply(srf_surf_file, display_rgb[n_hemi_vx[0]:,:],hemi=ih, values=data, **kwargs) # rh
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
    if return_ply_file:
        return ply_file_2open

def dag_srf_to_ply(srf_file, rgb_vals=None, hemi=None, values=None, incl_rgb=True, **kwargs):
    '''
    dag_srf_to_ply
    Convert srf file to .ply
    
    '''
    x_offset = kwargs.get('x_offset', None)
    
    if not isinstance(values, np.ndarray):
        values = np.ones(rgb_vals.shape[0])
    with open(srf_file) as f:
        srf_lines = f.readlines()
    n_vx, n_f = srf_lines[1].split(' ')
    n_vx, n_f = int(n_vx), int(n_f)
    # Also creating an rgb str...
    rgb_str = ''    
    # Create the ply string -> following this format
    ply_str  = f'ply\n'
    ply_str += f'format ascii 1.0\n'
    ply_str += f'element vertex {n_vx}\n'
    ply_str += f'property float x\n'
    ply_str += f'property float y\n'
    ply_str += f'property float z\n'
    if incl_rgb:
        ply_str += f'property uchar red\n'
        ply_str += f'property uchar green\n'
        ply_str += f'property uchar blue\n'
    ply_str += f'property float quality\n'
    ply_str += f'element face {n_f}\n'
    ply_str += f'property list uchar int vertex_index\n'
    ply_str += f'end_header\n'

    if x_offset is None:
        if hemi==None:
            x_offset = 0
        elif 'lh' in hemi:
            x_offset = -50
        elif 'rh' in hemi:
            x_offset = 50
    # Cycle through the lines of the obj file and add vx + coords + rgb
    v_idx = 0 # Keep count of vertices     
    for i in range(2,len(srf_lines)):
        # If there is a '.' in the line then it is a vertex
        if '.' in srf_lines[i]:
            split_coord = srf_lines[i][:-2:].split(' ')                        
            coord_count = 0
            for coord in split_coord:
                if ('.' in coord) & (coord_count==0): # Add x_offset
                    ply_str += f'{float(coord)+x_offset:.6f} ' 
                    coord_count += 1
                elif '.' in coord:
                    ply_str += f'{float(coord):.6f} ' 
                    coord_count += 1                    
            
            # Now add the value of the parameters...
            # Now add the rgb values. as integers between 0 and 255
            if incl_rgb:
                ply_str += f' {int(rgb_vals[v_idx][0]*255)} {int(rgb_vals[v_idx][1]*255)} {int(rgb_vals[v_idx][2]*255)} '

            # RGB str
            rgb_str += f'{int(rgb_vals[v_idx][0]*255)},{int(rgb_vals[v_idx][1]*255)},{int(rgb_vals[v_idx][2]*255)}\n'
            
            ply_str += f'{values[v_idx]:.3f}\n'
            v_idx += 1 # next vertex
        
        else:
            # After we finished all the vertices, we need to define the faces
            # -> these are triangles (hence 3 at the beginning of each line)
            # -> the index of the three vx is given
            # For some reason the index is 1 less in .ply files vs .obj files
            # ... i guess like the difference between matlab and python
            ply_str += '3 ' 
            split_idx = srf_lines[i][:-1:].split(' ')
            ply_str += f'{int(split_idx[0])} '
            ply_str += f'{int(split_idx[1])} '
            ply_str += f'{int(split_idx[2])} '
            ply_str += '\n'
    return ply_str, rgb_str

def dag_srf_to_ply_basic(srf_file, hemi=None):
    '''dag_srf_to_ply_basic
    Convert srf file to .ply (not including values, or rgb stuff)
    
    '''
    with open(srf_file) as f:
        srf_lines = f.readlines()
    n_vx, n_f = srf_lines[1].split(' ')
    n_vx, n_f = int(n_vx), int(n_f)
    # Create the ply string -> following this format
    ply_str  = f'ply\n'
    ply_str += f'format ascii 1.0\n'
    ply_str += f'element vertex {n_vx}\n'
    ply_str += f'property float x\n'
    ply_str += f'property float y\n'
    ply_str += f'property float z\n'
    ply_str += f'property float quality\n'
    ply_str += f'element face {n_f}\n'
    ply_str += f'property list uchar int vertex_index\n'
    ply_str += f'end_header\n'

    if hemi==None:
        x_offset = 0
    elif 'lh' in hemi:
        x_offset = -50
    elif 'rh' in hemi:
        x_offset = 50
    # Cycle through the lines of the obj file and add vx + coords + rgb
    v_idx = 0 # Keep count of vertices     
    for i in range(2,len(srf_lines)):
        # If there is a '.' in the line then it is a vertex
        if '.' in srf_lines[i]:
            split_coord = srf_lines[i][:-2:].split(' ')                        
            coord_count = 0
            for coord in split_coord:
                if ('.' in coord) & (coord_count==0): # Add x_offset
                    ply_str += f'{float(coord)+x_offset:.6f} ' 
                    coord_count += 1
                elif '.' in coord:
                    ply_str += f'{float(coord):.6f} ' 
                    coord_count += 1                    
            ply_str += f'{0:.3f}\n'
            v_idx += 1 # next vertex
        
        else:
            # After we finished all the vertices, we need to define the faces
            # -> these are triangles (hence 3 at the beginning of each line)
            # -> the index of the three vx is given
            # For some reason the index is 1 less in .ply files vs .obj files
            # ... i guess like the difference between matlab and python
            ply_str += '3 ' 
            split_idx = srf_lines[i][:-1:].split(' ')
            ply_str += f'{int(split_idx[0])} '
            ply_str += f'{int(split_idx[1])} '
            ply_str += f'{int(split_idx[2])} '
            ply_str += '\n'
    return ply_str

def dag_calculate_rgb_vals(data, **kwargs):
    '''
    dag_calculate_rgb_vals:
        Create an array of RGB values to plot on the surface

        
    Arguments:
        data            np.ndarray      What are we plotting on the surface? 1D array, same length as the number of vertices in subject surface.

    **kwargs:
        under_surf      np.ndarray      What is underlay of the data? i.e., going underneath the data (e.g., what is the background)?
                                        could be the curvature...
        data_mask       bool array      Mask to hide certain values (e.g., where rsquared is not a good fit)
        data_alpha      np.ndarray      Alpha values for plotting. Where this is specified the undersurf is used instead

        *** COLOR
        cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                            Default: viridis
        vmin            float           Minimum value for colormap
                                                            Default: 10th percentile in data
        vmax            float           Max value for colormap
                                                            Default: 90th percentile in data
                                                                
    '''
    under_surf = kwargs.get('under_surf', np.zeros((data.shape[0], 4)))
    data_mask = kwargs.get('data_mask', np.ones_like(data, dtype=bool))
    data_alpha = kwargs.get('data_alpha', np.ones_like(data, dtype=float))
    data_alpha[~data_mask] = 0 # Make values to be masked have alpha = 0

    # Load colormap properties: (cmap, vmin, vmax)
    cmap = kwargs.get('cmap', 'viridis')    
    vmin = kwargs.get('vmin', np.percentile(data[data_mask], 10))
    vmax = kwargs.get('vmax', np.percentile(data[data_mask], 90))
    
    # Create rgb values mapping from data to cmap
    data_cmap = dag_get_cmap(cmap)
    # data_cmap = mpl.cm.__dict__[cmap] 
    data_norm = mpl.colors.Normalize()
    data_norm.vmin = vmin
    data_norm.vmax = vmax
    data_col = data_cmap(data_norm(data))

    # Create a color bar...
    plt.close('all')
    plt.colorbar(mpl.cm.ScalarMappable(norm=data_norm, cmap=data_cmap))
    data_col_bar = plt.gcf()
    rgb_vals = (data_col * data_alpha[...,np.newaxis]) + \
        (under_surf * (1-data_alpha[...,np.newaxis]))
    
    return rgb_vals, data_col_bar

def dag_get_rgb_str(rgb_vals):
    '''
    dag_srf_to_ply
    Convert srf file to .ply
    
    '''
    n_vx = rgb_vals.shape[0]
    # Also creating an rgb str...
    rgb_str = ''    
    for v_idx in range(n_vx):
        rgb_str += f'{rgb_vals[v_idx][0]},{rgb_vals[v_idx][1]},{rgb_vals[v_idx][2]}\n'
    return rgb_str    


def dag_make_overlay_str(**kwargs):        
    masked_data = kwargs.get('masked_data', None)
    cmap = kwargs.get('cmap', 'viridis')    
    if masked_data is not None:
        vmin = kwargs.get('vmin', np.percentile(masked_data, 10))
        vmax = kwargs.get('vmax', np.percentile(masked_data, 90))
    else:
        vmin = kwargs.get('vmin', 0)
        vmax = kwargs.get('vmax', 1)

    cmap_nsteps = kwargs.get('cmap_nsteps', 20)
    
    # Make custom overlay:
    # value - rgb triple...
    fv_param_steps = np.linspace(vmin, vmax, cmap_nsteps)
    fv_color_steps = np.linspace(0,1, cmap_nsteps)
    fv_cmap = dag_get_cmap(cmap, **kwargs)
    # fv_cmap = mpl.cm.__dict__[cmap]
    
    ## make colorbar - uncomment to save a png of the color bar...
    # cb_cmap = mpl.cm.__dict__[cmap] 
    # cb_norm = mpl.colors.Normalize()
    # cb_norm.vmin = vmin
    # cb_norm.vmax = vmax
    # plt.close('all')
    # plt.colorbar(mpl.cm.ScalarMappable(norm=cb_norm, cmap=cb_cmap))
    # col_bar = plt.gcf()
    # col_bar.savefig(opj(self.sub_surf_dir, f'lh.{surf_name}_colorbar.png'))

    overlay_custom_str = 'overlay_custom='
    overlay_to_save = '['
    # '''
    # Takes the form 
    # [
    #     {
    #         "r" : 128,
    #         "g" : 0,
    #         "b" : 128.
    #         "val" : -10
    #     },
    #     {
    #         ...
    #     }
    # ]
    # '''
    for i, fv_param in enumerate(fv_param_steps):
        this_col_triple = fv_cmap(fv_color_steps[i])
        this_str = f'{float(fv_param):.2f},{int(this_col_triple[0]*255)},{int(this_col_triple[1]*255)},{int(this_col_triple[2]*255)},'
        overlay_custom_str += this_str    
        #
        overlay_to_save += '\n\t{'
        overlay_to_save += f'\n\t\t"b": {int(this_col_triple[2]*255)},'
        overlay_to_save += f'\n\t\t"g": {int(this_col_triple[1]*255)},'
        overlay_to_save += f'\n\t\t"r": {int(this_col_triple[0]*255)},'
        overlay_to_save += f'\n\t\t"val": {float(fv_param):.2f}'
        overlay_to_save += '\n\t}'
        if fv_param!=fv_param_steps[-1]:
            overlay_to_save += ','
    overlay_to_save += '\n]'
    
    return overlay_custom_str, overlay_to_save