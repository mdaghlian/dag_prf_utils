import cortex
import cortex.freesurfer
import imageio
# from linescanning import (
#     # image,
#     # prf, 
#     utils,
#     plotting)
import nibabel as nb
import numpy as np
import os
import pandas as pd
import configparser
from scipy.spatial import ConvexHull
import sys
import time
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union
opj = os.path.join
from PIL import Image
import uuid

from dag_prf_utils.mesh_maker import *
from dag_prf_utils.mesh_format import *
from dag_prf_utils.cmap_functions import *
from dag_prf_utils.utils import *
'''
MUCH STOLEN FROM JHEIJ LINESCANNING!!!
'''
def dag_add_cmap_to_pyctx(cmap_name, pyctx_cmap_name=None):
    pyc_cmap_path = cortex.config.get('webgl', 'colormaps')
    if pyctx_cmap_name is None:
        pyctx_cmap_name = 'pyc_'+cmap_name
    # do reverse version
    for direction in ['fw', 'bw']:
        if direction=='fw':
            cname_1D = pyctx_cmap_name + '_1D.png'
            cname_2D = pyctx_cmap_name + '_2D.png'
            cval_arr = np.linspace(0,1,256)
        elif direction=='bw':
            cname_1D = pyctx_cmap_name + '_1D_r.png'
            cname_2D = pyctx_cmap_name + '_2D_r.png'
            cval_arr = np.linspace(1,0,256)
        # [1] 1D version
        oneD_rgba = dag_get_col_vals(cval_arr, cmap_name)
        oneD_rgba = oneD_rgba[np.newaxis,:,:]
        oneD_rgba = (oneD_rgba * 255).astype(np.uint8)
        image = Image.fromarray(oneD_rgba)
        image.save(opj(pyc_cmap_path, cname_1D))
        # [2] 2D version
        twoD_arr, alpha_val = np.meshgrid(cval_arr, np.linspace(1,0,256))
        twoD_rgba = dag_get_col_vals(twoD_arr,cmap_name)
        # -> enter alpha levels
        # twoD_rgba[:,-1,:-1] = 0
        # alpha_val[alpha_val<.5] = 0
        # alpha_val[alpha_val>.5] = 1
        twoD_rgba[:,:,-1] = alpha_val
        idx,idy = np.where(alpha_val==0)
        # set anywhere with 0 alpha to be black
        twoD_rgba[idx,idy,0] = 0
        twoD_rgba[idx,idy,1] = 0
        twoD_rgba[idx,idy,2] = 0

        twoD_rgba = (twoD_rgba * 255).astype(np.uint8)
        plt.figure()
        plt.imshow(twoD_rgba)
        # twoD_rgba[:,:,-1] = .1
        image = Image.fromarray(twoD_rgba)
        image.save(opj(pyc_cmap_path, cname_2D))


def set_ctx_path(p=None, opt="update"):
    """set_ctx_path

    Function that changes the filestore path in the cortex config file to make changing between projects flexible. Just specify the path to the new pycortex directory to change. If you do not specify a string, it will default to what it finds in the os.environ['CTX'] variable as specified in the setup script. You can also ask for the current filestore path with "opt='show_fs'", or the path to the config script with "opt='show_pn'". To update, leave 'opt' to 'update'.

    Parameters
    ----------
    p: str, optional
        either the path we need to set `filestore` to (in combination with `opt="update"`), or None (in combination with `opt="show_fs"` or `opt="show_pn"`)
    opt: str
        either "update" to update the `filestore` with `p`; "show_pn" to show the path to the configuration file; or "show_fs" to show the current `filestore`

    Example
    ----------
    >>> set_ctx_path('path/to/pycortex', "update")
    """
    # ************************
    cortex.options.config.set('basic', 'filestore', p)
    cortex.db.filestore=p
    # ************************
    if p == None:
        p = os.environ.get('CTX')

    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

    usercfg = cortex.options.usercfg
    config = configparser.ConfigParser()
    config.read(usercfg)

    # check if filestore exists
    try:
        config.get("basic", "filestore")
    except:
        config.set("basic", "filestore", p)
        with open(usercfg, 'w') as fp:
            config.write(fp)

    if opt == "show_fs":
        return config.get("basic", "filestore")
    elif opt == "show_pn":
        return usercfg
    else:
        if config.get("basic", "filestore") != p:
            config.set("basic", "filestore", p)
            with open(usercfg, 'w') as fp:
                config.write(fp)
            
            if not os.path.exists(p):
                os.makedirs(p, exist_ok=True)

            return config.get("basic", "filestore")
        else:
            return config.get("basic", "filestore")

def get_ctx_path():
    usercfg = cortex.options.usercfg
    config = configparser.ConfigParser()
    config.read(usercfg)
    return config.get("basic", "filestore")


    
# def create_ctx_transform # **** DELETED 

def get_thickness(thick_map, hemi, vertex_nr):
    """get_thickness

    Fetch the cortical thickness given a vertex in a certain hemisphere.

    Parameters
    ----------
    thick_map: str
        thickness.npz created by pycortex (to be implemented: map created by user)
    hemi: str
        which hemisphere do we need to fetch data from
    vertex: int
        which vertex in the specified hemisphere

    Returns
    ----------
    float
        thickness measure given a hemisphere and vertex number

    Example
    ----------
    >>> get_thickness("/path/to/thickness.npz", "left", 875)
    1.3451
    """

    import numpy as np

    thick = np.load(thick_map)
    thick_hemi = thick[hemi]
    val = thick_hemi[vertex_nr]

    # invert value as it is a negative value
    return abs(val)



class PyctxSaver():
    """PyctxSaver copied from JH "SavePycortexViews"
    SavePycortexViews

    Save the elements of a `dict` containing vertex/volume objects to images given a set of viewing settings. If all goes well, a browser will open. Just wait for your settings to be applied in the viewer. You can then proceed from there. If your selected orientation is not exactly right, you can still manually adapt it before running :func:`linescanning.pycortex.SavePycortexViews.save_all()`, which will save all elements in the dataset to png-files given `fig_dir` and `base_name`. Additionally, colormaps will be produced if `data_dict` contains instances of :class:`linescanning.pycortex.Vertex2D_fix` and will produce instances of :class:`linescanning.plotting.LazyColorbar` in a single figure for all elements in `data_dict`. This figure will also be saved when calling :func:`linescanning.pycortex.SavePycortexViews.save_all()` with suffix `_desc-colormaps.<ext>`, where <ext> is set by `cm_ext` (default = **pdf**).

    Parameters
    ----------
    data_dict: dict, cortex.dataset.views.Vertex, cortex.dataset.views.Volume, linescanning.pycortex.Vertex2D_fix
        Dictionary collecting objects to be projected on the surface or any object that is compatible with Pycortex plotting. Loose inputs will be automatically converted to dictionary.
    subject: str, optional
        Subject name as per Pycortex' filestore naming, by default None
    fig_dir: str, optional
        Output directory for the figures, by default None
    specularity: int, optional
        Level of 'glow' on the surface; ranges from 0-1, by default 0 (nothing at all)
    unfold: int, optional
        Level of inflation the surface needs to undergo; ranges from 0-1, by default 1 (fully inflated)
    azimuth: int, optional
        Rotation around the top-bottom axis, by default 185
    altitude: int, optional
        Rotation around the left-right axis, by default 90
    radius: int, optional
        distance of object, by default 163
    lh: int, bool, optional
        show left hemisphere. Default is 1 (True)
    rh: int, bool, optional
        show right hemisphere. Default is 1 (True)
    pivot: int, optional
        rotate the hemispheres away from one another (positive values = lateral view, negative values = medial view), by default 0
    size: tuple, optional
        size of image that's being saved, by default (4000,4000)
    data_name: str, optional
        name of dataset, by default "occipital_inflated"
    base_name: str, optional
        Basename for the images to save from the pycortex viewer. If None, we'll default to `<subject>`; `_desc-<>.png` is appended.
    rois_visible: int, optional
        Show the ROIs as per the 'overlays.svg' file on the FSAverage brain. Default = 0
    rois_labels: int, optional
        Show the ROIs labels as per the 'overlays.svg' file on the FSAverage brain. Default = 0
    sulci_visible: int, optional
        Show the sulci outlines on the FSAverage brain. Default = 1
    sulci_labels: int, optional
        Show the sulci labels on the FSAverage brain. Default = 0
    cm_ext: str, optional
        File extension to save the colormap images with. Default = "pdf". These colormaps will be produced if `data_dict` contains instances of :class:`linescanning.pycortex.Vertex2D_fix` and will produce instances of :class:`linescanning.plotting.LazyColorbar` in a single figure for all elements in `data_dict`
    cm_scalar: float, optional
        Decides more or less the height of the colorbars. By default 0.85. Higher values = thicker bar. This default value scales well with the font size
    cm_width: int, optional
        Width of the colormaps, by default 6
    cm_decimals: int, optional
        Decimal accuracy to use for the colormaps. Minimum and maximum values will be rounded up/down accordingly if the initial rounding resulted in values outside of the bounds of the data. This will operate on float values; integer values are left alone. Default = 2.
    cm_nr: int, optional
        Number of ticks to use in colormap. Default = 5
    viewer: bool, optional
        Open the viewer (`viewer=True`, default) or suppress it (`viewer=False`)

    Example
    ----------

    """

    def __init__(
        self,
        data_dict: Union[dict, cortex.Vertex,cortex.VertexRGB,cortex.Vertex2D],
        cms_dict: dict=None,
        subject: str=None,
        fig_dir: str=None,
        specularity: int=0,
        unfold: int=0,
        azimuth: int=180,
        altitude: int=90,
        radius: int=250,
        pivot: float=0,
        size: tuple=(2400,1200),
        data_name: str="occipital_inflated",
        base_name: str=None,
        rois_visible: int=0,
        rois_labels: int=0,
        sulci_visible: int=1,
        cm_scalar: float=0.85,
        cm_width: int=6,
        cm_ext: str="pdf",
        cm_decimals: int=2,
        cm_nr: int=5,        
        lh: bool=True,
        rh: bool=True,
        sulci_labels: int=0,
        viewer: bool=True,
        clicker: str="vertex",
        **kwargs):

        self.data_dict = data_dict
        self.cms_dict = cms_dict
        self.subject = subject
        self.fig_dir = fig_dir
        self.altitude = altitude
        self.radius = radius
        self.pivot = pivot
        self.size = size
        self.azimuth = azimuth
        self.unfold = unfold
        self.specularity = specularity
        self.data_name = data_name
        self.base_name = base_name
        self.show_left = lh
        self.show_right = rh
        self.rois_visible = rois_visible
        self.rois_labels = rois_labels
        self.sulci_labels = sulci_labels
        self.sulci_visible = sulci_visible
        self.cm_ext = cm_ext
        self.cm_scalar = cm_scalar
        self.cm_width = cm_width
        self.cm_nr = cm_nr
        self.cm_decimals = cm_decimals        
        self.clicker = clicker
        self.viewer = viewer

        if not isinstance(self.subject, str):
            raise ValueError("Please specify the subject ID as per pycortex' filestore naming")
        
        if not isinstance(self.base_name, str):
            self.base_name = self.subject

        if not isinstance(self.fig_dir, str):
            self.fig_dir = os.getcwd()            

        # check what kind of object they are; if Vertex2D_fix, make colormaps
        self.cms = {}
        self.cm_fig, self.cm_axs = plt.subplots(
            nrows=len(self.data_dict), 
            figsize=(self.cm_width,self.cm_scalar*len(self.cms_dict)),
            constrained_layout=True
        )
        
        for iK,key in enumerate(list(self.cms_dict.keys())):
            if len(self.cms_dict) == 1:
                ax = self.cm_axs
            else:
                ax = self.cm_axs[iK]
            
            if self.cms_dict[key] == 'pyc':
                ax.set_title('pyc')
                self.cms[key] = ax
            else:
                self.cms[key] = dag_cmap_plotter(
                    cmap=self.cms_dict[key]['cmap'], 
                    vmin=self.cms_dict[key]['vmin'], 
                    vmax=self.cms_dict[key]['vmax'], 
                    title=str(key), 
                    return_ax=True, 
                    ax=ax,
                )

        
        # check what do to with hemispheres
        for ii in ["left","right"]:
            attr = getattr(self, f"show_{ii}")
            if isinstance(attr, int):
                if attr == 1:
                    new_attr = True
                elif attr == 0:
                    new_attr = False
                else:
                    raise ValueError(f"Input must be 0 or 1, not '{attr}'")
                
                setattr(self, f"show_{ii}", new_attr)

        # set the views
        self.view = {
            self.data_name: {
                f'surface.{self.subject}.unfold': self.unfold, 
                f'surface.{self.subject}.pivot': self.pivot, 
                f'surface.{self.subject}.specularity': self.specularity,
                f'surface.{self.subject}.left': self.show_left, 
                f'surface.{self.subject}.right': self.show_right, 
                # 'camera.target':self.target,
                'camera.azimuth': self.azimuth,
                'camera.altitude': self.altitude, 
                'camera.radius': self.radius}}
                
        if self.subject == "fsaverage":
            for tt in ["rois","sulci"]:
                for tag in ["visible","labels"]:
                    lbl = getattr(self, f"{tt}_{tag}")
                    self.view[self.data_name][f'surface.{self.subject}.overlays.{tt}.{tag}'] = lbl

        self.view[self.data_name]['camera.Save image.Width'] = self.size[0]
        self.view[self.data_name]['camera.Save image.Height'] = self.size[1]

        if self.viewer:
            if isinstance(self.clicker, str):
                if self.clicker == "plot":
                    clicker_func = self.clicker_plot
                elif self.clicker == "vertex":
                    clicker_func = self.clicker_function

                else:
                    raise ValueError(f"clicker must be one of 'vertex' (just prints vertex to terminal) or 'plot' (plots position in visual space), not '{self.clicker}'")
            else:
                clicker_func = None
                
            self.js_handle = cortex.webgl.show(self.data_dict, pickerfun=clicker_func)
            # self.js_handle = cortex.webgl.show(self.data_dict)
            self.params_to_save = list(self.data_dict.keys())
            self.set_view()

    def clicker_function(
        self,
        voxel,
        hemi,
        vertex):

        #translate javascript indeing to python
        lctm, rctm = cortex.utils.get_ctmmap(self.subject, method='mg2', level=9)
        if hemi == 'left':
            index = lctm[int(vertex)]
        else:
            index = rctm[int(vertex)]
        
        print(f"vertex ID: {index} (hemi = {hemi})")


    def clicker_plot(
        self,
        voxel,
        hemi,
        vertex):

        #translate javascript indeing to python
        lctm, rctm = cortex.utils.get_ctmmap(self.subject, method='mg2', level=9)
        if hemi == 'left':
            index = lctm[int(vertex)]
        else:
            index = len(lctm)+rctm[int(vertex)]
        print(f"index {index}")#: {pars}")

        return
    def to_static(self, filename=None, *args, **kwargs):
        output_dir = kwargs.pop('output_dir', self.fig_dir)
        if filename is None:
            filename = f"{self.base_name}_desc-static"        
        print(f'Saving static to {opj(output_dir, filename)}')
        print(args)
        print(kwargs)
        cortex.webgl.make_static(
            opj(output_dir, filename),
            self.data_dict,
            *args,
            **kwargs)
        
    def save_all(
        self, 
        base_name=None,
        fig_dir=None,
        gallery=False,
        add_cms=False,
        *args,
        **kwargs):
        
        # set output directory
        if not isinstance(fig_dir, str):
            if not isinstance(self.fig_dir, str):
                fig_dir = os.getcwd()
            else:
                fig_dir = self.fig_dir

        # set base name
        if not isinstance(base_name, str):
            if not isinstance(self.base_name, str):
                base_name = self.subject
            else:
                base_name = self.base_name                

        # store output in separete imgs-directory to prevent clogging
        img_dir = opj(fig_dir, "imgs")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)

        self.imgs = {}
        for param_to_save in self.js_handle.dataviews.attrs.keys():
            print(f"saving {param_to_save}")
            self.save(
                param_to_save,
                fig_dir=img_dir,
                base_name=base_name)
            
            # store filenames in list
            filename = opj(img_dir, f"{base_name}_desc-{param_to_save}.png")
            self.imgs[param_to_save] = filename

        # save colormap figure
        if len(self.cms) > 0:
            filename = f"{base_name}_desc-colormaps.{self.cm_ext}"
            output_path = os.path.join(fig_dir, filename)            
            self.cm_fig.savefig(
                output_path,
                bbox_inches="tight",
                facecolor="white",
                dpi=300)
            
        if gallery:
            filename = opj(fig_dir, f"{base_name}_desc-brainmaps.pdf")

            # check if we should add the colorbars
            if add_cms:
                dd = self.tmp_dict
            else:
                dd = None

            self.make_gallery(
                self.imgs, 
                data_dict=dd,
                save_as=filename, 
                add_cms=add_cms,
                *args,
                **kwargs)
    
    def make_gallery(
        self,
        img_dict,
        data_dict=None,
        n_cols=3,
        cb=[0,900,350,1900],
        title="brain maps",
        save_as=None,
        add_cms=False,
        cm_inset=[0.99,0.15,0.05,0.75],
        *args,
        **kwargs):

        if len(img_dict) < n_cols:
            n_cols = len(img_dict)
            n_rows = 1
        else:
            n_rows = int(np.ceil(len(img_dict)/n_cols))

        fig = plt.figure(figsize=(n_cols*8,n_rows*6), constrained_layout=True)
        gs = fig.add_gridspec(ncols=n_cols, nrows=n_rows)

        for ix,(key,val) in enumerate(img_dict.items()):
            axs = fig.add_subplot(gs[ix])
            img_d = imageio.v2.imread(val)
            axs.imshow(img_d[cb[0]:cb[1],cb[2]:cb[3],:])
            axs.set_title(key, fontsize=24),
            axs.axis("off")

            if add_cms:
                if not isinstance(data_dict, dict):
                    raise TypeError(f"add_cms requires 'data_dict' to be a dictionary, not {data_dict} of type {type(data_dict)}")
                
                try:
                    min_max = [data_dict[key].vmin1,data_dict[key].vmax1]
                except:
                    min_max = [data_dict[key].vmin,data_dict[key].vmax]

                cm_ax = axs.inset_axes(cm_inset)
                dag_cmap_plotter(
                    cmap=self.cms_dict[key]['cmap'], 
                    vmin=self.cms_dict[key]['vmin'], 
                    vmax=self.cms_dict[key]['vmax'], 
                    title=str(key),   
                    ax=cm_ax,               
                )                

        plt.tight_layout()
        fig.suptitle(
            title, 
            fontsize=30,
            **kwargs)

        if isinstance(save_as, str):
            print(f"saving {save_as}")
            fig.savefig(
                save_as,
                bbox_inches="tight",
                dpi=300,
                facecolor="white"
            )

    def set_view(self):
        # set specified view
        time.sleep(10)
        for _, view_params in self.view.items():
            for param_name, param_value in view_params.items():
                time.sleep(1)
                self.js_handle.ui.set(param_name, param_value)   

    def save(
        self, 
        param_to_save,
        fig_dir=None,
        base_name=None):

        self.js_handle.setData([param_to_save])
        time.sleep(1)
        
        # Save images by iterating over the different views and surfaces
        filename = f"{base_name}_desc-{param_to_save}.png"
        output_path = os.path.join(fig_dir, filename)
            
        # Save image           
        self.js_handle.getImage(output_path, size=self.size)
    
        # the block below trims the edges of the image:
        # wait for image to be written
        while not os.path.exists(output_path):
            pass
        time.sleep(1)
        try:
            import subprocess
            subprocess.call(["convert", "-trim", output_path, output_path])
        except:
            pass    



# ****************************************
# ****************************************
# ****************************************
# print(f'Assuming the unique issue in pycortex is fixed...')
# print(f'see: https://github.com/gallantlab/pycortex/issues/341')
# print(f'If not fixed. change fixed_unique to False in pyctxmaker')

class PyctxMaker(GenMeshMaker):

    """
    Based on JHeij. "Vertex2D_fix" Some adaptations for colormaps 

    """

    def __init__(self, sub, fs_dir=os.environ['SUBJECTS_DIR'], output_dir=[], **kwargs):
        super().__init__(sub, fs_dir, output_dir)
        if not isinstance(self.sub, str):
            raise ValueError("Please specify the subject ID as per pycortex' filestore naming")
        self.subject =self.sub
        self.ctx_path = kwargs.get('ctx_path', None)
        if self.ctx_path is not None:
            set_ctx_path(self.ctx_path)
            self.ctx_path = opj(self.ctx_path, self.subject)
        else:
            self.ctx_path = opj(get_ctx_path(), self.sub)
        print(self.ctx_path)
        self.vertex_dict = {} 
        self.cmap_dict = {}
        self.fixed_unique = kwargs.get('fixed_unique', False) # Have we fixed the unique issue in pycortex? 
        self.dud = kwargs.get('dud', False)
        if not os.path.exists(self.ctx_path):
            # import subject from freesurfer (will have the same names)
            cortex.freesurfer.import_subj(
                fs_subject=self.sub,
                cx_subject=self.sub,
                freesurfer_subject_dir=self.fs_dir,
                whitematter_surf='smoothwm')
        
        # reload database after import
        cortex.db.reload_subjects()

        #this provides a nice workaround for pycortex opacity issues, at the cost of interactivity    
        # Get curvature
        self.curv = cortex.db.get_surfinfo(self.sub)
        if self.dud:
            # Add dud at the beggining
            self.add_vertex_obj(
                data=np.random.rand(self.total_n_vx),
                surf_name='dud'
            )

    def add_vertex_obj(self, data, surf_name, **kwargs):
        '''Add a pycortex surface to the dictionary self.vertex_dict
        Can use the inbuild pycortex functions (Vertex2D, Vertex)
        But sometimes that doesn't work. 
        So added an option to do the RGB by hand (also makes colormaps more flexible)
        The disadvantage of this is that it makes it non interactive. (i.e. cannot adjust threshold)
        
        
        
        See also 
        Issue 1: https://github.com/gallantlab/pycortex/issues/341

        TODO: add the pyc to colormaps
        '''
        ctx_method = kwargs.get('ctx_method', 'custom') # vertex1d, vertex2d
        if ctx_method.lower() not in ('custom', 'vertex1d', 'vertex2d'):
            raise Exception('ctx_method should be custom, vertex2d or vertex1d')
        
        pycortex_args = kwargs.get('pycortex_args', {})
        return_vx_obj = kwargs.get('return_vx_obj', False) # Return the vertex object? Or save it to self.vertex_dict 

        if ctx_method.lower()=='custom':
            # Use custom rgb method
            display_rgb,cmap_dict = self.return_display_rgb(
                data, 
                return_cmap_dict=True, 
                unit_rgb=False, 
                **kwargs)  
                
            this_vertex_dict = cortex.VertexRGB(
                red=display_rgb[:,0], 
                green=display_rgb[:,1], 
                blue=display_rgb[:,2], 
                subject=self.subject,
                # unit_rgb=False, 
                )     
            this_cmap_dict = cmap_dict

        else:
            # Use either vertex1d or vertex2d        
            data_mask = kwargs.get('data_mask', np.ones(self.total_n_vx, dtype=bool))
            data_alpha = kwargs.get('data_alpha', np.ones(self.total_n_vx))
            data_alpha[~data_mask] = 0 # Make values to be masked have alpha=0
            cmap = kwargs.get('cmap', None) # 'autumnblack_alpha_2D')    
            vmin1 = kwargs.get('vmin', np.nanmin(data[data_mask]))
            vmax1 = kwargs.get('vmax', np.nanmax(data[data_mask]))
            masked_value = kwargs.get('masked_value', vmin1-1) # What to set values outside mask to
            vmin2 = kwargs.get('vmin2', 0)
            vmax2 = kwargs.get('vmax2', 1)
            # dtype_to_set = kwargs.get('dtype_to_set', np.float32)            
            data = data.astype(np.float32)
            data[~data_mask] = masked_value # 
            # data[np.isnan(data)] = 0
            data_alpha = data_alpha.astype(np.float32)
            # data_alpha[np.isnan(data_alpha)] = 0
            # *** NOTE ON BUG ***
            # if not self.fixed_unique:
            #     print('Adding noise to dim1 and dim2 so that to prevent crashes ')
            #     print('np.random.rand(len(data))*1e-1000')
            #     print(f'see: https://github.com/gallantlab/pycortex/issues/341')
            #     print('to stop this set self.fixed_unique=True')
            #     data += np.random.rand(len(data))*1e-1000
            #     data_alpha += np.random.rand(len(data))*1e-1000

            if ctx_method.lower()=='vertex2d':
                # Vertex 2D 
                this_vertex_dict = cortex.Vertex2D(
                        dim1=data, 
                        dim2=data_alpha,
                        cmap=cmap, 
                        subject=self.subject,
                        vmin=vmin1, 
                        vmax=vmax1,                 
                        vmin2=vmin2, 
                        vmax2=vmax2,    
                        **pycortex_args,
                    )
                # TO FIX THE "unique" pycortex ISSUE...
                print(this_vertex_dict)

                this_vertex_dict.dim1.unique_id = np.random.rand(500)
                this_vertex_dict.dim2.unique_id = np.random.rand(500)
                this_cmap_dict = 'pyc'
            
            elif ctx_method.lower()=='vertex1d':
                print(data.min())
                print(vmin1)
                print(data)
                this_vertex_dict = cortex.Vertex(
                    data=data, 
                    cmap=cmap,                 
                    subject=self.sub,
                    vmin=vmin1, vmax=vmax1, 
                    **pycortex_args
                )
                this_vertex_dict.unique_id = np.random.rand(500)
                this_cmap_dict = 'pyc'
        if return_vx_obj:
            return this_vertex_dict, this_cmap_dict
        else:
            self.vertex_dict[surf_name] = this_vertex_dict
            self.cmap_dict[surf_name] = this_cmap_dict
        return 
                        
    def open(self, **kwargs):
        self.pyc = PyctxSaver(
            data_dict=self.vertex_dict,
            cms_dict = self.cmap_dict,
            subject=self.sub,
            **kwargs)
        
    def return_pyc_saver(self, **kwargs):
        '''
        '''
        output_dir = kwargs.pop('output_dir', self.output_dir)
        self.pyc = PyctxSaver(
            data_dict=self.vertex_dict,
            cms_dict = self.cmap_dict,
            subject=self.sub,
            fig_dir = output_dir,
            # viewer=False,
            **kwargs)

    
    def get_curv(self):
        return self.curv
    
    def clear_cache(self):
        cortex.db.clear_cache(self.sub)

    def make_fake_flat_map(self, flatmap_name='fake.patch.3d'):
        '''
        Pycotex uses flatmaps for a bunch of things
        But if you can't be bothered to do it properly, and just want
        to display freesurfer ROIs in pycortex you can do this

        It takes the spherical coordinates of each surface
        And uses them to generate the flatmap
        '''
        # Make a fake flatmap...
        # [1] Look for the overlays.svg if it exists back it up
        old_overlay = opj(self.ctx_path, 'overlays.svg')
        current_datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        bu_overlay = opj(self.ctx_path, f'overlays_BU_{current_datetime_str}.svg')
        if os.path.exists(old_overlay):
            print('backing up existing overlay')
            os.system(f'cp {old_overlay} {bu_overlay}')
        
        # [2] Write fake flat files into freesurfer...
        # self.fake_flat
        for hemi in ['lh', 'rh']:
            
            # r,elev,az = dag_cart2sph(
            #     x = self.mesh_info['sphere'][hemi]['x'],
            #     y = self.mesh_info['sphere'][hemi]['y'],
            #     z = self.mesh_info['sphere'][hemi]['z'],
            # )
            
            # if hemi=='lh':
            #     r *= -1 # Flip if lh
            # filename = opj(self.sub_surf_dir, f'{hemi}.{flatmap_name}')
            # dag_write_patch(
            #     filename=filename, 
            #     vertex_index=np.arange(1,len(r)+1, dtype='int32'), 
            #     x_coords=r.astype('float32'), 
            #     y_coords=elev.astype('float32'), 
            #     z_coords=np.zeros(len(r), dtype='float32')
            #     )            
            pts,polys = dag_fake_flatten(self.mesh_info['sphere'][hemi])
            # FROM PYCORTEX IMPORT FLAT
            flat = pts[:, [1, 0, 2]]
            # Flip Y axis upside down
            flat[:, 1] = -flat[:, 1]
            polys = cortex.freesurfer._remove_disconnected_polys(polys)
            flat = cortex.freesurfer._move_disconnect_points_to_zero(flat, polys)
            flat_surf_path = opj(self.ctx_path, 'surfaces', f'flat_{hemi}.gii')
            print("saving to %s"%flat_surf_path)
            cortex.formats.write_gii(flat_surf_path, pts=flat, polys=polys)

            # print(pts)
            # print(pts.shape)
            # print(polys)
            # print(polys.shape)
            # bloop


            
            

def dag_write_patch(filename, vertex_index, x_coords, y_coords, z_coords):
    """
    Writes vertex data to a binary file.
    
    Parameters:
        filename (str): Name of the file to write to.
        vertex_index (array-like): Array containing vertex indices.
        x_coords (array-like): Array containing x coordinates.
        y_coords (array-like): Array containing y coordinates.
        z_coords (array-like): Array containing z coordinates.
    """
    assert len(vertex_index) == len(x_coords) == len(y_coords) == len(z_coords)
    
    with open(filename, 'wb') as fp:
        # Write header (assuming a default value of 0 for now)
        header = 0
        fp.write(struct.pack('>i', header))
        
        # Write number of vertices
        nverts = len(vertex_index)
        fp.write(struct.pack('>i', nverts))
        
        # Write vertex data
        for i in range(nverts):
            fp.write(struct.pack('>i', vertex_index[i]))
            fp.write(struct.pack('>f', x_coords[i]))
            fp.write(struct.pack('>f', y_coords[i]))
            fp.write(struct.pack('>f', z_coords[i]))