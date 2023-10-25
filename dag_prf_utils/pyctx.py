
import numpy as np
import os
from scipy import stats
opj = os.path.join
from dag_prf_utils.utils import *
from dag_prf_utils.threading import *
import multiprocessing

try: 
    import cortex 
except ImportError:
    raise ImportError('Error importing pycortex... Not a problem unless you want to use pycortex stuff')    

class PyctxSurf(object):
    '''
    Load pycortex 
    simplified from JHeijs linescanning toolbox (https://github.com/gjheij/linescanning/blob/9b5c8e967533e0e8354583f2424ddd2ce803a9cf/linescanning/optimal.py#L93)
    '''
    def __init__(self, sub=None, fs_dir=None):
        """Initialize object"""

        # print(" Perform surface operations")
        self.sub = sub
        self.ctx_path = opj(cortex.database.default_filestore, self.sub)

        # check if we need to reload kernel to activate changes to filestore
        if os.environ.get("PROJECT") not in self.ctx_path:
            raise TypeError(f"Project '{os.environ.get('PROJECT')}' not found in '{self.ctx_path}'. This can happen if you changed the filestore, but haven't reloaded the kernel. Use 'call_ctxfilestore' to set the filestore (and reload window if running from VSCode)")
            
        if fs_dir == None:
            self.fs_dir = os.environ.get("SUBJECTS_DIR")
        else:
            self.fs_dir = fs_dir

        if not os.path.exists(self.ctx_path):
            # import subject from freesurfer (will have the same names)
            cortex.freesurfer.import_subj(
                fs_subject=self.sub,
                cx_subject=self.sub,
                freesurfer_subject_dir=self.fs_dir,
                whitematter_surf='smoothwm')
        
        # reload database after import
        cortex.db.reload_subjects()
        
        # self.curvature = cortex.db.get_surfinfo(self.sub, type="curvature")
        # self.thickness = cortex.db.get_surfinfo(self.sub, type="thickness")
        # self.depth = cortex.db.get_surfinfo(self.sub, type="sulcaldepth")
        self.surf_data = {}
        self.surf_data['lh'], self.surf_data['rh'] = cortex.db.get_surf(self.sub, 'fiducial')
        self.surf = {}
        self.surf['lh'], self.surf['rh'] = cortex.polyutils.Surface(self.surf_data['lh'][0], self.surf_data['lh'][1]), cortex.polyutils.Surface(self.surf_data['rh'][0], self.surf_data['rh'][1])
        # self.subsurfs = {'lh':[], 'rh': []}
        # self.subsurfs_v = {'lh':[], 'rh': []}
        n_vx = dag_load_nverts(self.sub, self.fs_dir)
        self.n_vx = {'lh':n_vx[0], 'rh':n_vx[1]} # [:self.n_vx['lh']] [self.n_vx['lh']:]
        self.total_n_vx = sum(n_vx)
        # self.surf_coords = np.vstack([self.lh_surf_data[0],self.rh_surf_data[0]])
        self.closest_vx = []

        # # Normal vector for each vertex (average of normals for neighboring faces)
        # self.lh_surf_normals = self.lh_surf.vertex_normals
        # self.rh_surf_normals = self.rh_surf.vertex_normals

    # def create_subsurface(self, roi, roi_mask=None, **kwargs):
    #     self.roi = roi
    #     if roi_mask is None:
    #         self.roi_mask  = dag_load_roi(self.sub, roi, self.fs_dir, split_LR=True, do_bool=True, **kwargs)   
    #     else:
    #         self.roi_mask = roi_mask
    #     self.roi_idx = {}
    #     self.roi_idx['lh_full'] = np.where(self.roi_mask['lh'])[0]     
    #     self.roi_idx['lh_part'] = np.where(self.roi_mask['lh'])[0]     
    #     self.roi_idx['rh_full'] = np.where(self.roi_mask['rh'])[0] +  self.n_vx['lh']   
    #     self.roi_idx['rh_part'] = np.where(self.roi_mask['rh'])[0]     

    #     for hemi in ['lh', 'rh']:
    #         self.subsurfs[hemi] = self.surf[hemi].create_subsurface(vertex_mask=self.roi_mask[hemi])
        
    #     # WHY!!!!!! Why are the index changing? bleugh....
    #     self.subsurfs_v['lh'] = np.where(self.subsurfs['lh'].subsurface_vertex_map != stats.mode(self.subsurfs['lh'].subsurface_vertex_map)[0][0])[0]
    #     self.subsurfs_v['lh_full'] = self.subsurfs_v['lh']
    #     self.subsurfs_v['rh'] = np.where(self.subsurfs['rh'].subsurface_vertex_map != stats.mode(self.subsurfs['rh'].subsurface_vertex_map)[0][0])[0] 
    #     self.subsurfs_v['rh_full'] = self.subsurfs_v['rh'] + self.subsurfs['lh'].subsurface_vertex_map.shape[-1]
    #     self.ivx_list = np.concatenate((self.subsurfs_v['lh_full'], self.subsurfs_v['rh_full']))
                
    def calculate_gdists(self, vx_mask, closest_x, n_threads=10):
        # split mask into l & r
        vx_mask_hemi = {}
        vx_mask_hemi['lh'] = vx_mask[:self.n_vx['lh']]
        vx_mask_hemi['rh'] = vx_mask[self.n_vx['lh']:]
        self.closest_x = closest_x

        hemi_dists = {}
        for hemi in ['lh', 'rh']:
            ivx_list = np.where(vx_mask_hemi[hemi])[0]
            # It runs faster if its been run before (something about cacheing...)
            # so run it here first...
            self.surf[hemi].geodesic_distance(0)
            # MAKE A WRAPPER:            
            if hemi=='lh':
                this_gdist_wrapper = self.gdist_wrapper_lh                
            elif hemi=='rh':
                this_gdist_wrapper = self.gdist_wrapper_rh
            gdist_threader = DagThreader(io_function=this_gdist_wrapper)

            print(n_threads)
            hemi_dists[hemi] = gdist_threader.run_threader(
                input_list = ivx_list,
                max_size = n_threads,
                # num_workers=n_threads
                )
        
        return hemi_dists

        #     for ivx in np.where(vx_mask_hemi[hemi])[0]:
                
        #         self.hemi_dists[hemi].append(self.subsurfs[hemi].geodesic_distance([i]))
        #     self.hemi_dists[hemi] = np.array(self.hemi_dists[hemi])                
        # # Pad the right hem with np.inf.
        # padL = np.pad(
        #     self.hemi_dists['lh'], ((0, 0), (0, self.hemi_dists['rh'].shape[-1])), constant_values=np.nan)
        # # pad the left hem with np.inf..
        # padR = np.pad(
        #     self.hemi_dists['rh'], ((0, 0), (self.hemi_dists['lh'].shape[-1], 0)), constant_values=np.nan)

        # self.distance_matrix = np.vstack([padL, padR])  # Now stack.
        # self.distance_matrix = (self.distance_matrix + self.distance_matrix.T)/2 # Make symmetrical        
    def gdist_wrapper_lh(self, ivx):
        ivx_dist  = self.surf['lh'].geodesic_distance(ivx)
        close_ivx = np.argpartition(ivx_dist, self.closest_x+1)[:self.closest_x+1] # +1 (excluding the )
        close_ivx = close_ivx[close_ivx!=ivx]
        close_val = ivx_dist[close_ivx]
        close_dict = {'close_ivx':close_ivx, 'close_val':close_val}
        return close_dict
    
    def gdist_wrapper_rh(self, ivx):
        ivx_dist  = self.surf['rh'].geodesic_distance(ivx)
        close_ivx = np.argpartition(ivx_dist, self.closest_x+1)[:self.closest_x+1] # +1 (excluding the )
        close_ivx = close_ivx[close_ivx!=ivx] 
        close_val = ivx_dist[close_ivx]
        close_dict = {'close_ivx':close_ivx + self.n_vx['lh'], 'close_val':close_val} # add extra
        return close_dict        

    # def geodesic_distace_wrapper(self, hemi, hemi_vx_mask, closest_x, ivx):
    #     ivx_dist = self.surf['lh'].geodesic_distance(ivx)



    # def get_gdist(self, hemi, ivx):
    #     this_gdist = self.surf[hemi].geodesic_distance(ivx)
    #     return this_gdist
        
    # def get_ivx_dist(self, ivx):
    #     ivx_dist = np.zeros(self.total_n_vx) * np.nan
    #     if ivx not in self.ivx_list:
    #         return ivx_dist
    #     # Need to only index the 
    #     ivx_inarray = np.where(self.ivx_list==ivx)[0]
    #     ivx_dist[self.ivx_list] = self.distance_matrix[ivx_inarray,:]
    #     return ivx_dist


    # def get_ivx_dist_closest(self, ivx, closest_x=5):
    #     if ivx not in self.ivx_list:
    #         close_ivx = np.zeros(closest_x) * np.nan
    #         close_val = np.zeros(closest_x) * np.nan
    #     else:        
    #         ivx_dist = self.get_ivx_dist(ivx)        
    #         close_ivx = np.argpartition(ivx_dist, closest_x+1)[:closest_x+1] # +1 (excluding the )
    #         close_ivx = close_ivx[close_ivx!=ivx]
    #         close_val = ivx_dist[close_ivx]

    #     return close_ivx, close_val



def pycortex_alpha_plotting(sub, data, data_weight, **kwargs):
    
    '''
    Function to make plotting in pycortex (using web gui) easier
    I found that using the "cortex.Vertex2D" didn't work (IDK why)
    Based on Sumiya's code -> extracts the curvature as a grey map
    -> puts your data on top of it...

    sub             subject to plot (pycortex id)
    data            data to plot (np.array size of the surface)
    data_weight     Used to mask your data. Can be boolean or a range (should be between 0 and 1.)
                    See other options
                    
    
    *** Optional ***
    Value           Default             Meaning
    --------------------------------------------------------------------------------------------------
    data_w_thresh   None                1 or 2 values (gives the threshold of values to include (lower & upper bound))                    
    vmin            None                Minimal value for data cmap
    vmax            None                Maximum value for data cmap
    cmap            Retinotopy_RYBCR    Color map to use for data
    bool_mask       True                Mask the data with absolute mask or a gradient
    scale_data_w    False               Scale the data weight between 0 and 1 
    '''
    data_w_thresh   = kwargs.get('data_w_thresh', None)
    if (not isinstance(data_w_thresh, list)) & (data_w_thresh is not None):
        data_w_thresh = [data_w_thresh]
    vmin            = kwargs.get('vmin', None)
    vmax            = kwargs.get('vmax', None)
    cmap            = kwargs.get('cmap', 'Retinotopy_RYBCR')
    bool_mask       = kwargs.get('bool_mask', True)
    scale_data_w    = kwargs.get('scale_data_w', False)

    # [1] Set up the mask / weighting for the data    
    if scale_data_w: 
        # Rescale data weights to be b/w 0 and 1
        dw = (data_weight-np.nanmin(data_weight)) / (np.nanmax(data_weight)-np.nanmin(data_weight))
    else:
        dw = data_weight.copy()
    # -> apply thresholds
    if data_w_thresh is None:
        pass
    elif len(data_w_thresh)==1:
        # lower bound only
        dw[dw<data_w_thresh] = 0
    else:
        dw[dw<data_w_thresh[0]] = 0
        dw[dw>data_w_thresh[1]] = 0    
    # remove any NaN values
    dw[np.isnan(dw)] = 0
    # -> bool mask?
    if bool_mask:
        dw = dw!=0

    # [2] Get the background for the figure (the curve data)    
    # -> data goes on top of this. Where there are gaps, this will show (e.g., fits are not good enough etc.)
    curv = cortex.db.get_surfinfo(sub)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map.
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray' # we want it to be grey underneath


    # [3] Create the data colormap
    # Create vx mask from data
    vx = cortex.Vertex(data, sub, vmin=vmin, vmax=vmax, cmap=cmap)

    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    # Mask out using alpha values (the alpha of your data is inverse to the curve)
    alpha_mask = dw.astype(np.float)

    # Alpha mask
    display_data = vx_rgb * alpha_mask + curv_rgb * (1-alpha_mask)

    # Create vertex RGB object out of R, G, B channels
    vx_fin = cortex.VertexRGB(*display_data, sub)
    # cortex.quickshow(vx_fin)
    # cortex.webgl.show(
    #         {
    #         'alpha' : vx_fin,
    #         'no_alpha' : vx,
    #         })    
    return vx_fin, vx

