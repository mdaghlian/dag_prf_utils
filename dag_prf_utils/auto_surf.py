import os
opj = os.path.join
import pickle
import sys
import numpy as np
import re
import scipy.io
from copy import copy
try:
    from prfpy_csenf.stimulus import PRFStimulus2D
    from prfpy_csenf.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel, CSenFModel
except:
    from prfpy.stimulus import PRFStimulus2D
    from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel
import nibabel as nib
import cortex
from dag_prf_utils.prfpy_functions import prfpy_params_dict, Prf1T1M
from dag_prf_utils.prfpy_ts_plotter import TSPlotter
from dag_prf_utils.mesh_dash import MeshDash, dag_mesh_pickle
from dag_prf_utils.fs_tools import FSMaker
from dag_prf_utils.pycortex import PyctxMaker, set_ctx_path

def dag_auto_surf_function(surf_type, **kwargs):
    '''
    ---------------------------
    Auto open a subject surface

    Args:
        surf_type               plot using 'dash' or 'fs'
        param_path              path to .pkl/.npy/.gii/.mgz file 
        specific_param_path     dict with paths to specific parameters
        sub                     subject number
        fs_dir                  freesurfer director
        output_dir               where to put it
        file_name               name of the file
        model                   prfpy model to use
        real_ts                 path to real timeseries
        dm_file                 path to design matrix file
        hemi_markers            How are hemispheres marked in file?
        dump                    dump the mesh object
        open                    open the surface
	    port 			what port to host dash server on
	    host 			what ip to host dash on
        ow_prfpy_model  overwrite the prfpy_model (if stored in pickle) 

    ''' 
    # Parse the arguments
    param_path = kwargs.pop('param_path', None)   
    specific_param_path = kwargs.pop('specific_param_path', None)
    sub = kwargs.pop('sub', None)
    fs_dir = kwargs.pop('fs_dir', os.environ['SUBJECTS_DIR'])    
    if not os.path.exists(fs_dir):
        print('Could not find SUBJECTS_DIR')
        print(fs_dir)
        sys.exit()
    output_dir = kwargs.pop('output_dir', os.getcwd())
    file_name = kwargs.pop('file_name', 'auto_surf')
    hemi_markers = kwargs.pop('hemi_markers', ['lh', 'rh'])
    # Sort out how we id hemisphere
    # -> people (me) are annoyingly inconsistent with how they hame there hemispheres (I'm working on it)
    dm_file = kwargs.pop('dm_file', None)
    model = kwargs.pop('model', None)
    real_ts = kwargs.pop('real_ts', None)
    dump = kwargs.pop('dump', False)
    open_surf = kwargs.pop('open', False)
    port = kwargs.pop('port', 8000)
    host = kwargs.pop('host', '127.0.0.1')
    pars_to_plot = kwargs.pop('pars_to_plot', None)
    if isinstance(pars_to_plot, str):
        pars_to_plot = [pars_to_plot]
    min_rsq = kwargs.pop('min_rsq', 0.1)
    max_ecc = kwargs.pop('max_ecc', 5)
    ow_prfpy_model = kwargs.pop('ow_prfpy_model', False)
    extra_kwargs = copy(kwargs)
    
    # Check for missing stuff in param_path name
    if param_path is not None:
        if sub is None:
            sub = 'sub-'
            sub += re.search(r'sub-(.*?)_', param_path).group(1)
        
        if (model is None) & ('model' in param_path):
            pattern = r'model-(.*?)_'
            model = re.search(pattern, param_path).group(1)
    elif specific_param_path is not None:
        key_to_check = list(specific_param_path.keys())[0]
        # Check if specific_param_path
        if sub is None:
            sub = 'sub-'
            sub += re.search(r'sub-(.*?)_', specific_param_path[key_to_check]).group(1)
        if (model is None) & ('model' in specific_param_path[key_to_check]):
            pattern = r'model-(.*?)_'
            model = re.search(pattern, specific_param_path[key_to_check]).group(1)

    # Load some data...
    data_info = {
        'pars':[], 
        'pars_dict' : {}, 
        'prfpy_model':None, 
        'real_ts':None, 
        'settings': {},
        }
    if param_path is not None:
        if '.pkl' in param_path:
            # Assume store under pars
            with open(param_path, 'rb') as f:
                pickle_dict = pickle.load(f)
            for k in data_info.keys():
                if k in pickle_dict.keys():
                    print(f'loading {k}')
                    data_info[k] = pickle_dict[k]            
            if ow_prfpy_model:
                data_info['prfpy_model'] = None
                print('Overwriting prfpy model')
        elif '.npy' in param_path:
            data_info['pars'] = np.load(param_path)

        elif ('.gii' in param_path) or ('.mgz' in param_path):
            data_info['pars'] = load_mgz_or_gii(param_path, hemi_markers=hemi_markers)
    
    elif specific_param_path is not None:
        for k in specific_param_path.keys():
            if '.npy' in specific_param_path[k]:
                data_info['pars_dict'][k] = np.load(specific_param_path[k])
            elif ('.gii' in specific_param_path[k]) or ('.mgz' in specific_param_path[k]):
                data_info['pars_dict'][k] = load_mgz_or_gii(specific_param_path[k], hemi_markers=hemi_markers)
        # Try making a npy array for the params
        if model is not None:
            n_pars = len(prfpy_params_dict()[model])
            n_vx = data_info['pars_dict'][list(data_info['pars_dict'].keys())[0]].shape[0]
            data_info['pars'] = np.zeros((n_vx, n_pars))
            data_info['pars'][:,-1] = 1 # for rsq thresh
            for k in data_info['pars_dict'].keys():
                idx = prfpy_params_dict()[model][k]
                data_info['pars'][:,idx] = data_info['pars_dict'][k].squeeze()
    # Check if we have a 2D array (for loop below)
    if len(data_info['pars'].shape)==1:
        data_info['pars'] = data_info['pars'][:,np.newaxis]
    # Can we find the real timeseries somewhere else?
    if (real_ts is not None) and (data_info['real_ts'] is None):
        data_info['real_ts'] = np.load(real_ts)
        # Quick check, is it nvx X time?
        if data_info['real_ts'].shape[0]<data_info['real_ts'].shape[1]:
            data_info['real_ts'] = data_info['real_ts'].T
    
    # Can we remake the prfpy model?
    if (data_info['prfpy_model'] is None) and (dm_file is not None) and (model is not None):
        if '.mat' in dm_file:
            dm_npy = scipy.io.loadmat(dm_file)['stim']
        else:
            dm_npy = np.load(dm_file)
        try:
            prfpy_stim = PRFStimulus2D(
                screen_size_cm=data_info['settings']['screen_size_cm'],
                screen_distance_cm=data_info['settings']['screen_distance_cm'],
                design_matrix=dm_npy, 
                axis=0,
                TR=data_info['settings']['TR']
                )    
            if 'gauss' in model:
                data_info['prfpy_model'] = Iso2DGaussianModel(prfpy_stim)
            elif 'norm' in model:
                data_info['prfpy_model'] = Norm_Iso2DGaussianModel(prfpy_stim)
            elif 'dog' in model:
                data_info['prfpy_model'] = DoG_Iso2DGaussianModel(prfpy_stim) 
            elif 'css' in model:
                data_info['prfpy_model'] = CSS_Iso2DGaussianModel(prfpy_stim)                                   
        except:
            print('Could not make prfpy model')                    

    # DASH OBJECT
    if surf_type == 'dash':
        
        # Make the mesh dash object
        fs = MeshDash(
            sub=sub, 
            fs_dir=fs_dir,
            output_dir=output_dir,
            )    

        fs.web_get_ready(**extra_kwargs)
        
        if model is not None:
            try:
                prf_obj = TSPlotter(
                    prf_params=data_info['pars'],
                    model=model,
                    prfpy_model=data_info['prfpy_model'],
                    real_ts = data_info['real_ts'],            
                    incl_hrf=True, 
                )
            except:
                prf_obj = TSPlotter(
                    prf_params=data_info['pars'],
                    model=model,
                    prfpy_model=data_info['prfpy_model'],
                    real_ts = data_info['real_ts'],            
                    incl_hrf=False, 
                )
            if pars_to_plot is None:
                pars_to_plot = list(prf_obj.pd_params.keys())

            for p in pars_to_plot:
                data        = prf_obj.pd_params[p].to_numpy()
                data4mask   = prf_obj.pd_params['rsq'].to_numpy()
                if p=='pol':
                    cmap = 'marco_pol'
                    vmin,vmax = -np.pi, np.pi
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
                elif p=='ecc':
                    cmap = 'ecc2'
                    vmin,vmax = 0, int(np.nanmax(data))
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
                elif p=='rsq':
                    cmap='plasma'
                    vmin,vmax = 0,1
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)                
                elif p in ('x', 'y'):
                    cmap = 'RdBu'
                    vmin,vmax = -int(np.nanmax(data)), int(np.nanmax(data))
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

                else:
                    kwargs = {}
                
                fs.web_add_vx_col(
                    data=data, 
                    # data_alpha=data_alpha, 
                    data4mask = data4mask,
                    vx_col_name=p,  
                    **kwargs,  
                )
                # break
            fs.web_add_mpl_fig_maker(
                mpl_func=prf_obj.prf_ts_plot, 
                mpl_key='plot',
                mpl_kwargs={'return_fig':True},
            )
        elif len(data_info['pars_dict']) > 0:
            for k in data_info['pars_dict'].keys():
                fs.web_add_vx_col(
                    data=data_info['pars_dict'][k], 
                    vx_col_name=k,    
                )
        else:            
            # We don't know what everything is... 
            if len(data_info['pars'].shape)==1:
                data_info['pars'] = data_info['pars'][:,np.newaxis]

            for p in np.arange(data_info['pars'].shape[1]):
                data        = data_info['pars'][:,p]
                data4mask   = data_info['pars'][:,-1]            
                fs.web_add_vx_col(
                    data=data, 
                    data4mask = data4mask,
                    vx_col_name=p,    
                )
        if dump:
            dag_mesh_pickle(fs, file_name=file_name)
        if open_surf:
            app = fs.web_launch_with_dash()
            # Open the app in a browser
            # Do not show it in the notebook
            print(f'http://localhost:{port}/')
            app.run_server(host=host, port=port, debug=False, use_reloader=False)             
    # ****************************************************
    # ****************************************************
    elif surf_type == 'pycortex':
        ctx_method = kwargs.pop('ctx_method', 'custom')
        # Make the mesh dash object
        fs = PyctxMaker(
            sub=sub, 
            fs_dir=fs_dir,
            output_dir=output_dir,
            )    
        
        if model is not None:
            try:
                prf_obj = Prf1T1M(
                    prf_params=data_info['pars'],
                    model=model,
                    incl_hrf=True, 
                )
            except:
                prf_obj = Prf1T1M(
                    prf_params=data_info['pars'],
                    model=model,
                    incl_hrf=False, 
                )
            if pars_to_plot is None:
                pars_to_plot = list(prf_obj.pd_params.keys())

            for p in pars_to_plot:
                data        = prf_obj.pd_params[p].to_numpy()
                data_rsq   = prf_obj.pd_params['rsq'].to_numpy()
                data_mask   = prf_obj.pd_params['rsq'].to_numpy()>min_rsq
                if 'ecc' in prf_obj.pd_params.keys():
                    data_mask &= prf_obj.pd_params['ecc'].to_numpy()<max_ecc
                if p=='pol':
                    cmap = 'marco_pol'
                    vmin,vmax = -np.pi, np.pi
                    ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
                elif p=='ecc':
                    cmap = 'ecc2'
                    vmin,vmax = 0, int(np.nanmax(data))
                    ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
                elif p=='rsq':
                    cmap='plasma'
                    vmin,vmax = 0,1
                    ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)                
                elif p in ('x', 'y'):
                    cmap = 'RdBu'
                    vmin,vmax = -int(np.nanmax(data)), int(np.nanmax(data))
                    ctx_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

                else:
                    ctx_kwargs = {}
                ctx_kwargs['ctx_method'] = ctx_method
                if ctx_method == 'vertex2d':
                    ctx_kwargs['data_alpha'] = data_rsq
                fs.add_vertex_obj(
                    data=data, 
                    data_mask=data_mask,
                    surf_name=p,  
                    **ctx_kwargs,  
                )
                # break
        elif len(data_info['pars_dict']) > 0:
            for k in data_info['pars_dict'].keys():
                fs.add_vertex_obj(
                    data=data_info['pars_dict'][k], 
                    vx_col_name=k,    
                )
        else:            
            # We don't know what everything is... 
            if len(data_info['pars'].shape)==1:
                data_info['pars'] = data_info['pars'][:,np.newaxis]

            for p in np.arange(data_info['pars'].shape[1]):
                data        = data_info['pars'][:,p]
                data_mask   = data_info['pars'][:,-1]>min_rsq            
                fs.add_vertex_obj(
                    data=data, 
                    data_mask = data_mask>min_rsq,
                    vx_col_name=p,    
                )
        if dump:
            # fs.return_pyc_saver(viewer=False)
            # fs.pyc.to_static(filename=file_name)
            cortex.webgl.make_static(
                file_name,
                fs.vertex_dict, 
                )
        if open_surf:            
            cortex.webgl.show(
                fs.vertex_dict, 
                port=np.random.randint(8000, 9000),
                open_browser=False,
                autoclose=False,
                )
    # ****************************************************
    # ****************************************************
    # FS OBJECT
    elif surf_type == 'fs':
        # FS OBJECT
        fs = FSMaker(
            sub=sub, 
            fs_dir=fs_dir,
            )
        
        if model is not None:
            try:
                prf_obj = TSPlotter(
                    prf_params=data_info['pars'],
                    model=model,
                    prfpy_model=data_info['prfpy_model'],
                    real_ts = data_info['real_ts'],            
                    incl_hrf=True, 
                )
            except:
                prf_obj = TSPlotter(
                    prf_params=data_info['pars'],
                    model=model,
                    prfpy_model=data_info['prfpy_model'],
                    real_ts = data_info['real_ts'],            
                    incl_hrf=False, 
                )            
            if pars_to_plot is None:
                pars_to_plot = list(prf_obj.pd_params.keys())
            
            for p in pars_to_plot:
                data        = prf_obj.pd_params[p].to_numpy()
                data_mask   = prf_obj.pd_params['rsq'].to_numpy()>min_rsq
                if 'ecc' in prf_obj.pd_params.keys():
                    data_mask &= prf_obj.pd_params['ecc'].to_numpy()<max_ecc
                if p=='pol':
                    cmap = 'marco_pol'
                    vmin,vmax = -np.pi, np.pi
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
                elif p=='ecc':
                    cmap = 'ecc2'
                    vmin,vmax = 0, 5
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
                elif p=='rsq':
                    cmap='plasma'
                    vmin,vmax = 0,1
                    kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
                else:
                    kwargs = {}
                
                fs.add_surface(
                    data=data, 
                    data_mask = data_mask,
                    surf_name=f'{file_name}_{p}',  
                    **kwargs,  
                )
        else:
            # We don't know what everything is... 
            # assume last column is rsq
            for p in np.arange(data_info['pars'].shape[1]):
                data        = data_info['pars'][:,p]
                fs.add_surface(
                    data=data, 
                    surf_name=f'{file_name}_{p}',
                )
        if open_surf:
             fs.open_fs_surface(fs.surf_list, **extra_kwargs)


def dag_auto_from_prf_obj(prf_obj, sub, **kwargs):
    '''
    ---------------------------
    Auto open a subject surface

    Args:
        sub                     subject number
        prf_obj                 prf object 
        surf_type               plot using 'dash' or 'fs'
        fs_dir                  freesurfer director
        output_dir               where to put it
        file_name               name of the file
        model                   prfpy model to use
        dump                    dump the mesh object
        open                    open the surface
	    port 			what port to host dash server on
	    host 			what ip to host dash on

    ''' 
    # Parse the arguments
    fs_dir = kwargs.pop('fs_dir', os.environ['SUBJECTS_DIR'])    
    if not os.path.exists(fs_dir):
        print('Could not find SUBJECTS_DIR')
        print(fs_dir)
        sys.exit()
    
    output_dir = kwargs.pop('output_dir', os.getcwd())
    file_name = kwargs.pop('file_name', 'auto_surf')
    surf_type = kwargs.pop('surf_type', 'dash')
    dump = kwargs.pop('dump', False)
    open_surf = kwargs.pop('open', False)
    port = kwargs.pop('port', 8000)
    host = kwargs.pop('host', '127.0.0.1')
    pars_to_plot = kwargs.pop('pars_to_plot', None)
    min_rsq = kwargs.pop('min_rsq', 0.1)
    max_ecc = kwargs.pop('max_ecc', 5)
    return_fs = kwargs.pop('return_fs', False)
    extra_kwargs = copy(kwargs)

    # DASH OBJECT
    if surf_type == 'dash':
        
        # Make the mesh dash object
        fs = MeshDash(
            sub=sub, 
            fs_dir=fs_dir,
            output_dir=output_dir,
            )    

        fs.web_get_ready(**extra_kwargs)
        
        if pars_to_plot is None:
            pars_to_plot = list(prf_obj.pd_params.keys())

        for p in pars_to_plot:
            data        = prf_obj.pd_params[p].to_numpy()
            if '-' in p:
                # This is a multi object. Only get the rsq for the specific one...
                prf_id = p.split('-')[0]
                data4mask   = prf_obj.prf_obj[prf_id].pd_params['rsq'].to_numpy()
            else:
                data4mask   = prf_obj.pd_params['rsq'].to_numpy()

            if 'pol' in p:
                cmap = 'marco_pol'
                vmin,vmax = -np.pi, np.pi
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif 'ecc' in p:
                cmap = 'ecc2'
                vmin,vmax = 0, int(np.nanmax(data))
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif 'rsq' in p:
                cmap='plasma'
                vmin,vmax = 0,1
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)                
            elif ('x' in p) or ('y' in p):
                cmap = 'RdBu'
                vmin,vmax = -int(np.nanmax(data)), int(np.nanmax(data))
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)

            else:
                kwargs = {}
            
            fs.web_add_vx_col(
                data=data, 
                # data_alpha=data_alpha, 
                data4mask = data4mask,
                vx_col_name=p,  
                **kwargs,  
            )
            # break
        if hasattr(prf_obj, 'id_list'):
            # It is a multi figure...
            for prf_id in prf_obj.id_list:
                fs.web_add_mpl_fig_maker(
                    mpl_func=prf_obj.prf_obj[prf_id].prf_ts_plot, 
                    mpl_key=f'{prf_id}_plot',
                    mpl_kwargs={'return_fig':True},
                )
        else:
            fs.web_add_mpl_fig_maker(
                mpl_func=prf_obj.prf_ts_plot, 
                mpl_key='plot',
                mpl_kwargs={'return_fig':True},
            )

        if dump:            
            dag_mesh_pickle(fs, file_name=file_name)
        if open_surf:
            app = fs.web_launch_with_dash()
            # Open the app in a browser
            # Do not show it in the notebook
            print(f'http://localhost:{port}/')
            app.run_server(host=host, port=port, debug=False, use_reloader=False)             
    
    else:
        # FS OBJECT
        fs = FSMaker(
            sub=sub, 
            fs_dir=fs_dir,
            )
         
        if pars_to_plot is None:
            pars_to_plot = list(prf_obj.pd_params.keys())
        
        for p in pars_to_plot:
            data        = prf_obj.pd_params[p].to_numpy()
            if '-' in p:
                # This is a multi object. Only get the rsq for the specific one...
                prf_id = p.split('-')[0]
                data_mask   = prf_obj.prf_obj[prf_id].pd_params['rsq'].to_numpy() > min_rsq
                if 'ecc' in prf_obj.prf_obj[prf_id].pd_params.keys():
                    data_mask &= prf_obj.prf_obj[prf_id].pd_params['ecc'].to_numpy() < max_ecc
            else:
                data_mask   = prf_obj.pd_params['rsq'].to_numpy()>min_rsq
                if 'ecc' in prf_obj.pd_params.keys():
                    data_mask &= prf_obj.pd_params['ecc'].to_numpy()<max_ecc
 
            if 'pol' in p:
                cmap = 'marco_pol'
                vmin,vmax = -np.pi, np.pi
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif 'ecc' in p:
                cmap = 'ecc2'
                vmin,vmax = 0, int(np.nanmax(data))
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            elif 'rsq' in p:
                cmap='plasma'
                vmin,vmax = 0,1
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)                
            elif ('x' in p) or ('y' in p):
                cmap = 'RdBu'
                vmin,vmax = -int(np.nanmax(data)), int(np.nanmax(data))
                kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                kwargs = {}
            
            fs.add_surface(
                data=data, 
                data_mask = data_mask,
                surf_name=f'{file_name}_{p}',  
                **kwargs,  
            )

        if open_surf:
             fs.open_fs_surface(fs.surf_list, **extra_kwargs)

    if return_fs:
        return fs
    
    


def load_mgz_or_gii(mgz_or_gii_path, hemi_markers=['lh', 'rh']):
    '''
    Load a .mgz or .gii file and return the data (as numpy array)
    Containing both hemispheres
    '''
    mlh = [i for i in hemi_markers if 'l' in i.lower()][0]
    mrh = [i for i in hemi_markers if 'r' in i.lower()][0]

    if mlh in mgz_or_gii_path:
        lh_file = copy(mgz_or_gii_path)
        rh_file = mgz_or_gii_path.replace(mlh, mrh)
    else:
        rh_file = copy(mgz_or_gii_path)
        lh_file = mgz_or_gii_path.replace(mrh, mlh)
    if '.gii' in lh_file:
        lh_data = nib.load(lh_file)
        lh_data = [i.data for i in lh_data.darrays]
        lh_data = np.vstack(lh_data).squeeze()
        rh_data = nib.load(rh_file)
        rh_data = [i.data for i in rh_data.darrays]
        rh_data = np.vstack(rh_data).squeeze()
        # mgz_or_gii_data = np.concatenate([lh_data, rh_data], axis=0)
    else:
        lh_data = nib.load(lh_file).get_fdata().squeeze()[...,np.newaxis]
        rh_data = nib.load(rh_file).get_fdata().squeeze()[...,np.newaxis]
    mgz_or_gii_data = np.concatenate([lh_data, rh_data], axis=0)

    return mgz_or_gii_data
