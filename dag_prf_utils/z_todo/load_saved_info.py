import numpy as np
import scipy.io
import os
import sys
import yaml
import pickle
opj = os.path.join

import nibabel as nb
from prfpy.stimulus import PRFStimulus2D, CSFStimulus

import linescanning.utils as lsutils
import pandas as pd
from .utils import print_p
# from collections import defaultdict as dd
# import cortex

from .utils import hyphen_parse, coord_convert

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'#os.getenv("DIR_DATA_SOURCE")
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'#os.getenv("DIR_DATA_DERIV")
freesurfer_dir = opj(derivatives_dir, 'freesurfer')
default_prf_dir = opj(derivatives_dir, 'prf')
dm_dir = opj(os.path.dirname(os.path.realpath(__file__)), 'dm_files' )
psc_tc_dir = opj(derivatives_dir, 'psc_tc')

class PrfParamGetterv2(object):
    '''
    Used to hold parameters for 1 subject, 1 task & 1 model
    & To return user specified masks 

    __init__ will set up the useful information into 3 pandas data frames
    >> including: all the parameters in the numpy arrays input model specific
        gauss: "x", "y", "a_sigma", "a_val", "bold_baseline", "rsq"
        norm : "x", "y", "a_sigma", "a_val", "bold_baseline", "c_val", "n_sigma", "b_val", "d_val", "rsq"
    >> & eccentricit, polar angle, 
        "ecc", "pol",
    
    Functions:
    return_vx_mask: returns a mask for voxels, specified by the user
    return_th_param: returns the specified parameters, masked
    '''
    def __init__(self,sub, task_list, model_list, **kwargs):
        '''
        params_LE/X        np array, of all the parameters in the LE/X condition
        model               str, model: e.g., gauss or norm
        '''
        self.sub = kwargs.get('sub', None)
        self.n_vox = amb_load_nverts(sub)
        self.params_np = amb_load_prf_params(
            sub=sub, task_list=task_list, model_list=model_list, **kwargs)
        
        self.params_dd = {}
        for task in task_list:
            self.params_dd[task] = {}
            for model in model_list:
                self.params_dd[task][model] = {}
                mod_labels = print_p()[model] 
                for key in mod_labels.keys():                    
                    self.params_dd[task][model][key] = self.params_np[task][model][:,mod_labels[key]]
                # Ecc, pol
                self.params_dd[task][model]['ecc'], self.params_dd[task][model]['pol'] = coord_convert(
                    self.params_dd[task][model]['x'],self.params_dd[task][model]['y'],
                    'cart2pol')
                
        # Convert to PD
        self.pd_params = {}
        for task in task_list:
            self.pd_params[task] = {}
            for model in model_list:                
                self.pd_params[task][model] = pd.DataFrame(self.params_dd[task][model])

    def return_vx_mask(self, th={}):
        '''
        return_vx_mask: returns a mask (boolean array) for voxels, specified by the user
        th keys must be split into 4 parts
        'task-model-comparison-param' : value
        e.g.: to exclude gauss fits with rsq less than 0.1
        th = {'pRFLE-gauss-min-rsq': 0.1 } 

        task        -> pRFLE, pRFRE
        model       -> gauss, norm
        comparison  -> min, max
        param       -> any of... (model dependent)
            "x", "y", "ecc", "pol"
            gauss: "a_sigma", "a_val", "bold_baseline", "rsq"
            norm : "a_sigma", "a_val", "bold_baseline", "c_val", "n_sigma", "b_val", "d_val", "rsq"            
            returns a boolean array, excluding all vx where rsq < 0.1 in LE condition
        
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        # ADD ALL_th to both LE and RE
        for th_key in th.keys():
            print(th_key)
            # th_key_str = list(th_key)[0] # convert to string...
            # print(th_key_str.split('-'))
            # sys.exit()
            # task,model,comp,p = th_key_str.split('-')
            # th_val = th[th_key]
            # if comp=='min':
            #     vx_mask &= self.pd_params[task][model][p].gt(th_val)
            # elif comp=='max':
            #     vx_mask &= self.pd_params[task][model][p].lt(th_val)
            # else:
            #     sys.exit()

        return vx_mask
    
    # def return_th_param(self, task, param, vx_mask=None):
    #     '''
    #     For a specified task (LE, RE, Ed)
    #     return all the parameters listed, masked by vx_mask        
    #     '''
    #     if vx_mask is None:
    #         vx_mask = np.ones(self.n_vox, dtype=bool)
    #     if not isinstance(param, list):
    #         param = [param]        
    #     param_out = []
    #     for i_param in param:
    #         # this_task = i_param.split('-')[0]
    #         # this_param = i_param.split('-')[1]
    #         param_out.append(self.pd_params[task][i_param][vx_mask].to_numpy())

    #     return param_out

def amb_load_fit_settings(sub, task_list, model_list, **kwargs):
    fit_settings = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, key='settings', **kwargs)
    return fit_settings

def amb_load_pred_tc(sub, task_list, model_list, **kwargs):
    pred_tc = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, key='predictions', **kwargs)
    return pred_tc

def amb_load_prf_params(sub, task_list, model_list, **kwargs):
    prf_params = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, key='pars', **kwargs)
    return prf_params

def amb_load_pkl_key(sub, task_list, model_list, key, **kwargs):
    if not isinstance(task_list, list):
        task_list = [task_list]
    if not isinstance(model_list, list):
        model_list = [model_list]        
    
    prf_dict = {}
    for task in task_list:
        prf_dict[task] = {}
        for model in model_list:
            this_pkl = amb_load_pkl(sub=sub, task=task, model=model, **kwargs)
            prf_dict[task][model] = this_pkl[key]
    return prf_dict

def amb_load_pkl(sub, task, model, **kwargs):
    '''
    linescanning toolbox nicely saves everything into a pickle
    this will load the correct pickle associated with the correct, sub, ses, model and task
    roi_fit specifies which fitting run was used.  
    '''    
    if 'pRF' in task:
        amb_prf_dir = opj(derivatives_dir, 'amb-prf')
    else:
        amb_prf_dir = opj(derivatives_dir, 'amb-csf')

    dir_to_search = opj(amb_prf_dir, sub, 'ses-1')
    include = kwargs.get("include", []) # any extra details to search for in file name
    exclude = kwargs.get("exclude", []) # any extra details to search for in file name
    roi_fit = kwargs.get('roi_fit', 'all')    
    fit_stage = kwargs.get('fit_stage', 'iter')

    # the folder specified in "dir_to_search" will contain several files
    # -> different fit types (grid vs iter), model (gauss vs norm) task (As0,AS1,AS2) and 
    # Now we need to find the relevant file, by filtering for key terms (see included)
    include += [sub, model, task, roi_fit, fit_stage] # Make sure we get the correct model and task (& subject)    
    exclude += ['avg_bold', '.txt'] # exclude grid fits and bold time series

    data_path = lsutils.get_file_from_substring(filt=include, path=dir_to_search, exclude=exclude)
    if isinstance(data_path, list):
        print(f'Error, more than 1 match ({len(data_path)} files)')
        sys.exit()

    pkl_file = open(data_path,'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()     

    return data    

def amb_load_real_tc(sub, task_list, clip_start=0):
    if not isinstance(task_list, list):
        task_list = [task_list]
    this_dir = opj(psc_tc_dir, sub, 'ses-1')
    real_tc = {}
    for task in task_list:
        real_tc_file = lsutils.get_file_from_substring([task, 'hemi-LR_desc-avg_bold'], this_dir)
        if isinstance(real_tc_file, list):
            print(f'Error, more than 1 match ({len(real_tc_file)} files)')
            sys.exit()
        unclipped = np.load(real_tc_file).T        
        real_tc[task] = np.copy(unclipped[:,clip_start::])

    return real_tc

def amb_load_real_tc_run(sub, task_list, run_list):
    if not isinstance(task_list, list):
        task_list = [task_list]
    if not isinstance(run_list, list):
        run_list=[run_list]
    unz_dir = opj(derivatives_dir, 'pybest', sub, 'ses-1', 'unzscored')
    real_tc = {}
    for task in task_list:
        real_tc[task] = []
        for run in run_list:
            LH_real_tc_file = lsutils.get_file_from_substring(
                [task, f'run-{run}', 'fsnative', 'hemi-R_desc-denoised_bold'], unz_dir)
            LH_tc = np.load(LH_real_tc_file)
            RH_real_tc_file = lsutils.get_file_from_substring(
                [task, f'run-{run}', 'fsnative', 'hemi-R_desc-denoised_bold'], unz_dir)
            RH_tc = np.load(RH_real_tc_file)
            real_tc[task].append(np.concatenate([LH_tc, RH_tc], axis=1).T)

    return real_tc


def amb_load_dm(dm_types):
    
    if not isinstance(dm_types, list):
        dm_types = [dm_types]
    
    dm = {}
    for dm_type in dm_types:
        if not dm_type in ['sf_vect', 'c_vect', 'prf', 'csf']:
            print('ERROR')
            sys.exit()
        if dm_type == 'sf_vect'        :
            dm[dm_type] = np.squeeze(scipy.io.loadmat(opj(dm_dir, 'sf_vect.mat'))['sf_vect'])
        elif dm_type == 'c_vect':
            dm[dm_type] = np.squeeze(scipy.io.loadmat(opj(dm_dir, 'contrasts_vect.mat'))['contrasts_vect'])
        elif dm_type == 'prf':
            dm[dm_type] = np.load(opj(dm_dir, 'prf_design_matrix.npy'))

        else:
            print('error')        
            sys.exit()
    return dm

def amb_load_prfpy_stim(dm_type='pRF', clip_start=0):
    if dm_type=='pRF':
        screen_info_path = opj(dm_dir, 'screen_info.yml')
        with open(screen_info_path) as f:
            screen_info = yaml.safe_load(f)

        dm_prf = amb_load_dm('prf')['prf'][:,:,clip_start::]    
        prfpy_stim = PRFStimulus2D(
            screen_size_cm    =screen_info['screen_size_cm'],
            screen_distance_cm=screen_info['screen_distance_cm'],
            design_matrix=dm_prf, 
            axis=0,
            TR=screen_info['TR']
            )
    elif dm_type=='CSF':
        csf_dm = amb_load_dm(['sf_vect', 'c_vect'])
        sf_vect = csf_dm['sf_vect'][clip_start::]
        c_vect = csf_dm['c_vect'][clip_start::]

        # Number of stimulus types:
        u_sfs = np.sort(list(set(sf_vect))) # unique SFs
        u_sfs = u_sfs[u_sfs>0]
        u_con = np.sort(list(set(c_vect)))
        u_con = u_con[u_con>0]
        prfpy_stim = CSFStimulus(
            SFs = u_sfs,#,
            CONs = u_con,
            SF_seq=sf_vect,
            CON_seq = c_vect,
            TR=1.5,
        )


    return prfpy_stim    

def amb_load_nverts(sub):
    n_verts = []
    for i in ['lh', 'rh']:
        surf = opj(freesurfer_dir, sub, 'surf', f'{i}.white')
        verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0]
        n_verts.append(verts)
    return n_verts

def amb_load_roi(sub, roi):
    '''
    Return a boolean array of voxels included in the specified roi
    array is vector with each entry corresponding to a point on the subjects cortical surface
    (Note this is L & R hemi combined)

    roi can be a list (in which case more than one is included)
    roi can also be exclusive (i.e., everything *but* x)

    TODO - conjunctive statements (not)
    '''
    # If *ALL* voxels to be included
    if roi=='all':
        total_num_vx = np.sum(amb_load_nverts(sub))

        roi_idx = np.ones(total_num_vx, dtype=bool)
        return roi_idx    
    # Else look for rois in subs freesurfer label folder
    roi_dir = opj(derivatives_dir, 'freesurfer', sub, 'label')
    if not isinstance(roi, list): # roi can be a list 
        roi = [roi]    
    
    roi_idx = []
    for this_roi in roi:    
        # Find the corresponding files
        if 'not' in this_roi:
            do_not = True
            this_roi = this_roi.split('-')[-1]
        else:
            do_not = False
        roi_file = {}
        roi_file['L'] = lsutils.get_file_from_substring([this_roi, '.thresh', '.label', 'lh'], roi_dir)    
        roi_file['R'] = lsutils.get_file_from_substring([this_roi, '.thresh', '.label', 'rh'], roi_dir)    
        n_verts = amb_load_nverts(sub)
        LR_bool = []
        for i,hemi in enumerate(['L', 'R']):
            with open(roi_file[hemi]) as f:
                contents = f.readlines()
            
            idx_str = [contents[i].split(' ')[0] for i in range(2,len(contents))]
            idx_int = [int(idx_str[i]) for i in range(len(idx_str))]
            this_bool = np.zeros(n_verts[i], dtype=bool)
            this_bool[idx_int] = True
            if do_not:
                this_bool = ~this_bool

            LR_bool.append(this_bool)
        this_roi_mask = np.concatenate(LR_bool)
        roi_idx.append(this_roi_mask)
    
    roi_idx = np.vstack(roi_idx)
    roi_idx = roi_idx.any(0)

    return roi_idx

# def 

# def load_prf_data(sub, ses, model, task, dir_to_search, roi_fit='all', dm_fit='standard', **kwargs):
#     '''
#     linescanning toolbox nicely saves everything into a pickle
#     this will load the correct pickle associated with the correct, sub, ses, model and task
#     roi_fit specifies which fitting run was used.  
#     I also ran the fitting using only the V1 voxels. 
#     This was to speed up the fitting, so that I could use the trust constrained on the normalization model too
#     BUT - this gives a problem, when we search for the parameter file, it now comes up with two matches; e.g., 
#     >> sub-01_ses-1_task-AS0_model-norm_stage-iter_desc-prf_params.pkl
#     >> sub-01_ses-1_task-AS0_roi-V1_model-norm_stage-iter_desc-prf_params.pkl

#     So to solve this, I am doing this...
#     >> adding roi_fit='all' to the default...    
#     '''
#     fit_stage = kwargs.get('fit_stage', 'iter')
#     include = kwargs.get("include", [])
#     exclude = kwargs.get("exclude", [])
    
#     # Caught exceptions - now lets continue...
#     # the folder specified in "dir_to_search" will contain several files
#     # -> different fit types (grid vs iter), model (gauss vs norm) task (As0,AS1,AS2) and 
#     # Now we need to find the relevant file, by filtering for key terms (see included)
#     include += [sub, model, task, roi_fit, fit_stage] # Make sure we get the correct model and task (& subject)    
#     exclude += ['avg_bold', '.txt'] # exclude grid fits and bold time series

#     data_path = utils.get_file_from_substring(filt=include, path=dir_to_search, exclude=exclude)

#     # If the exclude is used, it returns a list of length one
#     if isinstance(data_path, list) and (len(data_path)==1): # check whether this happened 
#         data_path = data_path[0] # (we only want to do this if the list is length 1 - we want errors if there are more than 1 possible matches...)         

#     if '.npy' in data_path:
#         # Load numpy data
#         data = np.load(data_path)
#     elif '.pkl' in data_path:
#         pkl_file = open(data_path,'rb')
#         data = pickle.load(pkl_file)
#         pkl_file.close()     

#     return data

# def get_fit_settings(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard'):
#     '''
#     This will get the fitting settings stored in the pickle file associated with this model & task
    
#     '''
#     if isinstance(task_list, str):
#         task_list = [task_list]
#     if isinstance(model_list, str):
#         model_list = [model_list]

#     fit_settings  = {}
#     for task in task_list:
#         if "AS" in task:
#             ses='ses-1'
#         elif "2R" in task:
#             ses='ses-2'            
#         this_dir = opj(prf_dir, sub, ses)
#         fit_settings[task] = {}
#         for model in model_list:   
#             this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit)
#             fit_settings[task][model] = this_pkl['settings']
#     return fit_settings

# def get_model_params(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard', fit_stage='iter'):
#     # Turn any strings into lists
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     if not isinstance(model_list, list):
#         model_list = [model_list]

#     model_params  = {}
#     for task in task_list:
#         if "AS" in task:
#             ses='ses-1'
#         elif "2R" in task:
#             ses='ses-2'            
#         this_dir = opj(prf_dir, sub, ses)

#         model_params[task] = {}
#         for model in model_list:
#             this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit, fit_stage=fit_stage)
#             model_params[task][model] = this_pkl['pars']            
#     return model_params

# def get_number_of_vx(sub):
#     # Do this by loading the ROI mask for v1
#     try:
#         roi_idx = get_roi(sub, label='V1')
#         num_vx = roi_idx.shape[0]
    
#     except:
#         real_tc = get_real_tc(sub, 'task-AS0')['task-AS0']
#         num_vx = real_tc.shape[0]
    
#     return num_vx

# def get_roi(sub, label):    
#     if label=='all':
#         num_vx = get_number_of_vx(sub)
#         roi_idx = np.ones(num_vx, dtype=bool)
#         return roi_idx
    
#     if not isinstance(label, list):
#         label = [label]
    
#     roi_idx = []
#     for this_label in label:
#         if "not" in this_label:
#             this_roi_file = opj(default_numpy_roi_idx_dir, sub, f'{this_label.split("-")[-1]}.npy')        
#             this_roi_idx = np.load(this_roi_file)
#             this_roi_idx = this_roi_idx==0
#         else:
#             this_roi_file = opj(default_numpy_roi_idx_dir, sub, f'{this_label}.npy')
#             this_roi_idx = np.load(this_roi_file)
        
#         roi_idx.append(this_roi_idx)
#     roi_idx = np.vstack(roi_idx)
#     roi_idx = roi_idx.any(0)
#     return roi_idx

# def get_real_tc(sub, task_list, prf_dir=default_prf_dir):
#     if isinstance(task_list, str):
#         task_list = [task_list]

#     real_tc  = {}
#     for task in task_list:'            
#         this_dir = opj(prf_dir, sub, ses)
#         real_tc_path = utils.get_file_from_substring([sub, ses, task, 'hemi-LR', 'desc-avg_bold'], this_dir, exclude='roi')
#         if isinstance(real_tc_path, list) and (len(real_tc_path)==1):
#             real_tc_path = real_tc_path[0]
#         real_tc[task] = np.load(real_tc_path).T

#     return real_tc

# def get_pred_tc(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard'):
#     # Turn any strings into lists
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     if not isinstance(model_list, list):
#         model_list = [model_list]

#     pred_tc  = {}
#     for task in task_list:
#         if "AS" in task:
#             ses='ses-1'
#         elif "2R" in task:
#             ses='ses-2'            
#         this_dir = opj(prf_dir, sub, ses)

#         pred_tc[task] = {}
#         for model in model_list:
#             this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit)
#             pred_tc[task][model] = this_pkl['predictions']            

#     return pred_tc

# def get_design_matrix_npy(task_list, prf_dir=default_prf_dir):
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     this_dir = opj(prf_dir)
#     dm_npy  = {}    
#     for task in task_list:
#         dm_path = utils.get_file_from_substring(['design', task], this_dir)        
#         dm_npy[task] = scipy.io.loadmat(dm_path)['stim']

#     return dm_npy

# def get_prfpy_stim(sub, task_list, prf_dir=default_prf_dir):
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     dm_npy = get_design_matrix_npy(task_list, prf_dir=prf_dir)
#     model_list = ['gauss']     # stimulus settings are the same for both norm & gauss models  (so only use gauss) 
#     stim_settings = get_fit_settings(sub,task_list, model_list=model_list, prf_dir=prf_dir)
#     prfpy_stim = {}
#     for task in task_list:
#         prfpy_stim[task] = PRFStimulus2D(
#             screen_size_cm=stim_settings[task][model_list[0]]['screen_size_cm'],
#             screen_distance_cm=stim_settings[task][model_list[0]]['screen_distance_cm'],
#             design_matrix=dm_npy[task], 
#             axis=0,
#             TR=stim_settings[task][model_list[0]]['TR']
#             )    
#     return prfpy_stim



# # # ********************
# # def load_params_generic(params_file, load_all=False, load_var=[]):
# #     """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

# #     if isinstance(params_file, str):
# #         if params_file.endswith('npy'):
# #             params = np.load(params_file)
# #         elif params_file.endswith('pkl'):
# #             with open(params_file, 'rb') as input:
# #                 data = pickle.load(input)
            
# #             if len(load_var)==1:
# #                 params = data[load_var[0]]
# #             elif len(load_var)>1:
# #                 params = {}
# #                 # Load the specified variables
# #                 for this_var in load_var:
# #                     params[this_var] = data[this_var]
# #             elif load_all:
# #                 params = {}
# #                 for this_var in data.keys():
# #                     params[this_var] = data[this_var]
# #             else:
# #                 params = data['pars']

# #     elif isinstance(params_file, np.ndarray):
# #         params = params_file.copy()
# #     elif isinstance(params_file, pd.DataFrame):
# #         dict_keys = list(params_file.keys())
# #         if not "hemi" in dict_keys:
# #             # got normalization parameter file
# #             params = np.array((params_file['x'][0],
# #                                 params_file['y'][0],
# #                                 params_file['prf_size'][0],
# #                                 params_file['A'][0],
# #                                 params_file['bold_bsl'][0],
# #                                 params_file['B'][0],
# #                                 params_file['C'][0],
# #                                 params_file['surr_size'][0],
# #                                 params_file['D'][0],
# #                                 params_file['r2'][0]))
# #         else:
# #             raise NotImplementedError()
# #     else:
# #         raise ValueError(f"Unrecognized input type for '{params_file}'")

# #     return params


