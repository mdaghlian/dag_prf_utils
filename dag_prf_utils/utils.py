import numpy as np
import os
import pandas as pd
import sys
opj = os.path.join

class Prf1T1M(object):
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
    def __init__(self, prf_params, model, **kwargs):
        '''
        prf_params     np array, of all the parameters, i.e., output from prfpy
        model          str, model: e.g., gauss or norm
        '''
        self.model = model        
        self.model_labels = dag_print_p()[self.model] # Get names for different model parameters...
        self.prf_params_np = prf_params
        #
        self.task = kwargs.get('task', None)
        self.n_vox = self.prf_params_np.shape[0]

        self.params_dd = {}
        mod_labels = dag_print_p()[f'{model}'] 
        for key in mod_labels.keys():                    
            self.params_dd[key] = self.prf_params_np[:,mod_labels[key]]
        
        # Calculate extra interesting stuff
        if self.model in ['gauss', 'norm', 'css', 'dog']:
            # Ecc, pol
            self.params_dd['ecc'], self.params_dd['pol'] = dag_coord_convert(
                self.params_dd['x'],self.params_dd['y'],'cart2pol')        
        if self.model=='norm':
            # -> size ratio:
            self.params_dd['size_ratio'] = self.params_dd['n_sigma'] / self.params_dd['a_sigma']
            self.params_dd['amp_ratio'] = self.params_dd['a_val'] / self.params_dd['c_val']
            self.params_dd['bd_ratio'] = self.params_dd['b_val'] / self.params_dd['d_val']
        if self.model=='CSF':
            self.params_dd['log10_sf0'] = np.log10(self.params_dd['sf0'])
            self.params_dd['log10_maxC'] = np.log10(self.params_dd['maxC'])
            self.params_dd['sfmax'] = np.nan_to_num(
                10**(np.sqrt(self.params_dd['log10_maxC']/(self.params_dd['width_r']**2)) + \
                                            self.params_dd['log10_sf0']))            
            self.params_dd['sfmax'][self.params_dd['sfmax']>100] = 100 # MAX VALUE
            self.params_dd['log10_sfmax'] = np.log10(self.params_dd['sfmax'])

        # Convert to PD           
        self.pd_params = pd.DataFrame(self.params_dd)

    def return_vx_mask(self, th={}):
        '''
        return_vx_mask: returns a mask (boolean array) for voxels, specified by the user
        th keys must be split into 2 parts
        'comparison-param' : value
        e.g.: to exclude gauss fits with rsq less than 0.1
        th = {'min-rsq': 0.1 } 
        comparison  -> min, max
        param       -> any of... (model dependent)
            "x", "y", "ecc", "pol"
            gauss: "a_sigma", "a_val", "bold_baseline", "rsq"
            norm : "a_sigma", "a_val", "bold_baseline", "c_val", "n_sigma", "b_val", "d_val", "rsq"            
            returns a boolean array, excluding all vx where rsq < 0.1 in LE condition
        
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if comp=='min':
                vx_mask &= self.pd_params[p].gt(th_val)
            elif comp=='max':
                vx_mask &= self.pd_params[p].lt(th_val)
            elif comp=='bound':
                vx_mask &= self.pd_params[p].gt(th_val[0])
                vx_mask &= self.pd_params[p].lt(th_val[1])
            else:
                sys.exit()

        return vx_mask.to_numpy()
    
    def return_th_param(self, param, vx_mask=None):
        '''
        return all the parameters listed, masked by vx_mask        
        '''
        if vx_mask is None:
            vx_mask = np.ones(self.n_vox, dtype=bool)
        if not isinstance(param, list):
            param = [param]        
        param_out = []
        for i_param in param:
            # this_task = i_param.split('-')[0]
            # this_param = i_param.split('-')[1]
            param_out.append(self.pd_params[i_param][vx_mask].to_numpy())

        return param_out

    

class Prf2T1M(object):
    '''
    Used to hold parameters for 1 subject, ***2 tasks***, for 1model
    & To return user specified masks 

    __init__ will set up the useful information into 3 pandas data frames
    >> including: all the parameters in the numpy arrays input model specific
        gauss: "x", "y", "a_sigma", "a_val", "bold_baseline", "rsq"
        norm : "x", "y", "a_sigma", "a_val", "bold_baseline", "c_val", "n_sigma", "b_val", "d_val", "rsq"
    >> & eccentricit, polar angle, 
        "ecc", "pol",

    >> In addition we will also add the mean and difference of the different tasks...
    
    Functions:
    return_vx_mask: returns a mask for voxels, specified by the user
    return_th_param: returns the specified parameters, masked
    '''
    def __init__(self, prf_params1, prf_params2, model, **kwargs):
        '''
        prf_params     np array, of all the parameters, i.e., output from prfpy
        model          str, model: e.g., gauss or norm
        '''
        self.model = model
        self.model_labels = dag_print_p()[self.model] # Get names for different model parameters...
        # What are the task names?
        self.task1 = kwargs.get('task1', 'task1')        
        self.task2 = kwargs.get('task2', 'task2')        
        # Store the model parameters as np arrays
        self.prf_params_np = {
            self.task1 : prf_params1,
            self.task2 : prf_params2,
        }
        # Get the number of voxels...
        self.n_vox = self.prf_params_np[self.task1].shape[0]
        
        # Now create dictionaries to turn into dataframes for easy retrieval of info...
        all_task_dict = {
            self.task1: {},
            self.task2: {},
            'diff' : {},
            'mean' : {},
        }
        for i_task in [self.task1, self.task2]:
            for i_label in self.model_labels.keys():
                all_task_dict[i_task][i_label] = self.prf_params_np[i_task][:,self.model_labels[i_label]]
                        
            # Now add other interesting stuff:
            if self.model in ['gauss', 'norm', 'css', 'dog']:
                # Ecc, pol
                all_task_dict[i_task]['ecc'],all_task_dict[i_task]['pol'] = dag_coord_convert(
                    all_task_dict[i_task]['x'], all_task_dict[i_task]['y'], 'cart2pol'
                )
            if self.model in ['norm', 'dog']:
                # -> size ratio:
                all_task_dict[i_task]['size_ratio'] = all_task_dict[i_task]['size_2'] / all_task_dict[i_task]['size_1']
                all_task_dict[i_task]['amp_ratio']  = all_task_dict[i_task]['amp_1']  / all_task_dict[i_task]['amp_2']
            if self.model=='norm':
                all_task_dict[i_task]['bd_ratio'] = all_task_dict[i_task]['b_val'] / all_task_dict[i_task]['d_val']
            if self.model=='CSF':
                all_task_dict[i_task]['log10_sf0']  = np.log10(all_task_dict[i_task]['sf0'])
                all_task_dict[i_task]['log10_maxC'] = np.log10(all_task_dict[i_task]['maxC'])
                all_task_dict[i_task]['sfmax'] = np.nan_to_num(
                    10**(np.sqrt(all_task_dict[i_task]['log10_maxC'] / (all_task_dict[i_task]['width_r']**2)) + \
                                                all_task_dict[i_task]['log10_sf0']))            
                all_task_dict[i_task]['sfmax'][all_task_dict[i_task]['sfmax']>100] = 100 # MAX VALUE
                all_task_dict[i_task]['log10_sfmax'] = np.log10(all_task_dict[i_task]['sfmax'])
        
        # Complete list of model labels
        self.model_labels_plus = list(all_task_dict[self.task1].keys())
        # Now get mean and diff of parameters...
        for i_label in self.model_labels:
            # MEAN
            all_task_dict['mean'][i_label] = (all_task_dict[self.task2][i_label] + all_task_dict[self.task1][i_label]) / 2
            
            # DIFFERENCE            
            all_task_dict['diff'][i_label] = all_task_dict[self.task2][i_label] - all_task_dict[self.task1][i_label]
                
        # Recalculate the interesting stuff for mean and diff
        for i_comp in ['mean', 'diff']:
            # Now add other interesting stuff:
            if self.model in ['gauss', 'norm', 'css', 'dog']:
                # Ecc, pol
                all_task_dict[i_comp]['ecc'], all_task_dict[i_comp]['pol'] = dag_coord_convert(
                    all_task_dict[i_comp]['x'], all_task_dict[i_comp]['y'], 'cart2pol'
                )
            if self.model in ['norm', 'dog']:
                # -> size ratio:
                all_task_dict[i_comp]['size_ratio'] = all_task_dict[i_comp]['size_2'] / all_task_dict[i_comp]['size_1']
                all_task_dict[i_comp]['amp_ratio']  = all_task_dict[i_comp]['amp_1']  / all_task_dict[i_comp]['amp_2']
            if self.model=='norm':
                all_task_dict[i_comp]['bd_ratio'] = all_task_dict[i_comp]['b_val'] / all_task_dict[i_comp]['d_val']
            if self.model=='CSF':
                all_task_dict[i_comp]['log10_sf0']  = np.log10(all_task_dict[i_comp]['sf0'])
                all_task_dict[i_comp]['log10_maxC'] = np.log10(all_task_dict[i_comp]['maxC'])
                all_task_dict[i_comp]['sfmax'] = np.nan_to_num(
                    10**(np.sqrt(all_task_dict[i_comp]['log10_maxC'] / (all_task_dict[i_comp]['width_r']**2)) + \
                                                all_task_dict[i_comp]['log10_sf0']))            
                all_task_dict[i_comp]['sfmax'][all_task_dict[i_comp]['sfmax']>100] = 100 # MAX VALUE
                all_task_dict[i_comp]['log10_sfmax'] = np.log10(all_task_dict[i_comp]['sfmax'])

        # Convert to PD
        self.pd_params = {}
        for i_task in all_task_dict.keys():
            self.pd_params[i_task] = pd.DataFrame(all_task_dict[i_task])

    def return_vx_mask(self, th={}):
        '''
        return_vx_mask: returns a mask (boolean array) for voxels, specified by the user        
        th keys must be split into 3 parts
        'task-comparison-param' : value
        e.g.: to exclude gauss fits with rsq less than 0.1
        th = {'task1-min-rsq': 0.1 } 
        task        -> task1, task2, diff, mean, all. (all means apply the threshold to both task1, and task2)
        comparison  -> min, max, bound
        param       -> any of... (model dependent e.g., 'x', 'y', 'ecc'...)
        

        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            task, comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if task=='all':
                # Apply to both task1 and task2:
                vx_mask &= self.return_vx_mask({
                    f'{self.task1}-{comp}-{p}' : th_val,
                    f'{self.task2}-{comp}-{p}' : th_val,
                    })
                continue # now next item in th_key...
            
            if comp=='min':
                vx_mask &= self.pd_params[task][p].gt(th_val)
            elif comp=='max':
                vx_mask &= self.pd_params[task][p].lt(th_val)
            elif comp=='bound':
                vx_mask &= self.pd_params[task][p].gt(th_val[0])
                vx_mask &= self.pd_params[task][p].lt(th_val[1])
            else:
                sys.exit()
        if not isinstance(vx_mask, np.ndarray):
            vx_mask = vx_mask.to_numpy()
        return vx_mask
    
    def return_th_param(self, task, param, vx_mask=None):
        '''
        return all the parameters listed, masked by vx_mask        
        '''
        if vx_mask is None:
            vx_mask = np.ones(self.n_vox, dtype=bool)
        if not isinstance(param, list):
            param = [param]        
        param_out = []
        for i_param in param:
            param_out.append(self.pd_params[task][i_param][vx_mask].to_numpy())

        return param_out


def dag_qprint(print_str):
    print(print_str, flush=True)

def dag_load_nverts(sub, fs_dir):
    n_verts = []
    for i in ['lh', 'rh']:
        # surf = opj(fs_dir, sub, 'surf', f'{i}.white')
        # verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0]
        # Alternative to nibabel method...
        # The first number on the last line +1 is the number of vertices...         
        surf = opj(fs_dir, sub, 'label', f'{i}.cortex.label')
        with open(surf) as f:
            contents = f.readlines()
        verts = int(contents[-1].split(' ')[0]) + 1
        n_verts.append(verts)
    return n_verts

def dag_load_roi(sub, roi, fs_dir):
    '''
    Return a boolean array of voxels included in the specified roi
    array is vector with each entry corresponding to a point on the subjects cortical surface
    (Note this is L & R hemi combined)

    roi can be a list (in which case more than one is included)
    roi can also be exclusive (i.e., everything *but* x)

    TODO - hemi specific idx...
    '''
    # Get number of vx in each hemi, and total overall...
    n_verts = dag_load_nverts(sub=sub, fs_dir=fs_dir)
    total_num_vx = np.sum(n_verts)
    
    # If *ALL* voxels to be included
    if roi=='all':
        roi_idx = np.ones(total_num_vx, dtype=bool)
        return roi_idx    
    
    # Else look for rois in subs freesurfer label folder
    roi_dir = opj(fs_dir, sub, 'label')    
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
        roi_file['L'] = dag_find_file_in_folder([this_roi, '.thresh', '.label', 'lh'], roi_dir)    
        roi_file['R'] = dag_find_file_in_folder([this_roi, '.thresh', '.label', 'rh'], roi_dir)    
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

def dag_hyphen_parse(str_prefix, str_in):
    if str_prefix in str_in:
        str_out = str_in
    else: 
        str_out = f'{str_prefix}-{str_in}'
    return str_out
    
def dag_print_p():
    '''
    Easy look up table for prfpy model parameters
    name to index...
    '''
    p_order = {}
    # [1] gauss. Note hrf_1, and hrf_2 are idx 5 and 6, if fit...
    p_order['gauss'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # size
        'amp_1'         :  3, # beta
        'bold_baseline' :  4, # baseline 
        'rsq'           : -1, # ... 
    }    
    # [2] css. Note hrf_1, and hrf_2 are idx 6 and 7, if fit...
    p_order['css'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # size
        'amp_1'         :  3, # beta
        'bold_baseline' :  4, # baseline 
        'n_exp'         :  5, # n
        'rsq'           : -1, # ... 
    }

    # [3] dog. Note hrf_1, and hrf_2 are idx 7 and 8, if fit...
    p_order['dog'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # prf_size
        'amp_1'         :  3, # prf_amplitude
        'bold_baseline' :  4, # bold_baseline 
        'amp_2'         :  5, # srf_amplitude
        'size_2'        :  6, # srf_size
        'rsq'           : -1, # ... 
    }

    p_order['norm'] = {
        'x'             :  0, # mu_x
        'y'             :  1, # mu_y
        'size_1'        :  2, # prf_size
        'amp_1'         :  3, # prf_amplitude
        'bold_baseline' :  4, # bold_baseline 
        'amp_2'         :  5, # srf_amplitude
        'size_2'        :  6, # srf_size
        'b_val'         :  7, # neural_baseline 
        'd_val'         :  8, # surround_baseline
        'rsq'           : -1, # rsq
    }            

    p_order['CSF']  ={
        'width_r'       :  0,
        'sf0'           :  1,
        'maxC'          :  2,
        'width_l'       :  3,
        'a_val'         :  4,
        'baseline'      :  5,
        'rsq'           : -1,
    }

    return p_order

def dag_rescale_bw(data_in, old_min=None, old_max=None, new_min=0, new_max=1):
    data_out = np.copy(data_in)
    if old_min is not None:
        data_out[data_in<old_min] = old_min
    else:
        old_min = np.nanmin(data_in)
    if old_max is not None:
        data_out[data_in>old_max] = old_max
    else:
        old_max = np.nanmax(data_in)
    
    data_out = (data_out - old_min) / (old_max - old_min) # Scaled bw 0 and 1
    data_out = data_out * (new_max-new_min) + new_min # Scale bw new values
    return data_out

def dag_get_rsq(tc_target, tc_fit):
    ss_res = np.sum((tc_target-tc_fit)**2, axis=-1)
    ss_tot = np.sum(
        (tc_target-tc_target.mean(axis=-1)[...,np.newaxis])**2, 
        axis=-1
        )
    rsq = 1-(ss_res/ss_tot)

    return rsq

def dag_coord_convert(a,b,old2new):
    ''' 
    Convert cartesian to polar and vice versa
    >> a,b          x,y or eccentricity, polar
    >> old2new      direction of conversion ('pol2cart' or 'cart2pol') 
    '''
    if old2new=="pol2cart":
        x = a * np.cos(b)
        y = a * np.sin(b)

        new_a = x
        new_b = y
    
    elif old2new=="cart2pol":            
        ecc = np.sqrt( a**2 + b**2 ) # Eccentricity
        pol = np.arctan2( b, a ) # Polar angle
        new_a = ecc
        new_b = pol
                            
    return new_a, new_b    

def dag_get_pos_change(old_x, old_y, new_x, new_y):
    dx = new_x - old_x
    dy = new_y - old_y
    dsize = np.sqrt(dx**2 + dy**2)
    return dsize

def dag_str2file(filename, txt):
    file2write = open(filename, 'w')
    file2write.write(txt)
    file2write.close()

def dag_find_file_in_folder(filt, path, return_msg='error', exclude=None):
    """get_file_from_substring
    Essentially copied from Jurjen's linescanning toolbox. except with the option for multiple exclusion criteria

    This function returns the file given a path and a substring. Avoids annoying stuff with glob. Now also allows multiple filters to be applied to the list of files in the directory. The idea here is to construct a binary matrix of shape (files_in_directory, nr_of_filters), and test for each filter if it exists in the filename. If all filters are present in a file, then the entire row should be 1. This is what we'll be looking for. If multiple files are found in this manner, a list of paths is returned. If only 1 file was found, the string representing the filepath will be returned. 

    Parameters
    ----------
    filt: str, list
        tag for files we need to select. Now also support a list of multiple filters. 
    path: str
        path to the directory from which we need to remove files
    return_msg: str, optional
        whether to raise an error (*return_msg='error') or return None (*return_msg=None*). Default = 'error'.
    exclude: str, list, optional:
        Specify string/s to exclude from options. This criteria will be ensued after finding files that conform to `filt` as final filter.

    Returns
    ----------
    str
        path to the files containing `string`. If no files could be found, `None` is returned

    list
        list of paths if multiple files were found

    Raises
    ----------
    FileNotFoundError
        If no files usingn the specified filters could be found

    Example
    ----------
    >>> get_file_from_substring("R2", "/path/to/prf")
    '/path/to/prf/r2.npy'
    >>> get_file_from_substring(['gauss', 'best_vertices'], "path/to/pycortex/sub-xxx")
    '/path/to/pycortex/sub-xxx/sub-xxx_model-gauss_desc-best_vertices.csv'
    >>> get_file_from_substring(['best_vertices'], "path/to/pycortex/sub-xxx")
    ['/path/to/pycortex/sub-xxx/sub-xxx_model-gauss_desc-best_vertices.csv',
    '/path/to/pycortex/sub-xxx/sub-xxx_model-norm_desc-best_vertices.csv']    
    """

    input_is_list = False
    if isinstance(filt, str):
        filt = [filt]

    if isinstance(filt, list):
        # list and sort all files in the directory
        if isinstance(path, str):
            files_in_directory = sorted(os.listdir(path))

        elif isinstance(path, list):
            input_is_list = True
            files_in_directory = path.copy()
        else:
            raise ValueError("Unknown input type; should be string to path or list of files")

        # the idea is to create a binary matrix for the files in 'path', loop through the filters, and find the row where all values are 1
        filt_array = np.zeros((len(files_in_directory), len(filt)), dtype=bool)
        for ix,f in enumerate(files_in_directory):
            for filt_ix,filt_opt in enumerate(filt):
                filt_array[ix,filt_ix] = filt_opt in f

        # now we have a binary <number of files x number of filters> array. If all filters were available in a file, the entire row should be 1, 
        # so we're going to look for those rows
        full_match_idx = np.all(filt_array,axis=1)
        
        # Now repeat the same thing, for exclusion criteria
        if exclude==None:
            excl_idx = np.zeros(len(files_in_directory), dtype=bool)
        else:
            if not isinstance(exclude, list):
                exclude = [exclude] # Convert into list for iterations

            excl_array = np.zeros((len(files_in_directory), len(exclude)), dtype=bool)
            for ix,f in enumerate(files_in_directory):
                for excl_ix, excl_opt in enumerate(exclude):
                    excl_array[ix,excl_ix] = excl_opt in f

            excl_idx = np.any(excl_array, axis=1)
        # Combine the 2 criterion
        full_match_idx &= ~excl_idx # 
        full_match_idc = np.where(full_match_idx)[0]

        # Didn't find any matches?         
        if (not isinstance(full_match_idc, np.ndarray)) or (full_match_idc.shape[0]==0):
            # If there 
            if return_msg == "error":
                raise FileNotFoundError(f"Could not find fig with tags: {filt}, excluding: {exclude}, in {path}")        
            else:
                return None        
        
        if input_is_list:
            match_list = [files_in_directory[i] for i in full_match_idc]
        else:
            match_list = [opj(path,files_in_directory[i]) for i in full_match_idc]
        
        # Don't return a list if there is only one element
        if isinstance(match_list, list) & (len(match_list)==1):
            match_list = match_list[0]
        
        return match_list