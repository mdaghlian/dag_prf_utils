import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
opj = os.path.join


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