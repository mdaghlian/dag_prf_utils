import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import struct
import string
import random
import time
opj = os.path.join

import subprocess
import shutil
from datetime import datetime

def dag_make_backup(source_path, ow=False):
    # Check if the source exists
    if not os.path.exists(source_path):
        print("Source does not exist.")
        return
    

    # Get the directory of the source file
    source_dir = os.path.dirname(source_path)
    
    # Get the filename of the source file
    filename = os.path.basename(source_path)
    
    # Define the backup directory (parent directory of the source directory)
    backup_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
    
    # Define the path for the backup file
    backup_file = os.path.join(source_dir, f"{filename}.backup")
    
    # Check if the backup file already exists
    if not ow:
        if os.path.exists(backup_file):
            # If the backup file already exists, append the current date and time to the filename
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"{filename}_{current_time}.backup")
    
    try:
    # If the source is a file, copy it to the backup path
        if os.path.isfile(source_path):
            shutil.copyfile(source_path, backup_file)
            print(f"Backup created: {backup_file}")
        # If the source is a directory, copy the entire directory to the backup path
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, backup_file)
            print(f"Backup created: {backup_file}")
    except Exception as e:
        print(f"Error creating backup: {e}")




def dag_get_cores_used():
    user_name = os.environ['USER']
    command = f"qstat -u {user_name}"  # Replace with your actual username
    output = subprocess.check_output(command, shell=True).decode('utf-8')
    if output == '':
        return 0

    lines = output.strip().split('\n')
    header = lines[0].split()    
    n_cols = len(lines[1].split())

    count = 0 # sometimes take a second to load...
    while 'qw' in output: # 
        time.sleep(5)
        count += 1
        
        output = subprocess.check_output(command, shell=True).decode('utf-8')
        if output == '':
            return 0    
        if 'Eqw' in output:
            print('EQW')    
            sys.exit()
        print(output)

        lines = output.strip().split('\n')
        header = lines[0].split()    
        if count > 10:
            print('bloop')
            break

    cores_index = header.index('slots')  # Or 'TPN' if 'C' is not available
    cores = 0
    for line in lines[2:]:
        columns = line.split()
        if columns:
            cores += int(columns[cores_index])

    return cores 

def dag_qprint(print_str):
    print(print_str, flush=True)

def dag_load_nverts(sub, fs_dir = os.environ['SUBJECTS_DIR']):    
    '''
    nverts (points) in a given mesh
    '''
    n_verts, n_faces = dag_load_nfaces_nverts(sub, fs_dir)
    return n_verts

def dag_load_nfaces(sub, fs_dir=os.environ['SUBJECTS_DIR']):
    '''
    nfaces (triangular) in a given mesh
    '''
    n_verts, n_faces = dag_load_nfaces_nverts(sub, fs_dir)
    return n_faces

def dag_load_nfaces_nverts(sub, fs_dir=os.environ['SUBJECTS_DIR']):
    """
    Adapted from pycortex https://github.com/gallantlab/pycortex
    Load the number of vertices and faces in a given mesh
    """    
    n_faces = []
    n_verts = []
    for i in ['lh', 'rh']:
        surf = opj(fs_dir, sub, 'surf', f'{i}.inflated')
        with open(surf, 'rb') as fp:
            #skip magic
            fp.seek(3)
            fp.readline()
            comment = fp.readline()            
            i_verts, i_faces = struct.unpack('>2I', fp.read(8))
            n_verts.append(i_verts)    
            n_faces.append(i_faces)    
    return n_verts, n_faces


def dag_load_roi(sub, roi, fs_dir=os.environ['SUBJECTS_DIR'], split_LR=False, do_bool=True, **kwargs):
    '''
    Return a boolean array of voxels included in the specified roi
    array is vector with each entry corresponding to a point on the subjects cortical surface
    (Note this is L & R hemi combined)

    roi can be a list (in which case more than one is included)
    roi can also be exclusive (i.e., everything *but* x)

    TODO - hemi specific idx...
    '''
    need_both_hemis = kwargs.get('need_both_hemis', False) # Need ROI in both hemispheres to return true
    # Get number of vx in each hemi, and total overall...
    n_verts = dag_load_nverts(sub=sub, fs_dir=fs_dir)
    total_num_vx = np.sum(n_verts)
    
    # ****************************************
    # SPECIAL CASES [all, occ, demo]
    if 'all' in roi :        
        if split_LR:
            roi_idx = {}
            roi_idx['lh'] = np.ones(n_verts[0], dtype=bool)
            roi_idx['rh'] = np.ones(n_verts[1], dtype=bool)
        else:
            roi_idx = np.ones(total_num_vx, dtype=bool)
        return roi_idx    
    elif roi=='occ':
        roi_idx = dag_id_occ_ctx(sub=sub, fs_dir=fs_dir, split_LR=split_LR, **kwargs)
        return roi_idx
    elif 'demo' in roi:
        if '-' in roi:
            n_demo = int(roi.split('-')[-1])
        else:
            n_demo = 100
        if split_LR:
            roi_idx = {}
            roi_idx['lh'] = np.zeros(n_verts[0], dtype=bool)
            roi_idx['rh'] = np.zeros(n_verts[1], dtype=bool)
            roi_idx['lh'][:n_demo] = True
            roi_idx['rh'][:n_demo] = True

        else:
            roi_idx = np.zeros(total_num_vx, dtype=bool)        
            roi_idx[:n_demo] = True

        return roi_idx
    
    elif '+' in roi:
        roi = roi.split('+')
    # ****************************************
        
    # Else look for rois in subs freesurfer label folder
    roi_dir = opj(fs_dir, sub, 'label')    
    if not isinstance(roi, list): # roi can be a list 
        roi = [roi]    

    roi_idx = []
    roi_idx_split = {'lh':[], 'rh':[]}
    for this_roi in roi:    
        # Find the corresponding files
        if 'not' in this_roi:
            do_not = True
            this_roi = this_roi.split('-')[-1]
        else:
            do_not = False
        roi_file = {}
        # try:
        #     roi_file['lh'] = dag_find_file_in_folder([this_roi, '.thresh', '.label', 'lh'], roi_dir, recursive=True)    
        #     roi_file['rh'] = dag_find_file_in_folder([this_roi, '.thresh', '.label', 'rh'], roi_dir, recursive=True)    
        # except:
        #     roi_file['lh'] = dag_find_file_in_folder([this_roi, '.label', 'lh'], roi_dir,exclude='._', recursive=True)    
        #     roi_file['rh'] = dag_find_file_in_folder([this_roi, '.label', 'rh'], roi_dir,exclude='._', recursive=True)    
        missing_hemi = False # Do we have an ROI for both hemis? 
        for hemi in ['lh', 'rh']:
            roi_file[hemi] = dag_find_file_in_folder([this_roi, '.thresh', '.label', hemi], roi_dir, recursive=True, return_msg=None)
            # Didn't find it? Try again without "thresh"
            if roi_file[hemi] is None:
                roi_file[hemi] = dag_find_file_in_folder([this_roi, '.label', hemi], roi_dir,exclude='._', recursive=True, return_msg = None)                
            # Did we find it now? 
            if roi_file[hemi] is None:
                # If not make a note - no entry for this hemi
                missing_hemi = True
            else:        
                # If more than 1 match print the matches and raise error
                if isinstance(roi_file[hemi], list):
                    print(f'Multiple matches for {this_roi} in {roi_dir}')
                    print(roi_file[hemi])
                    raise ValueError
        if need_both_hemis and missing_hemi:
            print(f'Missing ROI in one hemisphere')
            print(roi_file)
            raise ValueError

        LR_bool = []
        for i,hemi in enumerate(['lh', 'rh']):
            if roi_file[hemi] is None:
                idx_int = []
            else:
                with open(roi_file[hemi]) as f:
                    contents = f.readlines()            
                idx_str = [contents[i].split(' ')[0] for i in range(2,len(contents))]
                idx_int = [int(idx_str[i]) for i in range(len(idx_str))]
                # val_str = [contents[i].split(' ')[-1].split('\n')[0] for i in range(2,len(contents))]
                # val_int = [int(float(val_str[i])) for i in range(len(val_str))]

            if do_bool:
                this_bool = np.zeros(n_verts[i], dtype=int)
                this_bool[idx_int] = True
            # else:
            #     this_bool = np.zeros(n_verts[i], dtype=bool)
            #     this_bool[idx_int] = val_int
            if do_not:            
                this_bool = ~this_bool

            LR_bool.append(this_bool)
        this_roi_mask = np.concatenate(LR_bool)
        roi_idx.append(this_roi_mask)
        roi_idx_split['lh'].append(LR_bool[0]) 
        roi_idx_split['rh'].append(LR_bool[1])

    roi_idx = np.vstack(roi_idx)
    roi_idx_split['lh'] = np.vstack(roi_idx_split['lh'])
    roi_idx_split['rh'] = np.vstack(roi_idx_split['rh'])
    if do_bool:
        roi_idx = roi_idx.any(0)
        roi_idx_split['lh'] = roi_idx_split['lh'].any(0)
        roi_idx_split['rh'] = roi_idx_split['rh'].any(0)    
    else:
        roi_idx = np.squeeze(roi_idx)
        roi_idx_split['lh'] = np.squeeze(roi_idx_split['lh'])
        roi_idx_split['rh'] = np.squeeze(roi_idx_split['rh'])

    if split_LR:
        return roi_idx_split
    else:
        return roi_idx

def dag_id_occ_ctx(sub, fs_dir, split_LR=False, max_y=-35):
    '''
    Return the rough coordinates for the occipital cortex
    '''
    occ_idx = []
    occ_idx_split = {}
    for i_hemi in ['lh', 'rh']:
        surf = opj(fs_dir, sub, 'surf', f'{i_hemi}.inflated')
        mesh_info = dag_read_fs_mesh(surf)
        occ_idx.append(mesh_info['coords'][:,1]<=max_y)        
        occ_idx_split[i_hemi] = mesh_info['coords'][:,1]<=max_y
    
    occ_idx = np.concatenate(occ_idx)

    if split_LR:
        return occ_idx_split
    else:
        return occ_idx    

def dag_hyphen_parse(str_prefix, str_in):
    '''dag_hyphen_parse
    checks whether a string has a prefix attached.
    Useful for many BIDS format stuff, and when passing arguments on a lot 
    (sometimes it is not clear whether the prefix will be present or not...)

    E.g., I want to make sure that string "task_name" has the format "task-A" 
    part_task_name = "A"
    full_task_name = "task-A"
    
    dag_hyphen_parse("task", part_task_name)
    dag_hyphen_parse("task", full_task_name)

    Both output -> "task-A"
    
    '''
    if str_prefix in str_in:
        str_out = str_in
    else: 
        str_out = f'{str_prefix}-{str_in}'
    return str_out
    
def dag_rescale_bw(data_in, old_min=None, old_max=None, new_min=0, new_max=1):
    '''dag_rescale_bw

    
    '''
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
    '''dag_get_rsq
    Calculate the rsq (R squared)
    Of a fit time course (tc_fit), on a target (tc_target)    
    '''
    ss_res = np.sum((tc_target-tc_fit)**2, axis=-1)
    ss_tot = np.sum(
        (tc_target-tc_target.mean(axis=-1)[...,np.newaxis])**2, 
        axis=-1
        )
    rsq = 1-(ss_res/ss_tot)

    return rsq

def dag_filter_for_nans(array):
    """
    filter out NaNs from an array
    Copied from JH linescanning toolbox
    """

    if np.isnan(array).any():
        return np.nan_to_num(array)
    else:
        return array
    
def dag_get_corr(a, b):
    '''dag_get_corr
    '''
    corr = np.corrcoef(a,b)[0,1]
    return corr

def dag_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))    
    return result_str

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

def dag_pol_difference(pol, ref_pol):
    abs_diff = np.abs(ref_pol - pol)
    abs_diff = np.min(abs_diff, 2*np.pi-abs_diff)
    return abs_diff

def dag_merid_idx(x, y, wedge_angle=10, angle_type='deg'):
    """
    Categorize points based on their position relative to specified meridians.

    Parameters:
    - x: NumPy array of x-coordinates
    - y: NumPy array of y-coordinates
    - wedge_angle: Number of degrees around each meridian center (+/-)
    - angly_type: is wedge_angle specified in degrees or radians

    Returns:
    - Dictionary with meridians as keys and boolean NumPy arrays indicating points within each meridian's range
    """
    # Define meridian centers
    merid_centers = {'right': 0, 'upper': np.pi/2, 'left': np.pi, 'lower': -np.pi/2}
    if angle_type=='deg':
        # Convert degrees around meridian to rad
        wedge_angle *= np.pi/180
    # Calculate polar angle
    pol = np.arctan2(y, x) 
    
    merid_idx = {}
    for merid,merid_center in merid_centers.items():        
        # Get difference from meridian centre
        abs_diff = np.abs(merid_center - pol)
        abs_diff = np.min([abs_diff, 2*np.pi-abs_diff], axis=0)
        # print(abs_diff.shape)
        merid_idx[merid] = abs_diff <= wedge_angle

    merid_label = np.full(x.shape[0], 'na', dtype='object')
    
    for m, m_idx in merid_idx.items():
        merid_label[m_idx] = m      
    merid_idx['label'] = merid_label
    return merid_idx


def dag_pol_to_clock(pol):
    # Convert angles to the range [0, 2*pi)
    # rotate by 90
    pol = pol + np.pi/2
    pol = np.mod(pol, 2 * np.pi)

    # Convert angles to the range [0, 12)
    clock_values = (pol / (2 * np.pi)) * 12
    return clock_values

def dag_weighted_mean(w,x, axis='all'):
    # w_mean = np.sum(w * x) / np.sum(w) # original form
    if axis=='all':
        w_mean = np.nansum(w * x) / np.nansum(w)
    else:
        w_mean = np.nansum(w * x, axis=axis) / np.nansum(w, axis=axis)

    return w_mean


def dag_get_pos_change(old_x, old_y, new_x, new_y):
    dx = new_x - old_x
    dy = new_y - old_y
    dsize = np.sqrt(dx**2 + dy**2)
    return dsize

def dag_str2file(filename, txt):
    file2write = open(filename, 'w')
    file2write.write(txt)
    file2write.close()

def dag_find_file_in_folder_OLD(filt, path, return_msg='error', exclude=None):
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
    

def dag_find_file_in_folder(filt, path, return_msg='error', exclude=None, recursive=False, file_limit=9999):
    """get_file_from_substring
    Setup to be compatible with JH linescanning toolbox function (linescanning.utils.get_file_from_substring)
    

    This function returns the file given a path and a substring. Avoids annoying stuff with glob. Now also allows multiple filters 
    to be applied to the list of files in the directory. The idea here is to construct a binary matrix of shape (files_in_directory, nr_of_filters), and test for each filter if it exists in the filename. If all filters are present in a file, then the entire row should be 1. This is what we'll be looking for. If multiple files are found in this manner, a list of paths is returned. If only 1 file was found, the string representing the filepath will be returned. 

    Parameters
    ----------
    filt: str, list
        tag for files we need to select
    path: str
        path to the folder we are searching directory 
        OR a list of strings (files), which will be searched
    return_msg: str, optional
        whether to raise an error (*return_msg='error') or return None (*return_msg=None*). Default = 'error'.
    exclude: str, list, optional:
        Specify string/s to exclude from options. 

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

    """
    # [1] Setup filters (should be lists): 
    filt_incl = filt
    if isinstance(filt_incl, str):
        filt_incl = [filt_incl]
    filt_excl = exclude
    if (filt_excl!=None) and isinstance(filt_excl, str):
        filt_excl = [filt_excl]

    # [2] List & sort files in directory
    if isinstance(path, str):
        input_is_list = False
        folder_path = path
    elif isinstance(path, list):        
        # The list of files is specified...
        input_is_list = True
        files = path.copy()
    else:
        raise ValueError("Unknown input type; should be string to path or list of files")

    matching_files = []
    files_searched = 0
    
    if input_is_list:   # *** Prespecified list of files ***
        for file_name in files:
            # Check if the file name contains all strings in filt_incl
            if all(string in file_name for string in filt_incl):                
                # Check if the file name contains any strings in filt_excl, if provided
                if filt_excl is not None and any(string in file_name for string in filt_excl):
                    continue
                
                matching_files.append(file_name)
    
    else:               # *** Walk through folders ***
        for root, dirs, files in os.walk(folder_path):
            if not recursive and root != folder_path:
                break        
            
            for file_name in files:
                files_searched += 1
                file_path = os.path.join(root, file_name)

                # Check the inclusion & exclusion criteria
                file_match = dag_file_name_check(file_name, filt_incl, filt_excl)
                if file_match:
                    matching_files.append(file_path)

                # Check if the limit has been reached
                if files_searched >= file_limit:
                    sys.exit()

    # Sort the matching files
    match_list = sorted(matching_files)
    
    # Are there any matching files? -> error option
    no_matches = len(match_list)==0
    if no_matches:
        if return_msg == "error":
            raise FileNotFoundError(f"Could not find file with incl {filt_incl}, excluding: {filt_excl}, in {path}")        
        else:
            return None        
    
    # Don't return a list if there is only one element
    if isinstance(match_list, list) & (len(match_list)==1):
        match_list = match_list[0]


    return match_list


def dag_file_name_check(file_name, filt_incl, filt_excl):
    file_match = False
    # Check if the file name contains all strings in filt_incl
    if all(string in file_name for string in filt_incl):
        file_match = True
    
    # Check if the file name contains any strings in filt_excl
    if filt_excl is not None and any(string in file_name for string in filt_excl):
        file_match = False
    return file_match

def dag_merge_dicts(a: dict, b: dict, max_depth=3, path=[]):
    '''
    Merge two dictionaries recursively
    Adapted from
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries    
    '''    
    merged_dict = a.copy()  # Create a copy of dictionary 'a' to start with
    for key in b:
        if key in merged_dict:
            if isinstance(merged_dict[key], dict) and isinstance(b[key], dict):
                if len(path) < max_depth:
                    # Recursively merge dictionaries
                    merged_dict[key] = dag_merge_dicts(merged_dict[key], b[key], max_depth, path + [str(key)])
                else:
                    raise Exception('Max depth reached at ' + '.'.join(path + [str(key)]))
            elif merged_dict[key] != b[key]:
                raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            merged_dict[key] = b[key]  # If the key is not in 'merged_dict', add it
    return merged_dict    


# ***********************************************************************************************************************
# STUFF COPIED FROM NIBABEL
# ***********************************************************************************************************************
def dag_fread3(fobj):
    """Read a 3-byte int from an open binary file object

    Parameters
    ----------
    fobj : file
        File descriptor

    Returns
    -------
    n : int
        A 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, '>u1', 3)
    return (b1 << 16) + (b2 << 8) + b3


def dag_read_fs_mesh(filepath, return_xyz=False):
    """Adapted from https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py
    ...
    Read a triangular format Freesurfer surface mesh.

    Parameters
    ----------
    filepath : str
        Path to surface file.

    Returns
    -------
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    """

    TRIANGLE_MAGIC = 16777214
    with open(filepath, 'rb') as fobj:

        magic = dag_fread3(fobj)
        create_stamp = fobj.readline().rstrip(b'\n').decode('utf-8')
        fobj.readline()
        vnum = np.fromfile(fobj, '>i4', 1)[0]
        fnum = np.fromfile(fobj, '>i4', 1)[0]
        coords = np.fromfile(fobj, '>f4', vnum * 3).reshape(vnum, 3)
        faces = np.fromfile(fobj, '>i4', fnum * 3).reshape(fnum, 3)

    coords = coords.astype(np.float64)  # XXX: due to mayavi bug on mac 32bits

    mesh_info = {
        'vnum' : vnum,
        'fnum' : fnum,
        'coords' : coords,
        'faces' : faces,        
    }
    if return_xyz:
        new_mesh_info = {}                                    
        new_mesh_info['x']= mesh_info['coords'][:,0]
        new_mesh_info['y']= mesh_info['coords'][:,1]
        new_mesh_info['z']= mesh_info['coords'][:,2]
        new_mesh_info['i']= mesh_info['faces'][:,0]
        new_mesh_info['j']= mesh_info['faces'][:,1]
        new_mesh_info['k']= mesh_info['faces'][:,2]        
        mesh_info = new_mesh_info

    return mesh_info
# ***


def dag_read_fs_curv_file(curv_file):
    with open(curv_file, 'rb') as h_us:
        h_us.seek(15)
        curv_vals = np.fromstring(h_us.read(), dtype='>f4').byteswap().newbyteorder()
    return curv_vals

# ***********************************************************************************************

