import json
import os
opj = os.path.join
import numpy as np
import re
from copy import copy
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from dag_prf_utils.utils import dag_rescale_bw

# Load custom color maps
path_to_utils = os.path.abspath(os.path.dirname(__file__))
custom_col_path = opj(path_to_utils, 'cmaps.json')

with open(custom_col_path, 'r') as fp:
    custom_col_dict = json.load(fp)


def dag_cmap_from_str(cmap_name, **kwargs):
    '''from a string make a cmap on the fly!
    Codes

    _rev_ : reverse the colormap
    _rot-X : rotate the colormap by a certain amount
    _log_ : log scale the colormap

    '''
    try:
        cmap = dag_get_cmap(cmap_name, **kwargs)
        print(f'{cmap_name} exists')
        return cmap
    except:
        pass
    # Check for reverse
    if '_rev_' in cmap_name:
        do_reverse = True
        cmap_name = cmap_name.replace('_rev_', '')
    else:
        do_reverse = False
        
    if '_log_' in cmap_name:
        do_log = True
        cmap_name = cmap_name.replace('_log_', '')
    else:
        do_log = False        
    
    # Check for rotation
    # find string matching pattern '_rot-X' where X is a number    
    rot_str = re.search(r'_rot-?\d+', cmap_name)
    if rot_str is not None:
        do_rotation = int(rot_str.group().split('_rot')[1])
        cmap_name = cmap_name.replace(rot_str.group(), '')
    else:
        do_rotation = False
    
    get_cmap_kwargs = {
        'reverse': do_reverse,
        'log': do_log,
        'rotation': do_rotation,
    }
    save_cmap_kwargs = {
        'cmap_name' : 'temp',
        'save_cmap' : True,
        'ow' : True, 
    }
    if '&' in cmap_name: # Stack cmaps (existing)
        cmap_list = cmap_name.split('&')
        cmap = dag_stack_cmaps(cmap_list=cmap_list, **save_cmap_kwargs)    
        temp_cmap_name = 'temp'
    elif '*' in cmap_name: # Stack colors 
        cmap = dag_make_custom_cmap(col_list=cmap_name.split('*'), **save_cmap_kwargs)
        temp_cmap_name = 'temp'

    else:
        temp_cmap_name = cmap_name    

    cmap = dag_get_cmap(temp_cmap_name, **get_cmap_kwargs)    
    
    return cmap

def dag_get_col_vals(col_vals, cmap, vmin=None, vmax=None, str_search=False):
    try:
        cmap = dag_get_cmap(cmap)
    except:
        cmap = dag_cmap_from_str(cmap)
    # cmap = mpl.cm.__dict__[cmap]
    cnorm = mpl.colors.Normalize()
    if vmin is not None:
        cnorm.vmin = vmin
        cnorm.vmax = vmax
    col_out = cmap(cnorm(col_vals))
    return col_out

def dag_rotate_cmap(cmap, rot, **kwargs):
    '''dag_rotate_cmap
    Rotates a colormap by a certain amount

    Parameters
    ----------
    cmap : str
        Name of the colormap
    rot : float
        Amount to rotate the colormap by

    Returns
    -------
    cmap : matplotlib colormap
        Rotated colormap
    '''    
    cmap_name = kwargs.pop('cmap_name', f'{cmap}_rot{rot}')
    n_steps = 360
    col_steps = np.linspace(0,360,n_steps)
    col_vals = dag_get_col_vals(col_steps, cmap, vmin=0, vmax=360)
    col_vals = np.roll(col_vals, int(rot), axis=0)
    cmap = dag_make_custom_cmap(
        col_list=col_vals, 
        col_steps=col_steps, 
        cmap_name=cmap_name,
        **kwargs
        )
    return cmap

def dag_stack_cmaps(cmap_list, save_cmap=False, **kwargs):
    cmap_name = kwargs.pop('cmap_name', None)
    n_steps_per_cmap = kwargs.pop('n_steps_per_cmap', 100)
    n_cmaps = len(cmap_list)
    # Now list of col vals
    col_list = []
    for i_cmap in cmap_list:
        try:
            this_cmap = dag_get_cmap(i_cmap)
        except:
            this_cmap = dag_cmap_from_str(i_cmap)
        c_norm = mpl.colors.Normalize()
        c_norm.vmin = 0
        c_norm.vmax = 1
        this_rgb_cols = c_norm(this_cmap(np.linspace(0,1,n_steps_per_cmap)))
        this_r = np.around(this_rgb_cols[:,0],3)
        this_g = np.around(this_rgb_cols[:,1],3)
        this_b = np.around(this_rgb_cols[:,2],3)
        col_list += [
            (this_r[i], this_g[i], this_b[i]) for i in range(this_r.shape[0])
        ]

    col_steps = np.linspace(0,1,n_steps_per_cmap*n_cmaps)
    if cmap_name is None:
        cmap_name = '_'.join(cmap_list)
        cmap_name = str.replace(cmap_name, '_r', 'rev')
    new_cmap = dag_make_custom_cmap(
        col_list=col_list, 
        col_steps=col_steps, 
        cmap_name=cmap_name, 
        save_cmap=save_cmap,
        **kwargs,
        )    

    return new_cmap    


def dag_make_custom_cmap(col_list, col_steps=None, **kwargs):
    """Return a LinearSegmentedColormap
    col_list        list of colors (can be rgb tuples or something which can be converted to rgb tuples by mcolors.ColorConverter().to_rgb)
    col_steps
    
    inspired by: https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale

    Example
    ------

    """
    do_log = kwargs.pop('log', False)
    save_cmap = kwargs.pop('save_cmap', False)
    cmap_name = kwargs.pop('cmap_name', '')
    if col_steps is None:
        col_val = np.linspace(0,1, len(col_list)) # default to linear spacing
    elif isinstance(col_steps, list) or isinstance(col_steps, np.ndarray):
        col_val = np.array(col_steps) # use the specified steps
    
    # Rescale to 0-1 (with option to log scale)
    col_val = dag_rescale_bw(col_val, log=do_log)
    
    # Change any values to rgb tuple
    conv2rgb = mcolors.ColorConverter().to_rgb
    for i_col,v_col in enumerate(col_list):
        if isinstance(v_col, str) & ('(' in v_col) or ('[' in v_col):
            # Convert string to tuple
            v_col = eval(v_col)
            col_list[i_col] = v_col
        if isinstance(v_col, np.ndarray):
            v_col = tuple(v_col)
            col_list[i_col] = v_col
        if (not isinstance(v_col, tuple)) and (not isinstance(v_col, list)):
            col_list[i_col] = conv2rgb(v_col) 
            print(col_list[i_col])
    # Check whether it is in 255 format (should be b/w 0 and 1)
    is_255 = False
    for i_col, v_col in enumerate(col_list):
        for i_rgb, v_rgb in enumerate(v_col):
            if v_rgb > 1:
                is_255 = True
    if is_255:
        for i_col, v_col in enumerate(col_list):
            col_list[i_col] = dag_rgb(*col_list[i_col])     
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, list(zip(col_val, col_list)))
    # save as temp cmap then apply the kwargs
    # dag_save_cmap(
    #     cmap_name='temp',
    #     col_steps=col_val,
    #     col_list=col_list,
    #     ow=True,
    #     )
    # custom_cmap = dag_get_cmap('temp', **kwargs)
    if save_cmap:
        if cmap_name=='':
            print('specify name!')
            return        
        
        dag_save_cmap(
            cmap_name=cmap_name,
            col_steps=col_val,
            col_list=col_list,
            **kwargs
        )
        
    return custom_cmap

def dag_get_cmap(cmap_name, **kwargs):    
    '''dag_get_cmap
    Loads custom cmaps (specified in cmaps.json) 
    or default matplotlib versions
    or something specified with a dictionary

    return mpl.colors.LinearSegmentedColormap 
    '''
    if cmap_name is None:
        cmap_name = 'viridis'
    
    if isinstance(cmap_name, mpl.colors.Colormap):
        return cmap_name
    do_reverse = kwargs.get('reverse', False)     
    do_log = kwargs.get('log', False)
    do_rotation = kwargs.get('rotation', False)

    # Check for '_r' in cmap name
    if '_r'==cmap_name[-2:]:
        do_reverse = True
        cmap_name = cmap_name.split('_r')[0]
    if '_log'==cmap_name[-4:]:
        do_log = True
        cmap_name = cmap_name.split('_log')[0]

    if isinstance(cmap_name, dict):
        cdict_copy = cmap_name
        cmap_name = cdict_copy.get('cmap_name', '')        
        col_list = cdict_copy.get('col_list', None)
        col_steps = cdict_copy.get('col_steps', None)    
    else:
        col_list = kwargs.get('col_list', None)
        col_steps = kwargs.get('col_steps', None)        
    
    cc_dict = dag_load_custom_col_dict()    
    if col_list is not None:
        this_cmap = dag_make_custom_cmap(col_list=col_list, col_steps=col_steps, cmap_name=cmap_name)
    elif cmap_name in cc_dict.keys():
        col_list = cc_dict[cmap_name]['col_list']
        col_steps = cc_dict[cmap_name]['col_steps']
        if do_log:
            col_steps = dag_rescale_bw(col_steps, log=True)
        this_cmap = dag_make_custom_cmap(col_list=col_list, col_steps=col_steps, cmap_name=cmap_name)
    elif cmap_name in mpl.cm.__dict__.keys():
        this_cmap = mpl.cm.__dict__[cmap_name]
        if do_log:
            col_list = this_cmap(np.linspace(0,1,100))
            this_cmap = dag_make_custom_cmap(col_list=col_list, log=True)    
    if do_rotation:
        this_cmap = dag_rotate_cmap(this_cmap, do_rotation, **kwargs)

    if do_reverse:
        this_cmap = this_cmap.reversed()


    return this_cmap

def dag_delete_cmap(cmap_name, sure=False):
    cc_dict = dag_load_custom_col_dict()
    if cmap_name in cc_dict.keys():
        if not sure:
            print(f'Are you sure you want to delete {cmap_name}? (y/n)')
            delete_yn = input()
            if delete_yn!='y':
                return
    else:
        print(f'Cannot find {cmap_name}')
        return

    new_cc_dict = {}
    for i_cmap in cc_dict.keys():
        if i_cmap!=cmap_name:
            new_cc_dict[i_cmap] = cc_dict[i_cmap]
    print('deleting cmap...')
    with open(custom_col_path, 'w') as fp:
        json.dump(new_cc_dict, fp,sort_keys=True, indent=4)
    return    
    
def dag_save_cmap(cmap_name, col_list, col_steps=None, ow=False):
    if col_steps is None:
        col_steps = np.linspace(0,1, len(col_list))
    cc_dict = dag_load_custom_col_dict()
    while (cmap_name in custom_col_dict.keys()) and (not ow):
        print(f'{cmap_name} already exists, overwrite? (y/n)')
        overwrite = input()
        if overwrite=='y':
            break            
        print('Enter name for cmap')
        cmap_name = input()            
    print(f'saving cmap {cmap_name}')
    # print(f'col_list={col_list}')    
    # print(f'col_steps={col_steps}')    
    cc_dict[cmap_name] = {}
    cc_dict[cmap_name]['col_list'] = list(col_list)
    cc_dict[cmap_name]['col_steps'] = list(col_steps)    
    # Make a backup...
    os.system(f'cp {custom_col_path} {custom_col_path}.bu')    
    with open(custom_col_path, 'w') as fp:
        json.dump(cc_dict, fp,sort_keys=True, indent=4)
    return

def dag_load_custom_col_dict():
    with open(custom_col_path, 'r') as fp:
        cc_dict = json.load(fp)        
    return cc_dict

def dag_rgb(r,g,b):
    return [r/255,g/255,b/255]


# *************************** LEGACY ***************************
def LEGACY_dag_make_custom_cmap(col_list, col_steps=None, cmap_name='', save_cmap=False):
    """Return a LinearSegmentedColormap
    col_list        list of colors (can be rgb tuples or something which can be converted to rgb tuples by mcolors.ColorConverter().to_rgb)
    col_steps
    
    inspired by: https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale

    Example
    ------

    """
    if col_steps is None:
        col_val = np.linspace(0,1, len(col_list))
    elif isinstance(col_steps, list) or isinstance(col_steps, np.ndarray):
        col_val = np.array(col_steps)

    col_val = dag_rescale_bw(col_val) # recale to b/w 0 and 1
    # Change any values to rgb tuple
    conv2rgb = mcolors.ColorConverter().to_rgb
    for i_col,v_col in enumerate(col_list):
        if isinstance(v_col, np.ndarray):
            v_col = tuple(v_col)

        if (not isinstance(v_col, tuple)) and (not isinstance(v_col, list)):
            col_list[i_col] = conv2rgb(v_col) 
    # Check whether it is in 255 format (should be b/w 0 and 1)
    is_255 = False
    for i_col, v_col in enumerate(col_list):
        for i_rgb, v_rgb in enumerate(v_col):
            if v_rgb > 1:
                is_255 = True
    if is_255:
        for i_col, v_col in enumerate(col_list):
            col_list[i_col] = dag_rgb(*col_list[i_col]) 

    
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name,list(zip(col_val, col_list)))
    if save_cmap:
        if cmap_name=='':
            print('specify name!')
            return
        dag_save_cmap(
            cmap_name=cmap_name,
            col_steps=col_val,
            col_list=col_list
        )
        
    return custom_cmap

def LEGACY_dag_get_cmap(cmap_name, **kwargs):    
    '''dag_get_cmap
    Loads custom cmaps (specified in cmaps.json) or default matplotlib versions

    do_reverse : reverse the colormap

    '''
    if cmap_name is None:
        cmap_name = 'viridis'
    do_reverse = kwargs.get('reverse', False)    
    do_log = kwargs.get('log', False)
    do_rotation = kwargs.get('rotation', False)

    if isinstance(cmap_name, dict):
        cdict_copy = cmap_name
        cmap_name = cdict_copy.get('cmap_name', '')        
        col_list = cdict_copy.get('col_list', None)
        col_steps = cdict_copy.get('col_steps', None)    
    else:
        col_list = kwargs.get('col_list', None)
        col_steps = kwargs.get('col_steps', None)
    
    # Check for '_r' in cmap name
    if '_r' in cmap_name:
        do_reverse = True
        cmap_name = cmap_name.split('_r')[0]
    if '_log' in cmap_name:
        do_log = True
        cmap_name = cmap_name.split('_log')[0]
    
    cc_dict = dag_load_custom_col_dict()
    if col_list is not None:
        this_cmap = dag_make_custom_cmap(col_list=col_list, col_steps=col_steps, cmap_name=cmap_name)
    elif cmap_name in cc_dict.keys():
        col_list = cc_dict[cmap_name]['col_list']
        col_steps = cc_dict[cmap_name]['col_steps']
        if do_log:
            col_steps = np.log10(col_steps)
        this_cmap = dag_make_custom_cmap(col_list=col_list, col_steps=col_steps, cmap_name=cmap_name)
    elif cmap_name in mpl.cm.__dict__.keys():
        this_cmap = mpl.cm.__dict__[cmap_name]
        if do_log:
            col_list = this_cmap(np.linspace(0,1,100))
            this_cmap = dag_make_custom_cmap(col_list=col_list, col_steps=np.logspace(.01,1,100))
    if do_reverse:
        this_cmap = this_cmap.reversed()
    return this_cmap

def LEGACY_dag_make_diverge_cmap(low, high, mid='white'):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    custom_cmap = dag_make_custom_cmap([low, mid, high])
    return custom_cmap
