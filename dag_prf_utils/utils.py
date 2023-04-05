import numpy as np

def dag_qprint(print_str):
    print(print_str, flush=True)

def dag_roi_idx_from_dot_label(path2dotlabel):
    with open(path2dotlabel) as f:
        contents = f.readline()
    
    # first 2 lines are not useful (look something like this):
    # #!ascii label  , from subject sub-005 vox2ras=TkReg\n
    # 5838\n
    total_num_vx = int(contents[1])

    idx_str = [contents[i].split(' ')[0] for i in range(2,len(contents))]
    idx_int = [int(idx_str[i]) for i in range(len(idx_str))]

    # check that the length is the same...
    assert total_num_vx==len(idx_int)

    return idx_int

def dag_hyphen_parse(str_prefix, str_in):
    if str_prefix in str_in:
        str_out = str_in
    else: 
        str_out = f'{str_prefix}-{str_in}'
    return str_out
    
def dag_print_p():
    p_order = {}
    p_order['gauss'] = {
        'x' : 0,
        'y' : 1,
        'a_sigma' : 2,
        'a_val' : 3,
        'bold_baseline' : 4,
        'rsq' : 5,
    }    

    p_order['norm'] = {
        'x' : 0,
        'y' : 1,
        'a_sigma' : 2,
        'a_val' : 3,
        'bold_baseline' : 4,
        'c_val' : 5,
        'n_sigma' : 6,
        'b_val' : 7,
        'd_val' : 8,
        'rsq' : 9,
    }    

    p_order['CSS'] = {
        'x' : 0,
        'y' : 1,
        'a_sigma' : 2,
        'beta' : 3,
        'bold_baseline' : 4,
        'n_exp' : 5,
        'rsq' : 6,
    }        

    p_order['DOG'] = {
        'x' : 0,
        'y' : 1,
        'a_sigma' : 2,
        'a_val' : 3,
        'bold_baseline' : 4,
        'c_val' : 5,
        'n_sigma' : 6,
        'rsq' : 7,
    }

    p_order['CSF']  ={
        'width_r' : 0,
        'sf0' : 1,
        'maxC' : 2,
        'width_l' : 3,
        'a_val' : 4,
        'baseline' : 5,
        'rsq' : 6,
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
