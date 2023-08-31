import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys
opj = os.path.join

from .utils import *
from .plot_functions import *

def prfpy_params_dict():
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
        'hrf_deriv'     :  5, # *hrf_1
        'hrf_disp'      :  6, # *hrf_2
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
        'hrf_deriv'     :  6, # *hrf_1
        'hrf_disp'      :  7, # *hrf_2        
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
        'hrf_deriv'     :  7, # *hrf_1
        'hrf_disp'      :  8, # *hrf_2        
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
        'hrf_deriv'     :  9, # *hrf_1
        'hrf_disp'      : 10, # *hrf_2        
        'rsq'           : -1, # rsq
    }            

    p_order['csf']  ={
        'width_r'       :  0,
        'sf0'           :  1,
        'maxC'          :  2,
        'width_l'       :  3,
        'amp_1'         :  4,
        'bold_baseline' :  5,
        'hrf_deriv'     :  6, # *hrf_1
        'hrf_disp'      :  7, # *hrf_2        
        'rsq'           : -1,
    }

    return p_order

def set_tc_shape (tc_in, n_timepts = 225):
    '''set_tc_shape
    Force the timecourse to be n_timepts x n_units
    '''
    # *** ALWAYS n_units * n_time
    if tc_in.shape[0] == n_timepts:
        tc_out = tc_in.T
    else:
        tc_out = tc_in
    return tc_out

def mask_time_series(ts, mask, ts_axis = 1, zero_pad = False):    
    '''mask_time_series    
    Mask certain voxel time series for later fitting. This is useful if you want to fit a subset of voxels, 
    to speed up fitting.

    Input:
    ----------
    ts          np.ndarray          time series, default is nvx x time
    mask        np.ndarray, bool    which vx to include (=True)
    ts_axis     int                 which axis is time
    zero_pad    bool                return the masked vx as flat time series    
        
    Output:
    ----------
    ts_out      np.ndarray          masked time series
    '''
    if zero_pad:
        # initialize empty array and only keep the timecourses from label; 
        # keeps the original dimensions for simplicity sake!     
        ts_out = np.zeros_like(ts)

        # insert timecourses 
        lbl_true = np.where(mask == True)[0]
        if ts_axis==0:
            ts_out[:,lbl_true] = ts[:,lbl_true]
        elif ts_axis==1:
            ts_out[lbl_true,:] = ts[lbl_true,:]
        else:
            print('Bad ts_axis...')
            sys.exit()
    else:
        # Don't pad everything...
        if ts_axis==0:
            ts_out = np.copy(ts[:,mask])        
        elif ts_axis==1:
            ts_out = np.copy(ts[mask,:])
        else:
            print('Bad ts_axis...')
            sys.exit()
    
    return ts_out

def process_prfpy_out(prfpy_out, mask=None):    
    '''process_prfpy_out
    Fit parameters can come out with nans, and are in the wrong shape
    If we masked certain timeseries, we want to go back and put in empty values for fitted parameters. 
    This is so that the shape of the vector is nice... (i.e., fits on the surface)
    '''
    if mask is None:
        mask = np.ones(prfpy_out.shape[0], dtype=bool)
    total_n_vx = mask.shape[0]
    n_vx_fit = prfpy_out.shape[0]
    n_pars = prfpy_out.shape[1]
    n_vx_in_mask = mask.sum()
    assert n_vx_fit==n_vx_in_mask
    
    filled_pars = np.zeros((total_n_vx, n_pars))

    filled_pars[mask,:] = dag_filter_for_nans(prfpy_out)

    return filled_pars


def make_vx_wise_bounds(n_vx, bounds_in, **kwargs):
    '''make_vx_wise_bounds        
    In prfpy you normally you set the bounds for all voxels to be the same
    Sometimes we want to do voxel wise bounds (e.g., to fix one parameter but 
    fit the others)
    
    Input:
    ----------
    n_vx        Number of voxels/vertices being fit
    bounds_in   list of tuples, for each parameter a min and a max 
    Optional:
    model           Which model are we setting bounds for
    fix_param_dict  Dictionary of parameter, and the values you are fixing 
                    e.g., vx_bound_dict = {'hrf_deriv' : np.ndarray} where 
                    np.ndarray is the length of the number of voxels
    
    Output:
    ----------
    vx_wise_bounds  np.ndarray, n_vx x n_pars x 2
    '''
    model=kwargs.get('model', None)
    fix_param_dict = kwargs.get('fix_param_dict', None)
    
    if isinstance(bounds_in, list):
        bounds_in = np.array(bounds_in)
    vx_wise_bounds = np.repeat(bounds_in[np.newaxis, ...], n_vx, axis=0)

    if not fix_param_dict is None:
        # Fix the specific parameters
        model_idx = prfpy_params_dict()[model]
        for p in fix_param_dict.keys():
            # Upper and lower bound the same...
            vx_wise_bounds[:,model_idx[p],0] = fix_param_dict[p]
            vx_wise_bounds[:,model_idx[p],1] = fix_param_dict[p]

    return vx_wise_bounds

def load_params_generic(params_file, load_all=False, load_var=[]):
    """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

    if isinstance(params_file, str):
        if params_file.endswith('npy'):
            params = np.load(params_file)
        elif params_file.endswith('pkl'):
            with open(params_file, 'rb') as input:
                data = pickle.load(input)
            
            if len(load_var)==1:
                params = data[load_var[0]]
            elif len(load_var)>1:
                params = {}
                # Load the specified variables
                for this_var in load_var:
                    params[this_var] = data[this_var]
            elif load_all:
                params = {}
                for this_var in data.keys():
                    params[this_var] = data[this_var]
            else:
                params = data['pars']

    elif isinstance(params_file, np.ndarray):
        params = params_file.copy()
    elif isinstance(params_file, pd.DataFrame):
        dict_keys = list(params_file.keys())
        if not "hemi" in dict_keys:
            # got normalization parameter file
            params = np.array((params_file['x'][0],
                                params_file['y'][0],
                                params_file['prf_size'][0],
                                params_file['A'][0],
                                params_file['bold_bsl'][0],
                                params_file['B'][0],
                                params_file['C'][0],
                                params_file['surr_size'][0],
                                params_file['D'][0],
                                params_file['r2'][0]))
        else:
            raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized input type for '{params_file}'")

    return params

# ********** PRF OBJECTS
class Prf1T1M(object):
    '''Prf1T1M
    For use with prfpy. Labels determined by prfpy
    Class for parsing prfpy output for 1 subject, 1 task, 1 model
    It will hold everything in the original numpy array, and also 
    convert it to a pandas dataframe for easy plotting, and analysis.
    
    Includes useful functions for plotting and analysis:    
    >> return_vx_mask: returns a mask for voxels
    >> return_th_param: returns the specified parameters, masked by the vx_mask
    >> hist: plot a histogram of a parameter, masked by the vx_mask
    >> visual_field: plot voxels around the visual field of the voxels, masked by the vx_mask
        and colored by a parameter
    >> scatter: scatter plot of 2 parameters, masked by the vx_mask
    >> plot_ts: plot the time series of a voxel
    >> make_prf_str: make a string of the parameters for a voxel
    >> make_context_str: make a string of the task, model, and voxel index
    >> rsq_w_mean: calculate the weighted mean of a parameter, weighted by rsq

    '''
    def __init__(self, prf_params, model, **kwargs):
        '''__init__
        Input:
        ----------
        prf_params     np.ndarray, of all the parameters, i.e., output from prfpy        
        model          str, model: e.g., gauss or norm
        Optional:
        fixed_hrf      bool, if True, then the hrf parameters are not included
        incl_rsq       bool, if False, then the rsq is not included
        task           str, task name

        '''
        self.model = model        
        self.model_labels = prfpy_params_dict()[self.model] # Get names for different model parameters...
        self.prf_params_np = prf_params
        self.fixed_hrf = kwargs.get('fixed_hrf', False)
        self.incl_rsq = kwargs.get('incl_rsq', True)
        #
        self.task = kwargs.get('task', None)
        self.n_vox = self.prf_params_np.shape[0]

        self.params_dd = {}
        mod_labels = prfpy_params_dict()[f'{model}'] 
        for key in mod_labels.keys():
            if ('hrf' in key) and self.fixed_hrf:
                continue
            if ('rsq' in key) and not self.incl_rsq:
                continue                    
            self.params_dd[key] = self.prf_params_np[:,mod_labels[key]]
        
        # Calculate extra interesting stuff
        if self.model in ['gauss', 'norm', 'css', 'dog']:
            # Ecc, pol
            self.params_dd['ecc'], self.params_dd['pol'] = dag_coord_convert(
                self.params_dd['x'],self.params_dd['y'],'cart2pol')        
        if self.model in ('norm', 'dog'):
            # -> size ratio:
            self.params_dd['size_ratio'] = self.params_dd['size_2'] / self.params_dd['size_1']
            self.params_dd['amp_ratio'] = self.params_dd['amp_2'] / self.params_dd['amp_1']
        if self.model == 'norm':
            self.params_dd['bd_ratio'] = self.params_dd['b_val'] / self.params_dd['d_val']
            # Suppression index 
            self.params_dd['sup_idx'] = (self.params_dd['amp_1'] * self.params_dd['size_1']**2) / (self.params_dd['amp_2'] * self.params_dd['size_2']**2)
        if self.model=='csf':
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
        '''return_vx_mask
        Returns a mask (boolean array) for voxels
        
        Notes: 
        ----------
        th keys must be split into 2 parts
        'comparison-param' : value
        e.g.: to exclude gauss fits with rsq less than 0.1
        th = {'min-rsq': 0.1 } 
        comparison  -> min, max,bound
        param       -> any of... (model dependent, see prfpy_params_dict)
        value       -> float, or tuple of floats (for bounds)

        A special case is applied for roi, which is a boolean array you specified previously
        

        Input:
        ----------
        th          dict, threshold for parameters

        Output:
        ----------
        vx_mask     np.ndarray, boolean array, length = n_vx
        
        '''        

        # Start with EVRYTHING         
        vx_mask = np.ones(self.n_vox, dtype=bool) 
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str: # Input roi specification...                
                vx_mask &= th[th_key]
                continue # now next item in key

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
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()

        return vx_mask
    
    def return_th_param(self, param, vx_mask=None):
        '''return_th_param
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
    
    def hist(self, param, th={'min-rsq':.1}, ax=None, **kwargs):
        '''hist: Plot a histogram of a parameter, masked by th'''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)        
        ax.hist(self.pd_params[param][vx_mask].to_numpy(), **kwargs)
        ax.set_title(param)
        dag_add_ax_basics(ax=ax, **kwargs)

    def visual_field(self, th={'min-rsq':.1, 'max-ecc':5}, ax=None, dot_col='k', **kwargs):
        '''visual_field
        Plot the visual field of the voxels, masked by the vx_mask
        and colored by a parameter

        Notes:
        ----------
        Default vx mask is all voxels with rsq > 0.1 and ecc < 5

        Input:
        ----------
        Optional:
        th          dict, threshold for parameters
        ax          matplotlib.axes, if None, then plt.axes() is used
        dot_col     str, color of the dots
        kwargs      dict, kwargs for dag_visual_field_scatter
        '''
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        if isinstance(dot_col,str):
            if dot_col in self.pd_params.keys():
                dot_col = self.pd_params[dot_col][vx_mask].to_numpy()

        dag_visual_field_scatter(
            ax=ax, 
            dot_x=self.pd_params['x'][vx_mask],
            dot_y=self.pd_params['y'][vx_mask],
            dot_col = dot_col,
            **kwargs
        )        

    def scatter(self, px, py, th={'min-rsq':.1}, ax=None, **kwargs):
        '''scatter
        Scatter plot of 2 parameters, masked by the vx_mask
        Can also color by a third parameter

        Notes:
        ----------
        Default vx mask is all voxels with rsq > 0.1

        Input:
        ----------
        px          str, parameter to plot on x axis
        py          str, parameter to plot on y axis
        Optional:
        th          dict, threshold for parameters
        ax          matplotlib.axes, if None, then plt.axes() is used
        dot_col     str, color of the dots
        dot_alpha   float, alpha of the dots
        kwargs      dict, kwargs for dag_scatter

        '''

        # dot_col = kwargs.get('dot_col', 'k')
        # dot_alpha = kwargs.get('dot_alpha', None)
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        # ax.scatter(
        #     self.pd_params[px][vx_mask],
        #     self.pd_params[py][vx_mask],
        #     c = dot_col,
        #     alpha=dot_alpha,
        # )
        # corr_xy = np.corrcoef(
        #     self.pd_params[px][vx_mask],
        #     self.pd_params[py][vx_mask],
        #     )[0,1]
        
        # ax.set_title(f'corr {px}, {py} = {corr_xy:.3f}')
        # ax.set_xlabel(px)        
        # ax.set_ylabel(py)        
        # # dag_add_ax_basics(ax=plt.gca(), **kwargs)
        pc = kwargs.get('pc', None)        
        if pc is not None:
            kwargs['dot_col'] = self.pd_params[pc][vx_mask]
        dag_scatter(
            ax=ax,
            X=self.pd_params[px][vx_mask],
            Y=self.pd_params[py][vx_mask],
            **kwargs
        )        
    
    def plot_ts(self, ts, idx, ax=None, **kwargs):
        '''plot_ts
        Plot the time series of a voxel 

        Input:
        ----------
        ts          np.ndarray, time series, n_vx x n_time
        idx         int, which voxel to plot
        Optional:
        ax          matplotlib.axes, if None, then plt.axes() is used
        kwargs      dict, kwargs for plt.plot
        '''
        if ax==None:
            ax = plt.axes()
        context_str = self.make_context_str(idx)
        prf_str = self.make_prf_str(idx)

        kwargs['label'] = kwargs.get('label', context_str)
        ax.plot(ts[idx,:], **kwargs)
        ow = kwargs.get('ow', False)
        if not ow:
            old_str = ax.get_title()
            if old_str!='':
                prf_str = old_str + '\n' + prf_str
        ax.set_title(prf_str)

    def make_prf_str(self, idx, pid_list=None, add_context=False):
        '''make_prf_str
        Make a string of the parameters for a voxel

        Input:
        ----------
        idx         int, which voxel to plot
        Optional:
        add_context bool, if True, then add the task, model, and voxel index

        Output:
        ----------
        prf_str     str, string of the parameters for a voxel
        '''
        prf_str = ''
        if add_context:
            prf_str += self.make_context_str(idx=idx)
        param_count = 0
        if pid_list is None:
            pid_list = self.model_labels
        for param_key in pid_list:
            if param_key in self.pd_params.keys():
                param_count += 1
                prf_str += f'{param_key}= {self.pd_params[param_key][idx]:.2f}; '
            if param_count > 3:
                prf_str += '\n'
                param_count = 0
        return prf_str
    
    def make_context_str(self, idx):
        '''make_context_str: add task, model to the string'''
        if self.task is None:
            ctxt_task = ''
        else:
            ctxt_task = self.task        

        context_str = f'{ctxt_task}, {self.model},vx={idx}\n'
        return context_str
    
    def rsq_w_mean(self, pid_list, th={'min-rsq':.1}):
        '''rsq_w_mean
        Calculate the weighted mean of a parameter, weighted by rsq

        Input:
        ----------
        pid_list    list, parameter id list
        Optional:
        th          dict, threshold for parameters

        Output:
        ----------
        wm_param    dict, weighted mean of the parameters
        '''
        if not isinstance(pid_list, list):
            pid_list = [pid_list]

        vx_mask = self.return_vx_mask(th)
        wm_param = {}
        for i_param in pid_list:
            wm_param[i_param] = dag_weighted_mean(
                w=self.pd_params['rsq'][vx_mask],
                x=self.pd_params[i_param][vx_mask]
            )
            # if np.isnan(wm_param[i_param]):
            #     print('bloop')
            #     print(vx_mask.sum())
        self.wm_param = wm_param
        return wm_param    
    




class PrfMulti(object):
    '''PrfMulti
    Class for parsing prfpy output for multiple models/tasks for the *same subject*
    
    Notes:
    ----------
    It is important that there are the same number of voxels in each model/task    
    Create a list of Prf1T1M objects, and associated labels, which are all collected in
    this class.     
    It will hold all of the original Prf1T1M objects inside a dictionary    
    The idea is that it makes it easier to do comparisons across conditions/models

    Functions:
    ----------
    Data processing:
    return_vx_mask: returns a mask (boolean array) for voxels
    return_th_param: returns the specified parameters, masked by the vx_mask
    add_prf_diff: add a difference between 2 prf_obj (e.g., diff between 2 tasks)
    TODO: add_prf_mean: add a mean between 2 prf_obj (e.g., mean between 2 tasks)

    ** Plot functions **:
    hist: plot a histogram of a parameter
    scatter: scatter plot of 2 parameters
    multi_scatter: Several scatter plots... multiple comparisons...
    arrow: plot an arrow between 2 prf_obj

    TODO: ? visual_field: plot voxels around the visual field of the voxels, masked by the vx_mask
        and colored by a parameter
    '''
    def __init__(self,prf_obj_list, id_list=[]):
        '''__init__
        
        Input:
        ----------
        prf_obj_list    list, of Prf1T1M objects
        Optional:
        id_list         list, of strings, to label the prf_obj_list        
        '''
        self.id_list = id_list
        self.prf_obj = {}
        self.n_vox = prf_obj_list[0].n_vox
        for i,id in enumerate(id_list):
            self.prf_obj[id] = prf_obj_list[i]
    
    def return_vx_mask(self, th={}):
        '''return_vx_mask
        Returns a mask (boolean array) for voxels
        
        Notes: 
        ----------
        As in Prf1T1M, but with one extra part of the key:        
        th keys must be split into 3 parts
        'id-comparison-param' : value
        th = {'prf1-min-rsq': 0.1 } 
        id          -> which prf_obj to apply the threshold to
                        Can also be 'all', which applies to all prf_obj
        comparison  -> min, max,bound
        param       -> any of... (model dependent, see prfpy_params_dict)
        value       -> float, or tuple of floats (for bounds)

        A special case is applied for roi, which is a boolean array you specified previously
        

        Input:
        ----------
        th          dict, threshold for parameters

        Output:
        ----------
        vx_mask     np.ndarray, boolean array, length = n_vx
                Returns a mask (boolean array) for voxels
        
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key

            id, comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if id=='all':
                # Apply to both task1 and task2:                
                for prf_id in self.id_list:
                    if 'diff_' in prf_id: # skip the diff ones...
                        print('not applying threshold to diff')
                        continue

                    p_available = list(self.prf_obj[prf_id].pd_params.keys())
                    if p in p_available:
                        vx_mask &= self.prf_obj[prf_id].return_vx_mask({f'{comp}-{p}':th_val})
                    else:
                        print(f'Warning - {p} is not a paramer for {prf_id}, ignoring...')
                continue # now next item in th_key...
            vx_mask &= self.prf_obj[id].return_vx_mask({f'{comp}-{p}':th_val})

        if not isinstance(vx_mask, np.ndarray):
            vx_mask = vx_mask.to_numpy()
        return vx_mask
    
    def return_th_params(self, px_list, th=None, **kwargs):
        '''return_th_param
        return all the parameters listed, masked by vx_mask        
        '''
        px_id = [None] * len(px_list)
        px_p = [None] * len(px_list)
        for i,p in enumerate(px_list):
            px_id[i], px_p[i] = p.split('-')
                
        if th==None:
            min_rsq = kwargs.get('min_rsq', .1)
            th = {}
            for key in list(set(px_id)):
                th[f'{key}-min-rsq'] = min_rsq
        # add extra th from the default
        th_plus = kwargs.get('th_plus', {})
        th = {**th, **th_plus}
        # relevant mask 
        vx_mask = self.return_vx_mask(th)
        # create tmp dict with relevant stuff...
        tmp_dict = {}
        for i_px_id,i_px_p in zip(px_id, px_p):
            tmp_dict[f'{i_px_id}-{i_px_p}'] = self.prf_obj[i_px_id].pd_params[i_px_p][vx_mask].to_numpy()
        return tmp_dict
    
    def add_prf_diff(self, id1, id2, new_id=None):
        '''add_prf_diff
        Add a difference between 2 prf_obj (e.g., diff between 2 tasks)

        Input:
        ----------
        id1         str, id of the first prf_obj
        id2         str, id of the second prf_obj
        Optional:
        new_id      str, id of the new prf_obj
        '''
        if new_id is None:
            new_id = f'diff_{id1}_{id2}'
        if new_id in self.id_list:
            print(f'Already created {new_id}')
        else:
            self.prf_obj[new_id] = PrfDiff(
                self.prf_obj[id1], self.prf_obj[id2], diff_id=new_id,
            )
            self.id_list += [new_id]
    # TODO - add_prf_mean    

    def hist(self, px, th=None, ax=None, **kwargs):
        '''hist: Plot a histogram of a parameter, masked by th'''
        if ax==None:
            ax = plt.axes()
        px_id, px_p = px.split('-')                
        if th==None:
            th = {f'{px_id}-min-rsq':.1}            
        vx_mask = self.return_vx_mask(th)        
        label = kwargs.get('label', f'{px_id}-{px_p}')
        kwargs['label'] = label
        ax.hist(self.prf_obj[px_id].pd_params[px_p][vx_mask].to_numpy(), **kwargs)
        ax.set_title(f'{px_id}-{px_p}')
        dag_add_ax_basics(ax=ax, **kwargs)

    def scatter(self, px, py, th=None, ax=None, **kwargs):
        '''scatter: As in Prf1T1M, but can also specify across different prf_obj'''
        # dot_col = kwargs.get('dot_col', 'k')
        # dot_alpha = kwargs.get('dot_alpha', None)
        if ax==None:
            ax = plt.axes()
        px_id, px_p = px.split('-')
        py_id, py_p = py.split('-')
        pc = kwargs.get('pc', None) # dot_color
        if pc is not None:
            pc_id, pc_p = pc.split('-')


        if th==None:
            if 'diff' in (px_id, py_id):
                print('bloop')             
            min_rsq = kwargs.get('min_rsq', .1)
            th = {
                f'{px_id}-min-rsq':min_rsq,
                f'{py_id}-min-rsq':min_rsq,
            }
            if pc is not None:
                th[f'{pc_id}-min-rsq'] = min_rsq

        th_plus = kwargs.get('th_plus', None)
        if not th_plus is None:
            th = {**th, **th_plus}
        vx_mask = self.return_vx_mask(th)
        if pc is not None:
            kwargs['dot_col'] = self.prf_obj[pc_id].pd_params[pc_p][vx_mask]
        dag_scatter(
            ax=ax,
            X=self.prf_obj[px_id].pd_params[px_p][vx_mask],
            Y=self.prf_obj[py_id].pd_params[py_p][vx_mask],
            **kwargs
        )                
        ax.set_xlabel(px)        
        ax.set_ylabel(py)                        

    def multi_scatter(self, px_list, th=None, ax=None, **kwargs):
        '''multi_scatter
        Several scatter plots... multiple comparisons...
        i.e., creates a grid of scatter plots
        '''
        tmp_dict = self.return_th_params(px_list, th, **kwargs)
        fig, ax_list = dag_multi_scatter(tmp_dict, **kwargs)            
        return fig, ax_list
    
    def arrow(self, pold, pnew, ax=None, th=None, **kwargs):
        '''arrow: arrows from one prf_obj to another'''
        if ax==None:
            ax = plt.gca()
        if th is None:
            th = {
                f'{pold}-min-rsq':.1,
                f'{pold}-max-ecc': 5,
                f'{pnew}-min-rsq':.1,
                f'{pnew}-max-ecc': 5,
                }
        th_plus = kwargs.get('th_plus', {})
        th = dict(**th, **th_plus)        
            
        vx_mask = self.return_vx_mask(th)        
        kwargs['title'] = kwargs.get('title', f'{pold}-{pnew}')

        dag_arrow_plot(
            ax, 
            old_x=self.prf_obj[pold].pd_params['x'][vx_mask], 
            old_y=self.prf_obj[pold].pd_params['y'][vx_mask], 
            new_x=self.prf_obj[pnew].pd_params['x'][vx_mask], 
            new_y=self.prf_obj[pnew].pd_params['y'][vx_mask], 
            # arrow_col='angle', 
            **kwargs
            )


class PrfDiff(object):
    '''PrfDiff
    Used with PrfMulti, to contrast 2 conditions
    '''
    def __init__(self, prf_obj1, prf_obj2, diff_id, **kwargs):
        assert ('diff' in diff_id), 'Needs a diff'
        # if not 'diff_' in id:
        #     print('needs a diff_!')  

        self.id = id
        self.model_labels1 = list(prf_obj1.pd_params.keys())
        self.model_labels2 = list(prf_obj2.pd_params.keys())
        self.n_vox = prf_obj1.n_vox 
        self.pd_params = {}
        
        # Make mean and difference:
        for i_label in self.model_labels1:
            if i_label not in self.model_labels2:
                continue
            self.pd_params[i_label] = prf_obj1.pd_params[i_label] -  prf_obj2.pd_params[i_label]
        # For the position shift, find the direction and magnitude:
        if ('x' in self.model_labels1) and ('x' in self.model_labels2):
            self.pd_params['shift_mag'], self.pd_params['shift_dir'] = dag_coord_convert(
                self.pd_params['x'], self.pd_params['y'], 'cart2pol'
            )        
        # some stuff needs to be recalculated?: (because they don't scale linearly...?
        self.pd_params = pd.DataFrame(self.pd_params)

    def return_vx_mask(self, th={}):
        '''
        ... as before ...
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key

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
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()

        return vx_mask


class PrfMean(object):
    def __init__(self, prf_obj1, prf_obj2, id):
        self.id = id
        if not 'mean_' in id:
            print('needs a mean_!')
        self.model_labels1 = list(prf_obj1.pd_params.keys())
        self.model_labels2 = list(prf_obj2.pd_params.keys())
        self.n_vox = prf_obj1.n_vox 
        self.pd_params = {}
        
        # Make mean and difference:
        for i_label in self.model_labels1:
            if i_label not in self.model_labels2:
                continue
            self.pd_params[i_label] = (self.pd_params[self.id2][i_label] +  self.pd_params[self.id1][i_label]) / 2
        # some stuff needs to be recalculated?: (because they don't scale linearly...?
        self.pd_params = pd.DataFrame(self.pd_params)

    def return_vx_mask(self, th={}):
        '''
        ... as before ...
        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key

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
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()

        return vx_mask

    
# ******************************************************************************************************************
# ******************************************************************************************************************
# ******************************************************************************************************************
# OLD PRF OBJECTS..
# ******************************************************************************************************************
# ******************************************************************************************************************

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
        self.model_labels = prfpy_params_dict()[self.model] # Get names for different model parameters...
        self.fixed_hrf = kwargs.get('fixed_hrf', False)
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
                if ('hrf' in i_label) and self.fixed_hrf:
                    continue
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
            if self.model=='csf':
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
            if ('hrf' in i_label) and self.fixed_hrf:
                continue            
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
            if self.model=='csf':
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
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key

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

    def rapid_hist(self, task, param, th={'all-min-rsq':.1}, ax=None, **kwargs):
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)        
        ax.hist(self.pd_params[task][param][vx_mask].to_numpy())
        ax.set_title(f'{task}-{param}')
        dag_add_ax_basics(ax=ax, **kwargs)

    def rapid_arrow(self, ax=None, th={'all-min-rsq':.1, 'all-max-ecc':5}, **kwargs):
        if ax==None:
            ax = plt.gca()
        vx_mask = self.return_vx_mask(th)        
        plt.figure()        
        dag_arrow_plot(
            ax, 
            old_x=self.pd_params[self.task1]['x'][vx_mask], 
            old_y=self.pd_params[self.task1]['y'][vx_mask], 
            new_x=self.pd_params[self.task2]['x'][vx_mask], 
            new_y=self.pd_params[self.task2]['y'][vx_mask], 
            arrow_col='angle', 
            **kwargs
            )


class Prf1T1Mx2(object):
    def __init__(self,prf_obj1, prf_obj2, **kwargs):
        self.task1 = prf_obj1.task
        self.task2 = prf_obj2.task
        self.model1 = prf_obj1.model
        self.model2 = prf_obj2.model
        self.model_labels1 = list(prf_obj1.pd_params.keys())
        self.model_labels2 = list(prf_obj2.pd_params.keys())
        self.id1 = kwargs.get('id1', f'{self.task1}_{self.model1}')
        self.id2 = kwargs.get('id2', f'{self.task2}_{self.model2}')
        self.n_vox = prf_obj1.n_vox 
        self.pd_params = {}
        self.pd_params[self.id1] = prf_obj1.pd_params
        self.pd_params[self.id2] = prf_obj2.pd_params
        
        # Make mean and difference:
        comp_dict = {'mean':{}, 'diff':{}}
        for i_label in self.model_labels1:
            if i_label not in self.model_labels2:
                continue
            comp_dict['mean'][i_label] = (self.pd_params[self.id1][i_label] +  self.pd_params[self.id2][i_label]) / 2
            comp_dict['diff'][i_label] = self.pd_params[self.id2][i_label] -  self.pd_params[self.id1][i_label]
        # For the position shift, find the direction and magnitude:
        if ('x' in self.model_labels1) and ('x' in self.model_labels2):
            comp_dict['diff']['shift_mag'], comp_dict['diff']['shift_dir'] = dag_coord_convert(
                comp_dict['diff']['x'], comp_dict['diff']['y'], 'cart2pol'
            )        
        # # some stuff needs to be recalculated: (because they don't scale linearly... e.g. polar angle)
        # # -> check which models...
        # xy_model_list = ['gauss', 'norm', 'css', 'dog']
        # both_with_xy =  (self.model1 in xy_model_list) & (self.model2 in xy_model_list)
        # both_norm = (self.model1=='norm') & (self.model2=='norm')
        # both_dog = (self.model1=='dog') & (self.model2=='dog')
        # both_csf = (self.model1=='csf') & (self.model2=='csf')
        # for i_comp in ['mean', 'diff']:
        #     # Now add other interesting stuff:
        #     if both_with_xy:
        #         # Ecc, pol
        #         comp_dict[i_comp]['ecc'], comp_dict[i_comp]['pol'] = dag_coord_convert(
        #             comp_dict[i_comp]['x'], comp_dict[i_comp]['y'], 'cart2pol'
        #         )
        #     if both_norm or both_dog:
        #         # -> size ratio:
        #         comp_dict[i_comp]['size_ratio'] = comp_dict[i_comp]['size_2'] / comp_dict[i_comp]['size_1']
        #         comp_dict[i_comp]['amp_ratio']  = comp_dict[i_comp]['amp_1']  / comp_dict[i_comp]['amp_2']
        #     if both_norm:
        #         comp_dict[i_comp]['bd_ratio'] = comp_dict[i_comp]['b_val'] / comp_dict[i_comp]['d_val']
        #     if both_csf:
        #         comp_dict[i_comp]['log10_sf0']  = np.log10(comp_dict[i_comp]['sf0'])
        #         comp_dict[i_comp]['log10_maxC'] = np.log10(comp_dict[i_comp]['maxC'])
        #         comp_dict[i_comp]['sfmax'] = np.nan_to_num(
        #             10**(np.sqrt(comp_dict[i_comp]['log10_maxC'] / (comp_dict[i_comp]['width_r']**2)) + \
        #                                         comp_dict[i_comp]['log10_sf0']))            
        #         comp_dict[i_comp]['sfmax'][comp_dict[i_comp]['sfmax']>100] = 100 # MAX VALUE
        #         comp_dict[i_comp]['log10_sfmax'] = np.log10(comp_dict[i_comp]['sfmax'])            
        # Enter into pd data frame
        self.pd_params['mean'] = pd.DataFrame(comp_dict['mean'])
        self.pd_params['diff'] = pd.DataFrame(comp_dict['diff'])
    
    def return_vx_mask(self, th={}):
        '''
        return_vx_mask: returns a mask (boolean array) for voxels, specified by the user        
        th keys must be split into 3 parts
        'task-comparison-param' : value
        e.g.: to exclude gauss fits with rsq less than 0.1
        th = {'AS0_gauss-min-rsq': 0.1 } 
        task        -> task1, task2, diff, mean, all. (all means apply the threshold to both task1, and task2)
        comparison  -> min, max, bound
        param       -> any of... (model dependent e.g., 'x', 'y', 'ecc'...)
        

        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for th_key in th.keys():
            th_key_str = str(th_key) # convert to string... 
            if 'roi' in th_key_str:
                # Input roi specification...
                vx_mask &= th[th_key]
                continue # now next item in key

            id, comp, p = th_key_str.split('-')
            th_val = th[th_key]
            if id=='all':
                # Apply to both task1 and task2:
                if p in self.model_labels1:
                    vx_mask &= self.return_vx_mask({
                        f'{self.id1}-{comp}-{p}': th_val
                    })
                
                if p in self.model_labels2:
                    vx_mask &= self.return_vx_mask({
                        f'{self.id2}-{comp}-{p}': th_val
                    })

                continue # now next item in th_key...
            
            if comp=='min':
                vx_mask &= self.pd_params[id][p].gt(th_val)
            elif comp=='max':
                vx_mask &= self.pd_params[id][p].lt(th_val)
            elif comp=='bound':
                vx_mask &= self.pd_params[id][p].gt(th_val[0])
                vx_mask &= self.pd_params[id][p].lt(th_val[1])
            else:
                sys.exit()
        if hasattr(vx_mask, 'to_numpy'):
            vx_mask = vx_mask.to_numpy()
        return vx_mask
    
    def rapid_hist(self, id, param, th={'all-min-rsq':.1}, ax=None, **kwargs):
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)        
        label = kwargs.get('label', f'{id}-{param}')
        kwargs['label'] = label
        ax.hist(self.pd_params[id][param][vx_mask].to_numpy(), **kwargs)
        ax.set_title(f'{id}-{param}')
        dag_add_ax_basics(ax=ax, **kwargs)

    def rapid_arrow(self, ax=None, th={'all-min-rsq':.1, 'all-max-ecc':5}, **kwargs):
        if ax==None:
            ax = plt.gca()
        vx_mask = self.return_vx_mask(th)        
        kwargs['title'] = kwargs.get('title', f'{self.id1}-{self.id2}')

        # arrow_col = kwargs.get('arrow_col', None)
        # if isinstance(arrow_col:
        #     # [1] Get change in d2 scotoma 
        #     q_cmap = mpl.cm.__dict__['bwr_r']
        #     q_norm = mpl.colors.Normalize()
        #     q_norm.vmin = -1
        #     q_norm.vmax = 1
        #     arrow_col = q_cmap(q_norm(self.pd_params['diff'][f'd2s_{d2_task}']))
        #     kwargs['arrow_col'] = arrow_col[vx_mask,:]

        dag_arrow_plot(
            ax, 
            old_x=self.pd_params[self.id1]['x'][vx_mask], 
            old_y=self.pd_params[self.id1]['y'][vx_mask], 
            new_x=self.pd_params[self.id2]['x'][vx_mask], 
            new_y=self.pd_params[self.id2]['y'][vx_mask], 
            # arrow_col='angle', 
            **kwargs
            )
    def rapid_p_corr(self, px, py, th={'all-min-rsq':.1}, ax=None, **kwargs):
        # dot_col = kwargs.get('dot_col', 'k')
        # dot_alpha = kwargs.get('dot_alpha', None)
        if ax==None:
            ax = plt.axes()
        vx_mask = self.return_vx_mask(th)
        px_id, px_p = px.split('-')
        py_id, py_p = py.split('-')
        # ax.scatter(
        #     self.pd_params[px_id][px_p][vx_mask],
        #     self.pd_params[py_id][py_p][vx_mask],
        #     c = dot_col,
        #     alpha=dot_alpha,
        # )
        # corr_xy = np.corrcoef(
        #     self.pd_params[px_id][px_p][vx_mask],
        #     self.pd_params[py_id][py_p][vx_mask],
        #     )[0,1]
        
        # ax.set_title(f'corr {px}, {py} = {corr_xy:.3f}')
        dag_scatter(
            ax=ax,
            X=self.pd_params[px_id][px_p][vx_mask],
            Y=self.pd_params[py_id][py_p][vx_mask],
            **kwargs
        )                
        ax.set_xlabel(px)        
        ax.set_ylabel(py)


    # def rapid_scatter(self, th={'all-min-rsq':.1}, ax=None, dot_col='k', **kwargs):
    #     if ax==None:
    #         ax = plt.axes()
    #     vx_mask = self.return_vx_mask(th)
                
    #     dag_visual_field_scatter(
    #         ax=ax, 
    #         dot_x=self.pd_params['x'][vx_mask],
    #         dot_y=self.pd_params['y'][vx_mask],
    #         dot_col = dot_col,
    #         **kwargs
    #     )                          

