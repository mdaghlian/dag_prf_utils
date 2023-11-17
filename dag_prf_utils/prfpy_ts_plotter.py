import numpy as np
import sys
import matplotlib.pyplot as plt

# import matplotlib as mpl
# from matplotlib import patches
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import linescanning.plotting as lsplt
# import pandas as pd
# from scipy.stats import binned_statistic

from prfpy.rf import gauss2D_iso_cart
from prfpy.model import CSenFModel, Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy_csenf.rf import csenf_exponential

from dag_prf_utils.plot_functions import *
from dag_prf_utils.prfpy_functions import *

from .utils import *
# from .load_saved_info import *
from .plot_functions import *


class TSPlotter(Prf1T1M):

    def __init__(self, real_tc, prf_params, model, prf_model,  **kwargs):
        super().__init__(prf_params, model=model, **kwargs)
        # self.sub = sub
        self.real_tc = real_tc
        self.prf_model = prf_model
        self.prfpy_stim = prf_model.stimulus
        # # self.prfpy_stim = prfpy_stim
        # if model=='gauss':
        #     self.prf_model = Iso2DGaussianModel(stimulus=self.prfpy_stim)
        # elif model=='norm':
        #     self.prf_model = Norm_Iso2DGaussianModel(stimulus=self.prfpy_stim)
        # elif model=='dog':
        #     self.prf_model = Norm_Iso2DGaussianModel(stimulus=self.prfpy_stim)
        # elif model=='CSS':
        #     self.prf_model = CSS_Iso2DGaussianModel(stimulus=self.prfpy_stim)
        # elif model=='csf':
        #     self.prf_model = CSenFModel(stimulus=self.prfpy_stim)
        #     self.sf_x_lim = (.25,20) # sf
        #     self.con_y_lim = (1, 500) # con
    
    def prf_tc_plot(self, idx, time_pt=None, return_fig=False, **kwargs):
        if self.model=='gauss':
            self.gauss_tc_plot(idx, time_pt, return_fig, **kwargs)
        elif self.model=='norm':
            self.norm_tc_plot(idx, time_pt, return_fig)
        elif self.model=='dog':
            self.dog_tc_plot(idx, time_pt, return_fig)
        elif self.model=='CSS':
            self.CSS_tc_plot(idx, time_pt, return_fig)
        elif self.model=='csf':
            self.csf_tc_plot(idx, time_pt, return_fig, **kwargs)


    def gauss_tc_plot(self, idx, time_pt=None, return_fig=False):
        '''
        
        '''

        # [1] Create csf_curve, pred_tc & rf
        this_rf = np.rot90(gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.pd_params['x'][idx], self.pd_params['y'][idx]),
                sigma=self.pd_params['size_1'][idx],
                normalize_RFs=False).T,axes=(1,2))
        this_rf = np.squeeze(this_rf)
        this_pred_tc = np.squeeze(self.prf_model.return_prediction(*list(self.prf_params_np[idx,:-1])))
        this_real_tc = self.real_tc[idx,:]
        
        # Plotting stimuli?
        # do_current_stim = True
        fig, ax = plt.subplots(1, 2,gridspec_kw={'width_ratios': [1,5]})
        fig.set_size_inches(15,5)

        # Setup ax 0
        ax[0].set_aspect('equal')
        ax[0].set_title(f'PRF, vx={idx}')
        ax[0].imshow(this_rf, cmap='magma', vmin=0, vmax=1) # Plot csf curve
        ax[0].axis('off')

        param_text = self.make_prf_str(idx)

        # TC
        tc_x = np.arange(this_pred_tc.shape[-1]) * 1.5        
        ax[1].plot(tc_x,this_pred_tc, '-', markersize=10, lw=5, alpha=.5) # color=self.plot_cols[eye]        
        ax[1].plot(tc_x,this_real_tc, ':^', color='k', markersize=5, lw=2, alpha=.5)
        ax[1].plot((0,tc_x[-1]), (0,0), 'k')   
        ax[1].set_title(param_text)

        dag_update_fig_fontsize(fig, 15)        
        if return_fig:
            return fig
        return

    def csf_tc_plot(self, idx, time_pt=None, return_fig=False, **kwargs):
        time_pt_col = '#42eff5'
        GT_params = self.pd_params.iloc[idx]
        # Load the specified info 
        GT_info = get_ncsf_info(GT_params, csenf_model=self.prf_model)
        
        do_corr = kwargs.get('do_corr', False)
        # Set up figure
        nrows = 2
        ncols = 4
        width_ratios = [2,1,1,6]
        if do_corr:
            height_ratios = [2,1+1] # 2 for each GT model, 1 for stim info
            fig_height = 2 + 4 + 2 # extra plus 2
        else:
            height_ratios = [2,1]
            fig_height = 2 + 4 # extra plus 2
        fig_size = (21,fig_height) # height depends on number of GT models

        fig, ax = plt.subplots(
        nrows, ncols,
        gridspec_kw={'width_ratios': width_ratios, 'height_ratios':height_ratios},
        figsize=fig_size,
        )   
        # Turn off unwanted axes
        ax[-1][0].axis('off')
        ax[-1][1].axis('off')   
        corr_ax = ax[-1][1]
        ax[-1][2].axis('off')
        
        # *********** ax -1,2: Stimulus info ***********
        # Add the stimulus plots (independent of number of GT models)
        # -> SF sequence
        SF_seq = self.prfpy_stim.SF_seq.copy()
        SF_seq[SF_seq==0] = np.nan

        SF_ax = ax[-1][-1]
        SF_ax.set_yscale('log')
        SF_ax.plot(SF_seq, '*k')                
        SF_ax.set_xlabel('time (s)')
        SF_ax.set_ylabel('log SF', color='black')

        # -> 100/Contrast seq
        con_s_seq = self.prfpy_stim.CON_S_seq.copy()
        con_s_seq[con_s_seq==np.inf] = np.nan
        con_s_ax = SF_ax.twinx()                        
        con_s_ax.plot(con_s_seq, 'r')
        # set ylabel to red, also yticks
        con_s_ax.set_ylabel('100/Contrast', color='red')
        con_s_ax.tick_params(axis='y', colors='red')

        # Add grey patches corresponding to the nan values in con_s_seq
        x = np.arange(len(con_s_seq))
        y1 = np.ones_like(x)*np.nanmin(con_s_seq)
        y2 = np.ones_like(x)*np.nanmax(con_s_seq)
        con_s_ax.fill_between(x, y1, y2, where=np.isnan(con_s_seq), facecolor='grey', alpha=0.5)
        # set xlim
        con_s_ax.set_xlim(0, len(con_s_seq))    
        if time_pt is not None:
            con_s_ax.plot(
                (time_pt, time_pt), (y1[0], y2[0]),
                color=time_pt_col, linewidth=5, alpha=0.8)

        # put x axis for con_s_ax and SF_ax at the top of the axis
        # SF_ax.xaxis.tick_top()
        # ***********************************************************************
        i = 0
        
        # *********** ax 0,0: CSF curve + with imshow to display CRF curve ***********
        # ax.set_title(f'CSF curve')    
        ax[i][0].set_xlabel('SF (c/deg)')
        ax[i][0].set_ylabel('100/Contrast')
        ax[i][0].set_xscale('log')
        ax[i][0].set_yscale('log')

        # Scatter the points sampled
        ax[i][0].scatter(
            self.prfpy_stim.SF_seq, 100/self.prfpy_stim.CON_seq, color='r', alpha=0.8
        )
        # Plot the CSF curve
        ax[i][0].plot(
            self.prfpy_stim.SFs, 
            GT_info['csf_curve'][:,i], 
            color='g', linewidth=5)

        # Plot the full CSF 
        ax[i][0].scatter(
            GT_info['full_csf_info']['sf_grid'].ravel(),
            GT_info['full_csf_info']['con_grid'].ravel(),
            c=GT_info['full_csf_info']['full_csf'].ravel(),
            vmin=0, vmax=1,
            alpha=.1,
            cmap='magma'
        )        
        if time_pt is not None:
            ax[i][0].plot(
                self.prfpy_stim.SF_seq[time_pt],
                100/self.prfpy_stim.CON_seq[time_pt],
                color=time_pt_col, marker='*', markersize=20,
            )
        # Put a grid on the axis (only the major ones)
        ax[i][0].grid(which='both', axis='both', linestyle='--', alpha=0.5)
        # Make the axis square
        ax[i][0].set_aspect('equal', 'box') 
        ax[i][0].set_xticks([.1, 1,10,100])
        ax[i][0].set_yticks([.1, 1,10,100, 1000])
        ax[i][0].set_xlim([0.25, 50]) # [0.1, 100])
        ax[i][0].set_ylim([1, 500]) # [0.1, 1000])
        ax[i][0].legend()
        # ***********************************************************************
        

        # *********** ax 0,1: CSF in DM space ***********
        # RF - in DM space:
        ax[i][1].imshow(GT_info['csf_mat'][0,:,:], vmin=0, vmax=1, cmap='magma')#, alpha=.5)        
        # Add dm representing the times series
        # Add grids to show the "matrix" aspect
        ax[i][1].set_xticks(np.arange(len(self.prfpy_stim.SFs))-.5)
        # ax[i][1].set_xticklabels(self.prfpy_stim.SFs)
        ax[i][1].set_xticklabels([])
        ax[i][1].set_yticks(np.arange(len(self.prfpy_stim.CON_Ss))-.5)
        # ax[i][1].set_yticklabels(np.round(self.prfpy_stim.CON_Ss,2))
        ax[i][1].set_yticklabels([])
        # Grids, thick red lines
        ax[i][1].grid(which='major', axis='both', linestyle='-', color='r', linewidth=2)
        ax[i][1].set_title('CSF-DM space')
        if time_pt is not None:
            if self.prfpy_stim.SF_seq_id[time_pt]!=0:
                ax[i][1].plot(
                    self.prfpy_stim.SF_seq_id[time_pt]-1,  # -1 for indexing...
                    self.prfpy_stim.CON_seq_id[time_pt]-1, # -1 for indexing...
                    color=time_pt_col, marker='s', markersize=15,
                )        
        # ***********************************************************************


        # *********** ax 0,2: CRF  ***********
        ax[i][2].set_title(f'CRF')    
        ax[i][2].set_xlabel('contrast (%)')
        ax[i][2].set_ylabel('fMRI Response')
        ax[i][2].plot(
            np.linspace(0,100, 100), 
            GT_info['crf_curve'][0,:], 
            color='g', linewidth=5)
        # Put a grid on the axis (only the major ones)
        ax[i][2].grid(which='both', axis='both', linestyle='--', alpha=0.5)
        # Make the axis square
        # ax.set_aspect('equal', 'box') 
        ax[i][2].set_xticks([0, 50,100])
        ax[i][2].set_yticks([0, 0.5, 1.0])
        ax[i][2].set_xlim([0, 100])
        ax[i][2].set_ylim([0, 1])
        ax[i][2].legend()
        # ***********************************************************************


        # *********** ax 0,3: Time series ***********
        ax[i][-1].plot(GT_info['ts'][0,:time_pt], color='g', marker="*", markersize=2, linewidth=5, alpha=0.8)        
        ax[i][-1].plot(self.real_tc[idx,:time_pt], color='k', linestyle=':', marker='^', linewidth=3, alpha=0.8)
        ax[i][-1].set_xlim(0, GT_info['ts'].shape[-1])
        ax[i][-1].set_title('Time series')
        ax[i][-1].plot((0,GT_info['ts'].shape[-1]), (0,0), 'k')   
        # Find the time for 0 stimulation, add grey patches
        id_no_stim = self.prfpy_stim.SF_seq==0.0
        x = np.arange(len(id_no_stim))
        y1 = np.ones_like(x)*np.nanmin(GT_info['ts'])
        y2 = np.ones_like(x)*np.nanmax(GT_info['ts'])
        ax[i][-1].fill_between(x, y1, y2, where=id_no_stim, facecolor='grey', alpha=0.5)    
        if time_pt is not None:
            ax[i][-1].plot(
                (time_pt, time_pt), (y1[0], y2[0]),
                color=time_pt_col, linewidth=5, alpha=0.8)    
            # also plot a full invisible version, to keep ax dim...
            ax[i][-1].plot(GT_info['ts'][0,:], alpha=0)
            ax[i][-1].plot(self.real_tc[idx,:], alpha=0)

        # ***********************************************************************
        # Calculate rsq
        # print(dag_get_rsq(
        #     real_ts,
        #     GT_info['ts'], 
        # ))

        # *********** DM CORRELATION ***********
        if do_corr:
            og_dm = self.prfpy_stim.design_matrix.copy()
            reshaped_dm = og_dm.reshape(-1, 214)
            # -> convolve with hrf
            conv_rs_dm = self.prf_model.convolve_timecourse_hrf(
                reshaped_dm, 
                self.prf_model.hrf
            )
            # -> correlate with ts
            corr = np.zeros(conv_rs_dm.shape[0])
            for i_dm in range(conv_rs_dm.shape[0]):
                corr[i_dm] = np.corrcoef(conv_rs_dm[i_dm,:], self.real_tc[idx,:])[0,1]        

            # -> reshape back to [14 x 6 x 214]
            corr_dm = corr.reshape(14, 6)
            # -> plot
            corr_ax.imshow(corr_dm, vmin=-.5, vmax=.5, cmap='RdBu_r')
            # Add grids to show the "matrix" aspect
            corr_ax.set_xticks(np.arange(len(self.prfpy_stim.SFs))-.5)
            # ax[i][1].set_xticklabels(csenf_stim.SFs)
            corr_ax.set_xticklabels([])
            corr_ax.set_yticks(np.arange(len(self.prfpy_stim.CON_Ss))-.5)
            # ax[i][1].set_yticklabels(np.round(csenf_stim.CON_Ss,2))
            corr_ax.set_yticklabels([])
            # Grids, thick red lines
            corr_ax.grid(which='major', axis='both', linestyle='-', color='r', linewidth=2)
            corr_ax.set_title('TS - DM correlation')
            # Add a colorbar to the right of corr_ax
            cbar = fig.colorbar(corr_ax.images[0], ax=corr_ax, location='right')
        # ***********************************************************************

        # *********** Bottom left Text ***********
        gt_txt = f'width_r={GT_info["width_r"]:>8.2f}, \n' + \
            f'SFp={GT_info["SFp"]:>8.2f}, \n' + \
            f'CSp={GT_info["CSp"]:>8.2f}, \n' + \
            f'width_l={GT_info["width_l"]:>8.2f}, \n' + \
            f'crf_exp={GT_info["crf_exp"]:>8.2f}, \n' + \
            f'sfmax={GT_info["sfmax"]:>8.2f}, \n' + \
            f'rsq={GT_info["rsq"]:>8.2f}, \n' #+ \
            # f'aulcsf={GT_info["aulcsf"][i]:>8.2f}, \n' + \
            # f'ncsf_volume={GT_info["ncsf_volume"][i]:>8.2f}, \n' 

        # gt_txt = f'width_r={GT_info["width_r"][0]:>8.2f}, \n' + \
        #     f'SFp={GT_info["SFp"][0]:>8.2f}, \n' + \
        #     f'CSp={GT_info["CSp"][0]:>8.2f}, \n' + \
        #     f'width_l={GT_info["width_l"][0]:>8.2f}, \n' + \
        #     f'crf_exp={GT_info["crf_exp"][0]:>8.2f}, \n' + \
        #     f'sfmax={GT_info["sfmax"][0]:>8.2f}, \n' + \
        #     f'rsq={GT_info["rsq"][0]:>8.2f}, \n' #+ \
        #     # f'aulcsf={GT_info["aulcsf"][i]:>8.2f}, \n' + \
        #     # f'ncsf_volume={GT_info["ncsf_volume"][i]:>8.2f}, \n' 
        # Add the text to the right of the time series figure
        # Ax is in axis coordinates, so 0,0 is bottom left, 1,1 is top right
        ax[i][-1].text(1.35, 0.20, gt_txt, transform=ax[i][-1].transAxes, fontsize=20, va='center', ha='right', family='monospace',)
        # ***********************************************************************

        fig.set_tight_layout(True)

        return fig, ax            

    # def norm_tc_plot(self, idx, time_pt=None, return_fig=False):
    #     '''
        
    #     '''

    #     # [1] Create csf_curve, pred_tc & rf
    #     this_rf = np.rot90(gauss2D_iso_cart(
    #             x=self.prfpy_stim.x_coordinates[...,np.newaxis],
    #             y=self.prfpy_stim.y_coordinates[...,np.newaxis],
    #             mu=(self.pd_params['x'][idx], self.pd_params['y'][idx]),
    #             sigma=self.pd_params['size_1'][idx],
    #             normalize_RFs=False).T,axes=(1,2))        
    #     this_rf = np.squeeze(this_rf)
    #     this_srf = np.rot90(gauss2D_iso_cart(
    #             x=self.prfpy_stim.x_coordinates[...,np.newaxis],
    #             y=self.prfpy_stim.y_coordinates[...,np.newaxis],
    #             mu=(self.pd_params['x'][idx], self.pd_params['y'][idx]),
    #             sigma=self.pd_params['size_2'][idx],
    #             normalize_RFs=False).T,axes=(1,2))  
    #     this_srf = np.squeeze(this_srf)
    #     this_pred_tc = np.squeeze(self.prf_model.return_prediction(*list(self.prf_params_np[idx,:-1])))
    #     this_real_tc = self.real_tc[idx,:]
        
    #     # Plotting stimuli?
    #     # do_current_stim = True
    #     fig = plt.figure(constrained_layout=True)

    #     gspec1 = fig.add_gridspec(2,6)
    #     gspec2 = fig.add_gridspec(1,5)        
    #     ax_rf = fig.add_subplot(gspec1[0])
    #     ax_srf = fig.add_subplot(gspec1[6])
    #     ax_tc = fig.add_subplot(gspec2[1::])

    #     fig.set_size_inches(15,5)

    #     # Setup ax 0
    #     ax_rf.set_aspect('equal')
    #     ax_rf.set_title(f'PRF, vx={idx}')
    #     ax_rf.imshow(this_rf, cmap='magma', vmin=0, vmax=1) # Plot csf curve
    #     ax_rf.axis('off')
    #     ax_srf.set_aspect('equal')
    #     ax_srf.set_title(f'SRF')
    #     ax_srf.imshow(this_srf, cmap='magma', vmin=0, vmax=1) # Plot csf curve
    #     ax_srf.axis('off')


    #     # TC
    #     param_text = self.make_prf_str(idx)
    #     tc_x = np.arange(this_pred_tc.shape[-1]) * 1.5        
    #     ax_tc.plot(tc_x,this_pred_tc, '-', markersize=10, lw=5, alpha=.5) # color=self.plot_cols[eye]        
    #     ax_tc.plot(tc_x,this_real_tc, ':^', color='k', markersize=5, lw=2, alpha=.5)
    #     ax_tc.plot((0,tc_x[-1]), (0,0), 'k')   
    #     ax_tc.set_title(param_text)

    #     dag_update_fig_fontsize(fig, 15)        
    #     if return_fig:
    #         return fig
    #     return




# **************************************************************
# **************************************************************
# **************************************************************
# CSF SPECIAL FUNCTIONS
# Function which takes the dictionary of parameters and returns the extra useful info
def get_ncsf_info(ncsf_params, csenf_model):
    csenf_stim = csenf_model.stimulus
    # Check that we have everything and that everything is an array
    if not 'width_l' in ncsf_params.keys():
        ncsf_params['width_l'] = np.zeros_like(ncsf_params['width_r']) + 0.448
    if not 'amp_1' in ncsf_params.keys():
        ncsf_params['amp_1'] = np.ones_like(ncsf_params['width_r'])
    if not 'bold_baseline' in ncsf_params.keys():
        ncsf_params['bold_baseline'] = np.zeros_like(ncsf_params['width_r'])
    # Cycle through the keys and make sure that they are arrays
    for key in ncsf_params.keys():
        if not isinstance(ncsf_params[key], (list, np.ndarray)):
            ncsf_params[key] = np.array([ncsf_params[key]])

    ncsf_info = {}
    # Calculate derived parameters for GT csf models
    # -> functions for calculating derived parameters
    # Calculate log10 versions of parameters
    log10_SFp   = np.log10(ncsf_params['SFp'])
    log10_CSp  = np.log10(ncsf_params['CSp'])

    # Calculate CSF curves + matrix 
    csf_mat, csf_curve = csenf_exponential(
        log_SF_grid = csenf_stim.log_SF_grid, 
        CON_S_grid  = csenf_stim.CON_S_grid,
        width_r     = ncsf_params['width_r'], 
        SFp         = ncsf_params['SFp'], 
        CSp        = ncsf_params['CSp'], 
        width_l     = ncsf_params['width_l'], 
        crf_exp     = ncsf_params['crf_exp'],
        return_curve=True,
        )    
    
    # Full RF (same as above, but we sample more densly, for plotting) i.e., not in DM space
    sf_grid = np.logspace(np.log10(csenf_stim.SFs[0]),np.log10(csenf_stim.SFs[-1]), 50)
    con_grid = np.logspace(np.log10(csenf_stim.CON_Ss[-1]),np.log10(csenf_stim.CON_Ss[0]), 50)
    # sf_grid = np.logspace(np.log10(.25),np.log10(20), 50)
    # con_grid = np.logspace(np.log10(.1),np.log10(500), 50)    
    sf_grid, con_grid = np.meshgrid(sf_grid,con_grid)
    full_csf = csenf_exponential(
        log_SF_grid = np.log10(sf_grid), 
        CON_S_grid = con_grid, 
        width_r     = ncsf_params['width_r'], 
        SFp         = ncsf_params['SFp'], 
        CSp        = ncsf_params['CSp'], 
        width_l     = ncsf_params['width_l'], 
        crf_exp     = ncsf_params['crf_exp'],
        return_curve = False)
    full_csf_info = {
        'sf_grid'   : sf_grid,
        'con_grid'  : con_grid,
        'full_csf'  : full_csf,
    }

    # crf_curve
    crf_curve = calculate_crf_curve(ncsf_params['crf_exp'])

    # Log CSF curve
    logcsf_curve = np.log10(csf_curve)    
    logcsf_curve[logcsf_curve<0 ] = 0

    # Calculate AULCSF (area under log CSF)
    aulcsf = calculate_aulcsf(
        log_SF_grid = csenf_stim.log_SF_grid, 
        CON_S_grid  = csenf_stim.CON_S_grid,
        width_r     = ncsf_params['width_r'], 
        SFp         = ncsf_params['SFp'], 
        CSp        = ncsf_params['CSp'], 
        width_l     = ncsf_params['width_l'], 
        crf_exp     = ncsf_params['crf_exp'],
        )    

    # Calculate nCSF_volume (like aulcsf, but takes into account the CRF)
    ncsf_volume = calculate_nCSF_volume(
        log_SF_grid = csenf_stim.log_SF_grid,
        CON_S_grid  = csenf_stim.CON_S_grid,
        width_r     = ncsf_params['width_r'],
        SFp         = ncsf_params['SFp'],
        CSp        = ncsf_params['CSp'],
        width_l     = ncsf_params['width_l'],
        crf_exp     = ncsf_params['crf_exp'],
        )

    # Calculate sfmax
    sfmax = calculate_sfmax(
        width_r     = ncsf_params['width_r'], 
        SFp         = ncsf_params['SFp'], 
        CSp        = ncsf_params['CSp'], 
        )

    # Calculate the time series for the GT parameters
    if 'hrf_1' in ncsf_params.keys():
        hrf_1 = ncsf_params['hrf_1']
        hrf_2 = ncsf_params['hrf_2']
    else:
        hrf_1 = None
        hrf_2 = None
    ts = csenf_model.return_prediction(
        width_r     = ncsf_params['width_r'],
        SFp         = ncsf_params['SFp'],
        CSp        = ncsf_params['CSp'],
        width_l     = ncsf_params['width_l'],
        crf_exp     = ncsf_params['crf_exp'],
        beta        = ncsf_params['amp_1'],
        baseline    = ncsf_params['bold_baseline'],
        hrf_1       = hrf_1,
        hrf_2       = hrf_2,
    )
        

    # Put it all together in a dict
    ncsf_info = {
        **ncsf_params,
        'log10_SFp'     : log10_SFp,
        'log10_CSp'    : log10_CSp,
        'csf_mat'       : csf_mat,
        'csf_curve'     : csf_curve,
        'crf_curve'     : crf_curve,
        'logcsf_curve'  : logcsf_curve,
        'aulcsf'        : aulcsf,
        'ncsf_volume'   : ncsf_volume,    
        'sfmax'         : sfmax,
        'ts'            : ts,
        'full_csf_info' : full_csf_info,
    }
    return ncsf_info    

def calculate_sfmax(width_r, SFp, CSp, max_sfmax=50):
    """calculate_sfmax
    High frequency cutoff. Useful summary statistic of whole CSF curve
    Can be infinte (with low width_r), so we set a max value 
    """
    log10_CSp = np.log10(CSp)
    log10_SFp = np.log10(SFp)
    sfmax = 10**((np.sqrt(log10_CSp/(width_r**2)) + log10_SFp))
    if len(sfmax.shape)>=1:
        sfmax[sfmax>max_sfmax] = max_sfmax
    elif sfmax>max_sfmax:
        sfmax = max_sfmax        
    return sfmax

def calculate_aulcsf(log_SF_grid, CON_S_grid, width_r, SFp, CSp, width_l, **kwargs):
    """calculate_aulcsf
    area under log contrast sensitivity function
    Useful summary statistic
    Note - does not take into account the way that we make the edge of the curve (e.g., CRF, exp...)
    """
    csf_rfs, csf_curve = csenf_exponential(
        log_SF_grid = log_SF_grid, 
        CON_S_grid = CON_S_grid, 
        width_r = width_r, 
        SFp = SFp, 
        CSp = CSp, 
        width_l = width_l, 
        return_curve = True,
        **kwargs # for crf_exp
        )
    logcsf_curve = np.log10(csf_curve)    
    logcsf_curve[logcsf_curve<0] = 0 # cannot have negative?
    aulcsf = np.trapz(logcsf_curve, x=log_SF_grid[0,:], axis=0) # NOTE -> TO TAKE INTO ACCOUNT THE CRF? NEEDS SOME KIND OF WEIGHTING...
    # idx = np.where(aulcsf<0)[0][0]
    # print(csf_curve.shape)
    # plt.plot(logcsf_curve[:,idx])
    # plt.figure()
    # plt.imshow(np.squeeze(csf_rfs[idx,:,:]))

    # n_x = log_SF_grid.shape[1]
    # maxaulcsf = np.trapz(np.ones(n_x)*np.log10(200), x=log_SF_grid[0,:], axis=0) # NOTE -> TO TAKE INTO ACCOUNT THE CRF? NEEDS SOME KIND OF WEIGHTING...
    # aulcsf = aulcsf / maxaulcsf
    return aulcsf

def calculate_nCSF_volume(log_SF_grid, CON_S_grid, width_r, SFp, CSp, width_l, **kwargs):
    """calculate_nCSF_volume
    Analagous to aulcsf, but takes into account the way that we make the edge of the curve (e.g., CRF, exp...)
    Useful summary statistic
    """
    csf_rfs, _ = csenf_exponential(
        log_SF_grid = log_SF_grid, 
        CON_S_grid = CON_S_grid, 
        width_r = width_r, 
        SFp = SFp, 
        CSp = CSp, 
        width_l = width_l, 
        return_curve = True,
        **kwargs # for crf_exp
        )
    nCSF_volume = csf_rfs.sum(axis=(1,2)) # / (csf_rfs.shape[1] * csf_rfs.shape[2])
    return nCSF_volume


def calculate_crf_curve(crf_exp, Q=20, C=np.linspace(0,100,100)):
    '''calculate_crf_curve
    To calculate the CRF curve for a given exponent and Q
    Used for plotting
    # a = 1;
    # C = 0.25:1:100;   % RMS contrast
    # Q = 20;
    # q_true = true_vals(4);
    # resp_true = a.*((C.^q_true)./((C.^q_true)+(Q.^q_true)));    
    '''
    # If crf_exp is not an array, make it one
    if not isinstance(crf_exp, (list, np.ndarray)):
        crf_exp = np.array([crf_exp])
    return (C**crf_exp[...,np.newaxis])/((C**crf_exp[...,np.newaxis])+(Q**crf_exp[...,np.newaxis]))
