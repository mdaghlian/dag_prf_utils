import numpy as np
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
try: 
    from prfpy_csenf.rf import *
    from prfpy_csenf.stimulus import *
    from prfpy_csenf.model import *
except:
    from prfpy.rf import *
    from prfpy.stimulus import *
    from prfpy.model import *

from dag_prf_utils.prfpy_functions import *
from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *


class TSPlotter(Prf1T1M):

    def __init__(self, prf_params, model, prfpy_model=None, real_ts=None,  **kwargs):
        super().__init__(prf_params, model=model, **kwargs)
        self.real_ts = deepcopy(real_ts)
        self.prfpy_model = deepcopy(prfpy_model)
        if self.prfpy_model is not None:
            self.prfpy_stim = prfpy_model.stimulus
            if 'csf' in self.model:
                self.edge_type = self.prfpy_model.edge_type
            self.TR_in_s = self.prfpy_stim.TR
        else: 
            print('MAKING UP A PRFPY STIMULUS FOR PLOTTING PURPOSE')
            if 'csf' in self.model:
                self.edge_type = kwargs.get('edge_type', 'gaussian')
                self.TR_in_s = kwargs.get('TR_in_s', 1.5)
                self.prfpy_stim = CSenFStimulus(
                    SF_seq=[0,0],
                    CON_seq=[0,0],
                    TR=self.TR_in_s,
                )
                self.prfpy_model = CSenFModel(
                    stimulus=self.prfpy_stim,
                    edge_type=self.edge_type,
                )
            else:
                self.TR_in_s = kwargs.get('TR_in_s', 1.5)
                self.prfpy_stim = PRFStimulus2D(
                    TR=self.TR_in_s,
                    design_matrix=np.zeros((100,100,3)),
                    screen_size_cm=60, # Roughly 10 deg radius
                    screen_distance_cm=200,
                )
                if self.model == 'gauss':
                    self.prfpy_model = Iso2DGaussianModel(stimulus=self.prfpy_stim)
                elif self.model == 'css':
                    self.prfpy_model = CSS_Iso2DGaussianModel(stimulus=self.prfpy_stim)
                elif self.model == 'norm':
                    self.prfpy_model = Norm_Iso2DGaussianModel(stimulus=self.prfpy_stim)
                elif self.model == 'dog':
                    self.prfpy_model = DoG_Iso2DGaussianModel(stimulus=self.prfpy_stim)
            
        if (self.real_ts is not None) and (self.prfpy_model is not None):
            # check for same number of time points
            if self.model != 'csf':
                assert self.real_ts.shape[-1] == self.prfpy_model.stimulus.design_matrix.shape[-1], 'real_ts and prfpy_model have different number of time points'
            else:
                assert self.real_ts.shape[-1] == self.prfpy_model.stimulus.n_TRs, 'real_ts and prfpy_model have different number of time points'

        if 'csf' in self.model:
            self._sort_SF_list(**kwargs)            

        if self.incl_rsq:
            self.pred_idx = -1 # When generating predictions don't include the last value (i.e., rsq)
        else:
            self.pred_idx = None
    
    def return_preds(self, idx=None):
        if idx is None:
            idx = np.ones(self.n_vox, dtype=bool)
        preds = self.prfpy_model.return_prediction(*list(self.prf_params_np[idx,:self.pred_idx]))
        return preds
        

    def prf_ts_plot(self, idx, time_pt=None, return_fig=True, **kwargs):
        if self.prfpy_model is None:
            # No model - just return the parameters
            fig = plt.figure(figsize=(15,5))
            param_text = self.make_prf_str(idx)
            fig.text(x=0,y=.5, s=param_text)  
                        
        elif self.model in ['gauss', 'css']:
            fig = self.gauss1_ts_plot(idx, return_fig, **kwargs)
        elif self.model in ['norm', 'dog']:
            fig = self.gauss2_ts_plot(idx, time_pt, return_fig, **kwargs)        
        elif 'csf' in self.model:
            fig = self.csf_ts_plot(idx, return_fig, time_pt, **kwargs)            

        if return_fig:
            return fig

    def real_ts_plot(self, ax, idx, **kwargs):
        try:
            this_real_ts = self.real_ts[idx,:]
            ts_x = np.arange(this_real_ts.shape[-1]) * self.TR_in_s
            ax.plot(ts_x,this_real_ts, ':^', color='k', markersize=5, lw=2, alpha=.5)        
        except:
            pass

    def gauss1_ts_plot(self, idx, return_fig=False, **kwargs):
        '''
        Plot time series for PRF model with 1 RF 
        > i.e., gauss, css
        '''
        do_dm = kwargs.pop('do_dm', False)
        ts_kwargs = dict(linestyle='-', markersize=10, lw=5, alpha=.5)
        ts_kwargs_in = kwargs.get('ts_kwargs', {})
        ts_kwargs.update(ts_kwargs_in)

        # [1] 
        this_rf = np.rot90(gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.pd_params['x'][idx], self.pd_params['y'][idx]),
                sigma=self.pd_params['size_1'][idx],
                normalize_RFs=self.prfpy_model.normalize_RFs).T,axes=(1,2))
        this_rf = np.squeeze(this_rf)
        this_pred_ts = np.squeeze(self.prfpy_model.return_prediction(*list(self.prf_params_np[idx,:-1])))
        ts_x = np.arange(this_pred_ts.shape[-1]) * self.TR_in_s
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
        ax[1].text(1.35, 0.20, param_text, transform=ax[1].transAxes, fontsize=10, va='center', ha='right', family='monospace',)
        # ts
        ax[1].plot(ts_x,this_pred_ts, **ts_kwargs) 
        if self.real_ts is not None:
            self.real_ts_plot(ax=ax[1], idx=idx)
        ax[1].plot((0,ts_x[-1]), (0,0), 'k')   
        # Do dm?
        dag_update_fig_fontsize(fig, 15)        
        if do_dm:
            ax[1].set_xticks(np.arange(ts_x[0], ts_x[-1],15))            
            dag_add_dm_to_ts(                
                fig, 
                ax=ax[1], 
                dm=self.prfpy_stim.design_matrix, 
                TR_in_s=self.TR_in_s, 
                dx_axs=2, do_time=False, 
                move_y=-.1
                )
            ax[1].set_xticks(np.arange(ts_x[0], ts_x[-1],50))            
            # ax[1].set_xticks([])
        if return_fig:
            return fig
        return


    def gauss2_ts_plot(self, idx, time_pt=None, return_fig=False):
        '''
        
        '''

        # [1]
        this_arf = np.rot90(gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.pd_params['x'][idx], self.pd_params['y'][idx]),
                sigma=self.pd_params['size_1'][idx],
                normalize_RFs=self.prfpy_model.normalize_RFs).T,axes=(1,2))
        this_arf = np.squeeze(this_arf)
        this_srf = np.rot90(gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.pd_params['x'][idx], self.pd_params['y'][idx]),
                sigma=self.pd_params['size_2'][idx],
                normalize_RFs=self.prfpy_model.normalize_RFs).T,axes=(1,2))
        this_srf = np.squeeze(this_srf)
        this_rf = [this_arf, this_srf]

        this_pred_ts = np.squeeze(self.prfpy_model.return_prediction(*list(self.prf_params_np[idx,:-1])))
        
        # Plotting stimuli?
        # do_current_stim = True
        fig  = plt.figure()
        fig.set_size_inches(15,5)
        gspec_rf = fig.add_gridspec(2,2,width_ratios=[1,5])
        gspec_ts = fig.add_gridspec(1,2,width_ratios=[1,5])
        rf_ax = []
        rf_ax.append(fig.add_subplot(gspec_rf[0]))
        rf_ax.append(fig.add_subplot(gspec_rf[2]))
        ts_ax = fig.add_subplot(gspec_ts[1])

        # Setup ax 0
        for i in range(2):
            rf_ax[i].set_aspect('equal')
            # rf_ax[i].set_title(f'PRF, vx={idx}')
            rf_ax[i].imshow(this_rf[i], cmap='magma', vmin=0, vmax=1) # 
            rf_ax[i].axis('off')

        param_text = self.make_prf_str(idx)
        ts_ax.text(1.35, 0.20, param_text, transform=ts_ax.transAxes, fontsize=10, va='center', ha='right', family='monospace',)
        # ts
        ts_x = np.arange(this_pred_ts.shape[-1]) * 1.5        
        ts_ax.plot(ts_x,this_pred_ts, '-', markersize=10, lw=5, alpha=.5) # color=self.plot_cols[eye]        
        if self.real_ts is not None:
            self.real_ts_plot(ax=ts_ax, idx=idx)        
        ts_ax.plot((0,ts_x[-1]), (0,0), 'k')   


        dag_update_fig_fontsize(fig, 15)        
        if return_fig:
            return fig
        return
    
    # ****************************************
    # CSF MODEL 
    def _sort_SF_list(self, **kwargs):
        self.SF_list = kwargs.get('SF_list', None)
        self.SF_cmap_name = kwargs.get('SF_cmap', 'viridis')
        if self.SF_list is None:
            self.SF_list = np.array([ 0.5,  1.,  3.,   6.,  12.,  18. ])
            if self.prfpy_model is not None:
                if self.prfpy_model.stimulus.discrete_levels:
                    self.SF_list = self.prfpy_model.stimulus.SFs        
        self.SF_cmap = mpl.cm.__dict__[self.SF_cmap_name]
        self.SF_cnorm = mpl.colors.Normalize()
        self.SF_cnorm.vmin, self.SF_cnorm.vmax = self.SF_list[0],self.SF_list[-1] # *1.5 
        self.SF_cols = {}
        for iSF, vSF in enumerate(self.SF_list):
            this_SF_col = self.SF_cmap(self.SF_cnorm(vSF))
            self.SF_cols[vSF] = this_SF_col        

    def _get_SF_cols(self, v):
        closest_key = None
        min_difference = float('inf')  # Initialize with infinity

        for key in self.SF_cols.keys():
            difference = abs(key - v)
            if difference <= 0.1 and difference < min_difference:
                min_difference = difference
                closest_key = key
        if closest_key is not None:
            this_col = self.SF_cols[closest_key]
        else:
            this_col = None
        return this_col

    def _add_SF_colorbar(self, ax):        
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=self.SF_cnorm, cmap=self.SF_cmap_name), ax=ax)
        cbar.set_label('SF')

    def csf_ts_plot_get_info(self, idx):
        '''Calculate various stuff used when plotting the CSF
        '''
        csenf_stim = self.prfpy_model.stimulus
        ncsf_info = {}
        for key in self.pd_params.keys():
            # if not isinstance(ncsf_info[key], (list, np.ndarray)):
            # ncsf_info[key] = np.array([ncsf_info[key]])                
            ncsf_info[key] = self.pd_params[key][[idx]].to_numpy()
        
        # [1] CSF in design matrix space:
        ncsf_info['part_csf_curve'] = asymmetric_parabolic_CSF(
            SF_seq = self.SF_list, 
            width_r     = ncsf_info['width_r'], 
            SFp         = ncsf_info['SFp'], 
            CSp         = ncsf_info['CSp'], 
            width_l     = ncsf_info['width_l'],                         
        ).squeeze()

        # [2] Smooth form of nCSF, i.e. not just sampling those points in stimulus
        sf_grid = np.logspace(np.log10(self.SF_list[0]),np.log10(50), 100)
        con_grid = np.logspace(np.log10(.1),np.log10(100), 100)
        full_csf = nCSF_response_grid(
            SF_list     = sf_grid, 
            CON_list    = con_grid,
            width_r     = ncsf_info['width_r'], 
            SFp         = ncsf_info['SFp'], 
            CSp         = ncsf_info['CSp'], 
            width_l     = ncsf_info['width_l'], 
            crf_exp     = ncsf_info['crf_exp'],    
            edge_type   = self.edge_type,        
            )
        full_csf_curve = asymmetric_parabolic_CSF(
            SF_seq      = sf_grid, 
            width_r     = ncsf_info['width_r'], 
            SFp         = ncsf_info['SFp'], 
            CSp         = ncsf_info['CSp'], 
            width_l     = ncsf_info['width_l'],                         
        )     
        ncsf_info['sf_grid'],ncsf_info['con_grid']  = np.meshgrid(sf_grid, con_grid)
        ncsf_info['full_csf']          = full_csf
        ncsf_info['full_csf_curve']    = full_csf_curve

        # Calculate the time series for the parameters
        if 'hrf_1' in ncsf_info.keys():
            hrf_1 = ncsf_info['hrf_1']
            hrf_2 = ncsf_info['hrf_2']
        else:
            hrf_1 = None
            hrf_2 = None
        ncsf_info['ts'] = self.prfpy_model.return_prediction(
            width_r     = ncsf_info['width_r'],
            SFp         = ncsf_info['SFp'],
            CSp         = ncsf_info['CSp'],
            width_l     = ncsf_info['width_l'],
            crf_exp     = ncsf_info['crf_exp'],
            beta        = ncsf_info['amp_1'],
            baseline    = ncsf_info['bold_baseline'],
            hrf_1       = hrf_1,
            hrf_2       = hrf_2,
        )
        return ncsf_info
    
    def csf_ts_plot(self, idx, return_fig=False, time_pt=None, **kwargs):
        '''csf_ts_plot
        Do a nice representation of the CSF timeseries model
        '''
        TR_in_s = self.TR_in_s
        do_text     = kwargs.get('do_text', True)
        do_stim_info = kwargs.get('do_stim_info', True)
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')
        do_2_row = kwargs.get('do_2_row', False)
        # return_fig = kwargs.get('return_fig', True)
        dpi = kwargs.get('dpi', 100)
        # Load the specified info 
        ncsf_info = self.csf_ts_plot_get_info(idx)
        ts_x = np.arange(0, ncsf_info['ts'].shape[-1]) * TR_in_s
        
        # Set up figure
        grow_by = kwargs.get('grow_by', 1.8)
        width_ratios = [2, 2, 6]        
        if do_2_row:
            width_ratios = [2,2]
            if do_stim_info:
                height_ratios = [2,1,.5]
            else:
                height_ratios = [2,1]


            fig = plt.figure(figsize=(sum(width_ratios)*grow_by, sum(height_ratios)*grow_by), dpi=dpi)
            gs = mpl.gridspec.GridSpec(len(height_ratios), len(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios)
            csf_ax = fig.add_subplot(gs[0, 0])
            crf_ax = fig.add_subplot(gs[0, 1])
            ts_ax = fig.add_subplot(gs[1, :])
            if do_stim_info:
                SF_ax = fig.add_subplot(gs[2, :])

        elif do_stim_info:
            height_ratios = [2,1]
            fig,axs = plt.subplots(
                nrows=len(height_ratios), ncols=len(width_ratios), 
                gridspec_kw={'width_ratios': width_ratios, 'height_ratios':height_ratios},
                figsize=(sum(width_ratios)*grow_by, sum(height_ratios)*grow_by),
            )
            top_row = axs[0]
            axs[1][0].axis('off')
            axs[1][1].axis('off')
            SF_ax = axs[1][2]
            csf_ax  = top_row[0]
            crf_ax  = top_row[1]
            ts_ax   = top_row[2]

        else:
            height_ratios = [2]
            fig,top_row = plt.subplots(
                nrows=len(height_ratios), ncols=len(width_ratios), 
                gridspec_kw={'width_ratios': width_ratios, 'height_ratios':height_ratios},
                figsize=(sum(width_ratios)*grow_by, sum(height_ratios)*grow_by),
            )            
            csf_ax  = top_row[0]
            crf_ax  = top_row[1]
            ts_ax   = top_row[2]
        
        # *********** ax -1,2: Stimulus info ***********
        if do_stim_info:
            self.sub_plot_stim_info(
                ax=SF_ax, ncsf_info=ncsf_info, 
                time_pt=time_pt,kwargs=kwargs,
            )

        # CSF curve + with imshow to display CRF curve 
        self.sub_plot_csf(
            ax=csf_ax, ncsf_info=ncsf_info, 
            time_pt=time_pt, kwargs=kwargs,           
        )

        # CRF
        self.sub_plot_crf(
            ax=crf_ax, ncsf_info=ncsf_info, 
            time_pt=time_pt, kwargs=kwargs,            
        )        

        # Time series
        self.sub_plot_ts(
            ax=ts_ax, idx=idx, ncsf_info=ncsf_info, 
            time_pt=time_pt, kwargs=kwargs,       
        )


        if do_text:            
            ncsf_txt = self.make_prf_str(
                idx=idx, 
                pid_list=['width_r', 'SFp', 'CSp', 'width_l', 'crf_exp', 'aulcsf', 'rsq' ]
                )
            ts_ax.text(1.35, 0.20, ncsf_txt, transform=ts_ax.transAxes, fontsize=10, va='center', ha='right', family='monospace',)
        # ***********************************************************************
        update_fig_fontsize(fig, new_font_size=1.2, font_multiply=True)
        fig.set_tight_layout(True)
        if return_fig:
            return fig

    def sub_plot_stim_info(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        ts_x = np.arange(0, ncsf_info['ts'].shape[-1]) * self.TR_in_s

        SF_ax = ax
        # Add the stimulus plots
        SF_ax.set_yscale('log')
        SF_ax.set_xlabel('time (s)')
        SF_ax.set_ylabel('SF') # log SF', color='black')
        SF_ax.yaxis.set_label_position('right')
        SF_ax.set_yticks([])
        # -> SF sequence
        SF_seq = self.prfpy_model.stimulus.SF_seq.copy()
        if self.prfpy_model.stimulus.discrete_levels:
            # Find indices where the values change ( & are not to 0)
            change_indices = np.where((np.diff(SF_seq) != 0) & (SF_seq[1:] != 0))[0]
            # Create a list of labels corresponding to the changed values
            labels = [f'{value:0.1f}' for value in SF_seq[change_indices+1]]
            labels = [value.split('.0')[0] for value in labels]
            # Add text labels at the change points on the plot
            for id, label in zip(change_indices + 1, labels):
                SF_ax.text(
                    id*self.TR_in_s+3*self.TR_in_s, 
                    SF_seq[id], 
                    label,
                    color=self._get_SF_cols(SF_seq[id]),
                    ha='center', va='bottom', ) 
            

        SF_ax.plot(ts_x, SF_seq, 'k', linestyle='', marker='_')                
        # SF_ax.spines['right'].set_visible(False)
        SF_ax.spines['top'].set_visible(False)


        # -> contrast
        con_seq = self.prfpy_model.stimulus.CON_seq.copy()
        con_seq[con_seq==0] = np.nan
        con_ax = SF_ax.twinx()                        
        con_ax.plot(ts_x, con_seq, 'r')
        # set ylabel to red, also yticks
        con_ax.set_ylabel('contrast ', color='red', alpha=0.5)        
        con_ax.set_yscale('log')
        con_ax.tick_params(axis='y', colors='red')
        con_ax.spines['right'].set_visible(False)
        con_ax.spines['top'].set_visible(False)
        con_ax.yaxis.set_label_position('left')
        con_ax.yaxis.set_ticks_position('left')
        # Add grey patches corresponding to the nan values in con_s_seq
        y1 = np.ones_like(ts_x)*np.nanmin(con_seq)
        y2 = np.ones_like(ts_x)*np.nanmax(con_seq)
        con_ax.fill_between(ts_x, y1, y2, where=np.isnan(con_seq), facecolor='grey', alpha=0.5)
        # set xlim
        con_ax.set_xlim(0, ts_x[-1])    
        if time_pt is not None:
            con_ax.plot(
                (time_pt*self.TR_in_s, time_pt*self.TR_in_s), (y1[0], y2[0]),
                color=self.time_pt_col, linewidth=5, alpha=0.8)

        # put x axis for con_s_ax and SF_ax at the top of the axis
        # SF_ax.xaxis.tick_top()

    def sub_plot_csf(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        csf_ax = ax
        # Scatter the points sampled
        # csf_ax.scatter(
        #     self.prfpy_model.stimulus.SF_seq, 100/self.prfpy_model.stimulus.CON_seq, color='r', alpha=0.8
        # )
        csf_ax.plot(
            ncsf_info['sf_grid'][0,:],
            ncsf_info['full_csf_curve'].squeeze(),
            lw=5, color='g',
        )

        csf_ax.scatter(
            ncsf_info['sf_grid'].ravel(),
            100/ncsf_info['con_grid'].ravel(),
            c=ncsf_info['full_csf'].ravel(),
            # vmin=0, vmax=1,
            alpha=1,
            cmap='magma',
            lw=0, edgecolor=None,             
        )   
        if time_pt is not None:
            csf_ax.plot(
                self.prfpy_model.stimulus.SF_seq[time_pt],
                100/self.prfpy_model.stimulus.CON_seq[time_pt],
                color=time_pt_col, marker='*', markersize=20,
            )

        csf_ax.set_xlabel('SF (c/deg)')
        csf_ax.set_ylabel('contrast sensitivity')
        csf_ax.set_xscale('log')
        csf_ax.set_yscale('log')  
        xticklabels = ['0.5', '1', '10', '50']
        xticks = [float(i) for i in xticklabels]
        xlim = [xticks[0], xticks[-1]]
        yticklabels = ['1', '10', '100']
        yticks = [float(i) for i in yticklabels]
        ylim = [1, 500]
        csf_ax.set_box_aspect(1)
        csf_ax.set_xticks(xticks) 
        csf_ax.set_xticklabels(xticklabels) 
        csf_ax.set_xlim(xlim) 
        csf_ax.set_yticks(yticks)
        csf_ax.set_yticklabels(yticklabels)
        csf_ax.set_ylim(ylim)
        csf_ax.spines['right'].set_visible(False)
        csf_ax.spines['top'].set_visible(False)        

    def sub_plot_crf(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        crf_ax = ax
        # Contrast response function at different SFs 
        crf_ax.set_title(f'CRF')    
        crf_ax.set_xlabel('contrast (%)')
        crf_ax.set_ylabel('fMRI response (a.u.)')
        contrasts = np.linspace(0,100,100)
        for iSF, vSF in enumerate(self.SF_list):
            # Plot the CRF at each SF we sample in the stimulus
            # [1] Get the "Q" aka "C50" aka "semisaturation point"
            # -> i.e., where response=50%
            # -> we get this using the CSF curve
            this_Q = 100/ncsf_info['part_csf_curve'][iSF]
            this_crf = ncsf_calculate_crf_curve(
                crf_exp=ncsf_info['crf_exp'],
                Q=this_Q, 
                C=contrasts,
                edge_type=self.edge_type,
            )
            crf_ax.plot(
                contrasts, 
                this_crf.squeeze(), 
                alpha=0.8,
                color=self._get_SF_cols(vSF),
                label=f'{vSF:.1f}',
            )

        # Put a grid on the axis (only the major ones)
        crf_ax.grid(which='both', axis='both', linestyle='--', alpha=0.5)
        # ax.set_xscale('log')
        # Make the axis square
        crf_ax.set_box_aspect(1) 
        # ax.set_title('CRF')
        crf_ax.set_xticks([0, 50,100])
        crf_ax.set_yticks([0, 0.5, 1.0])
        crf_ax.set_xlim([0, 100]) # ax.set_xlim([0, 100])
        crf_ax.set_ylim([0, 1])
        crf_ax.set_xlabel('contrast (%)')
        crf_ax.set_ylabel('fMRI response (a.u.)')
        # 
        crf_ax.spines['right'].set_visible(False)
        crf_ax.spines['top'].set_visible(False)            
        if len(self.SF_cols) > 10:
            self._add_SF_colorbar(crf_ax)
        else:
            leg = crf_ax.legend(
                handlelength=0, handletextpad=0, fancybox=True,
                bbox_to_anchor=(1.3, 1), loc='upper right',
                )
            for item in leg.legendHandles:
                item.set_visible(False)        
            for color,text in zip(self.SF_cols.values(),leg.get_texts()):
                text.set_color(color)        
    
    def sub_plot_ts(self, ax=None, idx=None, ncsf_info=None, time_pt=None, **kwargs):
        time_pt_col = kwargs.get('time_pt_col', '#42eff5')    
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if ncsf_info is None:
            ncsf_info = self.csf_ts_plot_get_info(idx=idx)
        ts_ax = ax
        ts_ax.plot(ncsf_info['ts'][0,:time_pt], color='g', marker="*", markersize=2, linewidth=5, alpha=0.8)        
        if self.real_ts is not None:
            ts_ax.plot(self.real_ts[idx,:time_pt], color='k', linestyle=':', marker='^', linewidth=3, alpha=0.8)
        ts_ax.set_xlim(0, ncsf_info['ts'].shape[-1])
        ts_ax.set_title('')
        ts_ax.plot((0,ncsf_info['ts'].shape[-1]), (0,0), 'k')   
        # Find the time for 0 stimulation, add grey patches
        id_no_stim = self.prfpy_model.stimulus.SF_seq==0.0
        x = np.arange(len(id_no_stim))
        y1 = np.ones_like(x)*np.nanmin(ncsf_info['ts'])
        y2 = np.ones_like(x)*np.nanmax(ncsf_info['ts'])
        ts_ax.fill_between(x, y1, y2, where=id_no_stim, facecolor='grey', alpha=0.5)    
        if time_pt is not None:
            ts_ax.plot(
                (time_pt, time_pt), (y1[0], y2[0]),
                color=time_pt_col, linewidth=2, alpha=0.8)    
            # also plot a full invisible version, to keep ax dim...
            ts_ax.plot(ncsf_info['ts'][0,:], alpha=0)
            if self.real_ts is not None:
                ts_ax.plot(self.real_ts[idx,:], alpha=0)


