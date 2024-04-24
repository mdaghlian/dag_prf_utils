import numpy as np
import sys
import matplotlib.pyplot as plt
try: 
    from prfpy_csenf.rf import *
except:
    from prfpy.rf import *

from dag_prf_utils.prfpy_functions import *
from dag_prf_utils.utils import *
from dag_prf_utils.plot_functions import *


class TSPlotter(Prf1T1M):

    def __init__(self, prf_params, model, prfpy_model=None, real_ts=None,  **kwargs):
        super().__init__(prf_params, model=model, **kwargs)
        self.real_ts = real_ts
        self.prfpy_model = prfpy_model
        if self.prfpy_model is not None:
            self.prfpy_stim = prfpy_model.stimulus
            self.TR_in_s = self.prfpy_stim.TR
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

        if return_fig:
            return fig

    def real_ts_plot(self, ax, idx, **kwargs):
        this_real_ts = self.real_ts[idx,:]
        ts_x = np.arange(this_real_ts.shape[-1]) * self.TR_in_s
        ax.plot(ts_x,this_real_ts, ':^', color='k', markersize=5, lw=2, alpha=.5)        

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
                normalize_RFs=False).T,axes=(1,2))
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
                normalize_RFs=False).T,axes=(1,2))
        this_arf = np.squeeze(this_arf)
        this_srf = np.rot90(gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.pd_params['x'][idx], self.pd_params['y'][idx]),
                sigma=self.pd_params['size_2'][idx],
                normalize_RFs=False).T,axes=(1,2))
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
