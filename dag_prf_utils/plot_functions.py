from ctypes import RTLD_GLOBAL
from ctypes.wintypes import RGB
from matplotlib.colors import rgb2hex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
import linescanning.plotting as lsplt
from scipy.stats import binned_statistic
from highlight_text import HighlightText, ax_text, fig_text
from .utils import coord_convert, print_p
# import rgba
# import os 
# import nibabel as nb
# from collections import defaultdict as dd
# from pathlib import Path
# import yaml

# import cortex
# import seaborn as sns
# import pickle
# from datetime import datetime
# opj = os.path.join

def rgba(r,g,b,a):
    return [r/255,g/255,b/255,a]


def show_plot_cols():
    plot_cols = get_plot_cols()
    # fig, axs = plt.subplot()    
    plt.figure(figsize=(2,5))
    for i,key in enumerate(plot_cols.keys()):
        plt.scatter(0,i, s=500, color=plot_cols[key], label = key)
        plt.text(0, i+.1, key)
    
def get_plot_cols():

# rgba(252,141, 89, 1)
# rgba(227, 74, 51, 1)
# rgba(179,  0,  0, 1)
# rgba(123,204,196, 1)
# rgba( 67,162,202, 1)
# rgba(  8,104,172, 1)
# rgba(223,101,176, 1)
# rgba(221, 28,119, 1)
# rgba(152,  0, 67, 1)

    plot_cols = {
        "LE"            : rgba(252,141, 89, .8),#'#fd8d3c',
        "RE"            : rgba( 67,162,202, .8),#'#43a2ca',
        #
        "gauss"          : rgba( 27, 158, 119, 0.9),
        "norm"           : rgba(217,  95,   2, 0.5),
        "real"           : '#cccccc',
        }
    return plot_cols

def spatially_bin_img(x, y, param, axs, ecc_bounds, pol_bounds, scotoma_info=[], **kwargs):
    ''' 
    TODO...    
    '''
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    incl_patches = kwargs.get("incl_patches", False)
    incl_ticks = kwargs.get("incl_ticks", True)
    n_pix = kwargs.get("n_pix", 500)
    max_ecc = ecc_bounds.max()
    oneD_grid = np.linspace(-max_ecc,max_ecc,n_pix,endpoint=True)
    img_x_grid, img_y_grid = np.meshgrid(oneD_grid, oneD_grid)    
    img_ecc_grid, img_pol_grid = coord_convert(img_x_grid,img_y_grid,old2new="cart2pol")
    # img_ecc_grid = np.rot90(img_ecc_grid)
    # img_pol_grid = np.rot90(img_pol_grid)

    pol_ecc, pol_angle = coord_convert(x,y,old2new="cart2pol")
    total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)

    img_matrix = np.zeros((n_pix, n_pix)) * np.nan
    for i_ecc in range(len(ecc_bounds)-1):
        for i_pol in range(len(pol_bounds)-1):
            ecc_lower = ecc_bounds[i_ecc]
            ecc_upper = ecc_bounds[i_ecc + 1]      

            pol_lower = pol_bounds[i_pol]
            pol_upper = pol_bounds[i_pol + 1]   

            
            ecc_idx =(pol_ecc >= ecc_lower) & (pol_ecc <=ecc_upper)
            pol_idx =(pol_angle >= pol_lower) & (pol_angle <=pol_upper)                    
            bin_idx = pol_idx & ecc_idx

            img_ecc_idx =(img_ecc_grid >= ecc_lower) & (img_ecc_grid <=ecc_upper)
            img_pol_idx =(img_pol_grid >= pol_lower) & (img_pol_grid <=pol_upper)
            img_bin_idx = img_pol_idx & img_ecc_idx
            
            param_bin_mean = np.mean(param[bin_idx])
            img_matrix[img_bin_idx] = param_bin_mean

    # a = (img_x_grid>0) & (img_y_grid>0)
    # img_matrix *=0
    # img_matrix[a] = 900
    # axs.imshow(img_x_grid, vmin=0)
    # sys.exit()
    
    im_col = axs.imshow(np.flipud(img_matrix), extent=[-max_ecc, max_ecc, -max_ecc, max_ecc])
    fig = plt.gcf()
    fig.colorbar(im_col, ax=axs)

    # if len(ecc_bounds)>0:
    #     add_scot_patches_and_bin_lines(axs, ecc_bounds, pol_bounds, scotoma_info=scotoma_info, incl_patches=incl_patches, incl_ticks=incl_ticks)
    
    axs.set_aspect('equal')


def basic_spatially_binned_param(x, y, param, axs, ecc_bounds=[], pol_bounds=[], scotoma_info=[], **kwargs):
    ''' 
    TODO...    
    '''
    dot_alpha = kwargs.get("alpha", 0.5)
    incl_patches = kwargs.get("incl_patches", False)
    incl_ticks = kwargs.get("incl_ticks", True)
    bin_position = kwargs.get("bin_position", True)
    scale_dot_size = kwargs.get("scale_dot_size", True)
    
    if len(ecc_bounds)>0:
        # print("DOING BINNING")

        pol_ecc, pol_angle = coord_convert(x,y,old2new="cart2pol")
        total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)

        pos_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1, 2))
        param_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1))    
        bin_idx = []
        for i_ecc in range(len(ecc_bounds)-1):
            for i_pol in range(len(pol_bounds)-1):
                ecc_lower = ecc_bounds[i_ecc]
                ecc_upper = ecc_bounds[i_ecc + 1]      

                pol_lower = pol_bounds[i_pol]
                pol_upper = pol_bounds[i_pol + 1]   

                
                ecc_idx =(pol_ecc >= ecc_lower) & (pol_ecc <=ecc_upper)
                pol_idx =(pol_angle >= pol_lower) & (pol_angle <=pol_upper)        
                bin_idx = pol_idx & ecc_idx

                if bin_position:
                    x_mean = np.mean(x[bin_idx])
                    y_mean = np.mean(y[bin_idx])                
                else:
                    ecc_mean = (ecc_lower + ecc_upper)/2
                    pol_mean = (pol_lower + pol_upper)/2   
                    x_mean,y_mean = coord_convert(ecc_mean,pol_mean,old2new="pol2cart")
                
                pos_bin_mean[i_ecc, i_pol, 0] = x_mean
                pos_bin_mean[i_ecc, i_pol, 1] = y_mean
                param_bin_mean[i_ecc, i_pol] = np.mean(param[bin_idx])

        pos_bin_mean = np.reshape(pos_bin_mean, (total_n_bins, 2))
        param_bin_mean = np.reshape(param_bin_mean, (total_n_bins))

        x_bin = pos_bin_mean[:,0]
        y_bin = pos_bin_mean[:,1]
        p_bin = param_bin_mean

    else:
        x_bin = x
        y_bin = y
        p_bin = param
    
    max_dot_size = 200
    min_dot_size = 5
    if scale_dot_size==True:
        # Get min & max ecc -> then scale with the dot size...
        ecc = np.sqrt(x_bin**2 + y_bin**2)
        max_ecc = np.nanmax(ecc)
        min_ecc = np.nanmin(ecc)        
        ecc_norm = (ecc - min_ecc) / (max_ecc - min_ecc)
        dot_sizes = ecc_norm * (max_dot_size-min_dot_size) + min_dot_size
    else:
        dot_sizes = np.ones_like(x_bin)*max_dot_size
    
    scat_col = axs.scatter(x_bin, y_bin, c=p_bin, s=dot_sizes, alpha=dot_alpha)
    fig = plt.gcf()
    fig.colorbar(scat_col, ax=axs)

    if len(ecc_bounds)>0:
        add_scot_patches_and_bin_lines(axs, ecc_bounds, pol_bounds, scotoma_info=scotoma_info, incl_patches=incl_patches, incl_ticks=incl_ticks)
    
    axs.set_aspect('equal')

def independent_arrow_plot(x4bin, y4bin, old_x, old_y, new_x, new_y, axs, ecc_bounds, pol_bounds, scotoma_info=[], **kwargs):
    ''' 
    PLOT FUNCTION: Takes starting coords (x,y) and end coords (new_x, new_y) produces a plot, with arrows from old to new points 
    # Uses x4bin/y4bin as independent points for binning...
    >> axs                              specify axes to plot on
    >> ecc_bounds, pol_bounds           if included, the points will be binned (with ref to old coords)
    >> scotoma_info                         specifies  
    
    '''
    old_col =   kwargs.get("old_col", 'b')  
    new_col =   kwargs.get("new_col", 'r')      
    dot_alpha = kwargs.get("alpha", 0.5)
    incl_patches = kwargs.get("incl_patches", True)

    # BINNING

    ecc4bin, pol4bin = coord_convert(x4bin,y4bin,old2new="cart2pol") # convert coords into polar coords
    total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)

    old_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1, 2))
    new_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1, 2))    
    bin_idx = []
    for i_ecc in range(len(ecc_bounds)-1):
        for i_pol in range(len(pol_bounds)-1):
            ecc_lower = ecc_bounds[i_ecc]
            ecc_upper = ecc_bounds[i_ecc + 1]

            pol_lower = pol_bounds[i_pol]
            pol_upper = pol_bounds[i_pol + 1]            

            ecc_idx =(ecc4bin >= ecc_lower) & (ecc4bin <=ecc_upper)
            pol_idx =(pol4bin >= pol_lower) & (pol4bin <=pol_upper)        
            bin_idx = pol_idx & ecc_idx
            old_bin_mean[i_ecc, i_pol, 0] = np.mean(old_x[bin_idx])
            old_bin_mean[i_ecc, i_pol, 1] = np.mean(old_y[bin_idx])

            new_bin_mean[i_ecc, i_pol, 0] = np.mean(new_x[bin_idx])
            new_bin_mean[i_ecc, i_pol, 1] = np.mean(new_y[bin_idx])

    old_bin_mean = np.reshape(old_bin_mean, (total_n_bins, 2))
    new_bin_mean = np.reshape(new_bin_mean, (total_n_bins, 2))

    binned_old_x = old_bin_mean[:,0]
    binned_old_y = old_bin_mean[:,1]
    binned_new_x = new_bin_mean[:,0]
    binned_new_y = new_bin_mean[:,1]

    dx = binned_new_x - binned_old_x
    dy = binned_new_y - binned_old_y

    axs.scatter(binned_old_x, binned_old_y, color=old_col, s=100, alpha=dot_alpha)
    axs.scatter(binned_new_x, binned_new_y, color=new_col, s=100, alpha=dot_alpha)
    axs.quiver(binned_old_x, binned_old_y, dx, dy, scale_units='xy', angles='xy', alpha=dot_alpha, color='r', scale=1)

    add_scot_patches_and_bin_lines(axs, ecc_bounds, pol_bounds, scotoma_info=scotoma_info, incl_patches=incl_patches)

    return

def rapid_arrow(p1, p2):
    ''' 
    PLOT FUNCTION: Takes starting coords (x,y) and end coords (new_x, new_y) produces a plot, with arrows from old to new points 
    # Uses x4bin/y4bin as independent points for binning...
    >> axs                              specify axes to plot on
    >> ecc_bounds, pol_bounds           if included, the points will be binned (with ref to old coords)
    >> scotoma_info                         specifies  
    
    '''
    fig, axs = plt.subplots(1)
    fig.set_size_inches(5,5)

    ecc_bounds = np.linspace(0, 5, 7)
    pol_bounds = np.linspace(0, 2*np.pi, 13)
    old_col =   'b'
    new_col =   'r'
    dot_alpha = .5

    # BINNING
    rsq_mask = (p1[:,-1]<0.1) & (p2[:,-1]<0.1) 
    old_x=p1[rsq_mask,0]
    old_y=p1[rsq_mask,1]
    new_x=p2[rsq_mask,0]
    new_y=p2[rsq_mask,1]

    ecc4bin, pol4bin = coord_convert(old_x,old_y,old2new="cart2pol") # convert coords into polar coords
    total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)

    old_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1, 2))
    new_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1, 2))    
    bin_idx = []
    for i_ecc in range(len(ecc_bounds)-1):
        for i_pol in range(len(pol_bounds)-1):
            ecc_lower = ecc_bounds[i_ecc]
            ecc_upper = ecc_bounds[i_ecc + 1]

            pol_lower = pol_bounds[i_pol]
            pol_upper = pol_bounds[i_pol + 1]            

            ecc_idx =(ecc4bin >= ecc_lower) & (ecc4bin <=ecc_upper)
            pol_idx =(pol4bin >= pol_lower) & (pol4bin <=pol_upper)        
            bin_idx = pol_idx & ecc_idx
            old_bin_mean[i_ecc, i_pol, 0] = np.mean(old_x[bin_idx])
            old_bin_mean[i_ecc, i_pol, 1] = np.mean(old_y[bin_idx])

            new_bin_mean[i_ecc, i_pol, 0] = np.mean(new_x[bin_idx])
            new_bin_mean[i_ecc, i_pol, 1] = np.mean(new_y[bin_idx])

    old_bin_mean = np.reshape(old_bin_mean, (total_n_bins, 2))
    new_bin_mean = np.reshape(new_bin_mean, (total_n_bins, 2))

    binned_old_x = old_bin_mean[:,0]
    binned_old_y = old_bin_mean[:,1]
    binned_new_x = new_bin_mean[:,0]
    binned_new_y = new_bin_mean[:,1]

    dx = binned_new_x - binned_old_x
    dy = binned_new_y - binned_old_y

    axs.scatter(binned_old_x, binned_old_y, color=old_col, s=100, alpha=dot_alpha)
    axs.scatter(binned_new_x, binned_new_y, color=new_col, s=100, alpha=dot_alpha)
    axs.quiver(binned_old_x, binned_old_y, dx, dy, scale_units='xy', angles='xy', alpha=dot_alpha, scale=1)

    add_scot_patches_and_bin_lines(axs, ecc_bounds, pol_bounds)

    return


def basic_arrow_plot(old_x, old_y, new_x, new_y, axs, **kwargs):
    ''' 
    PLOT FUNCTION: Takes starting coords (x,y) and end coords (new_x, new_y) produces a plot, with arrows from old to new points 
    >> axs                              specify axes to plot on
    >> ecc_bounds, pol_bounds           if included, the points will be binned (with ref to old coords)
    >> scotoma_info                         specifies  
    
    '''
    do_binning = kwargs.get("do_binning", True)
    ecc_bounds = kwargs.get("ecc_bounds", np.linspace(0,5,5))
    pol_bounds = kwargs.get("pol_bounds", np.linspace(0,2*np.pi,13))
    scotoma_info = kwargs.get("scotoma_info", [])
    old_col = kwargs.get("old_col", "b")
    new_col = kwargs.get("new_col", "r")
    dot_alpha = kwargs.get("alpha", 0.5)
    incl_patches = kwargs.get("incl_patches", True)
    patch_col = kwargs.get("patch_col", 'k')

    if do_binning:
        # print("DOING BINNING")

        old_ecc, old_ang = coord_convert(old_x, old_y,old2new="cart2pol")
        total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)

        old_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1, 2))
        new_bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1, 2))    
        bin_idx = []
        for i_ecc in range(len(ecc_bounds)-1):
            for i_pol in range(len(pol_bounds)-1):
                ecc_lower = ecc_bounds[i_ecc]
                ecc_upper = ecc_bounds[i_ecc + 1]

                pol_lower = pol_bounds[i_pol]
                pol_upper = pol_bounds[i_pol + 1]            

                ecc_idx =(old_ecc >= ecc_lower) & (old_ecc <=ecc_upper)
                pol_idx =(old_ang >= pol_lower) & (old_ang <=pol_upper)        
                bin_idx = pol_idx & ecc_idx

                old_bin_mean[i_ecc, i_pol, 0] = np.mean(old_x[bin_idx])
                old_bin_mean[i_ecc, i_pol, 1] = np.mean(old_y[bin_idx])

                new_bin_mean[i_ecc, i_pol, 0] = np.mean(new_x[bin_idx])
                new_bin_mean[i_ecc, i_pol, 1] = np.mean(new_y[bin_idx])

        old_bin_mean = np.reshape(old_bin_mean, (total_n_bins, 2))
        new_bin_mean = np.reshape(new_bin_mean, (total_n_bins, 2))

        old_x2plot = old_bin_mean[:,0]
        old_y2plot = old_bin_mean[:,1]
        new_x2plot = new_bin_mean[:,0]
        new_y2plot = new_bin_mean[:,1]
    else:
        old_x2plot = old_x
        old_y2plot = old_y
        new_x2plot = new_x
        new_y2plot = new_y

    dx = new_x2plot - old_x2plot
    dy = new_y2plot - old_y2plot

    # Plot old pts and new pts (different colors)
    axs.scatter(old_x2plot, old_y2plot, color=old_col, s=100, alpha=dot_alpha)
    axs.scatter(new_x2plot, new_y2plot, color=new_col, s=100, alpha=dot_alpha)
    
    # Add the arrows 
    axs.quiver(old_x2plot, old_y2plot, dx, dy, scale_units='xy', angles='xy', alpha=dot_alpha, scale=1)

    if len(ecc_bounds)>0:
        add_scot_patches_and_bin_lines(axs, ecc_bounds, pol_bounds, scotoma_info=scotoma_info, incl_patches=incl_patches, patch_col=patch_col)

def add_scot_patches_and_bin_lines(axs, ecc_bounds=[], pol_bounds=[], scotoma_info=[], incl_patches=False, **kwargs):
    patch_col = kwargs.get("patch_col", 'k')
    if ecc_bounds==[]:
        ecc_bounds = np.linspace(0, 5, 7)
        pol_bounds = np.linspace(0, 2*np.pi, 13)
    incl_ticks = kwargs.get("incl_ticks", False)
    # **** ADD THE LINES ****
    if not incl_ticks:
        axs.set_xticks([])    
        axs.set_yticks([])
    else:
        axs.set_xticks(ecc_bounds, rotation = 90)
        axs.set_xticklabels([f"{ecc_bounds[i]:.2f}\N{DEGREE SIGN}" for i in range(len(ecc_bounds))], rotation=90)
        axs.set_yticks([])
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(False)        

    n_polar_lines = len(pol_bounds)

    for i_pol in range(n_polar_lines):
        i_pol_val = pol_bounds[i_pol]
        outer_x = np.cos(i_pol_val)*ecc_bounds[-1]
        outer_y = np.sin(i_pol_val)*ecc_bounds[-1]
        outer_x_txt = outer_x*1.1
        outer_y_txt = outer_y*1.1        
        outer_txt = f"{180*i_pol_val/np.pi:.0f}\N{DEGREE SIGN}"
        # Don't show 360, as this goes over the top of 0 degrees and is ugly...
        if not '360' in outer_txt:
            axs.plot((0, outer_x), (0, outer_y), color="k", alpha=0.3)
            if incl_ticks:
                axs.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')
    
    for i_ecc, i_ecc_val in enumerate(ecc_bounds):
        grid_line = patches.Circle((0, 0), i_ecc_val, color="k", alpha=0.3, fill=0)    
        axs.add_patch(grid_line)
    
    # **** PATCHES *****
    if incl_patches:
        aperture_line = patches.Circle((0, 0), scotoma_info["aperture_rad"], color=patch_col, linewidth=8, alpha=0.5, fill=0)    
        axs.add_patch(aperture_line)
        if scotoma_info['scotoma_centre']!=[]: # Only add scotoma info if it exists...       
            scot = patches.Circle(scotoma_info["scotoma_centre"], scotoma_info["scotoma_radius"], color=patch_col, linewidth=8, fill=False, alpha=0.6)
            axs.add_patch(scot)

def plot_bin_line(X,Y,bin_using, **kwargs):
    # GET PARAMETERS....
    axs = kwargs.get("axs", [])    
    line_col = kwargs.get("line_col", "k")    
    line_label = kwargs.get("line_label", "_")
    n_bins = kwargs.get("n_bins", 10)    
    xerr = kwargs.get("xerr", False)
    # Do the binning
    X_mean = binned_statistic(bin_using, X, bins=n_bins, statistic='mean')[0]
    X_std = binned_statistic(bin_using, X, bins=n_bins, statistic='std')[0]
    count = binned_statistic(bin_using, X, bins=n_bins, statistic='count')[0]
    Y_mean = binned_statistic(bin_using, Y, bins=n_bins, statistic='mean')[0]                
    Y_std = binned_statistic(bin_using, Y, bins=n_bins, statistic='std')[0]  #/ np.sqrt(bin_data['bin_X']['count'])              
    if xerr:
        axs.errorbar(
            X_mean,
            Y_mean,
            yerr=Y_std,
            xerr=X_std,
            color=line_col,
            label=line_label, 
            )
    else:
        axs.errorbar(
            X_mean,
            Y_mean,
            yerr=Y_std,
            # xerr=X_std,
            color=line_col,
            label=line_label,
            )        
    axs.legend()

    return None


def binned_plot(X,Y, **kwargs):
    # GET PARAMETERS....
    axs = kwargs.get("axs", [])    
    line_col = kwargs.get("line_col", ["b", "r"])    
    scat_col = kwargs.get("scat_col", "k")    
    x_label = kwargs.get("x_label", [])    
    y_label = kwargs.get("y_label", [])
    title = kwargs.get("title", "")    
    # y_lim=kwargs.get("y_lim", [])
    # x_lim=kwargs.get("x_lim", [])
    n_bins = kwargs.get("n_bins", 10)
    do_scatter = kwargs.get("do_scatter", True) 
    do_bin_lines = kwargs.get("do_bin_lines", True)
    do_bin_with_x = kwargs.get("do_bin_with_x", True)    
    do_bin_with_y = kwargs.get("do_bin_with_y", True)

    # [1] Scatter values...
    if do_scatter:
        axs.scatter(X,Y, color=scat_col, alpha=0.1)

    # [2] Plot bin lines 
    if do_bin_lines and do_bin_with_x:        
        plot_bin_line(
            X=X,
            Y=Y,
            bin_using=X, 
            axs=axs,
            line_col=line_col[0], 
            line_label=f'Bin using - {x_label}',
            n_bins=n_bins,
            )
    if do_bin_lines and do_bin_with_y:
        plot_bin_line(
            X=X,
            Y=Y,
            bin_using=Y, 
            axs=axs,
            line_col=line_col[1], 
            line_label=f'Bin using - {y_label}',
            n_bins=n_bins,
            )            
    max_val = np.nanmax([np.nanmax(X),np.nanmax(Y)]) 
    min_val = np.nanmin([np.nanmin(X),np.nanmin(Y)]) 
    axs.set_xlim(min_val, max_val)
    axs.set_ylim(min_val, max_val)       

    axs.plot(
        (0,axs.get_ylim()[1]),
        (0,axs.get_ylim()[1]),
        'k'
    )
    if legend_check(axs):
        axs.legend()
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.set_title(title)
    axs.set_aspect('equal')     


    return None

def legend_check(axs):
    do_legend = False
    for child in axs.get_children():
        if isinstance(child, mpl.lines.Line2D):
            if child.get_label()[0]!="_":
                do_legend = True
    for child in axs.collections:
        if child.get_label()[0]!="_":
            do_legend=True
    return do_legend
            



def time_series_plot(params, prfpy_stim, real_tc=[], pred_tc=[], model='gauss', scotoma_info=[], show_stim_frame_x=[], **kwargs):
    # GET PARAMETERS....
    # axs=kwargs.get("axs", [])    
    # col=kwargs.get("col", "k")    
    # line_label=kwargs.get("line_label", " ")
    # y_label=kwargs.get("y_label", "")
    # axs_title=kwargs.get("axs_title", "")
    # y_lim=kwargs.get("y_lim", [])
    title=kwargs.get("title", None)
    fmri_TR =kwargs.get("fmri_TR", 1.5)

    if model=='gauss':
        model_idx = print_p()['gauss']
    elif model=='norm':
        model_idx = print_p()['norm']

    # ************* PRF (+stimulus) PLOT *************
    fig = plt.figure(constrained_layout=True, figsize=(15,5))
    gs00 = fig.add_gridspec(1,2, width_ratios=[10,20])
    ax1 = fig.add_subplot(gs00[0])
    # Set vf extent
    aperture_rad = prfpy_stim.screen_size_degrees/2    
    ax1.set_xlim(-aperture_rad, aperture_rad)
    ax1.set_ylim(-aperture_rad, aperture_rad)

    if show_stim_frame_x !=[]:
        this_dm = prfpy_stim.design_matrix[:,:,show_stim_frame_x]
        this_dm_rgba = np.zeros((this_dm.shape[0], this_dm.shape[1], 4))
        this_dm_rgba[:,:,0] = 1-this_dm
        this_dm_rgba[:,:,1] = 1-this_dm
        this_dm_rgba[:,:,2] = 1-this_dm
        this_dm_rgba[:,:,3] = .5
        ax1.imshow(this_dm_rgba, extent=[-aperture_rad, aperture_rad, -aperture_rad, aperture_rad])
    
    # Add prfs
    prf_x = params[0]
    prf_y = params[1]
    # Add normalizing PRF (FWHM)
    if model=='norm':
        prf_2_fwhm = 2*np.sqrt(2*np.log(2))*params[model_idx['n_sigma']] # 
        if prf_2_fwhm>aperture_rad:
            ax1.set_xlabel('*Norm PRF is too larget to show - covers whole screen')
            norm_prf_label = '*Norm PRF'
        else:
            norm_prf_label = 'Norm PRF'
        prf_2 = patches.Circle(
            (prf_x, prf_y), prf_2_fwhm, edgecolor="r", 
            facecolor=[1,0,0,.5], 
            linewidth=8, fill=False,
            label=norm_prf_label,)    
        ax1.add_patch(prf_2)
    # Add activating PRF (fwhm)
    prf_fwhm = 2*np.sqrt(2*np.log(2))*params[model_idx['a_sigma']] # 
    prf_1 = patches.Circle(
        (prf_x, prf_y), prf_fwhm, edgecolor="b", facecolor=[1,1,1,0], 
        linewidth=8, fill=False, label='PRF')
    ax1.add_patch(prf_1)

    # add scotoma
    if scotoma_info['scotoma_centre'] != []:
        scot = patches.Circle(
            scotoma_info["scotoma_centre"], scotoma_info["scotoma_radius"], 
            edgecolor="k", facecolor="w", linewidth=8, fill=True, alpha=1,
            label='scotoma')
        ax1.add_patch(scot)
    
    # Add 0 lines...
    ax1.plot((0,0), ax1.get_ylim(), 'k')
    ax1.plot(ax1.get_xlim(), (0,0), 'k')
    ax1.legend()
    ax1.set_title(model)
    # ************* TIME COURSE PLOT *************
    # Check title - if not present use parameters...
    if title == None:
        param_count = 0
        set_title = ''
        for param_key in model_idx.keys():
            set_title += f'{param_key}={round(params[model_idx[param_key]],2)}; '

            if param_count > 3:
                set_title += '\n'
                param_count = 0
            param_count += 1
    else:
        set_title = title
    x_label = "time (s)"
    x_axis = np.array(list(np.arange(0,real_tc.shape[0])*fmri_TR)) 
    ax2 = fig.add_subplot(gs00[1])
    lsplt.LazyPlot(
        [real_tc, pred_tc],
        xx=x_axis,
        color=['#cccccc', 'r'], 
        labels=['real', 'pred'], 
        add_hline='default',
        x_label=x_label,
        y_label="amplitude",
        axs=ax2,
        title=set_title,
        # xkcd=True,
        # font_size=font_size,
        line_width=[0.5, 3],
        markers=['.', None],
        # **kwargs,
        )
    # If showing stim - show corresponding time point in timeseries
    if show_stim_frame_x !=[]:
        this_time_pt = show_stim_frame_x * fmri_TR
        current_ylim = ax2.get_ylim()
        ax2.plot((this_time_pt,this_time_pt), current_ylim, 'k')

    return fig, ax1, ax2    


    

def update_axs_fontsize(axs, new_font_size):
    for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
                axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(new_font_size)        
    for item in axs.get_children():
        if isinstance(item, mpl.legend.Legend):
            texts = item.get_texts()
            if not isinstance(texts, list):
                texts = [texts]
            for i_txt in texts:
                i_txt.set_fontsize(new_font_size)
        elif isinstance(item, mpl.text.Text):
            item.set_fontsize(new_font_size)                

def update_fig_fontsize(fig, new_font_size):
    fig_kids = fig.get_children()
    for i_kid in fig_kids:
        if isinstance(i_kid, mpl.axes.Axes):
            update_axs_fontsize(i_kid, new_font_size)
        elif isinstance(i_kid, mpl.figure.SubFigure):
            update_fig_fontsize(i_kid, new_font_size)
        elif isinstance(i_kid, mpl.text.Text):
            i_kid.set_fontsize(new_font_size)


def update_axs_dotsize(axs, new_dot_size):
    # print(type(axs))
    for i_kid in axs.get_children():
        # print(type(i_kid))
        if isinstance(i_kid, mpl.collections.PathCollection):
            # if hasattr(i_kid, 'set_sizes')
            # print(type(i_kid))
            i_kid.set_sizes(np.array([new_dot_size]))
        elif isinstance(i_kid, mpl.lines.Line2D):
            i_kid.set_markersize(np.array([new_dot_size]))


def update_fig_dotsize(fig, new_dot_size):
    for i_kid in fig.get_children():
        # print(i_kid)
        if isinstance(i_kid, mpl.axes.Axes):
            update_axs_dotsize(i_kid, new_dot_size)
        elif isinstance(i_kid, mpl.figure.SubFigure):
            update_fig_dotsize(i_kid, new_dot_size)     



