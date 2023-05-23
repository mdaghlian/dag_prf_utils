import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
from scipy.stats import binned_statistic
from .utils import *

default_ecc_bounds =  np.linspace(0, 5, 7)
default_pol_bounds = np.linspace(-np.pi, np.pi, 13)

def dag_add_ecc_pol_lines(ax, **kwargs):
    '''
    Add eccentricity and polar lines to a plot    
    '''
    ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)        
    line_col = kwargs.get("vf_line_col", 'k' )
    incl_ticks = kwargs.get("incl_ticks", False)
    aperture_rad = kwargs.get("aperture_rad", None)
    aperture_col = kwargs.get('aperture_col', 'k')
    # **** ADD THE LINES ****
    if not incl_ticks:
        ax.set_xticks([])    
        ax.set_yticks([])
    else:
        ax.set_xticks(ecc_bounds, rotation = 90)
        ax.set_xticklabels([f"{ecc_bounds[i]:.2f}\N{DEGREE SIGN}" for i in range(len(ecc_bounds))], rotation=90)
        ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)        

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
            ax.plot((0, outer_x), (0, outer_y), color=line_col, alpha=0.3)
            if incl_ticks:
                ax.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')

    for i_ecc, i_ecc_val in enumerate(ecc_bounds):
        grid_line = patches.Circle((0, 0), i_ecc_val, color=line_col, alpha=0.3, fill=0)    
        ax.add_patch(grid_line)                    
    if aperture_rad!=None:
        aperture_line = patches.Circle((0, 0), aperture_rad, color=aperture_col, linewidth=8, alpha=0.5, fill=0)    
        ax.add_patch(aperture_line)
    # Set equal ratio
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio) 

def dag_add_ax_basics(ax, **kwargs):        
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    title = kwargs.get("title", None)
    x_lim = kwargs.get("x_lim", None)
    y_lim = kwargs.get("y_lim", None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.legend()

def dag_update_fig_fontsize(fig, new_font_size):
    fig_kids = fig.get_children()
    for i_kid in fig_kids:
        if isinstance(i_kid, mpl.axes.Axes):
            dag_update_ax_fontsize(i_kid, new_font_size)
        elif isinstance(i_kid, mpl.text.Text):
            i_kid.set_fontsize(new_font_size)

def dag_update_ax_fontsize(ax, new_font_size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(new_font_size)        
    for item in ax.get_children():
        if isinstance(item, mpl.legend.Legend):
            texts = item.get_texts()
            if not isinstance(texts, list):
                texts = [texts]
            for i_txt in texts:
                i_txt.set_fontsize(new_font_size)
        elif isinstance(item, mpl.text.Text):
            item.set_fontsize(new_font_size)                

def dag_update_fig_fontsize(fig, new_font_size):
    fig_kids = fig.get_children()
    for i_kid in fig_kids:
        if isinstance(i_kid, mpl.axes.Axes):
            dag_update_ax_fontsize(i_kid, new_font_size)
        elif isinstance(i_kid, mpl.figure.SubFigure):
            dag_update_fig_fontsize(i_kid, new_font_size)
        elif isinstance(i_kid, mpl.text.Text):
            i_kid.set_fontsize(new_font_size)

def dag_update_ax_dotsize(ax, new_dot_size):
    # print(type(ax))
    for i_kid in ax.get_children():
        # print(type(i_kid))
        if isinstance(i_kid, mpl.collections.PathCollection):
            # if hasattr(i_kid, 'set_sizes')
            # print(type(i_kid))
            i_kid.set_sizes(np.array([new_dot_size]))
        elif isinstance(i_kid, mpl.lines.Line2D):
            i_kid.set_markersize(np.array([new_dot_size]))


def dag_update_fig_dotsize(fig, new_dot_size):
    for i_kid in fig.get_children():
        # print(i_kid)
        if isinstance(i_kid, mpl.axes.Axes):
            dag_update_ax_dotsize(i_kid, new_dot_size)
        elif isinstance(i_kid, mpl.figure.SubFigure):
            dag_update_fig_dotsize(i_kid, new_dot_size)

def dag_return_ecc_pol_bin(params2bin, ecc4bin, pol4bin, bin_weight=None, **kwargs):
    '''
    params2bin      list of np.ndarrays, to bin
    ecc4bin         eccentricity & polar angle used in binning
    pol4bin
    ecc_bounds      how to split into bins 
    pol_bounds
    bin_weight      Something used to weight the binning (e.g., rsq)
    Return the parameters binned by the specified ecc, and pol bounds 

    '''
    # TODO have option for converting X,Y...    
    ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)            
    n_params_gt_1 = False
    if not isinstance(params2bin, list):
        params2bin = [params2bin]
        n_params_gt_1 = True
    
    total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)
    params_binned = []
    for i_param in range(len(params2bin)):

        bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1))
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

                if bin_weight is not None:
                    bin_mean[i_ecc, i_pol] = (params2bin[i_param][bin_idx] * bin_weight[bin_idx]).sum() / bin_weight[bin_idx].sum()
                else:
                    bin_mean[i_ecc, i_pol] = np.mean(params2bin[i_param][bin_idx])

        bin_mean = np.reshape(bin_mean, total_n_bins)
        # REMOVE ANY NANS
        bin_mean = bin_mean[~np.isnan(bin_mean)]
        params_binned.append(bin_mean)
    if n_params_gt_1:
        params_binned = params_binned[0] 
    return params_binned

def dag_visual_field_scatter(ax, dot_x, dot_y, **kwargs):
    ''' 
    Plot a (prf) parameter around the visual field (e.g., size, rsquared etc)
    Using scatter for position dot_x, dot_y

    '''
    do_binning = kwargs.get("do_binning", False)
    ecc_bounds = kwargs.get("ecc_bounds", np.linspace(0, 5, 7) )
    pol_bounds = kwargs.get("pol_bounds", np.linspace(0, 2*np.pi, 13))            
    dot_alpha = kwargs.get("alpha", 0.5)
    dot_size = kwargs.get("dot_size",200)
    # -> add option for dot size scaling... ( & alpha scaling) ??
    bin_weight = kwargs.get("bin_weight", None)
    dot_col = kwargs.get("dot_col", 'k')   
    dot_vmin =  kwargs.get("dot_vmin", None)   
    dot_vmax =  kwargs.get("dot_vmax", None)   
    dot_cmap =  kwargs.get("dot_cmap", None)   
    if isinstance(dot_col, np.ndarray) & (dot_cmap==None):
        dot_cmap = 'viridis'

    if do_binning:

        dot_ecc, dot_pol = dag_coord_convert(dot_x,dot_y,old2new="cart2pol")
        total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)
        bin_x, bin_y = dag_return_ecc_pol_bin(params2bin=[dot_x, dot_y], 
                            ecc4bin=dot_ecc, 
                            pol4bin=dot_pol, 
                            bin_weight=bin_weight,
                            ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)
        if isinstance(dot_col, np.ndarray):
            bin_col = dag_return_ecc_pol_bin(params2bin=dot_col, 
                                ecc4bin=dot_ecc, 
                                pol4bin=dot_pol, 
                                bin_weight=bin_weight,
                                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)            
        else:
            bin_col = dot_col
        if isinstance(dot_size, np.ndarray):
            bin_size = dag_return_ecc_pol_bin(params2bin=dot_size, 
                                ecc4bin=dot_ecc, 
                                pol4bin=dot_pol, 
                                bin_weight=bin_weight,
                                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)            
        else:
            bin_size = dot_size
        if isinstance(dot_alpha, np.ndarray):
            bin_alpha = dag_return_ecc_pol_bin(params2bin=dot_alpha, 
                                ecc4bin=dot_ecc, 
                                pol4bin=dot_pol, 
                                bin_weight=bin_weight,
                                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)            
        else:
            bin_alpha = dot_alpha

    else:
        bin_x = dot_x
        bin_y = dot_y
        bin_col = dot_col
        bin_size = dot_size
        bin_alpha = dot_alpha
    
    # max_dot_size = 200
    # min_dot_size = 5
    # if scale_dot_size==True:
    #     # Get min & max ecc -> then scale with the dot size...
    #     ecc = np.sqrt(x_bin**2 + y_bin**2)
    #     max_ecc = np.nanmax(ecc)
    #     min_ecc = np.nanmin(ecc)        
    #     ecc_norm = (ecc - min_ecc) / (max_ecc - min_ecc)
    #     dot_sizes = ecc_norm * (max_dot_size-min_dot_size) + min_dot_size
    # else:
    #     dot_sizes = np.ones_like(x_bin)*max_dot_size
    
    scat_col = ax.scatter(bin_x, bin_y, c=bin_col, s=bin_size, alpha=bin_alpha, cmap=dot_cmap, vmin=dot_vmin, vmax=dot_vmax)
    cb = None
    if not isinstance(bin_col, str):
        fig = plt.gcf()
        cb = fig.colorbar(scat_col, ax=ax)        
    dag_add_ecc_pol_lines(ax, ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)    
    dag_add_ax_basics(ax, **kwargs)    
    return ax, cb

def dag_plot_bin_line(ax, X,Y, bin_using, **kwargs):    
    # GET PARAMETERS....
    line_col = kwargs.get("line_col", "k")    
    line_label = kwargs.get("line_label", None)
    lw= kwargs.get("lw", 5)
    n_bins = kwargs.get('n_bins', 10)
    bins = kwargs.get('bins', None)
    if not isinstance(bins, (np.ndarray, list)):
        bins = n_bins    
    xerr = kwargs.get("xerr", False)
    # Do the binning
    X_mean = binned_statistic(bin_using, X, bins=bins, statistic='mean')[0]
    X_std = binned_statistic(bin_using, X, bins=bins, statistic='std')[0]
    count = binned_statistic(bin_using, X, bins=bins, statistic='count')[0]
    Y_mean = binned_statistic(bin_using, Y, bins=bins, statistic='mean')[0]                
    Y_std = binned_statistic(bin_using, Y, bins=bins, statistic='std')[0]  #/ np.sqrt(bin_data['bin_X']['count'])              
    if xerr:
        ax.errorbar(
            X_mean,
            Y_mean,
            yerr=Y_std,
            xerr=X_std,
            color=line_col,
            label=line_label, 
            lw=lw,
            )
    else:
        ax.errorbar(
            X_mean,
            Y_mean,
            yerr=Y_std,
            xerr=X_std,
            color=line_col,
            label=line_label,
            lw=lw,
            )        
    ax.legend()    
    dag_add_ax_basics(ax, **kwargs)

def dag_arrow_plot(ax, old_x, old_y, new_x, new_y, **kwargs):
    ''' 
    PLOT FUNCTION: 
    Takes voxel position in condition 1 and 2  and end coords (new_x, new_y) produces a plot, with arrows from old to new points 
    Will also show the aperture of stimuli
    Parameters
    ---------------
    ax :           matplotlib axes         where to plot
    old_x,old_y     np.ndarray              Old x,y coord
    new_x,new_y     np.ndarray              New x,y coord
    
    OPTIONAL
    ---------------
    do_binning      bool                    Bin the position (or not)
    bin_weight      np.ndarray              Weighted mean in each bin, not just the average    
    do_scatter      bool                    Include scatters of the voxel positions
    do_scatter_old  ""
    do_scatter_new  ""
    do_arrows       bool                    Include arrows
    ecc_bounds      np.ndarays              If binning, how split the visual field
    pol_bounds                               
    old_col         any value for color     Gives color for points
    new_col        
    patch_col       any value for color     Color for the patch 
    dot_alpha       single float or array   Alpha for each point (not split by old and new)
    dot_size        single float or array   Size for each point (not split by old and new)
    arrow_col       string                  Color for the arrows. Another option is "angle", where arrows will be coloured depending on there angle
    arrow_kwargs    dict                    Another dict for arrow properties
        scale       Always 1, exact pt to pt
        width       Of shaft relative to plot
        headwidth   relative to shaft width
    '''
    # Get arguments related to plotting:
    do_binning = kwargs.get("do_binning", False)
    bin_weight = kwargs.get("bin_weight", None)
    do_scatter = kwargs.get("do_scatter", False)
    do_scatter_old = kwargs.get("do_scatter_old", True)
    do_scatter_new = kwargs.get("do_scatter_new", True)
    if not do_scatter:
        do_scatter_old = False
        do_scatter_new = False
    do_arrows = kwargs.get("do_arrows", True)
    ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)
    old_col = kwargs.get("old_col", 'k')
    new_col = kwargs.get("new_col", 'g')
    # -> for now only 1 alpha, size per dot
    dot_alpha = kwargs.get('dot_alpha', .5)
    dot_size = kwargs.get('dot_size', 500)

    # aperture_rad = kwargs.get("aperture_rad", None)
    # patch_col = kwargs.get("patch_col", "k")
    arrow_col = kwargs.get("arrow_col", 'b')
    arrow_kwargs = {
        'scale'     : 1,                                    # ALWAYS 1 -> exact pt to exact pt 
        'width'     : kwargs.get('arrow_width', .01),       # of shaft (relative to plot )
        'headwidth' : kwargs.get('arrow_headwidth', 1.5),    # relative to width

    }    
    
    if do_binning:
        # print("DOING BINNING") 
        old_ecc, old_pol = dag_coord_convert(old_x, old_y,old2new="cart2pol")
        total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)
        old_bin_x, old_bin_y, new_bin_x, new_bin_y = dag_return_ecc_pol_bin(
            params2bin=[old_x, old_y, new_x, new_y], 
            ecc4bin=old_ecc, 
            pol4bin=old_pol, 
            bin_weight=bin_weight,
            ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)
    else:
        old_bin_x, old_bin_y, new_bin_x, new_bin_y = old_x, old_y, new_x, new_y
    
    dx = new_bin_x - old_bin_x
    dy = new_bin_y - old_bin_y

    # Plot old pts and new pts (different colors)
    if do_scatter_old:
        ax.scatter(old_bin_x, old_bin_y, color=old_col, s=dot_size, alpha=dot_alpha, )# cmap=dot_cmap)
    if do_scatter_new:
        ax.scatter(new_bin_x, new_bin_y, color=new_col, s=dot_size, alpha=dot_alpha, )# cmap=dot_cmap)
    
    # Add the arrows 
    if do_arrows: # Arrows all the same color
        if arrow_col=='angle':
            # Get the angles for the arrows
            _, angle = dag_coord_convert(dx, dy, 'cart2pol')
            q_cmap = mpl.cm.__dict__['hsv']
            q_norm = mpl.colors.Normalize()
            q_norm.vmin = -3.14
            q_norm.vmax = 3.14
            q_col = q_cmap(q_norm(angle))                
        else:
            q_col = arrow_col
        print(old_bin_x.shape)
        ax.quiver(old_bin_x, old_bin_y, dx, dy, scale_units='xy', 
                    angles='xy', alpha=dot_alpha,color=q_col,  **arrow_kwargs)
        
        # # For the colorbar
        # if isinstance(dot_col, np.ndarray):
        #     scat_col = ax.scatter(
        #         np.zeros_like(LE_x2plot), np.zeros_like(LE_x2plot), s=np.zeros_like(LE_x2plot), 
        #         c=dot_col, vmin=dot_vmin, vmax=dot_vmax, cmap=dot_cmap)
        #     fig = plt.gcf()
        #     cb = fig.colorbar(scat_col, ax=ax)        
        #     cb.set_label(kwargs['dot_col'])

    dag_add_ecc_pol_lines(ax, ecc_bounds=ecc_bounds, pol_bounds=pol_bounds, **kwargs)        
    dag_add_ax_basics(ax, **kwargs)    
    # END FUNCTION 


import matplotlib.colors as mcolors

def dag_make_custom_cmap(col_list, col_steps=None, cmap_name=''):
    """Return a LinearSegmentedColormap
    col_list        list of colors (can be rgb tuples or something which can be converted to rgb tuples by mcolors.ColorConverter().to_rgb)
    col_steps
    
    inspired by: https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale

    Example
    ------

    """
    if col_steps==None:
        col_val = np.linspace(0,1, len(col_list))
    elif isinstance(col_steps, list):
        col_val = np.array(col_steps)
    # print(col_val)
    col_val = dag_rescale_bw(col_val) # recale to b/w 0 and 1
    # Change any values to rgb tuple
    conv2rgb = mcolors.ColorConverter().to_rgb
    for i_col,v_col in enumerate(col_list):
        if not isinstance(v_col, tuple):
            col_list[i_col] = conv2rgb(v_col)
    
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name,list(zip(col_val, col_list)))

    return custom_cmap

def dag_make_diverge_cmap(low, high, mid='white'):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    custom_cmap = dag_make_custom_cmap([low, mid, high])
    return custom_cmap

def dag_get_cmap(cmap_name, **kwargs):
    
    if isinstance(cmap_name, dict):
        cdict_copy = cmap_name
        cmap_name = cdict_copy.get('cmap_name', '')
        col_list = cdict_copy.get('col_list', None)
        col_steps = cdict_copy.get('col_steps', None)    
    else:
        col_list = kwargs.get('col_list', None)
        col_steps = kwargs.get('col_steps', None)

    if col_list is not None:
        this_cmap = dag_make_custom_cmap(col_list=col_list, col_steps=col_steps, cmap_name=cmap_name)
    elif cmap_name in custom_col_dict.keys():
        col_list = custom_col_dict[cmap_name]['col_list']
        col_steps = custom_col_dict[cmap_name]['col_steps']
        this_cmap = dag_make_custom_cmap(col_list=col_list, col_steps=col_steps, cmap_name=cmap_name)
    elif cmap_name in mpl.cm.__dict__.keys():
        this_cmap = mpl.cm.__dict__[cmap_name]
        
    return this_cmap

def dag_rapid_corr(ax, x,y, **kwargs):
    ax.scatter(x,y, **kwargs)
    corr_xy = dag_get_corr(x,y)
    kwargs['title'] = f'corr={corr_xy:.3f}'
    
    dag_add_ax_basics(ax=ax, **kwargs)


# ************ SAVED COLOR STUFF *************************
def dag_rgb(r,g,b):
    return [r/255,g/255,b/255]

ecc_custom_col_list = [
    dag_rgb(128, 0, 0),
    dag_rgb(255, 0, 0),
    dag_rgb(255, 255, 0),
    dag_rgb(0, 255, 0),
    dag_rgb(0, 128, 0),
    dag_rgb(0, 128, 128),
    dag_rgb(0, 0, 255),
    dag_rgb(0, 0, 128),
    dag_rgb(128, 0, 128)
    ]
ecc_custom_col_steps = [0, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5]
ecc_custom_dict = {
    'col_list' : ecc_custom_col_list,
    'col_steps' :  ecc_custom_col_steps
}
pol_custom_col_list = [
    dag_rgb(255, 0, 0),
    dag_rgb(255, 255, 0),
    dag_rgb(0, 128, 0),
    dag_rgb(0, 255, 255),
    dag_rgb(0, 0, 255),
    dag_rgb(238, 130, 238),
    dag_rgb(255, 0, 0),
    dag_rgb(255, 255, 0),
    dag_rgb(0, 128, 0),
    dag_rgb(0, 255, 255),
    dag_rgb(0, 0, 255),
    dag_rgb(238, 130, 238),
    dag_rgb(255, 0, 0), 
    ]

pol_custom_col_steps = [-3.14, -2.65, -2.09, -1.75, -1.05, -0.5, 0, 0.5, 1.05, 1.57, 2.09, 2.65, 3.14]
pol_custom_dict = {
    'col_list' : pol_custom_col_list,
    'col_steps' : pol_custom_col_steps
}
custom_col_dict = {
    'pol' : pol_custom_dict,
    'ecc' : ecc_custom_dict,
}