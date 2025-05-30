import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import patches
from scipy.stats import binned_statistic
import json
import os
opj = os.path.join


from dag_prf_utils.utils import *
from dag_prf_utils.cmap_functions import *

# Default bounds for visual field plotting
default_ecc_bounds =  np.linspace(0, 5, 7)
default_pol_bounds = np.linspace(-np.pi, np.pi, 13)

# def dag_rm_dag_cmaps_from_mpl():
#     # Add to matplotlib cmaps?
#     for cm_name in custom_col_dict.keys():
#         plt.unregister_cmap(cm_name)

def dag_cmap_plotter(cmap, vmin=None, vmax=None, title='', **kwargs):
    ax = kwargs.get('ax', None)
    return_ax = kwargs.get('return_ax', False)
    return_fig = kwargs.get('return_fig', False)
    plot_type = kwargs.get('plot_type', 'linear')
    default_fig_size = {'linear':(10,2), 'pol':(5,5), 'ecc' : (5,5)}    

    try:
        cmap = dag_get_cmap(cmap)
    except:
        cmap = dag_cmap_from_str(cmap)

    if ax is None:
        figsize = kwargs.get('figsize', default_fig_size[plot_type])
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw=dict(
                projection=None if plot_type=='linear' else 'polar'
                )
            )        

    if plot_type=='linear':

        mpl.colorbar.ColorbarBase(
            ax, orientation='horizontal', cmap=cmap, 
            norm=mpl.colors.Normalize(vmin, vmax)
            )
    elif plot_type=='pol':
        # POLAR ANGLE
        pol = np.linspace(-np.pi, np.pi, 100)
        ecc = np.linspace(0, 1, 100)
        pol, ecc = np.meshgrid(pol, ecc)
        cax = ax.pcolormesh(
            pol, ecc, pol, cmap=cmap, shading='auto', 
            norm=mpl.colors.Normalize(-np.pi, np.pi),            
            )
        ax.set_title(title)
        ax.set_yticks([])
        dag_update_ax_fontsize(ax, 20)
    elif plot_type=='ecc':
        pol = np.linspace(-np.pi, np.pi, 100)
        ecc_vmax = 5 if vmax is None else vmax
        ecc = np.linspace(0, ecc_vmax, 100)
        pol, ecc = np.meshgrid(pol, ecc)
        cax = ax.pcolormesh(
            pol, ecc, ecc, cmap=cmap, shading='auto', 
            norm=mpl.colors.Normalize(0, ecc_vmax),            
            )
        ax.set_title(title)
        ax.set_xticks([])
        dag_update_ax_fontsize(ax, 20)        

    ax.set_title(title)
    dag_update_ax_fontsize(ax, 20)
    if return_ax:
        return ax
    if return_fig:
        return fig
    

def dag_add_dag_cmaps_to_mpl():
    # Add to matplotlib cmaps?
    for cm_name in custom_col_dict.keys():
        this_cm = dag_get_cmap(cm_name)
        mpl.colormaps.register(cmap=this_cm)

def dag_add_ecc_pol_lines(ax, **kwargs):
    '''dag_add_ecc_pol_lines    
    Description:
        Add eccentricity and polar lines to a plot
        Useful for plotting visual field
    Input:        
        ax              matplotlib axes
        *Optional*
        ecc_bounds      np.ndarray      eccentricity bounds
        pol_bounds      np.ndarray      polar angle bounds
        vf_line_col     str             color of lines
        incl_ticks      bool            Include ticks on the plot
        aperture_rad    float           Radius of aperture
        aperture_col    str             Color of aperture
    Output:
        None    
    '''
    ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)        
    line_col = kwargs.get("vf_line_col", 'k' )
    line_alpha = kwargs.get("vf_line_alpha", 0.3)
    line_width = kwargs.get("vf_line_width", 1)
    incl_ticks = kwargs.get("incl_ticks", False)
    aperture_rad = kwargs.get("aperture_rad", None)
    aperture_col = kwargs.get('aperture_col', 'k')
    do_radians = kwargs.get('do_radians', True)
    # **** ADD THE LINES ****
    if not incl_ticks: # Don't include ticks
        ax.set_xticks([])    
        ax.set_yticks([])
    else:
        ax.set_xticks(ecc_bounds, rotation = 90) # 
        ax.set_xticklabels(
            [f"{ecc_bounds[i]:.2f}\N{DEGREE SIGN}" for i in range(len(ecc_bounds))], 
            rotation=90
            )   # Set the labels of  
        ax.set_yticks([])
    # Remove the spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)        

    # Add the lines
    n_polar_lines = len(pol_bounds)

    for i_pol in range(n_polar_lines): # Loop through polar lines
        i_pol_val = pol_bounds[i_pol] # Get the polar angle
        outer_x = np.cos(i_pol_val)*ecc_bounds[-1] # Get the x,y coords of the outer line
        outer_y = np.sin(i_pol_val)*ecc_bounds[-1] 
        outer_x_txt = outer_x*1.1 # Get the x,y coords of the text
        outer_y_txt = outer_y*1.1        
        if do_radians:
            outer_txt = f"{i_pol_val:.2f}" # Get the text
        else:
            outer_txt = f"{180*i_pol_val/np.pi:.0f}\N{DEGREE SIGN}" # Get the text
        # Don't show 360, as this goes over the top of 0 degrees and is ugly...
        if ('360' in outer_txt) or ('-3.14' in outer_txt):
            continue
        else:
            ax.plot(
                (0, outer_x), 
                (0, outer_y), 
                color=line_col, 
                alpha=line_alpha,
                lw=line_width,
                )
            if incl_ticks:
                ax.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')

        # if not '360' in outer_txt:
        #     ax.plot((0, outer_x), (0, outer_y), color=line_col, alpha=0.3)
        #     if incl_ticks:
        #         ax.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')

    for i_ecc, i_ecc_val in enumerate(ecc_bounds): # Loop through eccentricity lines
        grid_line = patches.Circle(
            (0, 0), i_ecc_val, 
            color=line_col, 
            alpha=line_alpha,
            lw=line_width,
            fill=0
            )    
        ax.add_patch(grid_line)                    
    
    if aperture_rad!=None: # Add the aperture
        aperture_line = patches.Circle(
            (0, 0), aperture_rad, color=aperture_col, linewidth=8, alpha=0.5, fill=0)    
        ax.add_patch(aperture_line)
    # Set equal ratio
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio) 

def dag_add_ax_basics(ax, **kwargs):    
    '''dag_add_ax_basics
    Description:
        Add basic features to a plot
        
    Input:
        ax              matplotlib axes
        *Optional*
        xlabel          str             x axis label
        ylabel          str             y axis label
        title           str             title
        x_lim           tuple           x axis limits
        y_lim           tuple           y axis limits
    Return:
        None
    '''    
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", None)
    title = kwargs.get("title", None)
    x_lim = kwargs.get("x_lim", None)
    y_lim = kwargs.get("y_lim", None)
    despine = kwargs.get('despine', True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)    
    # ax.legend()
    if despine:
        if 'right' in ax.spines.keys():
            ax.spines['right'].set_visible(False)
        if 'top' in ax.spines.keys():
            ax.spines['top'].set_visible(False)



def dag_update_fig_fontsize(fig, new_font_size, **kwargs):
    '''dag_update_fig_fontsize
    Description:
        Update the font size of a figure
    Input:
        fig             matplotlib figure
        new_font_size   int/float             
    Return:
        None        
    '''
    fig_kids = fig.get_children() # Get the children of the figure, i.e., the axes
    for i_kid in fig_kids: # Loop through the children
        if isinstance(i_kid, mpl.axes.Axes): # If the child is an axes, update the font size of the axes
            dag_update_ax_fontsize(i_kid, new_font_size, **kwargs)
        elif isinstance(i_kid, mpl.text.Text): # If the child is a text, update the font size of the text
            i_kid.set_fontsize(new_font_size)            

def dag_update_ax_fontsize(ax, new_font_size, include=None, do_extra_search=True):
    '''dag_update_ax_fontsize
    Description:
        Update the font size of am axes
    Input:
        ax              matplotlib axes
        new_font_size   int/float
        *Optional*
        include         list of strings     What to update the font size of. 
                                            Options are: 'title', 'xlabel', 'ylabel', 'xticks','yticks'
        do_extra_search bool                Whether to search through the children of the axes, and update the font size of any text
    Return:
        None        
    '''
    if include is None: # If no include is specified, update all the text       
        include = ['title', 'xlabel', 'ylabel', 'xticks','yticks']
    if not isinstance(include, list): # If include is not a list, make it a list
        include = [include]
    incl_list = []
    for i in include: # Loop through the include list, and add the relevant text to the list
        if i=='title': 
            incl_list += [ax.title]
        elif i=='xlabel':
            incl_list += [ax.xaxis.label]
        elif i=='ylabel':
            incl_list += [ax.yaxis.label]
        elif i=='xticks':
            incl_list += ax.get_xticklabels()
        elif i=='yticks':
            incl_list += ax.get_yticklabels()
        elif i=='legend':
            incl_list += ax.get_legend().get_texts()

    for item in (incl_list): # Loop through the text, and update the font size
        item.set_fontsize(new_font_size)        
    if do_extra_search:
        for item in ax.get_children():
            if isinstance(item, mpl.legend.Legend):
                texts = item.get_texts()
                if not isinstance(texts, list):
                    texts = [texts]
                for i_txt in texts:
                    i_txt.set_fontsize(new_font_size)
            elif isinstance(item, mpl.text.Text):
                item.set_fontsize(new_font_size)                



def dag_update_ax_dotsize(ax, new_dot_size):
    '''dag_update_ax_dotsize
    Description:
        Update the dot size of an axes
    Input:
        ax              matplotlib axes
        new_dot_size    int/float
    Return:
        None
    '''
    for i_kid in ax.get_children(): 
        if isinstance(i_kid, mpl.collections.PathCollection):
            # if hasattr(i_kid, 'set_sizes')
            # print(type(i_kid))
            i_kid.set_sizes(np.array([new_dot_size]))
        elif isinstance(i_kid, mpl.lines.Line2D):
            i_kid.set_markersize(np.array([new_dot_size]))


def dag_update_fig_dotsize(fig, new_dot_size):
    '''dag_update_fig_dotsize
    Description:
        Update the dot size of a figure
    Input:
        fig            matplotlib figure
        new_dot_size    int/float
    Return:
        None
    '''    
    for i_kid in fig.get_children():
        # print(i_kid)
        if isinstance(i_kid, mpl.axes.Axes):
            dag_update_ax_dotsize(i_kid, new_dot_size)
        elif isinstance(i_kid, mpl.figure.SubFigure):
            dag_update_fig_dotsize(i_kid, new_dot_size)


def dag_return_ecc_pol_bin_mid_pts(ecc4bin, pol4bin, **kwargs):
    ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)            

    bin_mid_x = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1)) 
    bin_mid_y = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1)) 
    total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1) 
    for i_ecc in range(len(ecc_bounds)-1):
        for i_pol in range(len(pol_bounds)-1):
            ecc_lower = ecc_bounds[i_ecc]
            ecc_upper = ecc_bounds[i_ecc + 1]

            pol_lower = pol_bounds[i_pol]
            pol_upper = pol_bounds[i_pol + 1]            

            ecc_idx =(ecc4bin >= ecc_lower) & (ecc4bin <=ecc_upper)
            pol_idx =(pol4bin >= pol_lower) & (pol4bin <=pol_upper)        
            bin_idx = pol_idx & ecc_idx
            # Calculate midpoints
            ecc_mid = (ecc_lower + ecc_upper) / 2
            pol_mid = (pol_lower + pol_upper) / 2

            # Ensure the polar angle stays within the range [-π, π]
            pol_mid = (pol_mid + np.pi) % (2 * np.pi) - np.pi

            ecc_mid = (ecc_lower + ecc_upper) / 2                    
            pol_mid = (pol_lower + pol_upper) / 2
            mid_x, mid_y = dag_coord_convert(ecc_mid, pol_mid, 'pol2cart')
            if bin_idx.sum()==0:
                bin_mid_x[i_ecc, i_pol] = np.nan
                bin_mid_y[i_ecc, i_pol] = np.nan
            else:
                bin_mid_x[i_ecc, i_pol] = mid_x
                bin_mid_y[i_ecc, i_pol] = mid_y


    bin_mid_x = np.reshape(bin_mid_x, total_n_bins)
    bin_mid_y = np.reshape(bin_mid_y, total_n_bins)
    # REMOVE ANY NANS
    bin_mid_x = bin_mid_x[~np.isnan(bin_mid_x)]
    bin_mid_y = bin_mid_y[~np.isnan(bin_mid_y)]

    return bin_mid_x, bin_mid_y

def dag_return_ecc_pol_bin(params2bin, ecc4bin, pol4bin, bin_weight=None, **kwargs):
    '''dag_return_ecc_pol_bin
    Description:
        Bin parameters by eccentricity and polar angle
    Input:
        params2bin      list of np.ndarrays, to bin
        ecc4bin         eccentricity & polar angle used in binning
        pol4bin         ""
        ecc_bounds      how to split into bins
        pol_bounds      ""
        bin_weight      Something used to weight the binning (e.g., rsq)
    Returns:
        params_binned   list of np.ndarrays, binned by ecc and pol
    '''
    # TODO have option for converting X,Y...    
    min_per_bin = kwargs.get('min_per_bin', False)
    return_by_bin = kwargs.get('return_by_bin', False)
    ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)            
    n_params_gt_1 = False # If there is more than 1 parameter to bin
    if not isinstance(params2bin, list): # If there is only 1 parameter to bin, make it a list
        params2bin = [params2bin]
        n_params_gt_1 = True

    total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1) 
    params_binned = []
    params_binned_no_reshape = []
    for i_param in range(len(params2bin)): # Loop through the parameters to bin
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
                if min_per_bin is not False:
                    if bin_idx.sum()<min_per_bin:
                        # Not enough vx per bin
                        print('Not enough vx per bing')
                        bin_idx *= 0 # 
                
                if bin_weight is not None: # If there is a bin weight, use it
                    bin_mean[i_ecc, i_pol] = (params2bin[i_param][bin_idx] * bin_weight[bin_idx]).sum() / bin_weight[bin_idx].sum()
                else:
                    # bin_mean[i_ecc, i_pol] = np.mean(params2bin[i_param][bin_idx])
                    bin_mean[i_ecc, i_pol] = np.median(params2bin[i_param][bin_idx])

        if not return_by_bin:
            # reshape, remove the nans
            bin_mean = np.reshape(bin_mean, total_n_bins)
            # -> REMOVE ANY NANS
            bin_mean = bin_mean[~np.isnan(bin_mean)]
        
        params_binned.append(bin_mean)

    if n_params_gt_1:
        params_binned = params_binned[0] 
    return params_binned

def dag_visual_field_scatter(dot_x, dot_y, **kwargs):
    '''dag_visual_field_scatter
    Description:
        Plot a scatter of points on a visual field (e.g., size, rsquared etc)
        With the option to do various creative things. e.g., binning, color coding, etc
    Input:
        dot_x           np.ndarray      x coord of points
        dot_y           np.ndarray      y coord of points
        *Optional*
        ax              matplotlib axes
        do_binning      bool            Whether to bin the points
        fix_bin_xy      bool            Use the middle pt of the bins for xy? 
        bin_weight      np.ndarray      Weighted mean in each bin, not just the average
        ecc_bounds      np.ndarray      eccentricity bounds
        pol_bounds      np.ndarray      polar angle bounds
        dot_alpha       float           alpha of points
        dot_size        float           size of points
        dot_col         str             color of points
        dot_vmin        float           min value for color map
        dot_vmax        float           max value for color map
        dot_cmap        str             color map
        do_hex_bin      bool            Whether to do hex binning
    Return:
        ax              matplotlib axes
        cb              colorbar

    '''
    ax = kwargs.get('ax', plt.gca())
    do_binning = kwargs.get("do_binning", False)
    do_hex_bin = kwargs.get("do_hex_bin", False)
    bin_weight = kwargs.get("bin_weight", None)
    fix_bin_xy = kwargs.get("fix_bin_xy", False)
    
    # -> add option for dot size scaling... ( & alpha scaling) ??
    ecc_bounds = kwargs.get("ecc_bounds", np.linspace(0, 5, 7) )
    pol_bounds = kwargs.get("pol_bounds", np.linspace(-np.pi, np.pi, 13))            
    # Extra dot properties:
    
    dot_props = {
        'dot_alpha' : kwargs.get("dot_alpha", 0.5),
        'dot_size'  : kwargs.get("dot_size",200),
        'dot_col'   : kwargs.get("dot_col", 'k') ,  
        'dot_vmin'  : kwargs.get("dot_vmin", None),   
        'dot_vmax'  : kwargs.get("dot_vmax", None),   
        'dot_cmap'  : kwargs.get("dot_cmap", None),   
    }
    
    if (len(dot_props['dot_col'])==len(dot_x)) & (dot_props['dot_cmap']==None):
        dot_props['dot_cmap'] = 'viridis'
    if dot_props['dot_cmap'] != None:
        dot_props['dot_cmap'] = dag_get_cmap(dot_props['dot_cmap'])

    bin_dot_props = {}
    if do_binning:
        dot_ecc, dot_pol = dag_coord_convert(dot_x,dot_y,old2new="cart2pol")
        if fix_bin_xy:
            bin_x, bin_y = dag_return_ecc_pol_bin_mid_pts(
                ecc4bin=dot_ecc, 
                pol4bin=dot_pol, 
                ecc_bounds=ecc_bounds, 
                pol_bounds=pol_bounds,
            )
        else:
            bin_x, bin_y = dag_return_ecc_pol_bin(
                params2bin=[dot_x, dot_y], 
                ecc4bin=dot_ecc, 
                pol4bin=dot_pol, 
                bin_weight=bin_weight,
                ecc_bounds=ecc_bounds, 
                pol_bounds=pol_bounds
                )        
        for p in ['dot_col', 'dot_size', 'dot_alpha']:
            if not hasattr(dot_props[p], 'shape'):
                bin_dot_props[p] = dot_props[p]
            elif len(dot_props[p])!=len(dot_x):
                bin_dot_props[p] = dot_props[p]
            else:
                bin_dot_props[p] = dag_return_ecc_pol_bin(
                    params2bin=dot_props[p], 
                    ecc4bin=dot_ecc, 
                    pol4bin=dot_pol, 
                    bin_weight=bin_weight,
                    ecc_bounds=ecc_bounds, 
                    pol_bounds=pol_bounds,
                    )   

    else:
        bin_x = dot_x
        bin_y = dot_y
        for p in ['dot_col', 'dot_size', 'dot_alpha']:
            bin_dot_props[p] = dot_props[p]

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
    if not do_hex_bin:
        scat_col = ax.scatter(
            bin_x, 
            bin_y, 
            c       = bin_dot_props['dot_col'], 
            s       = bin_dot_props['dot_size'], 
            alpha   = bin_dot_props['dot_alpha'], 
            cmap    = dot_props['dot_cmap'], 
            vmin    = dot_props['dot_vmin'], 
            vmax    = dot_props['dot_vmax']
            )
        cb = None
    else:
        scat_col = ax.hexbin(
            bin_x, 
            bin_y, 
            C=bin_dot_props['dot_col'],
            gridsize=kwargs.get('gridsize', 50), 
            cmap=dot_props['dot_cmap'], 
            alpha=dot_props['dot_alpha'], 
            edgecolors='face',
            vmin=dot_props['dot_vmin'],
            vmax=dot_props['dot_vmax'],
            )
        cb = None
    if not isinstance(bin_dot_props['dot_col'], str):
        fig = plt.gcf()
        cb = fig.colorbar(scat_col, ax=ax)        
    dag_add_ecc_pol_lines(ax, ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)    
    dag_add_ax_basics(**kwargs)    
    return ax, cb

def dag_plot_bin_line(ax, X,Y, bin_using, **kwargs):    
    '''dag_plot_bin_line
    Description:
        Plot X vs Y binned by "bin_using" 

    Input:
        ax              matplotlib axes
        X               np.ndarray      x coord of points
        Y               np.ndarray      y coord of points
        bin_using       np.ndarray      What to bin by (often you will bin by X)
        *Optional*
        line_col        str             color of line
        line_label      str             label of line
        lw              float           line width
        n_bins          int             number of bins
        bins            np.ndarray      bin edges
        do_basics       bool            Whether to add basic features to the plot (i.e., run dag_add_ax_basics)
        xerr            bool            Whether to include x error bars
        do_bars         bool            Whether to include error bars
        do_shade        bool            Do shading instead?
        err_args        dict            Arguments for error bars
                    'upper_vals' -> values for upper line
                    'lower_vals' -> values for lower line
                    'method'    -> ... TODO (method for setting upper & lower values)
    Return:
        None

    '''
    # GET PARAMETERS....
    line_col = kwargs.get("line_col", "k")    
    line_label = kwargs.get("line_label", None)
    lw= kwargs.get("lw", 5)
    line_kwargs = kwargs.get('line_kwargs', {})
    n_bins = kwargs.get('n_bins', 10)
    bins = kwargs.get('bins', None)
    bin_X = kwargs.get('bin_X', True)
    if not isinstance(bins, (np.ndarray, list)):
        bins = n_bins    
    do_basics = kwargs.get('do_basics', False)
    min_per_bin = kwargs.get('min_per_bin', False) # min number of points before setting to NAN
    # xerr = kwargs.get("xerr", False)
    do_bars = kwargs.get("do_bars", True)
    do_shade = kwargs.get("do_shade", False)
    summary_type = kwargs.get("summary_type", 'mean')
    err_type = kwargs.get("err_type", 'std')
    if 'pc' in err_type:
        summary_type = 'median'
    # bloop
    # Binned X values
    # -> mean or median? 
    X_mid = binned_statistic(bin_using, X, bins=bins, statistic=summary_type)[0]
    # -> Or just use the midpoint    
    if not bin_X:        
        if isinstance(bins, int):
            bins = np.linspace(np.nanmin(X), np.nanmax(X), bins)
        X_mid = (bins[:-1] + bins[1:]) / 2

    # Now calculate the spreads    
    if 'pc' in err_type:
        # Percentile
        Y_mid = binned_statistic(bin_using, Y, bins=bins, statistic=summary_type)[0]                      
        pc_lower = float(err_type.split('-')[-1])
        pc_upper = 100 - pc_lower
        if pc_lower > pc_upper:
            # Oops - flip them
            pc_lower, pc_upper = pc_upper, pc_lower
        pcLOWER_lambda = kwargs.get('pcLOWER_lambda', lambda data: np.percentile(data, pc_lower, axis=0))
        pcUPPER_lambda = kwargs.get('pcUPPER_lambda', lambda data: np.percentile(data, pc_upper, axis=0))
        Y_lower_shade = binned_statistic(bin_using, Y, bins=bins, statistic=pcLOWER_lambda)[0]              
        Y_upper_shade = binned_statistic(bin_using, Y, bins=bins, statistic=pcUPPER_lambda)[0]              
        Y_lower_bar = Y_mid - Y_lower_shade
        Y_upper_bar = Y_upper_shade - Y_mid
    
    elif err_type=='std':
        # Standard deviation
        Y_mid = binned_statistic(bin_using, Y, bins=bins, statistic=summary_type)[0]
        Y_std = binned_statistic(bin_using, Y, bins=bins, statistic='std')[0]  #/ np.sqrt(bin_data['bin_X']['count'])
        Y_lower_bar = Y_std        
        Y_upper_bar = Y_std
        Y_lower_shade = Y_mid - Y_std
        Y_upper_shade = Y_mid + Y_std
    # replace any nans with zeros or ymid
    # Y_lower_bar[np.isnan(Y_lower_bar)] = 0
    # Y_upper_bar[np.isnan(Y_upper_bar)] = 0
    # Y_lower_shade[np.isnan(Y_lower_shade)] = Y_mid[np.isnan(Y_lower_shade)]
    # Y_upper_shade[np.isnan(Y_upper_shade)] = Y_mid[np.isnan(Y_upper_shade)]

    # Apply minimum per bin
    XY_count = binned_statistic(bin_using, X, bins=bins, statistic='count')[0]
    if min_per_bin is not False:        
        for i_bin,bin_count in enumerate(XY_count):
            if bin_count<min_per_bin:
                Y_mid[i_bin] = np.nan
                Y_lower_bar[i_bin] = np.nan
                Y_upper_bar[i_bin] = np.nan
                Y_lower_shade[i_bin] = np.nan
                Y_upper_shade[i_bin] = np.nan
    

    if do_bars:
        mask = ~np.isnan(Y_mid)
        ax.errorbar(
            X_mid[mask],
            Y_mid[mask],
            yerr=[Y_lower_bar[mask], Y_upper_bar[mask]],
            # xerr=X_std,
            color=line_col,
            label=line_label, 
            lw=lw,
            **line_kwargs
            )
        
    if do_shade:        
        ax.plot(
            X_mid,
            Y_mid,
            color=line_col,
            label=line_label,
            # alpha=0.5,
            lw=lw,
            **line_kwargs,
            )
        # bleep
        ax.fill_between(
            X_mid,
            Y_lower_shade,
            Y_upper_shade,
            alpha=0.5,
            color=line_col,            
            label='_',
            lw=0,
            edgecolor=line_col,        
            )       


    else:
        ax.plot(
            X_mid,
            Y_mid,
            color=line_col,
            label=line_label,
            lw=lw,
            **line_kwargs,
            )
    ax.legend()
    if do_basics:    
        dag_add_ax_basics(ax, **kwargs)


def dag_arrow_plot(ax, old_x, old_y, new_x, new_y, **kwargs):
    '''dag_arrow_plot
    Description:
        Plot arrows from old to new points. Includes the option to bin the points
        Also various fun things such as color coding the arrows by angle, color coding the points, etc...
    Input:
        ax              matplotlib axes
        old_x           np.ndarray      x coord of old points
        old_y           np.ndarray      y coord of old points
        new_x           np.ndarray      x coord of new points
        new_y           np.ndarray      y coord of new points
        *Optional*
        do_binning      bool            Whether to bin the points
        bin_weight      np.ndarray      Weighted mean in each bin, not just the average
        ecc_bounds      np.ndarray      eccentricity bounds
        pol_bounds      np.ndarray      polar angle bounds
        do_scatter      bool            Whether to include scatters of the voxel positions
        do_scatter_old  ""              Whether to include scatters of the old voxel positions
        do_scatter_new  ""              Whether to include scatters of the new voxel positions
        old_col         color           Gives color for old points
        new_col         color           Gives color for new points
        patch_col       color           Color for the patch 
        dot_alpha       float or array  Alpha for each point (not split by old and new)
        dot_size        float or array  Size for each point (not split by old and new)        
        do_arrows       bool            Whether to include arrows
        patch_col       str             Color of the patch
        dot_alpha       float           alpha of points
        dot_size        float           size of points
        arrow_col       str             Color for the arrows. Another option is "angle", where arrows will be coloured depending on there angle
        arrow_kwargs    dict            Another dict for arrow properties
            scale       Always 1, exact pt to pt
            width       Of shaft relative to plot
            headwidth   relative to shaft width
    Return:
        None
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
    kwargs['ecc_bounds'] = ecc_bounds    
    kwargs['pol_bounds'] = pol_bounds
    add_grid_lines = kwargs.get('add_grid_lines', True) 

    old_col = kwargs.get("old_col", 'k')
    new_col = kwargs.get("new_col", 'g')
    # -> for now only 1 alpha, size per dot
    dot_alpha = kwargs.get('dot_alpha', .5)
    dot_size = kwargs.get('dot_size', 500)

    # aperture_rad = kwargs.get("aperture_rad", None)
    # patch_col = kwargs.get("patch_col", "k")
    arrow_col = kwargs.get("arrow_col", 'b')
    arrow_cmap = kwargs.get("arrow_cmap", 'hsv')
    arrow_kwargs = {
        'scale'     : 1,                                    # ALWAYS 1 -> exact pt to exact pt 
        'width'     : kwargs.get('arrow_width', .01),       # of shaft (relative to plot )
        'headwidth' : kwargs.get('arrow_headwidth', 1.5),    # relative to width
    }    
    
    # arrow_kwargs = {
    #     'width'     : kwargs.get('arrow_width', .01),       # of shaft (relative to plot )
    #     'head_width' : kwargs.get('arrow_headwidth', .3),    # relative to width
    #     'length_includes_head' : True, 
    # }    

    if do_binning:
        # print("DOING BINNING") 
        min_vx_per_bin = kwargs.get('min_vx_per_bin', False)
        old_ecc, old_pol = dag_coord_convert(old_x, old_y,old2new="cart2pol")
        old_bin_x, old_bin_y, new_bin_x, new_bin_y = dag_return_ecc_pol_bin(
            params2bin=[old_x, old_y, new_x, new_y], 
            ecc4bin=old_ecc, 
            pol4bin=old_pol, 
            bin_weight=bin_weight,
            ecc_bounds=ecc_bounds, pol_bounds=pol_bounds,
            min_vx_per_bin=min_vx_per_bin,
            )
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
            q_cmap = dag_get_cmap(arrow_cmap)#mpl.cm.__dict__['hsv']
            q_norm = mpl.colors.Normalize()
            q_norm.vmin = -3.14
            q_norm.vmax = 3.14
            q_col = q_cmap(q_norm(angle))                
        else:
            q_col = arrow_col

        arrows = ax.quiver(
            old_bin_x, old_bin_y, dx, dy, scale_units='xy',
            angles='xy', alpha=dot_alpha,color=q_col,  **arrow_kwargs)

        # arrows = []
        # for i in range(len(old_bin_x)):
        #     arrows.append(ax.arrow(
        #         x=old_bin_x[i], y=old_bin_y[i], dx=dx[i], dy=dy[i], 
        #         alpha=dot_alpha,color=q_col,
        #         # scale_units='xy',
        #         # angles='xy',
        #         **arrow_kwargs
        #         ))
        
        # # For the colorbar
        # if isinstance(dot_col, np.ndarray):
        #     scat_col = ax.scatter(
        #         np.zeros_like(LE_x2plot), np.zeros_like(LE_x2plot), s=np.zeros_like(LE_x2plot), 
        #         c=dot_col, vmin=dot_vmin, vmax=dot_vmax, cmap=dot_cmap)
        #     fig = plt.gcf()
        #     cb = fig.colorbar(scat_col, ax=ax)        
        #     cb.set_label(kwargs['dot_col'])
    else:
        arrows = None
    if add_grid_lines:
        dag_add_ecc_pol_lines(ax, **kwargs)        
    dag_add_ax_basics(ax, **kwargs)    
    return arrows

def dag_arrow_coord_getter(old_x, old_y, new_x, new_y, **kwargs):
    '''dag_arrow_plot
    Description:
        Plot arrows from old to new points. Includes the option to bin the points
        Also various fun things such as color coding the arrows by angle, color coding the points, etc...
    Input:
        old_x           np.ndarray      x coord of old points
        old_y           np.ndarray      y coord of old points
        new_x           np.ndarray      x coord of new points
        new_y           np.ndarray      y coord of new points
        *Optional*
        do_binning      bool            Whether to bin the points
        bin_weight      np.ndarray      Weighted mean in each bin, not just the average
        ecc_bounds      np.ndarray      eccentricity bounds
        pol_bounds      np.ndarray      polar angle bounds
    Return:
        old_bin_x, old_bin_y, new_bin_x, new_bin_y, dx, dy
    '''
    # Get arguments related to plotting:
    do_binning = kwargs.pop("do_binning", True)
    # bin_weight = kwargs.get("bin_weight", None)
    # ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    # pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)    
    if do_binning:
        # print("DOING BINNING") 
        old_ecc, old_pol = dag_coord_convert(old_x, old_y,old2new="cart2pol")
        old_bin_x, old_bin_y, new_bin_x, new_bin_y = dag_return_ecc_pol_bin(
            params2bin=[old_x, old_y, new_x, new_y], 
            ecc4bin=old_ecc, 
            pol4bin=old_pol, 
            # bin_weight=bin_weight,
            # ecc_bounds=ecc_bounds, pol_bounds=pol_bounds,
            # min_vx_per_bin=min_vx_per_bin,
            **kwargs
            )
    else:
        old_bin_x, old_bin_y, new_bin_x, new_bin_y = old_x, old_y, new_x, new_y
    
    dx = new_bin_x - old_bin_x
    dy = new_bin_y - old_bin_y    
    return old_bin_x, old_bin_y, new_bin_x, new_bin_y, dx,dy
# def dag_2d_density(X,Y, ax=None, **kwargs):
#     '''
#     https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67
#     '''
#     if ax is None:
#         ax = plt.gca()

#     deltaX = (np.max(X) - np.min(X)) 

#     return



def dag_scatter(X,Y,ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    do_scatter = kwargs.get('do_scatter', True)
    do_line = kwargs.get('do_line', False)
    do_id_line = kwargs.pop('do_id_line', False)
    do_ow = kwargs.get('ow', False)    
    dot_col = kwargs.get('dot_col', None)
    dot_alpha = kwargs.get('dot_alpha', None)    
    dot_cmap = kwargs.get('dot_cmap', None)
    dot_size = kwargs.get('dot_size', None)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    dot_label=kwargs.get('dot_label')
    do_corr = kwargs.get('do_corr', False)
    do_colbar = kwargs.get('do_colbar', False)

    # hexbin, kde, joint_sns
    alt_plot = kwargs.get('alt_plot', False)
    alt_kwargs = kwargs.get('alt_kwargs', {})


    if do_scatter:        
        scat_col = ax.scatter(
            X,Y, c=dot_col, alpha=dot_alpha, label=dot_label, s=dot_size,
            cmap=dot_cmap, vmin=vmin, vmax=vmax) # added **kwargs
        # if dot_col is not None:
        if do_colbar:
            fig = plt.gcf()
            cb = fig.colorbar(scat_col, ax=ax)        
            cb.set_alpha(1)
            cb.draw_all()
    if do_line:
        dag_plot_bin_line(ax=ax, X=X,Y=Y, bin_using=X, **kwargs)
    if do_corr:
        corr_xy = dag_get_corr(X,Y)
        corr_str = f'corr={corr_xy:.3f}'
    else:
        corr_str = ''
    if not do_ow:
        corr_str = ax.get_title() + corr_str

    kwargs['title'] = kwargs.get('title', '') + corr_str
    
    if alt_plot == 'kde':
        sns.kdeplot(
            x=X,y=Y, color=dot_col,

            **alt_kwargs
            )
    elif alt_plot == 'joint_sns':
        sns.jointplot(
            x=X, y=Y, ax=ax,
            **alt_kwargs) 
    elif alt_plot == 'hexbin':
        ax.hexbin(
            X,Y,             
            **alt_kwargs            
            )        
    
    if do_id_line:
        xlim = kwargs.get('x_lim', ax.get_xlim())
        ylim = kwargs.get('y_lim', ax.get_ylim())
        min_v = np.min([xlim[0], ylim[0]])
        max_v = np.max([xlim[1], ylim[1]])
        ax.plot((min_v,max_v), (min_v,max_v), 'k')
        ax.set_xlim(min_v,max_v)
        ax.set_ylim(min_v,max_v)
        ax.set_box_aspect(1)
    dag_add_ax_basics(ax=ax, **kwargs)



def dag_multi_scatter(data_in, **kwargs):
    '''
    Many parameters to correlate do x,y... etc    
    '''
    do_dag_scatter = kwargs.get('dag_scatter', False)
    skip_hist = kwargs.get('skip_hist', False)
    truths = kwargs.get('truths', None)
    # Which labels for x and y?
    p_labels = kwargs.get('p_labels', None)
    transpose_subplots = kwargs.get('transpose_subplots', False)
    do_all_comb = False
    if isinstance(data_in, np.ndarray):
        print('numpy array: only n x n. Assuming each column is a parameter')
        n_pdim = data_in.shape[-1]
        n_px = n_pdim
        n_py = n_pdim
        if p_labels is None:
            p_labels = np.arange(n_pdim)    
            p_labels = [f'p{i}' for i in p_labels]
        px_labels = kwargs.get('px_labels', p_labels)
        py_labels = kwargs.get('py_labels', p_labels)    

        data_dict = {}
        for i,p in enumerate(p_labels):
            data_dict[p] = data_in[:,i]
    else:
        p_labels = kwargs.get('p_labels', list(data_in.keys()))
        px_labels = kwargs.get('px_labels', p_labels)
        py_labels = kwargs.get('py_labels', p_labels)    
        data_dict = data_in
        n_px = len(px_labels)
        n_py = len(py_labels)        

        if (px_labels!=py_labels):
            skip_hist = False
            do_all_comb = True

    fig = kwargs.get('fig', plt.figure())    
    rows = n_py
    cols = n_px
    if skip_hist:
        px_labels = list(px_labels)[1::]
    if skip_hist:
        cols -= 1

    if transpose_subplots: # Transpose the subplots
        rows_new = cols
        cols_new = rows
        rows = rows_new
        cols = cols_new

    plot_i = 1
    ax_list = {}

    for i1,y_param in enumerate(py_labels):
        ax_list[i1] = {}
        for i2,x_param in enumerate(px_labels):
            ax = fig.add_subplot(rows,cols,plot_i)
            if transpose_subplots:
                i1 = i1 % n_px
                i2 = i2 % n_py
                
            if (i1>i2) and (not do_all_comb):
                ax.axis('off')
            else:
                ax.set_xlabel(x_param)
                # if (i1==i2) and (not skip_hist):
                #     ax.hist(data_dict[x_param])
                if (x_param==y_param) and (not skip_hist):
                    ax.hist(data_dict[x_param])
                    if truths is not None:
                        # Add vline
                        ax.vlines(truths[x_param], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1])
                else:
                    if not do_dag_scatter:
                        ax.set_ylabel(y_param)
                        ax.scatter(
                            data_dict[x_param],
                            data_dict[y_param],
                        )        
                        ax.set_title(
                            f'corr={np.corrcoef(data_dict[x_param],data_dict[y_param])[0,1]:.3f}')
                    else:
                        dag_scatter(
                            X= data_dict[x_param],
                            Y= data_dict[y_param],
                            ax=ax,
                            **kwargs
                        )           
                        ax.set_xlabel(x_param)
                        ax.set_ylabel(y_param)
                    if truths is not None:
                        ax.vlines(truths[x_param], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1])
                        ax.hlines(truths[y_param], xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1])
            plot_i += 1
        fig.set_tight_layout('tight')
    return fig, ax_list


def edit_pair_plot(axes, **kwargs):
    """
    Adds vertical and horizontal lines to each subplot in a Seaborn pairplot.

    Annoyingly the x and y axis aren't consistently named in pairplot...

    Parameters:
    - pairplot (sns.PairGrid): The Seaborn pairplot to modify.
    - lines_dict (dict): Dictionary where keys are column names in the dataset, 
                         and values are the x or y values where lines should be drawn.
    """
    lines_dict = kwargs.pop('lines_dict', None)    
    lim_dict = kwargs.pop('lim_dict', None)
    n_r, n_c = axes.shape
    x_labels = [['']*n_c for _ in range(n_r)]
    y_labels = [['']*n_c for _ in range(n_r)]
    # Sometimes each row has a label...    
    for iR,axR in enumerate(axes):
        for iC,ax in enumerate(axR):
            if ax is None:
                continue
            x_labels[iR][iC] = axes[iR,iC].get_xlabel()
            y_labels[iR][iC] = axes[iR,iC].get_ylabel()

    # Now lets consolidate
    x_by_col = []
    for iC in range(n_c):
        v = ''
        i = 0 
        while v == '':
            v = x_labels[i][iC]
            i += 1
        x_by_col.append(v)

    y_by_row = []
    for iR in range(n_r):
        v = ''
        i = 0 
        while v == '':
            v = y_labels[iR][i]
            i += 1
        y_by_row.append(v)
    if lines_dict is not None:
        label_list = list(lines_dict.keys())
        for iR,vR in enumerate(y_by_row):
            for iC,vC in enumerate(x_by_col):
                if axes[iR,iC] is None:
                    continue

                x_label = axes[iR,iC].get_xlabel()
                y_label = axes[iR,iC].get_ylabel()
                if (x_label in label_list) & (y_label in label_list):
                    # Both in there? then it must be a scatter plot 
                    # Plot them both
                    axes[iR,iC].axvline(x=lines_dict[x_label], **kwargs)
                    axes[iR,iC].axhline(y=lines_dict[y_label], **kwargs)
                else:
                    # Missing labels... must be a histogram
                    if vR in label_list:
                        axes[iR,iC].axvline(x=lines_dict[vR], **kwargs)
    if lim_dict is not None:
        label_list = list(lim_dict.keys())
        for iR,vR in enumerate(y_by_row):
            for iC,vC in enumerate(x_by_col):
                if axes[iR,iC] is None:
                    continue

                x_label = axes[iR,iC].get_xlabel()
                y_label = axes[iR,iC].get_ylabel()
                if (x_label in label_list) & (y_label in label_list):
                    # Both in there? then it must be a scatter plot 
                    # Plot them both
                    axes[iR,iC].set_xlim(lim_dict[x_label])
                    axes[iR,iC].set_ylim(lim_dict[y_label])
                else:
                    # Missing labels... must be a histogram
                    if vR in label_list:
                        axes[iR,iC].set_xlim(lim_dict[vR])

    return 


# VIOLIN STUFF
def dag_full_violin(pd2plot):
    sns.set_style("whitegrid")
    sns.violinplot(                                                                        
        data=pd2plot, width=1, linewidth=0, 
        inner=None,saturation=0.5,
        )                                     
    sns.boxplot(
        data=pd2plot, showfliers = False, width=.5, saturation=0.5,
        )
    sns.pointplot(
        data=pd2plot, estimator=np.median,
        )   

def dag_dict_key_to_pd_label(dict_in):
    new_pd = {}
    new_pd['key'] = []
    new_pd['val'] = []
    for key in dict_in.keys():    
        new_pd['key'] += [str(key)] * len(dict_in[key])
        new_pd['val'] += list(dict_in[key])
    new_pd = pd.DataFrame(new_pd)    
    return new_pd


def dag_half_violin(pd2plot, match_id, split_id, **kwargs):
    new_pd = {}
    new_pd['match_id'] = []
    new_pd['split_id'] = []
    new_pd['val'] = []
    for p in pd2plot.keys():
        pname = p.split('-')[0]
        #
        pmatch = []
        psplit = []
        if match_id[0] in pname:
            pmatch = match_id[0]
        elif match_id[1] in pname:
            pmatch = match_id[1]
        
        if split_id[0] in pname:
            psplit = match_id[0]
        elif split_id[1] in pname:
            psplit = split_id[1]

        new_pd['match_id'] += pmatch * len(pd2plot[p])
        new_pd['split_id'] += psplit * len(pd2plot[p])
        new_pd['val'] += list(pd2plot[p])
    new_pd = pd.DataFrame(new_pd)    
    sns.set_style("whitegrid")
    sns.violinplot(                                                                        
        data=new_pd, width=1, linewidth=0, inner=None,saturation=0.5,
        x='match_id', y='val', hue='split_id', split=True
        )                                     
    sns.boxplot(
        data=new_pd, showfliers = False, width=.5, saturation=0.5,
        x='match_id', y='val', hue='split_id', 
        )
    sns.pointplot(
        data=new_pd, x='match_id', y='val', hue='split_id',
        estimator=np.median,
        )    


def dag_add_axis_to_xtick(fig, ax, dx_axs=1, **kwargs):
    '''add_axis_to_xtick
    Inputs:
        fig, ax: matplotlib figure and axis
        dx_axs: size of the new axis in figure units
    Returns:
        xtick_out: xtick values that were used
        xticks_axs: list of new axes

    Example:
        fig, ax = plt.subplots()
        ax.plot(np.random.randn(100))
        # Specify how many axes you want by chaning the number of ticks
        ax.set_xticks([0, 25, 50, 75, 100])
        xtick_out, xticks_axs = add_axis_to_xtick(fig, ax, dx_axs=1)
        # Now you can add whatever you want to the new axes

    '''
    plt.draw() # Make sure the figure is drawn first

    inv = fig.transSubfigure.inverted() # Invert the figure transform
    if isinstance(fig, mpl.figure.SubFigure): # Check if the figure is a subfigure
        inv = fig.transSubfigure.inverted()   # Invert the subfigure transform
    else:                                     # If not, use the figure transform
        inv = fig.transFigure.inverted()      # Invert the figure transform
    # Get the x tick values
    xticks = ax.get_xticks()                             
    ymin, _ = kwargs.pop('ymin', ax.get_ylim())          # Get the y limits
    move_y = kwargs.pop('move_y', 0)
    xmin, xmax = ax.get_xlim()                           # Get the x limits
    pix_coord = ax.transData.transform([(xtick, ymin) for xtick in xticks])     # Transform the x tick values to pixel coordinates
    fig_coord = inv.transform(pix_coord)                                        # Transform the pixel coordinates to figure coordinates
    
    dx_fig = np.abs(fig_coord[0,0] - fig_coord[1,0])  * dx_axs                  # Calculate the size of the new axes in figure units

    xtick_out = []      
    xticks_axs = []         
    for i_tick in range(len(xticks)):  # Loop over the x ticks          
        if (xticks[i_tick]<xmin) or (xticks[i_tick]>xmax):      # If the x tick is outside the x limits, skip it
            continue
        new_ax_pos = [
            fig_coord[i_tick,0]-dx_fig/2,
            fig_coord[i_tick,1]-(dx_fig*1) + move_y, #  - i_tick/10, # option to stagger the axes with i_tick
            dx_fig,
            dx_fig
            ]                   # Calculate the position of the new axis
                
        nax = fig.add_axes(new_ax_pos)
        nax.set_xticks([])
        nax.set_yticks([])
        nax.set_aspect('equal')
        # nax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        xticks_axs.append(nax)
        xtick_out.append(xticks[i_tick])
    return xtick_out, xticks_axs

def dag_add_dm_to_x(dm, xtick_out, xtick_axs, xtick_axs_idx=None, **kwargs):
    '''add_dm_to_x
    Description:
        Adds a design matrix to a set of axes
    Inputs:
        dm   np.ndarray design matrix
        xtick_out: xtick values that were used
        xtick_axs: list of new axes
    Returns:
        None
    '''
    kwargs['cmap'] = kwargs.get('cmap', 'Greys')
    kwargs['alpha'] = kwargs.get('alpha', 1)
    kwargs['vmin'] = kwargs.get('vmin', 0)
    kwargs['vmax'] = kwargs.get('vmax', 1)
    kwargs['extent'] = kwargs.get('extent', [-5,5,-5,5])
    TR_in_s = kwargs.pop('TR_in_s', 1)
    do_time = kwargs.pop('do_time', True)
    
    max_t = dm.shape[-1]
    for i,xtick in enumerate(xtick_out): 
        if xtick_axs_idx is None:
            dm_idx = int(xtick/TR_in_s)
        else:
            dm_idx = xtick_axs_idx[i]
        if dm_idx>max_t:
            xtick_axs[i].axis('off')
        elif dm_idx==max_t:
            xtick_axs[i].imshow(
                dm[:,:,dm_idx-1], 
                **kwargs
                )
            if do_time:
                xtick_axs[i].set_xlabel(f'{int(xtick)}')
        else:
            xtick_axs[i].imshow(
                dm[:,:,dm_idx], 
                **kwargs

                )
            if do_time:
                xtick_axs[i].set_xlabel(f'{xtick:.0f}')            
        
        # xtick_axs[i].patch.set_alpha(0.5)

def dag_add_dm_to_ts(fig, ax, dm, dx_axs=1, **kwargs):
    '''Add dm to time series
    '''
    move_y = kwargs.pop('move_y', 0)
    xtick_out, ax_out = dag_add_axis_to_xtick(fig, ax, dx_axs,move_y=move_y, **kwargs)
    dag_add_dm_to_x(
        dm=dm, 
        xtick_out=xtick_out, 
        xtick_axs=ax_out, 
        xtick_axs_idx=None, 
        **kwargs
        )

def dag_change_fig_item_col(fig_item, old_col, new_col, depth=0):
    '''
    Cycle recursively through all items in a figure and change the color of anything that matches old_col to new_col
    '''
    if depth > 100:
        print('Max depth reached')
        return
    if fig_item is []:
        return        

    if isinstance(fig_item, list):
        for item in fig_item:
            dag_change_fig_item_col(item, old_col, new_col, depth=depth+1)
    else:
        if hasattr(fig_item, 'get_color'):
            # print(depth)
            # print(fig_item)
            if fig_item.get_color() == old_col:
                print(fig_item.get_color())
                fig_item.set_color(new_col)
        if hasattr(fig_item, 'get_children'):
            dag_change_fig_item_col(fig_item.get_children(), old_col, new_col, depth=depth+1)



def dag_get_row_col(plot_index, n_cols=None, n_rows=None, dir='col', start_idx=0):
    '''
    Get row and col for a subplot index
    If dir is 'row', then row is the first index to change
    If dir is 'col', then col is the first index to change
    '''
    if dir == 'col':                
        row = (plot_index // n_cols ) + start_idx
        col = (plot_index % n_cols) + start_idx
    elif dir == 'row':
        row = (plot_index % n_rows) + start_idx
        col = (plot_index // n_rows) + start_idx
    return row, col



def dag_set_all_fig_item_attributes(fig_item, set_attribute, new_value, depth=0):
    '''
    Cycle recursively through all items in a figure and change the color of anything that matches old_col to new_col
    '''
    if depth > 100:
        print('Max depth reached')
        return
    if fig_item is []:
        return        

    if isinstance(fig_item, list):
        for item in fig_item:
            dag_set_all_fig_item_attributes(item, set_attribute, new_value, depth=depth+1)
    else:
        if hasattr(fig_item, set_attribute):
            fig_item.__getattribute__(set_attribute)(new_value)
        if hasattr(fig_item, 'get_children'):
            dag_set_all_fig_item_attributes(fig_item.get_children(), set_attribute, new_value, depth=depth+1)


def dag_add_square_axis(main_obj, width_ratio, position_ratio):
    """
    Add a square axis inside the given axis.

    Parameters:
    - main_obj: The main object to which the square axis will be added.
        If figure add subplot(111)
    - width_ratio: The width of the square axis relative to the main axis.
    - position_ratio: The position of the square axis relative to the main axis.
    """
    plt.draw() # Make sure the figure is drawn first
    if isinstance(main_obj,plt.Figure):
        ax = main_obj.add_subplot(111)
    elif isinstance(main_obj, plt.Axes):
        ax = main_obj
    else:
        raise ValueError("Must be a figure or an axes")

    # Get the position and size of the main axis
    main_position = ax.get_position(original=True)
    main_width = main_position.width
    main_height = main_position.height

    # Calculate the width and height of the square axis
    square_width = main_width * width_ratio
    square_height = main_height * width_ratio

    # Calculate the position of the square axis
    square_x = main_position.x0 + main_width * position_ratio[0] 
    square_y = main_position.y0 + main_height * position_ratio[1]

    # Add the square axis
    square_ax = plt.axes([square_x, square_y, square_width, square_height])
    if isinstance(main_obj,plt.Figure):
        # remove the extra subplot
        ax.remove()
    return square_ax
    
def dag_add_compass(main_ax, width_ratio=0.5, position_ratio=[1,1], **kwargs):    
    # Set everything up
    pol_type = kwargs.get("pol_type", "radians")
    n_pol    = kwargs.get("n_pol", 9)     
    pol_list = kwargs.get("pol_list", None)   
    x_axis_only = kwargs.get("x_axis_only", False)
    wheel_only = kwargs.get("wheel_only", False)
    if pol_list is None:
        pol_list = np.linspace(-np.pi, np.pi, n_pol)
    line_col = kwargs.get("line_col", 'k' )
    incl_ticks = kwargs.get("incl_ticks", True)
    aperture_col = kwargs.get('aperture_col', 'k')
    cmap = kwargs.get('cmap', 'hsv')
    pie_alpha = kwargs.get("pie_alpha", 1)
    # X axis
    # Add a scatter color map...
    n_segments = 45
    ecc = np.ones(n_segments)-0.2
    pol = np.linspace(-np.pi, np.pi, n_segments)
    xs,ys = dag_coord_convert(ecc,pol, 'pol2cart')
    # Add to main axis...
    if not wheel_only:
        y_lim = main_ax.get_ylim()
        main_ax.scatter(
            pol, 
            np.ones_like(pol) * y_lim[0],
            c=pol, cmap=cmap,s=250,
        )
    if x_axis_only:
        return

    # Add the axis    
    ax = dag_add_square_axis(main_ax, width_ratio, position_ratio)
    ax.set_ylim(-1,1)
    ax.set_xlim(-1,1)
    # **** ADD THE LINES ****
    ax.set_xticks([])    
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)        

    # Add the lines
    n_polar_lines = len(pol_list)
    for i_pol in range(n_polar_lines): # Loop through polar lines
        i_pol_val = pol_list[i_pol] # Get the polar angle
        outer_x = np.cos(i_pol_val) # Get the x,y coords of the outer line
        outer_y = np.sin(i_pol_val) 
        outer_x_txt = outer_x*1.1 # Get the x,y coords of the text
        outer_y_txt = outer_y*1.1        
        if pol_type=="radians":
            outer_txt = f"{i_pol_val:.2f}" # Get the text
        elif pol_type=="degrees":
            outer_txt = f"{180*i_pol_val/np.pi:.0f}\N{DEGREE SIGN}" # Get the text
        # Don't show 360, as this goes over the top of 0 degrees and is ugly...
        if ('360' in outer_txt) or ('-3.14' in outer_txt):
            continue
        else:
            ax.plot((0, outer_x), (0, outer_y), color=line_col, alpha=0.3)
            if incl_ticks:
                ax.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')
    
    # Add the circle
    grid_line = patches.Circle((0, 0), 1, color=line_col, alpha=0.3, fill=0)    
    ax.add_patch(grid_line)                        

    # Add pie chart colormap
    # Generate data for the pie chart
    values = np.ones(n_segments)
    cwheel_colors = dag_get_col_vals(pol, vmin=-np.pi, vmax=np.pi, cmap=cmap)
    # Plot the pie chart
    ax.pie(
        values, 
        colors=cwheel_colors, 
        startangle=180, 
        counterclock=True, 
        radius=1.0, 
        wedgeprops=dict(width=1, fill=True, alpha=pie_alpha)
        )
    ax.scatter(x=xs,y=ys,c=pol,cmap=cmap, s=250)
    # Also scatter dots, for sanity check on pie...


def dag_box_around_ax_list(fig, ax_list, **kwargs):
    '''
    Add a rectangle around a set of axes    

    '''    
    pad     = kwargs.pop('pad', 0)
    padX    = kwargs.pop('padX', 0) + pad 
    padY    = kwargs.pop('padY', 0) + pad
    padX0   =-kwargs.pop('padX0', 0) - padX
    padX1   = kwargs.pop('padX1', 0)  + padX
    padY0   =-kwargs.pop('padY0', 0) - padY
    padY1   = kwargs.pop('padY1', 0)  + padY
    
    # Calculate the coordinates of the box
    x0 = min(ax.get_position().x0 for ax in ax_list) + padX0
    y0 = min(ax.get_position().y0 for ax in ax_list) + padY0
    x1 = max(ax.get_position().x1 for ax in ax_list) + padX1
    y1 = max(ax.get_position().y1 for ax in ax_list) + padY1

    kwargs['linewidth'] = kwargs.get('linewidth', 2)
    kwargs['edgecolor'] = kwargs.get('edgecolor', 'r')
    kwargs['facecolor'] = kwargs.get('facecolor', 'none')
    # Create a rectangle patch
    rect = mpl.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, 
        **kwargs,
        )
    fig.add_artist(rect)


def dag_shaded_line(line_data, xdata, **kwargs):
    """    

    Parameters:
    - line_data
    - ax (matplotlib.axes._subplots.AxesSubplot): The subplot on which to create the plot.
    - line_col (str, optional): Color of the CSF curve. Default is 'g'.
    - lw (float): width of csf plot
    - line_alpha: alpha of line
    - shade_alpha: alpha of shade
    - error_version (str, optional): Type of error to be used ('pc-5', 'iqr', 'bound', 'std', 'ste').
    - error_bar (str, optional): Type of error representation ('shade', 'none'). Default is 'shade'.
    Returns:
    None
    """ 
    # Kwargs
    ax = kwargs.get('ax', plt.gca())
    line_col         = kwargs.get('line_col', None)
    lw              = kwargs.get('lw', 1)
    line_alpha      = kwargs.get('line_alpha', 1)
    line_label      = kwargs.get('line_label', '_')
    shade_alpha     = kwargs.get('shade_alpha', 0.5)
    error_version   = kwargs.get('error_version', 'iqr')
    error_bar       = kwargs.get('error_bar', 'shade')
    line_kwargs     = kwargs.get('line_kwargs', {})
    shade_kwargs     = kwargs.get('shade_kwargs', {})
    # 
    m_line = np.median(line_data, axis=1)

    # Color shading for error
    if 'pc' in error_version:
        pc_lower = float(error_version.split('-')[-1])
        pc_upper = 100 - pc_lower
        lower_line = np.percentile(line_data, pc_lower, axis=1) 
        upper_line = np.percentile(line_data, pc_upper, axis=1) 
    elif 'iqr' in error_version:
        lower_line = np.percentile(line_data, 25, axis=1) 
        upper_line = np.percentile(line_data, 75, axis=1) 
    elif 'bound' in error_version:
        lower_line = np.min(line_data, axis=1)
        upper_line = np.max(line_data, axis=1)
    elif 'std' in error_version:
        line_std = np.nanstd(line_data, axis=1)
        lower_line = m_line - line_std
        upper_line = m_line + line_std
    elif 'ste' in error_version:
        line_ste = np.nanstd(line_data, axis=1) / np.sqrt(line_data.shape[1])
        lower_line = m_line - line_ste
        upper_line = m_line + line_ste

    ax.plot(
        xdata, 
        m_line, 
        alpha=line_alpha,
        color=line_col, 
        lw=lw,
        label=line_label,
        **line_kwargs,
        )
    # print(shade_kwargs)
    if error_bar=='shade':
        ax.fill_between(
            xdata, 
            lower_line,
            upper_line,
            alpha=shade_alpha,
            color=line_col,
            label='_',
            edgecolor=line_col,
            lw=0,
            **shade_kwargs,                    
            )



def dag_add_all_subfig_labels(axs, **kwargs):
    i_row, i_col = axs.shape    
    num_labels = (i_row+1) * (i_col+1) 
    labels = []
    if num_labels<=26:
        for i in range(num_labels):
            labels.append(
                string.ascii_uppercase[i]
            )
    else:
        for iR in range(i_row):
            for iC in range(i_col):
                labels.append(
                    string.ascii_uppercase[iR] + \
                    string.ascii_uppercase[iC]
                )   

    for iR in range(i_row):
        for iC in range(i_col):
            label = labels[iR * i_col + iC]
            dag_add_subfig_labels(
                ax=axs[iR,iC],
                label=label,
                **kwargs
            )

def dag_add_subfig_labels(ax, label, **kwargs):
    x=kwargs.pop('x', -0.15)
    y=kwargs.pop('y',  1.1)
    kwargs['fontsize'] = kwargs.get('fontsize', 12)
    kwargs['va'] = kwargs.get('va', 'top')
    kwargs['ha'] = kwargs.get('ha', 'right')
    ax.text(
        x=x,y=y,
        s=label, 
        transform=ax.transAxes,
        **kwargs
    )

def dag_sub_categories_xvalues_SIMPLE(nL1, nL2, nL3=1, **kwargs):
    '''
    Parameters:
        nL1 (int): Total number of level 1.
        nL2 (int): Total number of level 2.
        nL3 (int): Total number of level 3.
        
        **kwargs: Additional keyword arguments:
            L1_distance (float, optional): Distance between subgroups. 
            L2_distance (float, optional): Distance within each subgroup
            L3_distance (float, optional): Distance within each sub-subgroup
        
    Returns:
        list: List of x coordinates for subcategories.
    '''
    
    x_values = []  # List to store x-coordinates
    
    # Default values for optional parameters
    L1_distance = kwargs.get('L1_distance', 1)
    L2_prop = kwargs.get('L2_prop', 1.5)
    L3_prop = kwargs.get('L3_prop', 1.5)
    L2_distance = kwargs.get(
        'L2_distance',
        L1_distance / (nL2*L2_prop)
    )
    L3_distance = kwargs.get(
        'L3_distance',
        L2_distance / (nL3*L3_prop)
    )
    
    # Iterate over each category
    for i1 in range(nL1):
        for i2 in range(nL2):
            for i3 in range(nL3):
                x = L1_distance * i1 + L2_distance * i2 + L3_distance * i3
                x_values.append(x)
    
    return x_values    
    


def dag_sub_categories_xvalues(i_sub, n_category, n_sub, **kwargs):
    '''
    Returns the x coordinates for subcategories within a category.
    
    Parameters:
        i_sub (int): Index of the current subcategory.
        n_category (int): Total number of categories.
        n_sub (int): Total number of subcategories.
        **kwargs: Additional keyword arguments:
            i_sub_sub (int, optional): Index of the current sub-subcategory. Default is None.
            n_sub_sub (int, optional): Total number of sub-subcategories. Default is None.
            group_distance (float, optional): Distance between subgroups. 
            sub_distance (float, optional): Distance within each subgroup
            sub_sub_distance (float, optional): Distance within each sub-subgroup
        
    Returns:
        list: List of x coordinates for subcategories.
    '''
    
    x_values = []  # List to store x-coordinates
    
    # Extracting kwargs
    i_sub_sub = kwargs.get('i_sub_sub', None)
    n_sub_sub = kwargs.get('n_sub_sub', 1) # None
    
    # Default values for optional parameters
    group_distance = kwargs.get('group_distance', 1)
    sub_prop = kwargs.get('sub_prop', 1.5)
    sub_distance = kwargs.get(
        'sub_distance',
        group_distance / (n_sub*sub_prop)
    )
    sub_sub_prop = kwargs.get('sub_sub_prop', 1.5)
    sub_sub_distance = kwargs.get(
        'sub_sub_distance',
        sub_distance / (n_sub_sub*sub_sub_prop)
        )

    # Iterate over each category
    for iC in range(n_category):
        # Calculate the offset for the current subgroup within the category
        subgroup_offset = sub_distance * (i_sub - (n_sub - 1) / 2)
        
        # Calculate the x-coordinate for the current subcategory,
        # taking into account the subgroup distance and offset
        x = group_distance * iC + subgroup_offset
        
        # If there are sub-subcategories
        if i_sub_sub is not None:
            # Calculate the offset for the current sub-subcategory within the subgroup
            sub_subgroup_offset = sub_sub_distance * (i_sub_sub - (n_sub_sub - 1) / 2)
            x += sub_subgroup_offset

        # Append the x-coordinate for the subcategory
        x_values.append(x)
    
    return x_values    


def dag_draw_significance_bar(ax, text, i_sub1, i_sub2, y_value=None, **kwargs):
    '''
    Draws a significance bar connecting two subgroups within different categories on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to draw the significance bar on.
        text (str): The text to display near the significance bar.
        i_sub1 (dict): Dictionary specifying the indices for the first subgroup.
            Example: {'i_category': 1, 'i_sub': 0, 'i_sub_sub': 0} for category 1, subgroup 0, sub-subgroup 0.
        i_sub2 (dict): Dictionary specifying the indices for the second subgroup.
            Example: {'i_category': 1, 'i_sub': 1, 'i_sub_sub': 0} for category 1, subgroup 1, sub-subgroup 0.
        y_value (float, optional): The y-coordinate where the significance bar should be drawn. If not provided, it's set to 90% of the upper limit of the y-axis.
        **kwargs: Additional keyword arguments for customizing the bar and text appearance.
            bar_color (str): Color of the significance bar (Default: 'black').
            bar_width (float): Width of the significance bar (Default: 0.5).
            text_color (str): Color of the text (Default: 'black').
            text_size (float): Size of the text (Default: 12).
            extend_by (float): Extend bars either side by... 

    Returns:
        None
    '''
    if y_value is None:
        y_value = ax.get_ylim()[-1] * .9
    bar_color = kwargs.get('bar_color', 'black')
    do_tails = kwargs.get('do_tails', True)
    tails_depth = kwargs.get('tails_depth', None)    

    text_color = kwargs.get('text_color', 'black')
    text_size = kwargs.get('text_size', 12)
    extend_by = kwargs.get('extend_by', 0) 
    lw = kwargs.get('lw', 1)
    # Get x-coordinates for the two subgroups
    x_values1 = dag_sub_categories_xvalues(**i_sub1, **kwargs)
    x1 = x_values1[i_sub1['i_category']]
    x_values2 = dag_sub_categories_xvalues(**i_sub2, **kwargs)
    x2 = x_values2[i_sub2['i_category']]
    xmid = (x1+x2)/2     
    bar_width = np.abs(x1 - x2)+extend_by
    x1 = xmid-(bar_width/2)
    x2 = xmid+(bar_width/2)
    # Draw the significance bar
    ax.plot([x1, x2], [y_value, y_value], color=bar_color, lw=lw)
    if do_tails:
        if tails_depth is None:
            tails_depth = [bar_width * .2]
        elif not isinstance(tails_depth, list):
            tails_depth = [tails_depth]

        if len(tails_depth)==1:
            tails_depth += tails_depth
        ax.plot([x1, x1], [y_value, y_value-tails_depth[0]], color=bar_color, lw=lw)
        ax.plot([x2, x2], [y_value, y_value-tails_depth[1]], color=bar_color, lw=lw)


    # Add text near the significance bar
    ax.text(xmid, y_value, text, color=text_color, size=text_size, ha='center', va='bottom')


def dag_group_and_individual_3dict(ax, mdict, **kwargs):
    y_upper = kwargs.pop('y_upper', None)
    y_lower = kwargs.pop('y_lower', None)
    l2_kwargs = kwargs.pop('l2_kwargs', {})

    # Ok lets loop around and call 2 dict...
    nl2 = 0
    xlim0 = np.inf
    xlim1 = -np.inf
    x_standard = []
    x_keys = []
    
    for i,k in enumerate(mdict.keys()):
            
        this_tick_out = dag_group_and_individual_2dict(
                ax=ax,
                mdict=mdict[k],
                y_upper=y_upper[k],
                y_lower=y_lower[k],
                x_offset=nl2,
                return_ticks=True,
                **l2_kwargs.copy()
                )
        if (i==0) & ('x_cols' in l2_kwargs.keys()):
            
            line4leg = []
            label4leg = []
            
            for lab,lin in l2_kwargs['x_cols'].items():
                line4leg.append(plt.Line2D([0], [0], color=lin, lw=2))
                label4leg.append(lab)
            plt.legend(line4leg, label4leg)

            # Add legend...

        # Create object of same color as the line
        # bleep
        nl2 += len(mdict[k].keys())
        xlim0 = min(xlim0, this_tick_out['xlim0'])
        xlim1 = max(xlim1, this_tick_out['xlim1'])
        x_standard.append(this_tick_out['x_standard'])
        x_keys.append(this_tick_out['x_keys'])
    ax.set_xlim(xlim0, xlim1)
    x_standard = [np.mean(x) for x in x_standard]
    ax.set_xticks(x_standard)
    ax.set_xticklabels(mdict.keys())
    # red_line = plt.Line2D([0], [0], color='red', lw=2)  # Create a red line object
    # plt.legend([red_line, 'Y=0 Line', 'Sine Wave'], ['Y=0 Line', 'Sine Wave'])
    # plt.legend()
    # ax.set_xticklabels(x_keys)




def dag_group_and_individual_2dict(ax, mdict, **kwargs):
    '''
    Plot bars for overall mean/median
    Points for individuals s + error 
    mdict: dict
        dictionary with the following structure:
        mdict[subject][x] = y        
    '''
    do_bar = kwargs.get('do_bar', True)
    do_points = kwargs.get('do_points', True)
    return_ticks = kwargs.get('return_ticks', False)
    bar_width = kwargs.get('bar_width', 0.5)
    jitter_width = kwargs.get('jitter_width', 0.5)
    err_kwargs = kwargs.get('err_kwargs', 
        dict(linestyle='None', marker='o', markersize=5)
        )
    bar_kwargs = kwargs.get('bar_kwargs', 
        dict(alpha=0.5)
        )
    m_method = kwargs.get('m_method', np.nanmean)
    y_upper = kwargs.get('y_upper', None)
    y_lower = kwargs.get('y_lower', None)
    s_keys = kwargs.get('s_keys', list(mdict.keys()))
    s_cols = kwargs.get('s_cols', None)
    do_jitter = kwargs.get('do_jitter', True)
    keys_split = kwargs.pop('keys_split', None) # Split into groupings
    # Of keys in the dictionary, which to plot
    keys_to_plot = kwargs.get('keys_to_plot', None)
    # What keys are there? if not specified, we can find them from the dictionary
    x_keys = kwargs.get('x_keys', keys_to_plot)
    if x_keys is None:
        x_keys = []
        for s in s_keys:
            for x in mdict[s].keys():
                if x not in x_keys:
                    x_keys.append(x)    
    # position of the x key
    x_offset = kwargs.get('x_offset', 0)
    x_pos = kwargs.get('x_pos', {x:i+x_offset for i,x in enumerate(x_keys)})
    x_cols = kwargs.get('x_cols', {x:'b' for x in x_keys})
    if isinstance(x_cols, str):
        this_cols = x_cols
        x_cols = {x:this_cols for x in x_keys}

    if keys_split is not None:
        for k in keys_split:
            dag_group_and_individual_2dict(
                ax=ax,
                mdict=mdict,
                keys_to_plot=k,
                **kwargs
            )
        dag_group_and_i_xticks(
            ax=ax, 
            mdict=mdict, 
            s_keys=s_keys,
            bar_width=bar_width,
            x_pos=x_pos,
            )
        return ax
    
    do_scols = True
    if s_cols is None:
        do_scols = False
        s_cols = {s:'grey' for s in s_keys}
    else:
        if isinstance(s_cols, str):
            do_scols = False
            s_cols = {s:s_cols for s in s_keys}
        
    # Specify the x position of each x key, as an array
    x_standard = np.array([x_pos[x] for x in x_keys])
    # Specify the color of each x key
    bar_kwargs['color'] = [x_cols[x] for x in x_keys]

    if do_jitter:
        s_jitter = np.linspace(-bar_width*jitter_width,bar_width*jitter_width, len(s_keys))
    else:
        s_jitter = np.zeros(len(s_keys))
    # Now find the overall_m
    overall_m = []
    for x in x_keys:
        m = []
        for s in s_keys:
            m.append(mdict[s][x])
        overall_m.append(m_method(m))
    for iS,s in enumerate(s_keys):
        this_y = np.array([mdict[s][x] for x in x_keys])
        this_yUPPER = np.array([y_upper[s][x] for x in x_keys]) if y_upper is not None else None
        this_yLOWER = np.array([y_lower[s][x] for x in x_keys]) if y_lower is not None else None
        this_yERR = [this_y - this_yLOWER, this_yUPPER - this_y] if y_upper is not None else None
        
        if do_points:
            ax.errorbar(
                x=x_standard + s_jitter[iS],
                y=this_y,
                yerr=this_yERR,
                color=s_cols[s],
                label=s if do_scols else None,
                **err_kwargs,
            )
    if do_bar:
        ax.bar(
            x=x_standard,
            height=overall_m,
            width=bar_width,
            **bar_kwargs,        
        )

    ticks_out = dag_group_and_i_xticks(
            ax=ax, 
            mdict=mdict, 
            s_keys=s_keys,
            bar_width=bar_width,
            keys_to_plot=keys_to_plot,
            x_keys=x_keys,
            x_pos=x_pos,
            )
    if return_ticks:
        return ticks_out

    return ax


def dag_group_and_i_xticks(ax, mdict, **kwargs):
    '''
    Set the xticks for group and individual plots
    '''
    s_keys = kwargs.get('s_keys', list(mdict.keys()))
    bar_width = kwargs.get('bar_width', 0.5)
    keys_to_plot = kwargs.get('keys_to_plot', None)
    x_keys = kwargs.get('x_keys', keys_to_plot)
    return_ticks = kwargs.get('return_ticks', False)
    if x_keys is None:
        x_keys = []
        for s in s_keys:
            for x in mdict[s].keys():
                if x not in x_keys:
                    x_keys.append(x)
    
    # x_cols = kwargs.get('x_cols', {x:'b' for x in x_keys})
    # Specify the x position of each x key
    x_pos = kwargs.get('x_pos', {x:i for i,x in enumerate(x_keys)})
    x_standard = np.array([x_pos[x] for x in x_keys])
    ax.set_xticks(x_standard)
    ax.set_xticklabels(x_keys, rotation=45)
    xlim0 = np.min(x_standard) - bar_width*2
    xlim1 = np.max(x_standard) + bar_width*2
    ax.set_xlim([xlim0, xlim1])
    return dict(x_standard=x_standard, x_pos=x_pos, x_keys=x_keys, xlim0=xlim0, xlim1=xlim1)


def dag_merid_helper(main_ax,wedge_angle, colors, **kwargs):
    """
    Categorize points based on their position relative to specified meridians.

    Parameters:
    - wedge_angle: Number of degrees around each meridian center (+/-)
    - angly_type: is wedge_angle specified in degrees or radians

    """
    angle_type = kwargs.get("angle_type", "deg")
    width_ratio = kwargs.get("width_ratio", 0.5)
    position_ratio = kwargs.get("position_ratio", [1,1])
    # Add the axis    
    ax = dag_add_square_axis(main_ax, width_ratio, position_ratio)
    ax.set_ylim(-1,1)
    ax.set_xlim(-1,1)
    # **** ADD THE LINES ****
    ax.set_xticks([])    
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)   
    # ax.axhline(0, color='k')     
    # ax.axvline(0, color='k')     

    # Define meridian centers
    merid_centers = {'right': 0, 'upper': np.pi/2, 'left': np.pi, 'lower': -np.pi/2}
    if angle_type=='deg':
        # Convert degrees around meridian to rad
        wedge_angle *= np.pi/180
    i = 0
    for m,m_centre in merid_centers.items():
        if m in ['horizontal' , 'vertical']:
            continue
        wedge = mpl.patches.Wedge(
            (0,0), 1, 
            np.degrees(m_centre-wedge_angle), 
            np.degrees(m_centre+wedge_angle), 
            facecolor=colors[m], edgecolor='black')
        i += 1
        ax.add_patch(wedge)
    ax.set_box_aspect(1)