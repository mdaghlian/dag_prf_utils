import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import patches
from scipy.stats import binned_statistic
import json
import os
opj = os.path.join


from .utils import *
from .cmap_functions import *

# Default bounds for visual field plotting
default_ecc_bounds =  np.linspace(0, 5, 7)
default_pol_bounds = np.linspace(-np.pi, np.pi, 13)

# def dag_rm_dag_cmaps_from_mpl():
#     # Add to matplotlib cmaps?
#     for cm_name in custom_col_dict.keys():
#         plt.unregister_cmap(cm_name)


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
            ax.plot((0, outer_x), (0, outer_y), color=line_col, alpha=0.3)
            if incl_ticks:
                ax.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')

        # if not '360' in outer_txt:
        #     ax.plot((0, outer_x), (0, outer_y), color=line_col, alpha=0.3)
        #     if incl_ticks:
        #         ax.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')

    for i_ecc, i_ecc_val in enumerate(ecc_bounds): # Loop through eccentricity lines
        grid_line = patches.Circle((0, 0), i_ecc_val, color=line_col, alpha=0.3, fill=0)    
        ax.add_patch(grid_line)                    
    
    if aperture_rad!=None: # Add the aperture
        aperture_line = patches.Circle((0, 0), aperture_rad, color=aperture_col, linewidth=8, alpha=0.5, fill=0)    
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.legend()

def dag_update_fig_fontsize(fig, new_font_size):
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
            dag_update_ax_fontsize(i_kid, new_font_size)
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
    ecc_bounds = kwargs.get("ecc_bounds", default_ecc_bounds)
    pol_bounds = kwargs.get("pol_bounds", default_pol_bounds)            
    n_params_gt_1 = False # If there is more than 1 parameter to bin
    if not isinstance(params2bin, list): # If there is only 1 parameter to bin, make it a list
        params2bin = [params2bin]
        n_params_gt_1 = True

    total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1) 
    params_binned = []
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

                if bin_weight is not None: # If there is a bin weight, use it
                    bin_mean[i_ecc, i_pol] = (params2bin[i_param][bin_idx] * bin_weight[bin_idx]).sum() / bin_weight[bin_idx].sum()
                else:
                    # bin_mean[i_ecc, i_pol] = np.mean(params2bin[i_param][bin_idx])
                    bin_mean[i_ecc, i_pol] = np.median(params2bin[i_param][bin_idx])

        bin_mean = np.reshape(bin_mean, total_n_bins)
        # REMOVE ANY NANS
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
        bin_weight      np.ndarray      Weighted mean in each bin, not just the average
        ecc_bounds      np.ndarray      eccentricity bounds
        pol_bounds      np.ndarray      polar angle bounds
        dot_alpha       float           alpha of points
        dot_size        float           size of points
        dot_col         str             color of points
        dot_vmin        float           min value for color map
        dot_vmax        float           max value for color map
        dot_cmap        str             color map
    
    Return:
        ax              matplotlib axes
        cb              colorbar

    '''
    ax = kwargs.get('ax', plt.gca())
    do_binning = kwargs.get("do_binning", False)
    # -> add option for dot size scaling... ( & alpha scaling) ??
    bin_weight = kwargs.get("bin_weight", None)
    ecc_bounds = kwargs.get("ecc_bounds", np.linspace(0, 5, 7) )
    pol_bounds = kwargs.get("pol_bounds", np.linspace(-np.pi, np.pi, 13))            
    dot_alpha = kwargs.get("alpha", 0.5)
    dot_size = kwargs.get("dot_size",200)
    dot_col = kwargs.get("dot_col", 'k')   
    dot_vmin =  kwargs.get("dot_vmin", None)   
    dot_vmax =  kwargs.get("dot_vmax", None)   
    dot_cmap =  kwargs.get("dot_cmap", None)   
    
    if isinstance(dot_col, np.ndarray) & (dot_cmap==None):
        dot_cmap = 'viridis'
    if dot_cmap != None:
        dot_cmap = dag_get_cmap(dot_cmap)

    if do_binning:
        dot_ecc, dot_pol = dag_coord_convert(dot_x,dot_y,old2new="cart2pol")
        total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)
        bin_x, bin_y = dag_return_ecc_pol_bin(params2bin=[dot_x, dot_y], 
                            ecc4bin=dot_ecc, 
                            pol4bin=dot_pol, 
                            bin_weight=bin_weight,
                            ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)

        # if isinstance(dot_col, np.ndarray):
        if not isinstance(dot_col, str):
            bin_col = dag_return_ecc_pol_bin(params2bin=dot_col, 
                                ecc4bin=dot_ecc, 
                                pol4bin=dot_pol, 
                                bin_weight=bin_weight,
                                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)   
        else:
            bin_col = dot_col
        # if isinstance(dot_size, np.ndarray):
        if hasattr(dot_size, 'len'):
            bin_size = dag_return_ecc_pol_bin(params2bin=dot_size, 
                                ecc4bin=dot_ecc, 
                                pol4bin=dot_pol, 
                                bin_weight=bin_weight,
                                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)            
        else:
            bin_size = dot_size
        # if isinstance(dot_alpha, np.ndarray):
        if hasattr(dot_size, 'len'):            
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
    '''dag_plot_bin_line
    Description:
        Plot a line, binned by a variable. 
        e.g., plot the mean rsquared, binned by eccentricity

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
    do_basics = kwargs.get('do_basics', False)
    if not isinstance(bins, (np.ndarray, list)):
        bins = n_bins    
    xerr = kwargs.get("xerr", False)
    do_bars = kwargs.get("do_bars", True)
    do_shade = kwargs.get("do_shade", False)
    summary_type = kwargs.get("summary_type", 'mean')
    err_type = kwargs.get("err_type", None)
    # Do the binning
    X_mean = binned_statistic(bin_using, X, bins=bins, statistic=summary_type)[0]
    X_std = binned_statistic(bin_using, X, bins=bins, statistic='std')[0]
    # count = binned_statistic(bin_using, X, bins=bins, statistic='count')[0]
    Y_mean = binned_statistic(bin_using, Y, bins=bins, statistic=summary_type)[0]  
    Y_std = binned_statistic(bin_using, Y, bins=bins, statistic='std')[0]  #/ np.sqrt(bin_data['bin_X']['count'])              
    if do_bars:
        if xerr:
            ax.errorbar(
                X_mean,
                Y_mean,
                yerr=Y_std,
                xerr=X_std,
                color=line_col,
                label=line_label, 
                lw=lw,
                **line_kwargs
                )
        else:
            ax.errorbar(
                X_mean,
                Y_mean,
                yerr=Y_std,
                # xerr=X_std,
                color=line_col,
                label=line_label,
                lw=lw,
                **line_kwargs
                )        

    elif do_shade:
        
        X_mid_pt = (bins[:-1] + bins[1:]) / 2
        if 'pc' in err_type:            
            Y_mid = binned_statistic(bin_using, Y, bins=bins, statistic=np.median)[0]                      
            pc_lower = float(err_type.split('-')[-1])
            pc_upper = 100 - pc_lower

            pcLOWER_lambda = lambda data: np.percentile(data, pc_lower)
            pcUPPER_lambda = lambda data: np.percentile(data, pc_upper)
            Y_lower = binned_statistic(bin_using, Y, bins=bins, statistic=pcLOWER_lambda)[0]              
            Y_upper = binned_statistic(bin_using, Y, bins=bins, statistic=pcUPPER_lambda)[0]              
        if err_type=='mean':
            Y_mid = Y_mean
            Y_lower = Y_mean - Y_std
            Y_upper = Y_mean + Y_std
            

        ax.plot(
            X_mid_pt,
            Y_mid,
            color=line_col,
            label=line_label,
            alpha=0.5,
            lw=lw,
            **line_kwargs,
            )
        ax.fill_between(
            X_mid_pt,
            Y_lower,
            Y_upper,
            alpha=0.5,
            color=line_col,            
            label='_'        
            )       

    else:
        ax.plot(
            X_mean,
            Y_mean,
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

    old_col = kwargs.get("old_col", 'k')
    new_col = kwargs.get("new_col", 'g')
    # -> for now only 1 alpha, size per dot
    dot_alpha = kwargs.get('dot_alpha', .5)
    dot_size = kwargs.get('dot_size', 500)

    # aperture_rad = kwargs.get("aperture_rad", None)
    # patch_col = kwargs.get("patch_col", "k")
    arrow_col = kwargs.get("arrow_col", 'b')
    arrow_cmap = kwargs.get("arrow_cmap", 'magma_magmarev')
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
            q_cmap = dag_get_cmap(arrow_cmap)#mpl.cm.__dict__['hsv']
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

    dag_add_ecc_pol_lines(ax, **kwargs)        
    dag_add_ax_basics(ax, **kwargs)    
    # END FUNCTION 

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
    do_kde = kwargs.get('do_kde', False)    
    do_id_line = kwargs.get('do_id_line', False)
    do_ow = kwargs.get('ow', False)
    do_scatter = kwargs.get('do_scatter', True)
    do_line = kwargs.get('do_line', False)
    dot_col = kwargs.get('dot_col', None)
    dot_alpha = kwargs.get('dot_alpha', None)    
    dot_cmap = kwargs.get('dot_cmap', None)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    dot_label=kwargs.get('dot_label')
    do_corr = kwargs.get('do_corr', False)

    if do_scatter:        
        scat_col = ax.scatter(
            X,Y, c=dot_col, alpha=dot_alpha, label=dot_label,
            cmap=dot_cmap, vmin=vmin, vmax=vmax) # added **kwargs
        if dot_col is not None:
            fig = plt.gcf()
            # cb = fig.colorbar(scat_col, ax=ax)        
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
    if do_kde:
        sns.kdeplot(X,Y, color=dot_col)
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
    dag_scatter = kwargs.get('dag_scatter', False)
    skip_hist = kwargs.get('skip_hist', False)
    if isinstance(data_in, np.ndarray):
        n_pdim = data_in.shape[-1]
        p_labels = kwargs.get('p_labels', np.arange(n_pdim))
        data_dict = {}
        for i,p in enumerate(p_labels):
            data_dict[p] = data_in[:,i]
    else:
        data_dict = data_in
        p_labels = kwargs.get('p_labels', data_dict.keys())
        n_pdim = len(p_labels)

    fig = kwargs.get('fig', plt.figure())    
    rows = n_pdim
    cols = n_pdim
    plot_i = 1
    ax_list = {}

    for i1,y_param in enumerate(p_labels):
        ax_list[i1] = {}
        for i2,x_param in enumerate(p_labels):
            ax = fig.add_subplot(rows,cols,plot_i)
            ax_list[i1][i2] = ax
            if i1>i2:
                ax.axis('off')
            else:
                ax.set_xlabel(x_param)
                if i1==i2:
                    ax.hist(data_dict[x_param])
                else:
                    if not dag_scatter:
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
            plot_i += 1
        fig.set_tight_layout('tight')
    return fig, ax_list



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
    ymin, _ = kwargs.get('ymin', ax.get_ylim())          # Get the y limits
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
            fig_coord[i_tick,1]-(dx_fig*1), #  - i_tick/10, # option to stagger the axes with i_tick
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

    
