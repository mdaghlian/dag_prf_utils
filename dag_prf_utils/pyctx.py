
import numpy as np
try: 
    import cortex 
except ImportError:
    raise ImportError('Error importing pycortex... Not a problem unless you want to use pycortex stuff')    


def pycortex_alpha_plotting(sub, data, data_weight, **kwargs):
    
    '''
    Function to make plotting in pycortex (using web gui) easier
    I found that using the "cortex.Vertex2D" didn't work (IDK why)
    Based on Sumiya's code -> extracts the curvature as a grey map
    -> puts your data on top of it...

    sub             subject to plot (pycortex id)
    data            data to plot (np.array size of the surface)
    data_weight     Used to mask your data. Can be boolean or a range (should be between 0 and 1.)
                    See other options
                    
    
    *** Optional ***
    Value           Default             Meaning
    --------------------------------------------------------------------------------------------------
    data_w_thresh   None                1 or 2 values (gives the threshold of values to include (lower & upper bound))                    
    vmin            None                Minimal value for data cmap
    vmax            None                Maximum value for data cmap
    cmap            Retinotopy_RYBCR    Color map to use for data
    bool_mask       True                Mask the data with absolute mask or a gradient
    scale_data_w    False               Scale the data weight between 0 and 1 
    '''
    data_w_thresh   = kwargs.get('data_w_thresh', None)
    if (not isinstance(data_w_thresh, list)) & (data_w_thresh is not None):
        data_w_thresh = [data_w_thresh]
    vmin            = kwargs.get('vmin', None)
    vmax            = kwargs.get('vmax', None)
    cmap            = kwargs.get('cmap', 'Retinotopy_RYBCR')
    bool_mask       = kwargs.get('bool_mask', True)
    scale_data_w    = kwargs.get('scale_data_w', False)

    # [1] Set up the mask / weighting for the data    
    if scale_data_w: 
        # Rescale data weights to be b/w 0 and 1
        dw = (data_weight-np.nanmin(data_weight)) / (np.nanmax(data_weight)-np.nanmin(data_weight))
    else:
        dw = data_weight.copy()
    # -> apply thresholds
    if data_w_thresh is None:
        pass
    elif len(data_w_thresh)==1:
        # lower bound only
        dw[dw<data_w_thresh] = 0
    else:
        dw[dw<data_w_thresh[0]] = 0
        dw[dw>data_w_thresh[1]] = 0    
    # remove any NaN values
    dw[np.isnan(dw)] = 0
    # -> bool mask?
    if bool_mask:
        dw = dw!=0

    # [2] Get the background for the figure (the curve data)    
    # -> data goes on top of this. Where there are gaps, this will show (e.g., fits are not good enough etc.)
    curv = cortex.db.get_surfinfo(sub)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map.
    curv.vmin = -1
    curv.vmax = 1
    curv.cmap = 'gray' # we want it to be grey underneath


    # [3] Create the data colormap
    # Create vx mask from data
    vx = cortex.Vertex(data, sub, vmin=vmin, vmax=vmax, cmap=cmap)

    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    # Mask out using alpha values (the alpha of your data is inverse to the curve)
    alpha_mask = dw.astype(np.float)

    # Alpha mask
    display_data = vx_rgb * alpha_mask + curv_rgb * (1-alpha_mask)

    # Create vertex RGB object out of R, G, B channels
    vx_fin = cortex.VertexRGB(*display_data, sub)
    # cortex.quickshow(vx_fin)
    # cortex.webgl.show(
    #         {
    #         'alpha' : vx_fin,
    #         'no_alpha' : vx,
    #         })    
    return vx_fin, vx

