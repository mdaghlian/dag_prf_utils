import numpy as np
from scipy.stats import t
from scipy.fftpack import dct, idct

def dag_dct_detrending(ts_au, n_trend_to_remove, do_psc=True, baseline_pt=None):
    """
    Perform detrending using Discrete Cosine Transform (DCT) and optionally Percentage Signal Change (PSC).

    Parameters:
    - ts_au (numpy.ndarray): Input time series data, shape (n_vx, n_time_points).
    - n_trend_to_remove (int or False): Number of DCT coefficients to remove for detrending. If False/0, no detrending is performed.
    - do_psc (bool, optional): Whether to perform Percentage Signal Change (PSC) after detrending. Default is True.
    - baseline_pt (np.ndarray, int, list, or None, optional): Baseline points used for PSC calculation. If None, all points are considered as baseline.
        If 1 value: taske baseline values from 0 to baseline_pt
        If 2 value: it represents the range of points [start, stop] as baseline.
        If more   : it represents specific points as baseline.

    Returns:
    - numpy.ndarray: Detrended time series data, shape (n_vx, n_time_points).
    """
    if ts_au.ndim == 1:
        ts_au = ts_au.reshape(-1, 1)

    if n_trend_to_remove!=0:
        # Preparation: demean the time series
        ts_au_centered = ts_au - np.mean(ts_au, axis=1, keepdims=True)

        # Compute the DCT of the time series
        dct_values = dct(ts_au_centered, type=2, norm='ortho', axis=1)

        # Remove the specified number of coefficients
        dct_values[:, :n_trend_to_remove] = 0

        # Inverse DCT to obtain detrended time series
        ts_detrend = idct(dct_values, type=2, norm='ortho', axis=1)

        # Add the mean back to the detrended series
        ts_detrend = ts_detrend + np.mean(ts_au, axis=1, keepdims=True)
    else:
        ts_detrend = ts_au.copy()

    # Perform Percentage Signal Change (PSC) if specified
    if do_psc:
        ts_detrend = dag_psc(ts_detrend, baseline_pt)

    return ts_detrend

def dag_psc(ts_in, baseline_pt=None):
    """
    Calculate Percentage Signal Change (PSC) for the input time series.

    Parameters:
    - ts_in (numpy.ndarray): Input time series data, shape (n_vx, n_time_points).
    - baseline_pt (np.ndarray, int, list, or None, optional): Baseline points used for PSC calculation. If None, all points are considered as baseline.
        If 1 value: taske baseline values from 0 to baseline_pt
        If 2 value: it represents the range of points [start, stop] as baseline.
        If more   : it represents specific points as baseline.

    Returns:
    - numpy.ndarray: Time series data after Percentage Signal Change (PSC) normalization, shape (n_vx, n_time_points).
    """
    if ts_in.ndim == 1:
        ts_in = ts_in.reshape(-1, 1)

    # Define the baseline points
    if baseline_pt is None:
        baseline_pt = np.arange(ts_in.shape[-1])
    elif isinstance(baseline_pt, (int, np.integer)):
        baseline_pt = np.arange(0, baseline_pt)
    elif len(baseline_pt) == 2:
        baseline_pt = np.arange(baseline_pt[0], baseline_pt[1], dtype=int)
    else:
        baseline_pt = np.array(list(baseline_pt), dtype=int)

    # Calculate the mean of baseline points
    baseline_mean = np.mean(ts_in[:, baseline_pt], axis=1, keepdims=True)

    # Perform Percentage Signal Change (PSC) normalization
    ts_out = (ts_in - baseline_mean) / baseline_mean * 100

    # Handle NaN values resulting from division by zero
    nan_rows = np.isnan(ts_out).any(axis=1)
    ts_out[nan_rows, :] = 0

    return ts_out

def dag_paired_ttest(x, y, **kwargs):
    '''dag_paired_ttest
    sim#ple paired t-test, with option to override to correct for voxel-to-surface upsampling
    
    ow_n                    Specify the 'n' by hand
    upsampling_factor       n will be obtained from len(x)/upsampling factor 
    side                    'two-sided', 'greater', 'less'. Which sided test.

    # Test that it works (not doing adjusting)
    x= np.random.rand(100)
    y = x +  (np.random.rand(100) -.5 )     
    for side in ['greater', 'less', 'two-sided']:
        print(dag_paired_ttest(x, y,side=side))
        print(stats.ttest_rel(x, y, alternative=side))
    '''
    ow_n = kwargs.get('ow_n', None)
    upsampling_factor = kwargs.get('upsampling_factor', None)
    side = kwargs.get('side', 'two-sided') # 

    actual_n = len(x)    
    if upsampling_factor is not None:
        n = actual_n / upsampling_factor
    elif ow_n is not None:
        n = ow_n # overwrite the n that is given
    else:
        n = actual_n
    df = n-1 # Degrees of freedom = n-1 
    
    diffs = x - y
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)  # Delta degrees of freedom
    standard_error = std_diff / np.sqrt(n)
    t_statistic = mean_diff / standard_error

    p_value = dag_t_to_p(t_statistic, df, side)

    stats = {
        'n'             : n,
        'mean_diff'     : mean_diff,
        't_statistic'   : t_statistic,
        'p_value'       : p_value,
        'df'            : df,
    }
    return stats


def dag_rapid_slope(x,y):
    '''dag_rapid_slope
    Calculate the slope as quickly as possible
    '''
    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    # Calculate the slope...
    slope = numerator / denominator
    return slope

def dag_slope_test(x,y, **kwargs):
    '''ncsf_slope
    Calculate the slope, intercept, associated t-stat, and p-values
    
    Options:
    ow_n                Overwrite the "n" to your own
    upsampling_factor       n will be obtained from len(x)/upsampling factor 
    side                    'two-sided', 'greater', 'less'. Which sided test.

    '''
    ow_n = kwargs.get('ow_n', None)
    upsampling_factor = kwargs.get('upsampling_factor', None)
    side = kwargs.get('side', 'two-sided') # 

    actual_n = len(x)    
    if upsampling_factor is not None:
        n = actual_n / upsampling_factor
    elif ow_n is not None:
        n = ow_n # overwrite the n that is given
    else:
        n = actual_n
    df = n-2 # Degrees of freedom

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    # Calculate the slope...
    slope = numerator / denominator
    # Calculate the intercept
    intercept = y_mean - (slope * x_mean)
    # Calculate predictions
    y_pred  = slope * x + intercept
    # Now calculate the residuals 
    residuals = y - y_pred

    # Standard error of the slope
    std_error_slope = np.sqrt(np.sum(residuals**2) / (df * np.sum((x - x_mean) ** 2)))

    # t-statistic
    t_statistic = slope / std_error_slope

    p_value = dag_t_to_p(t_statistic, df, side)

    stats = {
        'n' : n,
        'df' : df,
        'slope' : slope, 
        'intercept' : intercept, 
        't_statistic' : t_statistic, 
        'p_value' : p_value,
    }
    return stats

def dag_t_to_p(t_statistic, df, side):
    # Caclulate the p-value
    if side=='two-sided':
        p_value = 2 * (1 - t.cdf(np.abs(t_statistic), df))
    elif side=='less':
        p_value = t.cdf(t_statistic, df)
    elif side=='greater':
        p_value = t.cdf(-t_statistic, df)        
    return p_value