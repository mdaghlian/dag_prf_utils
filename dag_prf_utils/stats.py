import numpy as np
from scipy.stats import t

from scipy.fftpack import dct, idct

def dag_dct_detrending(ts_au, n_trend_to_remove):
    if n_trend_to_remove is not False:
        # Preparation: - demean
        ts_au_centered = ts_au - np.mean(ts_au, axis=1, keepdims=True)

        # [2] Get the DCT of the time series (specify axis=1 to do it over time)
        dct_values = dct(ts_au_centered, type=2, norm='ortho', axis=1)

        # [3] Remove the coefficients specified
        dct_values[:,:n_trend_to_remove] = 0

        # [4] Get the inverse DCT
        ts_detrend = idct(dct_values, type=2, norm='ortho', axis=1)
        # print(np.mean(ts_detrend, axis=1))
        # [5] Add the mean back
        ts_detrend = ts_detrend + np.mean(ts_au, axis=1, keepdims=True)
    else:
        ts_detrend = ts_au.copy()
    # [6] PSC
    ts_detrend = dag_psc(ts_detrend)

    return ts_detrend

def dag_psc(ts_in):
    ts_out = (ts_in - np.mean(ts_in, axis=1, keepdims=True)) / np.mean(ts_in, axis=1, keepdims=True) * 100
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