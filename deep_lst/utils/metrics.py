import numpy as np

def bias(pred, targ, mean = False, *args, **kwargs):
    """Calculate bias between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray or xarray.DataArray
        Predicted values.
    targ : np.ndarray or xarray.DataArray
        Target (true) values.

    Returns
    -------
    float
        The bias value (mean of differences). 
    """
    if mean:
        return np.nanmean(targ.values - pred.values)
    else:
        return targ - pred
    
def absolute_error(pred, targ, mean = False, *args, **kwargs):
    """Calculate absolute error between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray or xarray.DataArray
        Predicted values.
    targ : np.ndarray or xarray.DataArray
        Target (true) values.

    Returns
    -------
    float
        The absolute error value (mean of absolute differences).
    """
    if mean:
        return np.nanmean(np.abs(targ.values - pred.values))
    else:
        return np.abs(targ - pred)
    
def rmse(pred, targ, *args, **kwargs):
    """Calculate root mean square error between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray or xarray.DataArray
        Predicted values.
    targ : np.ndarray or xarray.DataArray
        Target (true) values.

    Returns
    -------
    float
        The RMSE value.
    """
    return np.sqrt(np.nanmean((targ.values - pred.values) ** 2))

def mae(pred, targ, *args, **kwargs):
    """Calculate mean absolute error between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray or xarray.DataArray
        Predicted values.
    targ : np.ndarray or xarray.DataArray
        Target (true) values.

    Returns
    -------
    float
        The MAE value.
    """
    return np.nanmean(np.abs(targ.values - pred.values))

def coefficient_of_variation(pred, targ, *args, **kwargs):
    """Calculate coefficient of variation between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray or xarray.DataArray
        Predicted values.
    targ : np.ndarray or xarray.DataArray
        Target (true) values.

    Returns
    -------
    float
        The coefficient of variation value.
    """
    mae_value = mae(pred, targ)
    mean_targ = np.nanmean(targ.values)
    return mae_value / mean_targ

def r_squared(pred, targ, *args, **kwargs):
    """Calculate R-squared (coefficient of determination) between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray or xarray.DataArray
        Predicted values.
    targ : np.ndarray or xarray.DataArray
        Target (true) values.

    Returns
    -------
    float
        The R-squared value.
    """
    ss_res = np.nansum((targ.values - pred.values) ** 2)
    ss_tot = np.nansum((targ.values - np.nanmean(targ.values)) ** 2)
    return 1 - (ss_res / ss_tot)

def anomaly_correlation(pred, targ, *args, **kwargs):
    """Calculate anomaly correlation between predictions and targets.

    Parameters
    ----------
    pred : np.ndarray or xarray.DataArray
        Predicted values.
    targ : np.ndarray or xarray.DataArray
        Target (true) values.

    Returns
    -------
    float
        The anomaly correlation value.
    """
    pred_anom = pred.values - np.nanmean(pred.values)
    targ_anom = targ.values - np.nanmean(targ.values)
    numerator = np.nansum(pred_anom * targ_anom)
    denominator = np.sqrt(np.nansum(pred_anom ** 2) * np.nansum(targ_anom ** 2))
    return numerator / denominator  