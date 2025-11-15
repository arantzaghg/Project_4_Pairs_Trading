import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def correlation(data, window):
    """
    Calculate the rolling correlation between two time series.
    
    Parameters:
    data : pandas.DataFrame: A DataFrame with two columns representing the time series.
    window : int: The rolling window size.

    Returns:
    float: The average rolling correlation over the specified window.
    """
    
    data = data.copy()
    corr_window = data.iloc[:, 0].rolling(window).corr(data.iloc[:, 1])
    corr = corr_window.mean()

    return corr

def ols_adf(data):
    """
    Perform OLS regression between two time series and conduct ADF test on the residuals.
    
    Parameters:
    data : pandas.DataFrame: A DataFrame with two columns representing the time series.
    
    Returns:
    tuple: A tuple containing the residuals from the OLS regression and the p-value from the ADF test.
    """

    data = data.copy()
    y = data.iloc[:, 0]
    x = data.iloc[:, 1]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    res = model.resid
    adf_results = adfuller(res)
    adf_pvalue = adf_results[1] 

    return res, adf_pvalue

def johansen(data, det_order=0, k_ar_diff=1):
    """
    Perform the Johansen cointegration test on two time series.

    Parameters:
    data : pandas.DataFrame: A DataFrame with two columns representing the time series.
    det_order : int: The deterministic trend order to include in the test.
    k_ar_diff : int: The number of lagged differences to include in the test.

    Returns:
    tuple: A tuple containing the first eigenvector, the 95% critical value, and the trace statistic.
    """

    data = data.copy()
    johansen_res = coint_johansen(data, det_order, k_ar_diff)
    eigenvector = johansen_res.evec[:, 0]
    critical_value95 = johansen_res.cvt[0, 1]
    trace_stat = johansen_res.lr1[0]
    return eigenvector, critical_value95, trace_stat
