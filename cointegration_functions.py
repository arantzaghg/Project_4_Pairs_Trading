import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def correlation(data, window):
    
    data = data.copy()
    corr_window = data.iloc[:, 0].rolling(window).corr(data.iloc[:, 1])
    corr = corr_window.mean()

    return corr

def ols_adf(data):

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

    data = data.copy()
    johansen_res = coint_johansen(data, det_order, k_ar_diff)
    eigenvector = johansen_res.evec[:, 0]
    critical_value95 = johansen_res.cvt[0, 1]
    trace_stat = johansen_res.lr1[0]
    return eigenvector, critical_value95, trace_stat
