'''
Created on Dec 24, 2018

@author: mkim1
'''

import numpy as np 
from scipy.stats import sem
import statsmodels.formula.api as smf    
import pandas as pd

from uncertainties import ufloat
from uncertainties.umath import exp as uexp
from statsmodels.regression.quantile_regression import QuantReg
import logging
logging.basicConfig(level=logging.ERROR) #For the case when verbose=False
logger = logging.getLogger(__name__)  #Generate logger for this module

from scipy.special import comb
    
def coeffs(M):
    """
    Generate the "Smooth noise-robust differentiators" as defined in Pavel
    Holoborodko's formula for c_k
    Parameters
    ----------
    M : int
        the order of the differentiator
    c : float array of length M
        coefficents for k = 1 to M
    """
    m = (2*M - 2)/2
    k = np.arange(1, M+1)
    c = 1./2.**(2*m + 1)*(comb(2*m, m - k + 1) - comb(2*m, m - k - 1))
    return c

def holo_diff(x,y,M=2):
    """
    Implementation of Pavel Holoborodko's method of "Smooth noise-robust
    differentiators" see
    http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/
    smooth-low-noise-differentiators
    Creates a numerical approximation to the first derivative of a function
    defined by data points.  End point approximations are found from
    approximations of lower order.  Greater smoothing is achieved by using a
    larger value for the order parameter, M.
    Parameters
    ----------
    x : float array or scalar
        abscissa values of function or, if scalar, uniform step size
    y : float array
        ordinate values of function (same length as x if x is an array)
    M : int, optional (default = 2)
        order for the differentiator - will use surrounding 2*M + 1 points in
        creating the approximation to the derivative
    Returns
    -------
    dydx : float array
        numerical derivative of the function of same size as y
    """
    if np.isscalar(x):
        x = x*np.arange(len(y))
    assert len(x) == len(y), 'x and y must have the same length if x is ' + \
            'an array, len(x) = {}, len(y) = {}'.format(len(x),len(y))
    N = 2*M + 1
    m = (N - 3)/2
    c = coeffs(M)
    df = np.zeros_like(y)
    nf = len(y)
    fk = np.zeros((M,(nf - 2*M)))
    for i,cc in enumerate(c):
        # k runs from 1 to M
        k = i + 1
        ill = M - k
        ilr = M + k
        iul = -M - k
        # this formulation is needed for the case the k = M, where the desired
        # index is the last one -- but range must be given as [-2*M:None] to
        # include that last point
        iur = ((-M + k) or None)
        fk[i,:] = 2.*k*cc*(y[ilr:iur] - y[ill:iul])/(x[ilr:iur] - 
                x[ill:iul])
    df[M:-M] = fk.sum(axis=0)
    # may want to incorporate a variety of methods for getting edge values,
    # e.g. setting them to 0 or just using closest value with M of the ends.
    # For now we recursively calculate values closer to the edge with
    # progressively lower order approximations -- which is in some sense
    # ideal, though maybe not for all cases
    if M > 1:
        dflo = holo_diff(x[:2*M],y[:2*M],M=M-1)
        dfhi = holo_diff(x[-2*M:],y[-2*M:],M=M-1)
        df[:M] = dflo[:M]
        df[-M:] = dfhi[-M:]
    else:
        df[0] = (y[1] - y[0])/(x[1] - x[0])
        df[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return df


class init(object):
    def __init__(self, q, t, correction_ET=None, verbose=False):
        self.q = q
        self.t = t 
        
        if correction_ET is None:
            self.correction = np.ones_like(q)
        else:
            self.correction = correction_ET
            
        if verbose:
            logger.setLevel('DEBUG')
        
    def def_period(self, exclude=None, merge_gap=0, min_length=0):
        if exclude is None:
            exclude = np.zeros_like(self.t, dtype = bool) 
                        
        dec = ~exclude
        
        # find the start and end of the recession events
        idx_st = np.where(np.logical_and(dec[1:], ~dec[:-1]))[0]+1
        if dec[0]:
            idx_st = np.append(0, idx_st)
            
        idx_end = np.where(np.logical_and(dec[:-1], ~dec[1:]))[0]+1
        if dec[-1]:
            idx_end = np.append(idx_end,len(dec)) 
            
    
        # merge events if they are mergable
        self.merge_gap = merge_gap
        if merge_gap>0:
            gaps = (idx_st[1:] - idx_end[:-1]) - 1
            mergable = np.where(gaps<=merge_gap,True,False) #True when mergable
            #need to check if [y:x+1] is okay
            mergable_exclude = ~np.array([True in exclude[y:x+1] for x, y in zip(idx_st[1:], idx_end[:-1])] ) #True when mergable
            idx_st = idx_st[np.append(True,~np.logical_and(mergable, mergable_exclude))]
            idx_end = idx_end[np.append(~np.logical_and(mergable, mergable_exclude),True)]            
            
                
        # remove if they are too short
        if min_length>1:
            too_short = (idx_end - idx_st)<min_length
            idx_end = idx_end[~too_short]
            idx_st = idx_st[~too_short]

                        
        self.events = [event_obj(idx_st[i] , idx_end[i] , self.q[idx_st[i]:idx_end[i]+1] , self.t[idx_st[i]:idx_end[i]+1], self.correction[idx_st[i]:idx_end[i]+1]) for i in range(len(idx_st))]


    def est_dqdt(self, grad_mult = 1.0, method = None):

        self.corh = {}
        self.th = {}
        self.qh = {}
        self.dq_dt = {}
                        
        for idx,item in enumerate(method):
            self.corh[item] = np.array([])
            self.th[item] = np.array([])
            self.qh[item] = np.array([])
            self.dq_dt[item] = np.array([])
                        
        
        for R in self.events: 

            R.getdqdt_method(method = method, grad_mult =  grad_mult)

            for _,item in enumerate(method):            
                self.corh[item] = np.append(self.corh[item], R.corh[item])
                self.th[item] = np.append(self.th[item], R.th[item])
                self.qh[item] = np.append(self.qh[item], R.qh[item])
                self.dq_dt[item] = np.append(self.dq_dt[item], R.dq_dt[item])
            

                
    def _fitgfunc_set(self, obj, g_func, p0=None,method_dqdt = None, method_fit = None, regression_function = None, uncertainty = True):
        if (method_fit == 'ols') or (method_fit == 'quantile'):
            if len(obj.dq_dt[method_dqdt])>0:
        
                x = np.log(np.array(obj.qh[method_dqdt])) 
                y = np.log(obj.dq_dt[method_dqdt] * obj.corh[method_dqdt])
    
                #Remove data when discharge <= 0
                y = y[~np.isinf(x)]
                x = x[~np.isinf(x)]

                x = x[~np.isinf(y)]
                y = y[~np.isinf(y)]
                                
                if len(x)>0:
                    d = {'x': x, 'y': y}
                    df = pd.DataFrame(data=d)
                    
                    if regression_function == 'ln(-dqdt) = a + b * ln(Q)':                
                        if method_fit == 'ols':                    
                            res = smf.ols(formula='y ~ x',   data=df).fit()
                        elif method_fit == 'quantile':
                            res = smf.quantreg('y ~ x', data=df).fit(q=.10)
                        else:
                            logger.error('Wrong fitting method')
                            
                        popt = [np.exp(res.params[0]), res.params[1] - 1] #modifying popt due to log and g(Q)*Q        
    
                        if uncertainty:
                            ci = res.conf_int(alpha=0.05, cols=None)
                            tmp = ufloat(np.log(popt[0]), ci[1][0] - np.log(popt[0]))
                            tmp  = uexp(tmp)
                            std = tmp.std_dev
                            popt_low = [popt[0] - std, ci[0][1] - 1]
                            popt_high = [popt[0] + std,ci[1][1] - 1]
                        else:
                            popt_low, popt_high = None, None 
                            
                    elif regression_function == 'ln(-dqdt) ~ a + b * ln(Q) + c * ln(Q)^2':
                        if method_fit == 'ols':                    
                            res = smf.ols(formula='y ~ x + np.power(x,2)',   data=df).fit()
                        elif method_fit == 'quantile':
                            res = smf.quantreg('y ~ x + np.power(x,2)', data=df).fit(q=.10)
                        else:
                            logger.error('Wrong fitting method')
                            logger.error(method_fit)
                                                
                        popt = [res.params[0], res.params[1]-1 , res.params[2]] #modifying popt due to log and g(Q)*Q        
                        
                        if uncertainty:
                            ci = res.conf_int(alpha=0.05, cols=None)
                            popt_low = [ci[0][0], ci[0][1] -1 , ci[0][2]]
                            popt_high = [ci[1][0], ci[1][1] -1, ci[1][2]]
                        else:
                            popt_low, popt_high = None, None                        
                                            
                    else:
                        logger.error('Wrong regression function')
                else:

                    if regression_function == 'ln(-dqdt) = a + b * ln(Q)':                
                        popt = [0.,0.0]
                        popt_high = [0,0]
                        popt_low = [0,0] 
                    elif regression_function == 'ln(-dqdt) ~ a + b * ln(Q) + c * ln(Q)^2':
                        popt = [0.,0.0,0.]
                        popt_high = [0,0,0.]
                        popt_low = [0,0,0.] 
                        
                    logger.debug('No data - fitgfunc')

                                  
            else: #
                if regression_function == 'ln(-dqdt) = a + b * ln(Q)':                
                    popt = [0.,0.0]
                    popt_high = [0,0]
                    popt_low = [0,0] 
                elif regression_function == 'ln(-dqdt) ~ a + b * ln(Q) + c * ln(Q)^2':
                    popt = [0.,0.0,0.]
                    popt_high = [0,0,0.]
                    popt_low = [0,0,0.] 
                    
                logger.debug('No data - fitgfunc')
            
            return lambda x: g_func(x, popt), popt, popt_low, popt_high, None, None, None,None, None
        
        elif method_fit == 'wls':

            x = np.array(obj.qh[method_dqdt])
            y = obj.dq_dt[method_dqdt] * obj.corh[method_dqdt] 

            y = y[x>0]  #Remove data when discharge <=0
            x = x[x>0]  
    
            #sort y based on x and sort x as well.
            temp = x.argsort()
            temp = np.flipud(temp)
            
            y = y[temp]
            x = x[temp]
    
            xlog = np.log(x)
            
            binx = []
            biny = []
            binvar = []
            binvarlog = []
            bin_stderr_divQ = []
            binnobs = []
            
            bin_x_range = [np.nanmax(xlog)]
            xlog_min = np.nanmin(xlog)
            onepercent_range = (np.nanmax(xlog) - np.nanmin(xlog)) / 100.
    
            flag_cont = True
        
            idx_here = 0
            while flag_cont:
                
                #Check if there is enough data
                std_err = 0.0
                
                #First guess on the bin
                bin_upper = bin_x_range[idx_here]
                bin_lower = bin_x_range[idx_here] - onepercent_range
                
                if bin_lower > xlog_min:
                    #adjust the range based on standard error               
                    flag_criteria = True
                    bin_upper_idx = next(xx[0] for xx in enumerate(xlog) if xx[1] <= bin_upper) 
                    if idx_here>0:
                        bin_upper_idx = bin_upper_idx + 1
                        
                    bin_lower_idx = next(xx[0] for xx in enumerate(xlog) if xx[1] <= bin_lower)
                        
                    bin_lower = xlog[bin_lower_idx]
                    while flag_criteria:
                        if len(y[bin_upper_idx:bin_lower_idx]) > 1:
                            std_err_y = sem(y[bin_upper_idx:bin_lower_idx])
                            half_mean = np.nanmean(y[bin_upper_idx:bin_lower_idx]) * 0.5
                            x_mean = np.nanmean(x[bin_upper_idx:bin_lower_idx])
                        else:
                            std_err_y = np.inf
                            half_mean = 0.0
                            
                        if std_err_y <= half_mean:
                            flag_criteria = False
                        else:
                            bin_lower_idx = bin_lower_idx + 1
                            
                        if bin_lower_idx >= len(x):
                            flag_criteria = False 
                            flag_cont = False
                            x_mean = np.nan
                            
                    #add stats to the arrays
                    if ~np.isnan([np.float64(x_mean), np.float64(half_mean * 2.0),np.power(np.float64(std_err_y),2.0)]).any(): #how is this possible? happen when bin_low_idx>=len(x) above?
                        binx.append(np.float64(x_mean))
                        biny.append(np.float64(half_mean * 2.0))
                        binvar.append(np.power(np.float64(std_err_y),2.0))
                        binvarlog.append(np.power(np.float64(sem(np.log(y[bin_upper_idx:bin_lower_idx]))),2.0))
                        bin_stderr_divQ.append(np.float64(sem(np.array(y[bin_upper_idx:bin_lower_idx])/np.array(x[bin_upper_idx:bin_lower_idx]))))
                        bin_x_range.append(bin_lower)
                        binnobs.append(bin_lower_idx-bin_upper_idx)
                        idx_here = idx_here + 1
                        
                else: # didnt include the last bin for now
                    flag_cont = False 
                
                if idx_here >= len(x):
                    flag_cont = False
                
            d = {'x': np.log(binx), 'y': np.log(biny)}
            df = pd.DataFrame(data=d)
    
            if regression_function == 'ln(-dqdt) = a + b * ln(Q)':                
                wls_res = smf.wls('y ~ x', data =df, weights = 1./np.array(binvarlog)).fit()  #maybe I need the variance in the log space...
                popt = [np.exp(wls_res.params[0]), wls_res.params[1] - 1] #modifying popt due to log and g(Q)*Q
        
                ci = wls_res.conf_int(alpha=0.05, cols=None)  
                tmp = ufloat(np.log(popt[0]), ci[1][0] - np.log(popt[0]))
                tmp  = uexp(tmp)
                std = tmp.std_dev
                
                popt_low = [popt[0] - std, ci[0][1] - 1] 
                popt_high = [popt[0] + std,ci[1][1] - 1]
        
            elif regression_function == 'ln(-dqdt) ~ a + b * ln(Q) + c * ln(Q)^2':

                wls_res = smf.wls('y ~ x + np.power(x,2)', data =df, weights = 1./np.array(binvarlog)).fit()  #maybe I need the variance in the log space...
                popt = [wls_res.params[0], wls_res.params[1] - 1, wls_res.params[2]] #modifying popt due to log and g(Q)*Q
        
                ci = wls_res.conf_int(alpha=0.05, cols=None)  
                popt_low = [ci[0][0], ci[0][1] - 1, ci[0][2]] 
                popt_high = [ci[1][0], ci[1][1] - 1, ci[1][2]]
            
            else:
                logger.error('Wrong regression function')
                            
            return lambda x: g_func(x, popt), popt, popt_low, popt_high, np.array(binx), np.array(biny) / np.array(binx), binnobs, np.sqrt(np.array(binvar)), bin_stderr_divQ
        
        else:
            logger.error('Wrong fittig method')
            
            
    def getgf(self, g_func, p0=None, method = None, func = None):


        method_T = np.transpose(method)    
        method_name = [x+','+y for x, y in zip(method_T[0], method_T[1])]

        self.g_func_set = {}
        self.g_func_params_set = {} 
        self.g_func_prop_wls = {}
        for idx,item in enumerate(method_name):
            self.g_func_set[item], self.g_func_params_set[(item,'mean')], self.g_func_params_set[(item,'low')], self.g_func_params_set[(item,'high')], self.g_func_prop_wls[(item,'gfuncx')], self.g_func_prop_wls[(item,'gfuncy')], self.g_func_prop_wls[(item,'nobs')], self.g_func_prop_wls[(item,'std_err')], self.g_func_prop_wls[(item,'bin_stderr_divQs')] = self._fitgfunc_set(self, g_func,p0, method_dqdt = method_T[0][idx], method_fit = method_T[1][idx], regression_function = func, uncertainty = True)

        for R in self.events:
            R.g_func, R.g_func_params, R.g_func_params_low, R.g_func_params_high,_,_,_,_,_ = self._fitgfunc_set(R, g_func,p0, method_dqdt = 'C1', method_fit = 'ols', regression_function = func, uncertainty = False)                


class event_obj(object):
    
    def __init__(self, i_start, i_end, q, t, correction):
        self.i_start = i_start
        self.i_end = i_end
        self.q = q
        self.t = t
        self.correction = correction
            
        
    def getdqdt_method(self, method = None, M = 2, grad_mult = 1.0): #alpha is M
                
        self.dq_dt = {}
        self.qh = {}
        self.th = {}
        self.corh = {}

        for _, item in enumerate(method):

            if item == 'C1': #C2
                self.qh[item] = self.q  #Q = (Q[0] + Q[1]) / 2
                self.th[item] = self.t               
                self.corh[item] = self.correction  
                self.dq_dt[item] = -np.gradient(self.q, self.t)[1:-1]  #in terms of indexes: q[+1] -  q[0]

                self.qh[item] = self.qh[item][1:-1]  #Q = (Q[0] + Q[1]) / 2
                self.th[item] = self.th[item][1:-1]               
                self.corh[item] = self.corh[item][1:-1]   
            
            elif item == 'B1': #F1
                self.dq_dt[item] = -np.diff(self.q)/np.diff(self.t)  #in terms of indexes: q[+1] -  q[0]
                self.corh[item] = (self.correction[:-1]+self.correction[1:])/2
                self.qh[item] = (self.q[:-1]+self.q[1:])/2
                self.th[item] = (self.t[:-1]+self.t[1:])/2                                  
                            
            elif item == 'holo':
                if len(self.t)>3:
                    self.dq_dt[item] = holo_diff(self.t,self.q, M = M) * -1.0                
                    self.qh[item] = self.q #[MSK] Maybe this as well.. Do I need to use the smoothed Q?
                    self.th[item] = self.t
                    self.corh[item] = self.correction  #Maybe i am missing something here if I haven't altered correction in def ... holo [MSK]
                else:
                    self.dq_dt[item] = [np.nan, np.nan]       
                    self.qh[item] = [np.nan, np.nan]
                    self.th[item] = [np.nan, np.nan]
                    self.corh[item] = [np.nan, np.nan]                
            else:
                logger.error('Wrong method of estimating dqdt')


            discard = self.dq_dt[item]<=0
            self.dq_dt[item] = self.dq_dt[item][~discard]
            self.th[item] = self.th[item][~discard]
            self.qh[item] = self.qh[item][~discard]
            self.corh[item] = self.corh[item][~discard]
            
            discard = np.isnan(self.dq_dt[item])
            self.dq_dt[item] = self.dq_dt[item][~discard]
            self.th[item] = self.th[item][~discard]
            self.qh[item] = self.qh[item][~discard]        
            self.corh[item] = self.corh[item][~discard]
    
            self.dq_dt[item] = self.dq_dt[item] * grad_mult
        