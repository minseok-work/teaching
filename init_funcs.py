import numpy as np
from uncertainties import unumpy

from scipy.special import erf
from scipy.special import erfi
from scipy.special import erfinv


def g_func_pow(Qt, params_here):
    return params_here[0] * np.power(Qt,params_here[1])  

def g_func_2nd(Q,params_here):        
    return np.exp(params_here[0] + params_here[1] * np.log(Q) + params_here[2] * np.power(np.log(Q),2.0))

def ERF(QQ,params_here):
    return unumpy.erf((-1. + params_here[1] + 2. * params_here[2] * np.log(QQ))/(2.*unumpy.sqrt(params_here[2])))

def ERFi(QQ,params_here):
    return np.array([mp.erfi((-1. + params_here[1] + 2. * params_here[2] * mp.log(QQQ))
                    /(2.*mp.sqrt(-1. * params_here[2]))) for QQQ in QQ])
        
def SQ_2nd_order_ERF(QQ, params_here): #for Q0
    return 1./(2.*unumpy.sqrt(params_here[2])) * unumpy.exp((-1.*params_here[0] + (np.power(params_here[1]-1.0,2.0))/(4.*params_here[2]))) *np.sqrt(np.pi)* ( 1. + ERF(QQ,params_here))

def SQ_2nd_order_ERFi(QQ, params_here):
#    return 1./(2.*np.sqrt(-1.0*params_here[2])) * np.exp(-1.*params_here[0] + np.power(params_here[1]-1.0,2.0)/(4.*params_here[2])) *np.sqrt(np.pi)* (ERFi([QQ[0]],params_here)-1.0* ERFi(QQ,params_here))
    return [np.trapz(1/g_func_2nd(QQ[:i+1],params_here),QQ[:i+1]) for i in range(len(QQ))]

def SQ_power_original(QQ, params_here):
    return  1./params_here[0] * 1./(2. - (params_here[1] + 1.)) * np.power(QQ,2.0 - (params_here[1] + 1.))  
#    return  1./params_here[0] * 1./(2. - (params_here[1] + 1.)) * np.power(QQ,2.0 - (params_here[1] + 1.))  

def SQ_power(QQ, params_here):
    return  SQ_power_original(QQ,params_here) - SQ_power_original(QQ[0],params_here)

def SQ_2nd_order(QQ,params_here):
    if params_here[2]>0:
        return SQ_2nd_order_ERF(QQ,params_here) - SQ_2nd_order_ERF(QQ[0],params_here)
    else:
        return SQ_2nd_order_ERFi(QQ,params_here)