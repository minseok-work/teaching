'''
Created on Dec 24, 2018

@author: mkim1
'''

import numpy as np

def one_minus_em(evaluation, simulation, efficiency_measure = 'KGE', return_all=False, log_trans = False): #Modified from spotpy 1.3.27
    """
    Kling-Gupta Efficiency

    Corresponding paper: 
    Gupta, Kling, Yilmaz, Martinez, 2009, Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling

    output:
        kge: Kling-Gupta Efficiency
    optional_output:
        cc: correlation 
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """

    if log_trans:
        simulation[np.where(simulation == 0.0)] = 0.000001
        evaluation[np.where(evaluation == 0.0)] = 0.000001    
        
        simulation, evaluation = np.copy(np.log(simulation)),np.copy(np.log(evaluation))
    else:
        simulation, evaluation = np.copy(simulation),np.copy(evaluation)
        
    
    simulation[np.where(np.isnan(evaluation))] = np.nan
    evaluation[np.where(np.isnan(simulation))] = np.nan    

#     simulation[np.where(evaluation <= np.log(thres_q))] = np.nan
#     evaluation[np.where(evaluation <= np.log(thres_q))] = np.nan    
# 
#     evaluation[np.where(simulation <= np.log(thres_q))] = np.nan    
#     simulation[np.where(simulation <= np.log(thres_q))] = np.nan
    

    simulation = simulation[np.where(~np.isnan(simulation))]
    evaluation = evaluation[np.where(~np.isnan(evaluation))] 

#    print len(simulation), len(evaluation)

#    plt.figure()
#    plt.plot(simulation)
#    plt.plot(evaluation)
#    plt.show()
    
    if len(evaluation) == len(simulation):

        if efficiency_measure == 'KGE':
            cc = np.corrcoef(evaluation, simulation)[0, 1]
            alpha = np.std(simulation) / np.std(evaluation)
            beta = np.sum(simulation) / np.sum(evaluation)
            kge = 1 - np.sqrt((cc - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            if return_all:
                return kge, cc, alpha, beta
            else:
                return 1 - kge
        elif efficiency_measure == 'NSE':
            return sum((simulation-evaluation)**2)/sum((evaluation-np.mean(evaluation))**2)
    else:
        raise SyntaxError("evaluation and simulation lists does not have the same length.")
        return np.nan    
    
    
def Bucket_ode(Q,ttt,params_here,nt,jj,et,Unit_conv, g_func_here, thres_q = 0.0):
    if (Q<thres_q):       
        Q = thres_q
        
    if ((int(ttt*Unit_conv<(nt-1))) and  (int(ttt*Unit_conv)>=0)): #and (Q>=thres_q)): 
        jhere = np.interp(ttt,[int(ttt*Unit_conv),int(ttt*Unit_conv)+1],[jj[int(ttt*Unit_conv)], jj[int(ttt*Unit_conv)+1] ])
        ethere = np.interp(ttt,[int(ttt*Unit_conv),int(ttt*Unit_conv)+1],[et[int(ttt*Unit_conv)], et[int(ttt*Unit_conv)+1] ])
        
#        if functype == 'Power':
#            dQdt = g_func_here(Q, params_here)*(jhere - ethere - Q)
#        elif functype == '2nd-order':
#            dQdt = g_func_2nd(Q, params_here)*(jhere - ethere - Q)                
        dQdt = g_func_here(Q, params_here)*(jhere - ethere - Q)
    else:
        dQdt = 0.0 

                      
    return dQdt



def plot_JQET(t,j,q,et,axT,fill_bet_J = True):
    
    if fill_bet_J:
        axT.fill_between(t,j.fillna(0), step="pre", alpha=0.4,  color = '#0089d0')

    p1, = axT.step(t,j.fillna(0), color = '#0089d0', label = 'J')
    axT.set_ylim([np.nanmax(j)*2.0,0.0])

    axT2 = axT.twinx()    
    p2, = axT2.plot(t,q, color = 'grey', label = 'Q')
    p3, = axT2.plot(t,et, color = '#f37021', label = 'ET')
    axT2.set_ylim([0.0,np.nanmax(q)*2.0])

    return axT, axT2, p1, p2, p3