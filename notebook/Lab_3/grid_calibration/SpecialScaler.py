import numpy as np

def scale_params(params, lb, ub):
    l = len(params)
    res = np.zeros(l)
    for i in range(l):
        res[i] = (2*params[i] - (ub[i] + lb[i])) / (ub[i] - lb[i])        
    return res

def back_params(params, lb, ub):
    l = len(params)
    res = np.zeros(l)
    for i in range(l):
        res[i] = 0.5*(ub[i]-lb[i])*params[i] + 0.5*(ub[i]+lb[i])
    return res
