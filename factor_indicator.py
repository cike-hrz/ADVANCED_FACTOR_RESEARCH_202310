import pandas as pd
import numpy as np

# wl-zcy IC definition
def ic(R:np.ndarray, X:np.ndarray):
    '''
    input
    -----
    `R` : np.ndarray;
    Shape of (N,T), returns(day/week/month) of investment target.
    
    `X` : np.ndarray;
    Shape of (N,T), factor values of target, should be in format
    of matrix; called 'macro factors' if in shape of (1,T), and
    is roughly considered np.tile(macro,(N,1)) before-head input
    in this function.
    
    output
    ------
    `ic_arr` : np.ndarray;
    Shape of (T,), IC values defined in Wu's Quant courses.
    '''
    R_bar = np.roll((R - np.mean(R,axis=0)),shift=-1,axis=1)
    X_bar = X - np.mean(X,axis=0)
    ic_arr = np.sum(R_bar*X_bar,axis=0)\
        /np.sqrt((R_bar**2).sum(axis=0)*(X_bar**2).sum(axis=0))
    return ic_arr

def rank_ic(R:np.ndarray, X:np.ndarray):
    R_bar = np.roll((R - np.mean(R,axis=0)),shift=-1,axis=1)
    rankX = (X.T).argsort().argsort().T
    X_bar = rankX - np.mean(rankX,axis=0)
    rank_ic_arr = np.sum(R_bar*X_bar,axis=0)\
        /np.sqrt((R_bar**2).sum(axis=0)*(X_bar**2).sum(axis=0))
    return rank_ic_arr

def cum_ic(R:np.ndarray, X:np.ndarray):
    return ic(R,X).cumsum()

def cum_rank_ic(R:np.ndarray, X:np.ndarray):
    return rank_ic(R,X).cumsum()

def group_reg(R:np.ndarray, X:np.ndarray, group:int):
    N = R.shape[1]
    num = N//group
    matrix_R_ascending_X = np.take_along_axis(R,X.argsort(axis=0),axis=0)
    tmp = list()
    for k in range(group):
        if k==group-1:
            group_k = matrix_R_ascending_X[k*num:N].mean(axis=0)
        else:
            group_k = matrix_R_ascending_X[k*num:(k+1)*num].mean(axis=0)
        tmp.append(group_k)
    cum_ret = np.cumprod(np.stack(tmp)+1,axis=1)
    return cum_ret

# APM IC definition


class 