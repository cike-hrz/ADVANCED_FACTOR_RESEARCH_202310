import numpy as np
import pandas as pd

def drawdown(cum_ret:np.ndarray):
    """
    Calculate MAX Drawdown and MIN Drawdown over the 1D 
    data (dtype:ndarray).
    
    Inputs
    ------
    `cum_ret` : np.ndarray,
        Cummulative return of portfolio,should be in 
        shape of (T,) with no NAN value.
    
    Outputs
    -------
    `max_drawdown `: float,
        Max-drawdown of the input data;
    `max_dd_start` : int,
        Start date of max-drawdown period;    
    `max_dd_end` : int,
        End date of max-drawdown period;
    `second_max_drawdown` : float,
        Second max-drawdown after the period where
        max-drawdown is determined of the input data;
    `sec_max_dd_start` : int,
        Start date of second max-drawdown period;    
    `sec_max_dd_end` : int,
        End date of second max-drawdown period;
    """    
    drawdown_arr_1 = np.maximum.accumulate(cum_ret)-cum_ret
    if np.max(drawdown_arr_1) == 0:
        return (0,0,0,0,0,0)
    
    elif np.max(drawdown_arr_1) > 0:
        pointer_min_1 = np.argmax(drawdown_arr_1)
        pointer_max_1 = np.argmax(cum_ret[0:pointer_min_1])
        tmp_accumulate_arr_1 = \
            np.maximum.accumulate(cum_ret[pointer_min_1:-1])
        dd_1 = (np.max(drawdown_arr_1) \
                    / cum_ret[pointer_max_1])*100
        
        drawdown_arr_2 = tmp_accumulate_arr_1\
                            - cum_ret[pointer_min_1:-1]
        if (len(drawdown_arr_2) == 0 or np.max(drawdown_arr_2) == 0):
            return (round(dd_1,2),
                    pointer_max_1,pointer_min_1,
                    0,0,0)
        
        elif np.max(drawdown_arr_2) > 0:
            pointer_min_2 = \
                pointer_min_1 + np.argmax(drawdown_arr_2)
            pointer_max_2 = \
                pointer_min_1 + np.argmax(cum_ret[pointer_min_1:pointer_min_2])
            dd_2 = (np.max(drawdown_arr_2)\
                        / cum_ret[pointer_max_2])*100
            return (round(dd_1,2),
                    pointer_max_1,pointer_min_1,
                    round(dd_2,2),
                    pointer_max_2,pointer_min_2)

def annual_sharpe_ratio(cum_ret:np.ndarray):
    T   = len(cum_ret)
    ret = (cum_ret[-1]/cum_ret[0])**(T/252)-1
    std = np.nanstd(cum_ret)*((252/T)**(1/2))
    return round(ret/std,4)

def annual_sortino_ratio(cum_ret:np.ndarray):
    T   = len(cum_ret)
    ret = (cum_ret[-1]/cum_ret[0])**(T/252)-1
    loss_arr = cum_ret[np.insert(np.diff(cum_ret)<0,0,False)]
    T_loss = len(loss_arr)
    if T_loss!=0:
        std = np.nanstd(cum_ret)*np.sqrt((252/T_loss))
        return (round(ret/std,4), T_loss)
    else:
        return (0,0)

def net_analysis_DailyInput(portfolio:pd.DataFrame):
    """
    Inputs
    ------
    `portfolio` : pd.DataFrame,
        Should be in shape of (1,T), index should be 
        portfolio name, and columns daily timestamps.
    
    Outputs
    -------
    `result` : pd.DataFrame,
        Indicators that assess performance of assets.
    """
    ret  = np.squeeze(portfolio.values)
    ret  = np.nan_to_num(ret, 0)
    mean = np.mean(ret).round(4)
    std  = np.std(ret).round(4)
    cum_ret_arr = np.cumprod(1+ret)
    cum_ret = (cum_ret_arr[-1] - 1).round(4)
    T = ret.shape[0]
    sharpe = mean/std
    ann_ret = round((cum_ret-1)*252/T,4)
    ann_std = round(std*np.sqrt(252/T),4)
    ann_sharpe = annual_sharpe_ratio(cum_ret_arr)
    ann_sortino,T_loss = annual_sortino_ratio(cum_ret_arr)
    win_rate = round((T-T_loss)/T,4)
    max_dd, max_dd_start, max_dd_end, \
        sec_dd, sec_dd_start, sec_dd_end \
            = drawdown(cum_ret_arr)
    max_dd_duration = max_dd_end - max_dd_start
    sec_dd_duration = sec_dd_end - sec_dd_start
    timestamp = portfolio.columns
    max_dd_start = timestamp[max_dd_start].strftime('%Y-%m-%d')
    max_dd_end   = timestamp[max_dd_end].strftime('%Y-%m-%d')
    sec_dd_start = timestamp[sec_dd_start].strftime('%Y-%m-%d')
    sec_dd_end   = timestamp[sec_dd_end].strftime('%Y-%m-%d')
    
    result = pd.DataFrame([{
        '日收益率均值': mean,
        '日收益率标准差': std,
        '累计收益率': cum_ret,
        '夏普比率'  : sharpe,
        '年化收益率': ann_ret,
        '年化标准差': ann_std,
        '年化夏普比率': ann_sharpe,
        '年化索提诺率': ann_sortino,
        '最大回撤(%)': max_dd,
        '最大回撤区间': [max_dd_start,max_dd_end],
        '最大回撤天数': max_dd_duration,
        '次大回撤(%)': sec_dd,
        '次大回撤区间': [sec_dd_start,sec_dd_end],
        '次大回撤天数': sec_dd_duration,        
        '下行天数': T_loss,
        '胜率': win_rate,
    }]).T
    result.columns = portfolio.index
    
    return result



if __name__=='__main__':
    # 随机生成一行“日收益率”序列
    arr = np.random.randn(1000)
    arr = np.clip(arr,-10,10)*0.01
    date_series = pd.date_range(start='2010-01-10', periods=1000)
    df = pd.DataFrame([arr],
                      columns = date_series,
                      index = ['portfolio-1'])
    print(net_analysis_DailyInput(df))
