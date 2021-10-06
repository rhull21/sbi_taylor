import numpy as np

def _q_Mean(y):
    '''
    Mean Annual Discharge
    '''
    # print('Mean Annual Discharge')
    return y.mean()
    
def _q_Mean_W(y):
    '''
    Mean Winter Discharge
    (Need to think about indexing here)
    '''
    print('Mean Winter Discharge')
    print('Method not available yet')
    return None
    
def _q_95(y):
    '''
    95 Quantile Flow
    '''
    # print('95 Quantile Flow')
    return np.quantile(y, 0.95)
    
def _q_Date_Half(y):
    '''
    Mean Half Flow Date
    
    (Court 1962) 'date on which cumulative discharge since October first reaches
        half of the annual discharge'
    
    Edited from `np.where(y == _q_any(y, q=0.5))[0][0]` -- 07092021
    '''
    # print('Mean Half Flow Date')
    # np.where(y == _q_any(y, q=0.5))[0][0]
    sumy = np.sum(y)
    cumsumy = np.cumsum(y)
    return np.where(cumsumy >= sumy/2)[0][0]
    
def _q_05(y):
    '''
    5 Quantile Flow
    '''
    # print('5 Quantile Flow')
    return np.quantile(y, 0.05)
    
def _q_any(y, q=0.5):
    '''
    Any Quantile Flow (default = 50)
    '''
    return np.quantile(y, q)
    
def _q_Date_Peak(y):
    '''
    Max Flow Date
    07192021 - duplicate of _q_peak_time
    '''
    return np.where(y == y.max())[0][0]
    
def _q_99(y):
    '''
    99 Quantile Flow
    Added 07122021 to see if it makes an impact
    '''
    return np.quantile(y, 0.99)
    
def _q_peak_flow(y):
    '''
    Hydrograph Peak Outflow (9 - q_peak_flow)
    '''
    return y.max()

def _q_peak_time(y):
    '''
    Hydrograph Peak Time (10 - q_peak_time)
    07192021 - duplicate of _q_Date_Peak
    '''
    return np.where(y == y.max())[0][0]

def _q_flow_total(y):
    '''
    Hydrograph Total Outflow (11 - q_flow_total)  
    '''
    return y.sum()
 
    
def _operations(y,typ, q=0.5):
    '''
    this is a wrapper that selects summary statistics methods based on user definition
    '''
    return {
        1: lambda: _q_Mean(y),
        2: lambda: _q_Mean_W(y),
        3: lambda: _q_95(y),
        4: lambda: _q_Date_Half(y),
        5: lambda: _q_05(y),
        6: lambda: _q_any(y, q),
        7: lambda: _q_Date_Peak(y),
        8: lambda: _q_99(y),
        9: lambda: _q_peak_flow(y),
        10: lambda: _q_peak_time(y),
        11: lambda: _q_flow_total(y)
    }.get(typ, lambda: 'Not a valid operation')()


def summary(y, typ=1):
    '''
    User-Defined 'Summary Statistics'
    y = series of time-domain streamflow data numpy array shape `(n_days,)`
    typ = Integer type of streamflow signature see dictionary [Default=1, `_q_Mean()`]
    '''
    # generate statistic based off of defined input
    stat = _operations(y,typ)
    return stat


def setStatSim(y_o, stat_typ):
    '''
    stat_typ
    '''
    stat_sim = []
    for stat in stat_typ:
        stat_sim.append(summary(y_o, typ=stat))
    return stat_sim
