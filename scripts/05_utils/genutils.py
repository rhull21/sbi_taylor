import numpy
import torch
from parflowio.pyParflowio import PFData

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def PFread(pfb_data):
    '''
    read in pfb 'data' and get numpy array
    optionally - will read in a path-like string and convert pfb_data fo PFData type, else leave as is
    '''
    if type(pfb_data) == str:
        pfb_data = PFData(pfb_data)
    pfb_data.loadHeader()
    pfb_data.loadData()
    data_arr = pfb_data.moveDataArray()
    pfb_data.close()
    return data_arr
    
def plot_stuff(df_in, path = '.', save=False, same=True, ylabel='empty', title=None):
    '''
    To be used to plot something, based on date axis
    df_in : name of dataframe
    path : name of path (for saving only)
    save : True is saving, False if not 
    same : Plot everything on one, or plot on multiple
    ylabal : For 'same' only, the label
    title : title for plot
    ''' 
    
    import warnings
    warnings.filterwarnings("ignore")
    
    if same:
        plt.subplots(figsize=(10,5))
        for label in df_in.columns:
            # time, data, color
            
            plt.plot(df_in.index, df_in[label], label=label, alpha=0.5)
            plt.xlabel('time')
            plt.ylabel(ylabel)
            if title is None:
                plt.title('empty')
            else:
                plt.title(title)
        
        plt.legend(loc = 2)
        if save:
            plt.savefig(path+ylabel+'.png')
            plt.savefig(path+ylabel+'.eps', format='eps')
        plt.show()
        plt.close()
            
    else:
        for label in df_in.columns:
            plt.subplots(figsize=(10,5))
            plt.plot(df_in.index, df_in[label])
            plt.xlabel('time')
            plt.ylabel(label)
            if title is None:
                plt.title('empty')
            else:
                plt.title(title)
            if save:
                plt.savefig(path+label+'.png')
                plt.savefig(path+label+'.eps', format='eps')
            plt.show()
            plt.close()
    return None
            
def seriesLength(dataY, member_num):
    '''
    Takes --
    dataY : Tensor or numpy array containing y data (i.e. dataY_test)
    member_num : number of members in dataY (i.e. test_num)
    Returns --
    The length of a timeseries after window creation
    '''
    num_rows = dataY.shape[0]
    series_len = num_rows/member_num
    series_len_int = int(series_len)
    if series_len / series_len_int != 1:
        print('non-whole series Length number')
    return series_len_int

def _numDims(value):
    '''
    returns number of dimensions
    value : can pass scalar or array
    returns scalar value of number of dimensions
    '''
    try:
        num_dims = len(value)
    except:
        num_dims = 1
    return num_dims
    
def convertNumpy(data_arr, toTorch=True):
    '''
    checks to see if an array is a numpy array or a torch array
    data_arr : A torch or numpy array
    toTorch : (Default True) Converts to torch
        if False, converts to numpy
        
    HELP: 
        https://stackoverflow.com/questions/18922407/boolean-and-type-checking-in-python-vs-numpy
        https://medium.com/axinc-ai/conversion-between-torch-and-numpy-operators-ce189b3882b1
    '''
    if toTorch:
        if not isinstance(data_arr, torch.Tensor):
            data_arr = torch.from_numpy(data_arr)
    
    else:
        if not isinstance(data_arr, numpy.ndarray):
            data_arr = data_arr.detach().numpy()
            
    return data_arr
            
    
    
    
    
