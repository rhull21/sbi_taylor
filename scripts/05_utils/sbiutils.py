# for machine learning
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # for refactoring x and y
from torch.utils.data import DataLoader # for batch submission
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Independent, Uniform
from torch.distributions.log_normal import LogNormal

# for SBI
from sbi import utils as utils
from sbi import analysis as analysis
from sbi import inference
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.types import Array, OneOrMore, ScalarFloat

import numpy as np
from datetime import datetime
import sys
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from summaryutils import summary, setStatSim


def retStatTyp(stat_method, stat_typ=None, out_dim=None, embed_type=None):
    '''
    stat_method : 'summary', 'full', 'embed' - method for choosing statistics
    stat_typ : None, but add numpy array (like np.array([])
        of potentail summary statistics for state_method=='summary'
    out_dim : None, but should be an integer > 0 for stat_method=='embed'
    embed_type : None, but 'MLP', 'RNN', 'CNN' if stat_method=='embed'
    
    if stat_method:
        stat_typ is passed directly
    elif stat_method 'summary', this defines which 
        type of summary statistics to use (must define)
    elif stat_method 'embed', must have out_dim and embed_type
    '''
    if stat_method == 'summary':
        if (type(stat_typ)==str) or (out_dim is not None) or (embed_type is not None):
            print('No valid summary stats passed')
            return None
        else:
            stat_typ = stat_typ
    elif stat_method == 'full':
        if (stat_typ is not None) or (out_dim is not None) or (embed_type is not None):
            print('Invalid non-null values passed')
            return None
        else:
            stat_typ = stat_method
    elif stat_method == 'embed':
        if (stat_typ is not None) or (out_dim is None) or (embed_type is None):
            print('Invalid combo of null and non-null values passed')
            return None
        else:
            stat_typ = f'{stat_method}_{embed_type}_{out_dim}'
    else:
        stat_typ = 'No Valid Method Passed'
    
    return stat_typ
    

def parseListDf(list_df_cond):
    '''
    Harvest data from LSTM ensemble list
    '''
    DataX_test = list_df_cond[0].loc['test']['DataX']
    DataY_test = list_df_cond[0].loc['test']['DataY']
    series_len = list_df_cond[0].loc['test']['series_len'].min()
    lstm_out_list = []
    for m in range(len(list_df_cond)):
        lstm_out_list.append(list_df_cond[m].loc['train']['lstm_out'])
    
    return DataX_test, DataY_test, series_len, lstm_out_list


def parseUniqueParams(DataX_test, series_len):
    '''
    Create list of observations summarized by stat_typ
    '''
    all_length = DataX_test.shape[0]
    num_params = DataX_test.shape[2]-8 # where 8 is the number of forcings
    # print(num_params)
    num_unique = int(all_length / series_len)
    # print(num_unique)
    unique_params = torch.empty(num_unique, num_params)
    for i in range(num_unique):
        unique_params[i, :] = DataX_test[i*series_len, 0, -num_params:]
        # print(unique_params[i, :])
    
    #return DataX_test with null values in params
    DataX_test = DataX_test[:series_len,:,:]
    DataX_test[:,:,-num_params:] = np.nan
        
    return unique_params, num_params, num_unique, DataX_test
    
'''
create statistics summarized by stat_typ
'''
def reshape_y(y_in):
    '''
    Takes a one-dimensional numpy array in representing a timeseries (num_days,)
    Puts out a 2D torch arry (1,num_days) for inference
    Due to concerns with Shape 0806
    
    0906 - can also be used with torch tensors to reshape
    '''
    y_out = y_in.T.reshape(1,-1)
    return torch.tensor(y_out)


'''
prep y_hat
'''
def createYHat(y_hat, stat_method, stat_typ=None, embed_type=None):
    '''
    Create y_hat values to take sbi slices against observations
    '''
    if stat_method == 'summary': 
        stat_test = setStatSim(y_hat, stat_typ)
    elif stat_method == 'full':
        stat_test = reshape_y(y_hat)
    elif stat_method == 'embed':
        stat_test = reshape_y(y_hat)
    # print(f'True y_hat :', stat_test)
    return stat_test
    
def createYHatList(DataY_test, series_len, num_unique, stat_method, stat_typ=None, embed_type=None):
    '''
    Create a list of 'observations' to call on for later slicing
    '''
    unique_series = []
    for n in range(num_unique):
        y_hat = DataY_test[n*series_len:(n+1)*series_len,:]
        y_hat_out = createYHat(y_hat, stat_method, stat_typ=stat_typ, embed_type=embed_type)
        unique_series.append(y_hat_out)
    return unique_series

'''
set theta value on import data
'''
def setTheta(DataX, theta):
    '''
    DataX : Data array where last len(theta) values are theta and need set
    theta : theta values to set
    '''
    num_theta = len(theta)
    DataX[:,:,-num_theta:] = theta
    return DataX

