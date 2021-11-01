# for DL
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset # for refactoring x and y
from torch.utils.data import DataLoader # for batch submission
import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

from random import randint



def _sliding_windows(data, seq_length, future=1):
    """This takes in a dataset and sequence length, and uses this
    to dynamically create (multiple) sliding windows of data to fit
    """
    x = []
    y = []

    for i in range(len(data)-seq_length-future):
        tx = data[i:(i+seq_length)]
        ty = data[i+seq_length+future-1]
        x.append(tx)
        y.append(ty)

    return np.array(x),np.array(y)
    

# break up into Predictors and Predictands
def sliding_windows(data_window, seq_length, fut_length=1):
    """This uses the basic function _sliding_windows to create sliding
    sliding windows for datasets with large feature spaces where
    data_window.columns = [Feature1, Feature2..., Featuren, Y]
    (i.e., the final column is the 'target' value to be predicted)
    """
    it = 0
    cols = data_window.columns

    if len(cols) > 1:
        for idx in cols:
            x_0, y = _sliding_windows(data_window[idx].to_numpy(), seq_length, fut_length)
            if idx == cols[0]:
                x = np.zeros((x_0.shape[0],x_0.shape[1],len(cols)))
            x[:,:,it] = x_0
            it = it+1
    else:
        x, y = _sliding_windows(data_window.to_numpy(), seq_length, fut_length)
    return x, y
    
def delnull(x_in, y_in):
    '''
    This sub for removing columns with null values
    in_x -> input x array from the sliding_windows procecure)
    in_y -> input y array from the sliding windows procedure)
    '''
    
    # keep track of dimensions
    num_x_axes = len(x_in.shape)
    if num_x_axes > 1:
        axes_tup = []
        for shp in range(1,num_x_axes,1):
            axes_tup.append(shp)
        axes_tup = tuple(axes_tup)
    else:
        axes_tup = 0        
    
# #     print(axes_tup)

    # figure out null values
    y_boolean = np.isnan(y_in)
    x_isnan = np.sum(np.isnan(x_in),(1,2))
    x_boolean = np.zeros(x_isnan.shape)
    x_boolean[x_isnan > 0] = 1
    x_boolean = np.array(x_boolean, dtype=bool)
    
#     print(x.shape)
#     print(x_boolean.shape)
#     print(y.shape)
    
    # convert all booleans
    all_boolean = np.zeros(y_boolean.shape)
    all_boolean[x_boolean | y_boolean] = 1
    all_boolean = np.array(all_boolean, dtype=bool)

    # drop y and x values
    x_del = np.delete(x_in, all_boolean, axis=(0))
    y_del = np.delete(y_in, all_boolean, axis=(0))
    
#     print(x_del.shape)
#     print(y_del.shape)
    
    return x_del, y_del, all_boolean

def randomidx(num, num_total, num_taken_in=[]):
    '''
    randomly generates a list for filtering
        num : number of idxes 
        num_total : total number to generate range from
        num_taken_in : [], unless there are already indexes selected
    '''
    # these are the indexes to be taken out
    num_taken_out = []
    # this is the range of unique values to get
    for idx in range(num):
        rand_int = randint(0, num_total-1)
        # this chackes to see if a number is unique
        isnt_unique = True
        while isnt_unique:
            if (rand_int in num_taken_in) | (rand_int in num_taken_out):
                rand_int = randint(0, num_total-1)
            else:
                isnt_unique = False
        num_taken_out.append(rand_int)
    return num_taken_out

def selectData(data, labelist, name_ens_l, AOC_ens_l, AOC_ens_scale_l, idx_taken):
    '''
    selects data from a dataframe array, name_ens_l, AOC_ens_l, and AOC_ens_scale_l for later use
    useful for train, validation, test splits
    returns selected data as numpy array
    '''
    
    name_ens_l_idx = [name_ens_l[i] for i in idx_taken]
    AOC_ens_l_idx = np.take(AOC_ens_l, idx_taken, axis=0)
    AOC_ens_scale_l_idx = np.take(AOC_ens_scale_l, idx_taken, axis=0)
    
    # add member name list
    member_name_list = labelist[:-1]
    for i in range(len(name_ens_l_idx)):
        member_name_list.append(f'Flow_{name_ens_l_idx[i]}')
        
    data_out = data[member_name_list]
    
    # print(AOC_ens_scale_idx[0,:])
    # for i in range(len(AOC_ens_scale_idx)):
    #     x_y_idx = np.where(x[:,:,-num_params:] == AOC_ens_scale_idx[i:])
    # print(len(x_y_idx[0]), len(x_y_idx[1]), len(x_y_idx[2]))
    
    return data_out, member_name_list, name_ens_l_idx, AOC_ens_l_idx, AOC_ens_scale_l_idx # , x_y_idx, x, y, t_bool, 

def arrangeData(data, name_ens_l, AOC_ens_l, AOC_ens_scale_l, labelist, seq_length, fut_length, shuffle_it=False):
    '''
    Function for assmebling dataset for PyTorch
    Variables verbatin defined as per earlier in this script
    Useful Functions:
        # https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d
        # https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
        # convert to torch tensors
        # * This is the first step of the PyTorch procedure
        # * Sets training data equal to normalized flow data
        # * Uses the function sliding_windows to define the set of training and test data
        # * Sets variables as Torch Tensors
    '''
    # define sliding windows (note, these are defined in the globals) 
    # note - make sure target (in this case flow) is the last entry
    # note - make sure to add the scaler for the AOC
    for idx in range(len(name_ens_l)):
        # ensemble member name, Attribute of Concern (K), and scale value
        member_name = f'Flow_{name_ens_l[idx]}'
        # print(AOC_ens_l.shape)
        AOC = AOC_ens_l[idx, :]
        AOC_scale = AOC_ens_scale_l[idx, :]
        # extract data
        data_window = data[labelist[:-1]]
        # loop through all AOC members
        for l in range(len(AOC_scale)):
            data_window[f'AOC_{l}'] = AOC_scale[l]
        # append target data
        data_window['Y'] = data[member_name]
        # create sliding windows
        x_temp, y_temp = sliding_windows(data_window, seq_length, fut_length)
#         print(x_temp)
        # create x and y to pass on to next step
        if idx == 0:
            x, y = x_temp, y_temp
        else:
            x, y = np.append(x,x_temp, axis=0), np.append(y,y_temp, axis=0)
        # clean up
        del x_temp, y_temp, member_name, AOC, data_window, AOC_scale

    # reset x (i.e. don't include streamflow as a predictive variable)
    x = x[:,:,:-1]
    
    # shuffle?
    if shuffle_it:
        x, y = shuffle(x, y)

    # drop na values
    x, y, t_bool = delnull(x, y)

    if len(x.shape) < 3:
        x = np.expand_dims(x, axis=2)
        
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    
    return x, y, t_bool 

def data_tensor(x, y):
    '''
    passess array x and y attributes
    converts to tensor
    '''
    dataX = Variable(torch.Tensor(x))
    dataY = Variable(torch.Tensor(y))
    
    return dataX, dataY
    
def trainloader(dataX, dataY, bs):
    '''
    passess tensor x and y and batch size bs
    returns trainloader 
    '''
    # playing with DataLoader and TensorDataset
    train_ds = TensorDataset(dataX, dataY)
    train_dl = DataLoader(train_ds, batch_size = bs)
    
    return train_ds, train_dl

def moduleLoad(load_Path):
    '''
    for loading other lstm models
    Antiquated - 09072021
    '''
    with open(load_PATH+'/params.txt') as f:
        lines = f.readlines()
        print('model parameter description \n', lines, '\n')
        
    num_classes = int(lines[-1][-4:-1].replace('=',''))
    num_layers = int(lines[-2][-4:-1].replace('=',''))
    hidden_size = int(lines[-3][-4:-1].replace('=',''))
    input_size = int(lines[-4][-4:-1].replace('=',''))
    seq_length = int(lines[-9][-3:-1].replace('h',''))
    print(num_classes, num_layers, hidden_size, input_size, seq_length)
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
    lstm.load_state_dict(torch.load(load_PATH+'/model.txt'))
    return lstm

def recon_Y(t_bool,y_in):
    '''
    A sub for reconstructing time series of Y
    '''
    y_out = np.empty((t_bool.shape[0],1))
    
    it = 0
    for idx in range(t_bool.shape[0]):
        if t_bool[idx]:
            y_out[idx,0] = np.nan
        else:
            y_out[idx,0] = y_in[it, 0]
            it = it + 1
    
    return y_out

def _moduleLoad(load_PATH, ret_seq_info=False):
    '''
    for loading other lstm models and reading from model parameter file
    load_PATH : file location of parameter file
    ret_seq_info : boolean (set to False by default) for returning just the sequence length information
        otherwise returns an lstm model object preloaded with architecture and training
    '''
    with open(load_PATH+'/params.txt') as f:
        lines = f.readlines()
        # print('model parameter description \n', lines, '\n')

    num_classes = int(lines[-1][-4:-1].replace('=',''))
    num_layers = int(lines[-2][-4:-1].replace('=',''))
    hidden_size = int(lines[-3][-4:-1].replace('=',''))
    input_size = int(lines[-4][-4:-1].replace('=',''))
    fut_length = int(lines[-8][-3:-1].replace('h',''))
    seq_length = int(lines[-9][-3:-1].replace('h',''))

    # print(num_classes, num_layers, hidden_size, input_size, seq_length)
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
    lstm.load_state_dict(torch.load(load_PATH+'/model.txt'))
    if ret_seq_info:
        return seq_length, fut_length
    else:
        return lstm
        
def findMinMaxidx(AOC_ens_scale_l):
    '''
    Finds Min and Max of idx for exclusion (if desired)
    '''
    # find indices of idx
    idx_find = np.where((AOC_ens_scale_l == 0.) | (AOC_ens_scale_l == 1.))[0].tolist()
    
    # remove replicates
    idx_find = list(set(idx_find))
    return idx_find
