# ANTIQUATED

'''
This script encapsultes the entire sbi workflow
    from setting LSTM globals and training LSTM
    to running sbi
'''



import pandas as pd
import pickle
import numpy as np
from sklearn.utils import shuffle
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # for refactoring x and y
from torch.utils.data import DataLoader # for batch submission
from torch.autograd import Variable

from datetime import datetime
from random import randint

import sys
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from lstmutils import _sliding_windows, sliding_windows, delnull, LSTM, randomidx, arrangeData, selectData, data_tensor, trainloader
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/')
from lstm_build import runLSTM

'''
Set LSTM globals
'''
lstm_name = '0830_01_test'
save_path = f'/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_name}/' 
save = True
shuffle_it_in = False
module_load = False

# for choosing which huc to train on
ensemble_name = '0819_01'
ensemble_path = f'/home/qh8373/SBI_TAYLOR/data/03_ensemble_out/_ensemble_{ensemble_name}/'

data = pd.read_csv(f'{ensemble_path}df_l_scaled.csv')
AOC_ens_l = pickle.load(open(f'{ensemble_path}AOC_ens_l.pkl', 'rb')) #numpy
AOC_ens_scale_l = pickle.load(open(f'{ensemble_path}AOC_ens_scale_l.pkl', 'rb')) #numpy
labelist = pickle.load(open(f'{ensemble_path}labelist.pkl', 'rb')) #list
name_ens_l = pickle.load(open(f'{ensemble_path}member_name_ens.pkl', 'rb')) #list
num_params = AOC_ens_l.shape[1] #usually twoa

# sliding sequence length (by default, the same as windows set in globals for previous step)
seq_length = 7 # window of history (num days back)
fut_length = 1 # when prediction (num days future)

# batch size
bs = 20

# For LSTM
num_epochs = 100 # number of times iterating through the model
learning_rate = 0.001 # rate of learning
input_size = 10 # nodes on the input (should be number of features)
hidden_size = 10 # number of nodes in hidden layer 
num_layers = 1 # number of hidden layers
num_classes = 1 # nodes in output (should be 1)

'''
Arrange Data
'''
# fraction of data to include in train, validate, and test datasets
train_frac, val_frac, test_frac = 0.7, 0.2, 0.1 # must sum to 1
train_num, val_num, test_num = int(train_frac*len(name_ens_l)), int(val_frac*len(name_ens_l)), int(test_frac*len(name_ens_l))
test_idx = randomidx(num=test_num, num_total=np.sum([train_num, val_num, test_num]))
train_val_idx = randomidx(num=np.sum([train_num, val_num]), num_total=np.sum([train_num, val_num, test_num]), num_taken_in=test_idx)

# sample data from test and train_val periods
data_out_test, member_name_list_test, name_ens_l_idx_test, AOC_ens_l_idx_test, AOC_ens_scale_l_idx_test = selectData(data, 
            labelist, name_ens_l, AOC_ens_l, AOC_ens_scale_l, test_idx)
data_out_train_val, member_name_list_train_val, name_ens_l_idx_train_val, AOC_ens_l_idx_train_val, AOC_ens_scale_l_idx_train_val = selectData(data, 
            labelist, name_ens_l, AOC_ens_l, AOC_ens_scale_l, train_val_idx)

# test data only subset into numpy arrays
x_test, y_test, t_bool_test = arrangeData(data_out_test, name_ens_l_idx_test, 
                          AOC_ens_l_idx_test, AOC_ens_scale_l_idx_test,
                          labelist, seq_length, fut_length, shuffle_it=shuffle_it_in)
dataX_test, dataY_test = data_tensor(x_test, y_test)

'''
Save LSTM globals
'''
if save:
    try:
        os.mkdir(save_path)
    except:
        print('warning: file exists')
        pass
    print('saving')
    file = open(save_path+"params.txt", "w")
    file.write(f'Ensemble Name {ensemble_name} \n')
    file.write(f'Ensemble Members {name_ens_l}\n')
    file.write(f'Run Datetime {datetime.now()}\n')
    file.write('\n')
    file.write(f'Batch Size {bs}\n')
    file.write(f'Sequence Length {seq_length}\n')
    file.write(f'Prediction Length {fut_length}\n')
    file.write(f'test fraction {test_frac}\n')
    file.write(f'LSTM Params: Number of epochs - {num_epochs}\n')
    file.write(f'LSTM Params: learning rate - {learning_rate}\n')
    file.write(f'LSTM Params: input_size = {input_size}\n')
    file.write(f'LSTM Params: hidden_size = {hidden_size}\n')
    file.write(f'LSTM Params: num_layers = {num_layers}\n')
    file.write(f'LSTM Params: num_classes = {num_classes}\n')
    file.close()
    
    file = open(save_path+"params_2.txt", "w")
    file.write(f'train fraction {train_frac}\n')
    file.write(f'validate fraction {val_frac}\n')
    file.write(f'test fraction {test_frac}\n')
    file.write(f'train number {train_num}\n')
    file.write(f'validate number {val_num}\n')
    file.write(f'test number {test_num}\n')
    file.close()

    
    with open(save_path+"test_idx.pkl", "wb") as fp:   
        pickle.dump(test_idx, fp)
        
    with open(save_path+"train_val_idx.pkl", "wb") as fp:   
        pickle.dump(train_val_idx, fp)
    
    test_title = ['data_out_test', 'member_name_list_test', 'name_ens_l_idx_test', 'AOC_ens_l_idx_test', 'AOC_ens_scale_l_idx_test', 
                'x_test', 'y_test', 't_bool_test',
                'dataX_test', 'dataY_test']
    test_data = [data_out_test, member_name_list_test, name_ens_l_idx_test, AOC_ens_l_idx_test, AOC_ens_scale_l_idx_test, 
                x_test, y_test, t_bool_test,
                dataX_test, dataY_test]
    for td in range(len(test_data)):
        if len(test_title) != len(test_data):
            break
        with open(f'{save_path}{test_title[td]}.pkl', "wb") as fp:   
            pickle.dump(test_data[td], fp)
            
    train_val_title = ['data_out_train_val', 'member_name_list_train_val', 'name_ens_l_idx_train_val', 'AOC_ens_l_idx_train_val', 'AOC_ens_scale_l_idx_train_val']
    train_val_data = [data_out_train_val, member_name_list_train_val, name_ens_l_idx_train_val, AOC_ens_l_idx_train_val, AOC_ens_scale_l_idx_train_val]
    for tvd in range(len(train_val_title)):
        if len(train_val_title) != len(train_val_data):
            break
        with open(f'{save_path}{train_val_title[tvd]}.pkl', "wb") as fp:   
            pickle.dump(train_val_data[tvd], fp)


'''
Train ensemble of LSTMs or load them
'''
setnum = 1
for i in range(setnum):
    save_path_out = save_path+'a/'
    
    
    if module_load:
        print('nothing')
    else:
        lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
        criterion = torch.nn.MSELoss()# MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate) # Adam as optimizer
        
        runLSTM(lstm, criterion, optimizer, train_dl, save, save_path_out)

'''
Assess fit of LSTMs to reality
'''






