import pandas as pd
import pickle
import numpy as np
from sklearn.utils import shuffle
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # for refactoring x and y
from torch.utils.data import DataLoader # for batch submission
from torch.autograd import Variable

from sklearn.metrics import r2_score, mean_squared_error

from datetime import datetime
from random import randint

import sys
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from lstmutils import _sliding_windows, sliding_windows, delnull, randomidx, arrangeData, selectData, data_tensor, trainloader, findMinMaxidx
from assessutils import compute_stats
from genutils import seriesLength, plot_stuff
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/')
from lstm_build_utils import LSTM, runLSTM, evallstm


'''
This SCRIPT builds and tests an ensemble of LSTM members for use in inference
    * To be used in conjunction with 
    `/home/qh8373/SBI_TAYLOR/sbi_taylor/runs/lstm_sbi.py` script
'''

def buildLSTM(lstm_name, save_path, save, shuffle_it_in, num_members, ensemble_name, ensemble_path, random_flag=True, load_test_idx=False, lstm_load_path=None):
    '''
    This function builds LSTM (note the hyper parameters are here-in contained, some things that should be broken up)
    ------
    build LSTM
    1. Define HyperParmaeters
    2. Arrange Data
    3. Save LSTM Build Information
    4. Train test routine for LSTMs
        1. Randomly set train / validation splits
        2. Train LSTM
    '''

    # '''
    # Define HyperParameters
    # '''  

    data = pd.read_csv(f'{ensemble_path}df_l_scaled.csv')
    AOC_ens_l = pickle.load(open(f'{ensemble_path}AOC_ens_l.pkl', 'rb')) #numpy
    AOC_ens_scale_l = pickle.load(open(f'{ensemble_path}AOC_ens_scale_l.pkl', 'rb')) #numpy
    labelist = pickle.load(open(f'{ensemble_path}labelist.pkl', 'rb')) #list
    name_ens_l = pickle.load(open(f'{ensemble_path}member_name_ens.pkl', 'rb')) #list
    num_params = AOC_ens_l.shape[1] #usually twoa
    
    # sliding sequence length (by default, the same as windows set in globals for previous step)
    seq_length = 14 # window of history (num days back)
    fut_length = 1 # when prediction (num days future)
    
    # batch size
    bs = 50
    
    # For LSTM
    num_epochs = 300 # number of times iterating through the model
    learning_rate = 0.001 # rate of learning
    input_size = 10 # nodes on the input (should be number of features)
    hidden_size = 10 # number of nodes in hidden layer 
    num_layers = 1 # number of hidden layers
    num_classes = 1 # nodes in output (should be 1)
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
    criterion = torch.nn.MSELoss()# MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate) # Adam as optimizer

    # '''
    # Arrange Data
    # -- (add functions?)
    # '''
    # fraction of data to include in train, validate, and test datasets
    train_frac, val_frac, test_frac = 0.7, 0.2, 0.1 # must sum to 1
    train_num, val_num, test_num = int(train_frac*len(name_ens_l)), int(val_frac*len(name_ens_l)), int(test_frac*len(name_ens_l))
    
    # randomly | (removing 0-1 from test selection set)
    if random_flag == False:
        reserve_idx = findMinMaxidx(AOC_ens_scale_l)
    else:
        reserve_idx = []
    
    
    # set up the idxs 
    if load_test_idx:
        test_idx = pickle.load(open(f'{lstm_load_path}test_idx.pkl', 'rb')) # list
    else:
        test_idx = randomidx(num=test_num, num_total=np.sum([train_num, val_num, test_num]), num_taken_in=reserve_idx)
    
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
    
    # define the length of a time series after window-creation
    series_len = seriesLength(y_test, test_num)

    # '''
    # Save LSTM **Build** Information
    # '''
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
        file.write(f'Criterion {criterion}\n')
        file.write(f'Batch Size {bs}\n')
        file.write(f'Sequence Length {seq_length}\n')
        file.write(f'Prediction Length {fut_length}\n')
        file.write(f'Optimizer {optimizer}\n')
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
        file.write(f'load test idx? {load_test_idx}\n')
        file.write(f'idx from where? {lstm_load_path}\n')
        file.close()
    
        file = open(save_path+"series_len.txt", "w")
        file.write(f'series_len (i.e. number of days)\n')
        file.write(f'{series_len}')
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
                
        with open(save_path+"lstm_empty.pkl", "wb") as fp:   # empty LSTM model
            pickle.dump(lstm, fp)
                
        
    # '''
    # Train / Test routine of LSTMs
    # -- (implement validation step?)
    # '''
    list_df_cond = []
    for member in range(num_members):
        start_new = datetime.now()
        
        member_string = "{:02d}".format(member)
        lstm_member_name = f'{lstm_name}_{member_string}'
        print(lstm_member_name)
        # set save path
        save_path_out = save_path+f'{lstm_member_name}/' # edit to do this automatedly
        
        # '''
        # Randomly set train and validation split
        # '''
        # set idxs from train, validation split, with or without reserve_idx (note this is a strong sort)
        
        val_idx = randomidx(num=val_num, num_total=np.sum([train_num, val_num, test_num]), num_taken_in=(test_idx+reserve_idx))
        train_idx = randomidx(num=train_num, num_total=np.sum([train_num, val_num, test_num]), num_taken_in=(test_idx+val_idx))
        
        # print(type(reserve_idx))
        # reserve_idx.sort(reverse=True)
        # print(reserve_idx)
        # print('/n', len(test_idx))
        # test_idx.sort(reverse=True)
        # print(test_idx)
        # print('/n', len(val_idx))
        # val_idx.sort(reverse=True)
        # print(val_idx)
        # print('/n', len(train_idx))
        # train_idx.sort(reverse=True)
        # print(train_idx)
                
        # sample data from train, validation periods
        data_out_train, member_name_list_train, name_ens_l_idx_train, AOC_ens_l_idx_train, AOC_ens_scale_l_idx_train = selectData(data, 
                    labelist, name_ens_l, AOC_ens_l, AOC_ens_scale_l, train_idx)
        data_out_val, member_name_list_val, name_ens_l_idx_val, AOC_ens_l_idx_val, AOC_ens_scale_l_idx_val = selectData(data, 
                    labelist, name_ens_l, AOC_ens_l, AOC_ens_scale_l, val_idx)
                    
        # train data only subset into numpy arrays, prep for training
        x_train, y_train, t_bool_train = arrangeData(data_out_train, name_ens_l_idx_train, 
                                  AOC_ens_l_idx_train, AOC_ens_scale_l_idx_train,
                                  labelist, seq_length, fut_length, shuffle_it=shuffle_it_in)
        dataX_train, dataY_train = data_tensor(x_train, y_train)
        train_ds, train_dl = trainloader(dataX_train, dataY_train, bs)
        
        # validation data only subset into numpy arrays, prep for validation
        x_val, y_val, t_bool_val = arrangeData(data_out_val, name_ens_l_idx_val, 
                                  AOC_ens_l_idx_val, AOC_ens_scale_l_idx_val,
                                  labelist, seq_length, fut_length, shuffle_it=shuffle_it_in)
        dataX_val, dataY_val = data_tensor(x_val, y_val)
        valid_ds, valid_dl = trainloader(dataX_val, dataY_val, bs)
        
        # '''
        # Train LSTM
        # -- (speed up with GPU?)
        # '''
        lstm_out = runLSTM(lstm, criterion, optimizer, train_dl, num_epochs, save, save_path_out, valid_dl)


        # '''
        # Test LSTMs
        # '''
        # different conditions to test
        cond_list = ['test', 'train', 'val']
        column_list = ['DataX', 'DataY', 't_bool', 'member_name', 'r2', 'rmse-kge-nse', 'lstm_out', 'series_len', 'save_path_out', 'shuffle_it_in', 'lstm_member_name']
        cond_df = pd.DataFrame(columns=column_list, index=cond_list)
        cond_df['DataX'] = [dataX_test, dataX_train, dataX_val]
        cond_df['DataY'] = [dataY_test, dataY_train, dataY_val]
        cond_df['t_bool'] = [t_bool_test, t_bool_train, t_bool_val]
        cond_df['member_name']= [name_ens_l_idx_test, 
                                name_ens_l_idx_train, 
                                name_ens_l_idx_val]
        cond_df['lstm_out'] = lstm_out
        cond_df['series_len'] = series_len
        cond_df['save_path_out'] = save_path_out
        cond_df['shuffle_it_in'] = shuffle_it_in
        cond_df['lstm_member_name'] = lstm_member_name
        
        # explicitly tell the model we are evaluating it now
        lstm_out.eval()

        # evalute fit for all conditions
        for idx in cond_list:
            # set up temporary dataframe
            temp_df = cond_df.loc[idx]
            # read out stats
            r2_out, stats_out = evallstm(lstm=lstm_out, dataY=temp_df['DataY'], dataX=temp_df['DataX'],
                                cond=idx, save_path_out=temp_df['save_path_out'],
                                member_name_l=temp_df['member_name'], series_len=temp_df['series_len'],
                                shuffle_it_in=temp_df['shuffle_it_in'], save=True)
        
                    
            # add to a dataframe
            cond_df['r2'].loc[idx] = r2_out
            cond_df['rmse-kge-nse'].loc[idx] = stats_out
            del temp_df
            
        list_df_cond.append(cond_df)

        # '''
        # Save training, testing-specific datasets
        # data_out_train, member_name_list_train, etc...
        # data_out_val...
        # '''
        if save:
            try:
                os.mkdir(save_path_out)
            except:
                print('warning: file exists')
                pass
    
            print('saving ensemble members')
            with open(save_path_out+"train_idx.pkl", "wb") as fp:   
                pickle.dump(train_idx, fp)
                
            with open(save_path_out+"val_idx.pkl", "wb") as fp:   
                pickle.dump(val_idx, fp)
            
            train_title = ['data_out_train', 'member_name_list_train', 'name_ens_l_idx_train', 'AOC_ens_l_idx_train', 'AOC_ens_scale_l_idx_train', 
                        'x_train', 'y_train', 't_bool_train',
                        'dataX_train', 'dataY_train']
            train_data = [data_out_train, member_name_list_train, name_ens_l_idx_train, AOC_ens_l_idx_train, AOC_ens_scale_l_idx_train, 
                        x_train, y_train, t_bool_train,
                        dataX_train, dataY_train]
            for td in range(len(train_data)):
                if len(train_title) != len(train_data):
                    break
                with open(f'{save_path_out}{train_title[td]}.pkl', "wb") as fp:   
                    pickle.dump(train_data[td], fp)
            
            val_title = ['data_out_val', 'member_name_list_val', 'name_ens_l_idx_val', 'AOC_ens_l_idx_val', 'AOC_ens_scale_l_idx_val']
            val_data = [data_out_val, member_name_list_val, name_ens_l_idx_val, AOC_ens_l_idx_val, AOC_ens_scale_l_idx_val]
            for tvd in range(len(val_title)):
                if len(val_title) != len(val_data):
                    break
                with open(f'{save_path_out}{val_title[tvd]}.pkl', "wb") as fp:   
                    pickle.dump(val_data[tvd], fp)
               
            with open(save_path_out+"cond_df.pkl", "wb") as fp:   
                pickle.dump(cond_df, fp)
                
            with open(save_path_out+"lstm_trained.pkl", 'wb') as fp:
                pickle.dump(lstm_out.train(), fp)
            
            file = open(save_path_out+"start_each_time.txt", "w")
            file.write(f'{datetime.now() - start_new}\n')
            file.close()
            
 
            with open(save_path_out+"metrics_1.pkl", "wb") as fp:   
                pickle.dump(cond_df['rmse-kge-nse'], fp)
                
            with open(save_path_out+"metrics_2.pkl", "wb") as fp:   
                pickle.dump(cond_df['r2'], fp)
            
            
            print('saving complete')
            
            # clean up:
            del train_idx
            del val_idx
            
            for td in train_data:
                del td
            del train_data
            
            for tvd in val_data:
                del tvd
            del val_data
            
            del cond_df
            del lstm_out
            del file
        
        
    # '''
    # Save data relevant to sbi for all lstm ensemble members
    # '''
    if save:
        try:
            os.mkdir(save_path)
        except:
            print('warning: file exists')
            pass
        
        print('saving all ensemble')
        with open(save_path+"list_df_cond.pkl", "wb") as fp:   
            pickle.dump(list_df_cond, fp)
        print('saving complete')
        
        
    return list_df_cond # and other things"?

        
