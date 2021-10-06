from torch.utils.data import TensorDataset # for refactoring x and y
from torch.utils.data import DataLoader # for batch submission
import torch
import torch.nn as nn
from torch.autograd import Variable
from datetime import datetime

from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import numpy.ma as ma
import pandas as pd

import torch
import torch.nn as nn

import sys
import os

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset # for refactoring x and y
from torch.utils.data import DataLoader # for batch submission
from torch.autograd import Variable

sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from assessutils import compute_stats 
from genutils import plot_stuff, seriesLength

'''
Utilities for training LSTM Module in LSTM_build.py and lstm_sbi.py
'''

'''
LSTM classes:
'''
class LSTM(nn.Module):
    # input_size = the number of features in the input

    # num_classes = dimensions of the export layer

    # hidden_size = dimensions of the hidden layer

    # num_layers = number of layers to use
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        # Super
        super(LSTM, self).__init__()
        
        # attributes
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # the LSTM takes inputs and exports hidden states
        # note we are only passing in dimensions here
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        # the LSTM takes hidden states space to tag space
        self.fc = nn.Linear(hidden_size, num_classes)

      # this moves weights forward through network
      # I'm not sure how this works!!
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
    
class LSTM_2(nn.Module):
    # input_size = the number of features in the input

    # num_classes = dimensions of the export layer

    # hidden_size = dimensions of the hidden layer

    # num_layers = number of layers to use
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        # Super
        super(LSTM, self).__init__()
        
        # attributes
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # the LSTM takes inputs and exports hidden states
        # note we are only passing in dimensions here
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        # the LSTM takes hidden states space to tag space
        self.fc = nn.Linear(hidden_size, num_classes)

      # this moves weights forward through network
      # I'm not sure how this works!!
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


'''
This FUNCTION prepares LSTM structure to be used in LSTM training (later on) -
    should only need to do this once per LSTM 'class'
'''


def runLSTM(lstm, criterion, optimizer, train_dl, num_epochs, save, save_path, valid_dl=None):
    '''
    Takes --
        lstm : LSTM object
        criterion : loss function
        optimizer : optimizing function, with preset learning rate
        train_dl : training data loader
        valid_dl : optional validation step (if desired), if None will not execute
    Returns --
        trained model
    '''
    ## TO DO: Have an approach that requires early stopping?
    
    
    epoch_array = []
    loss_array = []
    v_loss_array = []
    min_valid_loss = np.inf
    
    starttime = datetime.now()
    
    # Train the model
    for epoch in range(num_epochs):
        
        train_loss = 0.0
        lstm.train()
        
        for xb, xy in train_dl:

            # make predictions
            outputs = lstm(xb)
            
            # obtain the loss function
            loss = criterion(outputs, xy)
            
            # back propogation
            loss.backward()
    
            # update the weights backward
            optimizer.step()
            
            # calculate loss
            train_loss += loss.item()
            
            # zero gradientsTypeError: cannot perform reduce with flexible type
            optimizer.zero_grad()

       # validation (if applicable)
        if valid_dl is not None:
            valid_loss = 0.0
            lstm.eval()
            for valx, valy in valid_dl:
                v_outputs = lstm(valx)
                v_loss = criterion(v_outputs, valy)
                valid_loss += v_loss.item()


        # store for next epoch
        epoch_array.append(epoch)

        loss_array.append(train_loss / len(train_dl))
        print("Epoch: %d, training loss: %1.5f" % (epoch, train_loss / len(train_dl)))

        v_loss_array.append(valid_loss / len(valid_dl))
        print("Epoch: %d, validation loss: %1.5f" % (epoch, valid_loss / len(valid_dl)))
    
    totaltime = datetime.now() - starttime
    print('')
    print(totaltime)
    print('the relationship between loss and epochs for given hyper parameters')
    plt.plot(epoch_array, loss_array)
    plt.plot(epoch_array, v_loss_array)
    plt.yscale('log')
    if save:
        try: 
            os.mkdir(save_path)
        except:
            # time.sleep might help here
            pass
        plt.savefig(save_path+'epochs.png')
        plt.show()
        plt.close()
        # save epoch file
        file = open(save_path+"epochs.txt", "w")
        file.write(str(epoch_array))
        file.close()
        # save loss array file
        file = open(save_path+"train_loss.txt", "w")
        file.write(str(loss_array))
        file.close()
        # save v_loss array file
        file = open(save_path+"val_loss.txt", "w")
        file.write(str(v_loss_array))
        file.close()
        # save time
        file = open(save_path+"runtime.txt", "w")
        file.write(str(totaltime))
        file.close()
        # save model
        torch.save(lstm.state_dict(), save_path+'/model.txt')
        
    return lstm

'''
Evaluate Trained LSTM Ensemble Members
    Scatter Plot
    Metrics
    Simulated v Real
'''

def fitStats(dataY_predict, dataY_hat):
    '''
    calculate r2 and rmse, nse, kge of y_pred, y_hat
    dataY_predict and dataY_hat 
    inherited from evallstm
    '''
    # set arguments
    arg_1 = dataY_hat[(dataY_hat.mask==False) & (dataY_predict.mask==False)]      
    arg_2 = dataY_predict[(dataY_predict.mask==False) & (dataY_predict.mask==False)]
    # compute r^2 and other statistics
    r2_compute = r2_score(arg_1, arg_2)
    stats_compute = compute_stats(arg_1, arg_2)
    return r2_compute, stats_compute

def scatterStats(dataY_predict, dataY_hat, cond, save_path_out, save=True):
    '''
    make scatter plots showing relationship between y_pred, y_hat
    dataY_predict, dataY_hat, cond, save, save_path_out
        inherited from evallstm
    '''
    fig, ax = plt.subplots()
    ax.scatter(dataY_hat, dataY_predict)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xlabel('y_hat'), ax.set_ylabel('y_pred')
    ax.set_aspect('equal')
    ax.set_title(f'Flow Comparison {cond}')
    
    if save:
        fig.savefig(f'{save_path_out}scatter_{cond}.png')
        fig.savefig(f'{save_path_out}scatter_{cond}.eps', format='eps')

    fig.show()
    plt.close()
    return None
    
def plotSeries(dataY_predict, dataY_hat, cond, member_name_l, series_len, save_path_out, save=True):
    '''
    for plotting series
    dataY_predict, dataY_hat, cond, member_name_l, series_len, save_path_out, save=True 
        are inherited from evallstm
    '''
    ## UPDATES NEEDED - ADDING functionality for shuffled data
    df_all = pd.DataFrame({'LSTM' : dataY_hat[:,0], 'ParFlow' : dataY_predict[:,0]})
    # Compare 'LSTM' and 'Parflow' streamflow data
    for idx in range(len(member_name_l)):
        idx_1, idx_2 = idx*series_len, (idx+1)*series_len
        nm = member_name_l[idx]
        plot_stuff(df_all.iloc[idx_1:idx_2,:], same=True, ylabel=f'flow-{cond}-{nm}', title=f'PFvLSTM-{cond}-{nm}',
                   save=True,path=save_path_out)
        del idx_1, idx_2, nm
    return None
    
def evallstm(lstm, dataY, dataX, cond, save_path_out, member_name_l, series_len, shuffle_it_in=False, save=True):
    '''
    Does a bunch of statistical things on 
    Takes:
        lstm : An LSTM model set to 'eval' that has already been trained
        dataY : a tensor of data to be tested against predictions
        dataX : a tensor of inputs to be input into an lstm model for prediction
        cond : a text indicator of the type of data (test, train, validation) here for saving
        member_name_l : list contining member names used for plotting
        series_len : length of series
        save : boolean telling whether or not to save
        save_path_out : where to save
        shuffle_it_in : Boolean if shuffled, default False
    '''
    print(cond)
    # make a prediction
    dataY_predict = lstm(dataX)
    
    # convert to numpy array
    dataY_predict = dataY_predict.data.numpy() # predicted
    dataY_hat = dataY.data.numpy() # 'true'
    
    # mask (just in case)
    dataY_predict = ma.masked_invalid(dataY_predict)
    dataY_hat = ma.masked_invalid(dataY_hat)
    
    # score LSTM
    r2_compute, stats_compute = fitStats(dataY_predict, dataY_hat)
    print(f'The {cond} R-squared value for lstm-parflow is: {r2_compute}')
    print(f'The {cond} summary stats (RMSE, NSE, KGE) for lstm-parflow are: \n {stats_compute}')
    
    # return scatter plots
    scatterStats(dataY_predict, dataY_hat, cond, save_path_out, save)
    
    # plot series
    # (an important step for getting sensical, non-shuffled results)
    if shuffle_it_in:
        print('Data is shuffled and will not generate useful comparisons of time series')
    else:
        plotSeries(dataY_predict, dataY_hat, cond, member_name_l, series_len, save_path_out, save=True)

    return r2_compute, stats_compute
