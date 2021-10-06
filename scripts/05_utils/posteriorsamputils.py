# -- Libararies  
import os
import sys
import pickle
import torch

# for SBI
from sbi import utils as utils
from sbi import analysis as analysis
from sbi import inference
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.types import Array, OneOrMore, ScalarFloat

from matplotlib import pyplot as plt

import numpy as np
import random
from numpy.random import normal
import matplotlib.pyplot as plt
from seaborn import pairplot
from seaborn import jointplot
import pandas as pd
from parflowio.pyParflowio import PFData
# import pygmmis <- This was the default but it doesn't work very well
from sklearn.mixture import GaussianMixture as GMM
import sys
import os
from datetime import datetime

# for machine learning
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # for refactoring x and y
from torch.utils.data import DataLoader # for batch submission
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Independent, Uniform
from torch.distributions.log_normal import LogNormal

# for scaling
from sklearn.preprocessing import MinMaxScaler

# for stats
from assessutils import compute_stats
import re
from scipy.stats import mode

# helpers
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/')
from sbi_build import simulate
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from genutils import convertNumpy

def genProbThetas(params_true_scaled, posterior_samples, log_probability, theta_prec=4, mult_factor=100):
    '''
    Passes: 
    - Object containing true thetas (NOT the unscaled-'measured' values)
    - torch tensors posterior_samples and log_probability
    - precision of tolerance for identifying log probability of true parameters, and mode - default is 4
    - mult_factor is a a multiplication factor for bracketing 'reasonable' values
    Generates some summary statistics for later usage, including:
    - log_prob_true_thetas_ind : A 1D numpy array of length `n_thetas` showing the mean of the log probability of each
    - log_prob_true_thetas_all : A single scalar of the probability of the combination of 'n_thetas'. 
    - log_prob_true_thetas_flag : If 'True', this means that the combination is improbable (no pairing found),
        and that the probability of log_prob_true_thetas is the minimum of the dataset
    '''
    
    ## NOTE need to figure out how to do log_prob_true_thetas_all for more than two (i.e. dynamic number) of thetas

    # don't run this routine if working with unknown parameter values
    if params_true_scaled == None:
        return None, None, None
    else:
        # convert
        params_true_scaled = convertNumpy(params_true_scaled, toTorch=False)
        posterior_samples = convertNumpy(posterior_samples, toTorch=False)
        log_probability = convertNumpy(log_probability, toTorch=False) 
        
        # generating log_prob_true_thetas_ind
        idx_list = []
        log_prob_true_thetas_ind = np.empty(len(params_true_scaled))
        # type(params_true_scaled)
        for i in range(len(params_true_scaled)):
            # round and identify parameter of interest, and log_probability
            
            theta_i = np.round(posterior_samples[:,i], theta_prec)
            param_i = np.round(params_true_scaled[i], theta_prec)
            # type(theta_i)
            # type(param_i)
            bracket_factor = 10**(-theta_prec)*mult_factor
            # find location of true theta criterion, append and save for later use
            theta_idx = np.where((theta_i > (param_i - bracket_factor)) & (theta_i < (param_i + bracket_factor)))
            idx_list.append(theta_idx[0])
            # find the log probabilities of acceptable theta, may be multiple so average taken
            log_prob_true_thetas_ind[i] = np.take(log_probability, theta_idx)[0].mean()
            
            # # find acceptable thetas
            # theta_acc = np.take(theta_i, theta_idx)

        # generate log_prob_true_thetas_all
        log_prob_true_thetas_all = np.intersect1d(idx_list[0],idx_list[1])

        # flag log_prob_true_thetas_flag, and if True set to lowest probability in dataset
        if len(log_prob_true_thetas_all) == 0:
            log_prob_true_thetas_all = log_probability.min()
            log_prob_true_thetas_flag = True
        else: 
            # take indexes and calculate mean
            log_prob_true_thetas_all = np.take(log_probability, log_prob_true_thetas_all).mean()
            log_prob_true_thetas_flag = False
            
        # convert
        log_prob_true_thetas_ind = convertNumpy(log_prob_true_thetas_ind, toTorch=True)
        # log_prob_true_thetas_all = convertNumpy(log_prob_true_thetas_all, toTorch=True)
         
        return log_prob_true_thetas_ind, log_prob_true_thetas_all, log_prob_true_thetas_flag
        
# max prob theta, 
def maxProbTheta(posterior_samples, log_probability):
    '''
    Passess two numpy vectors, finds the index of the maximum probability
    returns: 
    a - the value of the maximum probability (scalar, log prob)
    b- and the theta(s) at said probability (vector, 1D)
    '''
    log_prob_max = log_probability.max()
    log_prob_max_idx = np.where(log_probability[:,0] == log_prob_max)[0][0]
    thetas_max_prob = posterior_samples[log_prob_max_idx,:]
    return torch.from_numpy(thetas_max_prob), log_prob_max
    
def thetasMode(posterior_samples, log_probability, theta_prec=4):
    '''
    Returns the most comonly occuring theta values within a precision of theta_prec
    '''
    dims = posterior_samples.shape[1]
    mode_theta = np.empty(dims)
    for i in range(dims):
        mode_theta_idx = mode(np.round(posterior_samples[:,i], theta_prec))
        mode_theta[i] = mode_theta_idx[0]
    log_prob_mode = None
    return torch.from_numpy(mode_theta), log_prob_mode
    
def randTheta(posterior_samples, log_probability):
    '''
    This is a randomly selected theta pair and its probability
    '''
    idx_random = random.randint(0, len(log_probability)-1)
    theta_rand = posterior_samples[idx_random, :]
    log_prob_rand = log_probability[idx_random]
    return torch.from_numpy(theta_rand), torch.from_numpy(log_prob_rand)
    
    
def statTheta(posterior_samples, log_probability):
    thetaTypeList = ['mu', 'median', 'max', 'mode', 'random']
    thetaStatsList = []
    probStatsList = []
    # mean
    mu_posterior = torch.mean(posterior_samples, axis=0)
    thetaStatsList.append(mu_posterior)
    probStatsList.append(None)
    # median
    median_posterior = torch.median(posterior_samples, 0)[0] # median
    thetaStatsList.append(median_posterior)
    probStatsList.append(None)
    # max_prob
    max_prob_posterior, log_prob_max = maxProbTheta(posterior_samples=posterior_samples.numpy(),
                                                  log_probability=log_probability.numpy()) # max probability of thetas, max probability value
    thetaStatsList.append(max_prob_posterior)
    probStatsList.append(log_prob_max)
    # mode
    mode_posterior, log_prob_mode = thetasMode(posterior_samples.numpy(),
                                    log_probability=log_probability.numpy()) # mode thetas
    thetaStatsList.append(mode_posterior)
    probStatsList.append(log_prob_mode)
    # random
    rand_posterior, log_prob_rand = randTheta(posterior_samples=posterior_samples.numpy(),
                                          log_probability=log_probability.numpy()) # random theta
    thetaStatsList.append(rand_posterior)
    probStatsList.append(log_prob_rand)
    
    return thetaTypeList, thetaStatsList, probStatsList
    
    
def gen_Series_Wrapper(lstm_out_list, thetaStatsList,
                DataX_test, series_len):
    '''
    returns seriesarr 
        (timeseries, len(lstm_out_list), len(thetaStatsList)
        (350,        10,          5)
    '''
    # create an array to hold data
    seriesarr = torch.empty((series_len,
                        len(lstm_out_list),
                        len(thetaStatsList)))

    # run forward simulations for each lstm model
    for i in range(len(lstm_out_list)):
        lstm = lstm_out_list[i]
        # for each type (see thetaTypeList)
        for j in range(len(thetaStatsList)):
            theta = thetaStatsList[j]
            y_o = simulate(DataX_test, theta, lstm)
            seriesarr[:, i, j] = y_o[:,0]
        
    return seriesarr
    

def gen_Fit_Wrapper(seriesarr, y_hat_full):
    '''
    return fitarr
        (evalfit, len(lstm_out_list), len(thetaStatsList)
        (3,        10,          5)  
    '''
    # create empty array of shape
        # (RMSE_NSE_KGE, lstm_number, theta_type)
        # (3,        10,          5)
        
    fitarr = np.empty((3,
                        seriesarr.shape[1],
                        seriesarr.shape[2]))
    
    y_hat_full = convertNumpy(y_hat_full[0,:], toTorch=False)
    
    # for each lstm:
    for j in range(seriesarr.shape[1]):
        # simulate for each theta type:
        for k in range(seriesarr.shape[2]):
            y_o = seriesarr[:,j,k]
            y_o = convertNumpy(y_o, toTorch=False)
            fitarr[:,j,k] = compute_stats(y_o, y_hat_full)

    fitarr = convertNumpy(fitarr, toTorch=True)

    return fitarr


def gen_Fit_Series_Wrapper(lstm_out_list, DataX_test, y_hat_full,
                            true_theta, series_len,
                            thetaTypeList, thetaStatsList):
    '''
    This function:
    Generates full series of results (forward simulations 
    based off of inferred parameters)
    & 
    Fit Results
    '''
    
    # create array of y_o
    seriesarr = gen_Series_Wrapper(lstm_out_list, thetaStatsList,
                DataX_test, series_len)
                
    
    # create fit of array
    fitarr = gen_Fit_Wrapper(seriesarr, y_hat_full)
    
    
    return seriesarr, fitarr

