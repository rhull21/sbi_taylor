# modified from SBI_Interp_Agg

import os
import os.path
import sys
import shutil
from pprint import pprint
from datetime import datetime
from copy import copy
from copy import deepcopy
import pickle

from parflowio.pyParflowio import PFData

import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import glob


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# -- Libararies  
import os
import sys
import pickle
import torch
from random import *

# for SBI
from sbi import utils as utils
from sbi import analysis as analysis
from sbi import inference
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.types import Array, OneOrMore, ScalarFloat

from matplotlib import pyplot as plt

import numpy as np
from numpy import cov
from numpy.linalg import det, norm # determinant, norm (used to calculate 'euclidean' distance)

import random
from numpy.random import normal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
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
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from assessutils import compute_stats
import re
from scipy.stats import mode

# helpers
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/')
from sbi_build import simulate

# Path to the SandTank Repo
dev_path = '/home/SHARED/ML_TV/HydroGEN/modules/'
#Add Sand Tank path to the sys path
sys.path.append(dev_path)
from transform import float32_clamp_scaling

# user defined functions
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from genutils import PFread, plot_stuff
from ensembleutils import assembleYears, assembleAllForcings, parseAllMetadata, returnDF_ens, returnAOC_ens, _ret_MinMax, _ret_AOCMinMax
from assessutils import compute_stats
from scalerutils import scaled_ens_Values, scaledForcingData, scaledAOCData
from posteriorsamputils import statTheta, genProbThetas, gen_Fit_Series_Wrapper
from summaryutils import summary, setStatSim
from sbiutils import retStatTyp, parseListDf, parseUniqueParams
from sbiutils import reshape_y, createYHatList, setTheta, setNoise


'''
Data Grabbing Functions
'''

# function for retrieving data
def getData_old(sub, sub_sub, sbi_dir):
    '''
    function for retrieving data, and data to plot in time series
    
    For a single observation sampled at a single posterior
    
    Specifically, focuses on calculating the minumum, maximum and mean of 
        an ensemble of selected forward simulations so as to compare 
        to an observation
        
    Returns:
        minarr, maxarr, meanarr, y_hat_plot
        
    Depricated
    '''
    sbi_dir_sub = f'{sbi_dir}{sub}/'
    sbi_dir_sub_sub = f'{sbi_dir_sub}{sub_sub}/'


    with open(sbi_dir+'DataX_test.pkl', 'rb') as fp:
        DataX_test = pickle.load(fp)

    with open(sbi_dir+'test_params.pkl', 'rb') as fp:
        test_params = pickle.load(fp)

    with open(sbi_dir+'lstm_out_list.pkl', 'rb') as fp:
        lstm_out_list = pickle.load(fp)

    with open(sbi_dir_sub_sub+'y_hat_full.pkl', 'rb') as fp:
        y_hat = pickle.load(fp)[0,:]
    #     print(y_hat)

    with open(sbi_dir_sub_sub+'seriesarr.pkl', 'rb') as fp:
        seriesarr = pickle.load(fp)
    
    minarr = torch.min(torch.min(seriesarr, dim=2)[0], dim=1)[0].detach().numpy()
    maxarr = torch.max(torch.max(seriesarr, dim=2)[0], dim=1)[0].detach().numpy()
    meanarr = torch.mean(torch.mean(seriesarr, dim=2), dim=1).detach().numpy()

    y_hat_plot = y_hat.detach().numpy()
    
    return minarr, maxarr, meanarr, y_hat_plot

def grabDataarrays(sbi_dir, sub, sub_sub_int):
    '''
    For exporting multiple charcteristics of import from a 
    single observation sampling a single posterio
    
    Returns:
        DataX_test, test_params, lstm_out_list, y_hat, seriesarr, log_probability, posterior_samples
    
    '''
    sbi_dir_sub = f'{sbi_dir}{sub}/'
    sub_sub = "{:02d}".format(sub_sub_int)
    sbi_dir_sub_sub = f'{sbi_dir_sub}{sub_sub}/'
    
    
    with open(sbi_dir+'DataX_test.pkl', 'rb') as fp:
        DataX_test = pickle.load(fp)

    with open(sbi_dir+'test_params.pkl', 'rb') as fp:
        test_params = pickle.load(fp)

    with open(sbi_dir+'lstm_out_list.pkl', 'rb') as fp:
        lstm_out_list = pickle.load(fp)
        

    with open(sbi_dir_sub_sub+'y_hat_full.pkl', 'rb') as fp:
        y_hat = pickle.load(fp)[0,:]
    #     print(y_hat)

    with open(sbi_dir_sub_sub+'seriesarr.pkl', 'rb') as fp:
        seriesarr = pickle.load(fp)
        
    with open(sbi_dir_sub_sub+'log_probability.pkl', 'rb') as fp:
        log_probability = pickle.load(fp)

    with open(sbi_dir_sub_sub+'posterior_samples.pkl', 'rb') as fp:
        posterior_samples = pickle.load(fp)
        
    return DataX_test, test_params, lstm_out_list, y_hat, seriesarr, log_probability, posterior_samples


def genSimulation(DataX_test,boots_params,lstm_out_list,add_noise=False,f_noise=1e-02,rand_lstm=False,lstm_idx=0,y_len=350):
    '''
    simulates for all instantiations of boots_params
    '''
    n = len(boots_params)
    y_sims = np.empty((n,y_len))
    
    for i in range(n):
        if (len(lstm_out_list) > 1) and rand_lstm:
            idx_slct = randint(0, len(lstm_out_list)-1)
            lstm_out = lstm_out_list[idx_slct]
        else:
            lstm_out = lstm_out_list[lstm_idx]
        theta = boots_params[i,:]
        
        y_sim = simulate(DataX_test,theta,lstm_out)
        
        # decides whether or not to add noise
        if add_noise:
            # print('make some noise')
            y_sim = setNoise(y_sim, f_noise)
        
        y_sim = y_sim.detach().numpy()[:,0]
        
        y_sims[i,:] = y_sim
    
#     print(y_sims.shape)
    return y_sims

def randBootstrap(posterior_samples, log_probability, n=50):
    '''
    random bootstrap sampling (w/o replacement)
    '''
    idxs = []
    dim = len(posterior_samples)-1

    boots_params = torch.empty((n, posterior_samples.shape[1]))
    boots_logprob = torch.empty((n,log_probability.shape[1]))

    for i in range(n):
        # create idx
        idx = randint(0,dim)
        # save idxbootstrap
        idxs.append(idx)
        # slice and save log_prob and param
        boots_params[i,:] = posterior_samples[idx,:]
        boots_logprob[i,:] = log_probability[idx,:]
        
    return boots_params, boots_logprob, idxs

def getData(sub, sub_sub_int, sbi_dir, ret_arrays=False, add_noise=False, f_noise=1e-02):
    '''
    function for retrieving data, and data to plot in time series
    
    For a single observation sampled at a single posterior
    
    * Extracts Data
    * Takes a bootstrap of params
    * Extracts truth
    * Generates a bunch of forward simulations
        
    Returns:
        y_sims, y_hat_plot (optionally log_probability and posterior_samples)
    '''
    sbi_dir_sub = f'{sbi_dir}{sub}/'
    sub_sub = "{:02d}".format(sub_sub_int)
    sbi_dir_sub_sub = f'{sbi_dir_sub}{sub_sub}/'
    
    # extract data
    DataX_test, test_params, lstm_out_list, y_hat, seriesarr, log_probability, posterior_samples = grabDataarrays(sbi_dir, sub, sub_sub_int)
    
    # bootstrap
    boots_params, boots_logprob, idxs = randBootstrap(posterior_samples, log_probability)
    
    # get 'truth'
    y_hat_plot = y_hat.detach().numpy()
    y_len = len(y_hat_plot)
    
    # generate forward simulations
    if len(lstm_out_list) > 1:
        print('Caution, multiple emulators available')    
    y_sims = genSimulation(DataX_test,boots_params,lstm_out_list,add_noise=add_noise,f_noise=f_noise,lstm_idx=0,y_len=y_len)
    
    if ret_arrays==False:
        return y_sims, y_hat_plot
    else:
        return y_sims, y_hat_plot, boots_params, boots_logprob
        
        
        
'''
Data Processing Functions
'''

def getDataCov(sbi_dir, sub, sub_sub_int):
    '''
    Processes a dataset and extracts important characteristics of the 
        distribution and relationship between parameter array inferences
        and true parameters
    '''
    sbi_dir_sub = f'{sbi_dir}{sub}/'
    sub_sub = "{:02d}".format(sub_sub_int)
    sbi_dir_sub_sub = f'{sbi_dir_sub}{sub_sub}/'

    
    DataX_test, test_params, lstm_out_list, y_hat, seriesarr, log_probability, posterior_samples = grabDataarrays(sbi_dir, sub, sub_sub_int)
    
    # detach posterior samples
    posterior_samples = posterior_samples.detach().numpy()
    log_probability = log_probability.detach().numpy()
    
    # Summary statistics of param and the mean of those sampled
#     print(sub_sub_int)
    test_param = test_params.detach().numpy()[sub_sub_int,:]
    mean_param = posterior_samples.mean(axis=0)
    median_param = np.quantile(posterior_samples, q=0.5, axis=0)

    # probability stuff
    log_prob_true_thetas_ind, log_prob_true_thetas_all, log_prob_true_thetas_flag = genProbThetas(test_param,
                                        posterior_samples, log_probability, theta_prec=4, mult_factor=10)

    # covariance of the distribution
    cov_sample = cov(posterior_samples.T)
#     print(cov_sample)
    # determinant of the distribution
    det_sample = det(cov_sample)

    # diagonal varaince
    var_param1, var_param2 = cov_sample[0,0], cov_sample[1,1]

    # off diagonal variance
    var_12, var_21 = cov_sample[0,1], cov_sample[0,1]

    # mismatch data
    theta_mu_diff_param1, theta_mu_diff_param2 = abs(test_param[0]-mean_param[0]), abs(test_param[1]-mean_param[1])
    theta_med_diff_param1, theta_med_diff_param2 = abs(test_param[0]-median_param[0]), abs(test_param[1]-median_param[1])

    # normalized mismatch
    var_mu_norm_param1, var_mu_norm_param2 = theta_mu_diff_param1/var_param1, theta_mu_diff_param2/var_param2
    var_med_norm_param1, var_med_norm_param2 = theta_med_diff_param1/var_param1, theta_med_diff_param2/var_param2

    out_dict = {'covariance' : cov_sample, 
                'determinant' : det_sample,
                'true param' : test_param,
                'mean param' : mean_param,
                'median_param' : median_param,
                'dia var 11' : var_param1,
                'dia var 22' : var_param2,
                'mismatch, mu, param1' : theta_mu_diff_param1,
                'mismatch, mu, param2' : theta_mu_diff_param2,
                'mismatch, med, param1' : theta_med_diff_param1,
                'mismatch, med, param2' : theta_med_diff_param2,
                'norm mismatch, mu, param1' : var_mu_norm_param1,
                'norm mismatch, mu, param2' : var_mu_norm_param2,
                'norm mismatch, med, param1' : var_med_norm_param1,
                'norm mismatch, med, param2' : var_med_norm_param2,
                'log_prob_true_thetas_ind' : log_prob_true_thetas_ind, 
                'log_prob_true_thetas_all' : log_prob_true_thetas_all 
               }
    
    
    return out_dict
    
def getMetricArray(y_sims_T, y_hat_plot):
    '''
    y_sims_T : transposed array from getData
    y_hat_plot : 'true' value
    
    return array of metrics (idx 1 == nse)
    '''
    
    temp_metric_arr = np.empty((3,y_sims_T.shape[1]))
    for sim_idx in range(y_sims_T.shape[1]):
        y_sim = y_sims_T[:,sim_idx]
        temp_metric_arr[:,sim_idx] = compute_stats(y_sim, y_hat_plot)

    return temp_metric_arr

def getStatTypArray(stat_typ, y_hat_plot):
    '''
    stat_typ : an array of length (n) containing indices that
        refer to summaryutils keys
    y_hat_plot : the 'measured' synthetic lstm-generated value
        at a parameter value
    '''
    stat_sim = np.array(setStatSim(torch.tensor(y_hat_plot), stat_typ))
    return stat_sim
    
    
def getData_Surface(save_dir, sbi_dir, sub, sub_sub_num, stats_bool=False, stat_typ=None, scaled_bool=True, save_bool=True):
    '''
    For getting data out to plot as a 2D surface
    '''
    if save_bool:
        # read in 
        try:
            os.mkdir(save_dir)
        except:
            print('warning: file exists')
            pass

        try:
            os.mkdir(f'{save_dir}{sub}/')
        except:
            print('warning: file exists')
            pass

    # true param array
    param_arr = np.empty((sub_sub_num, 2))
    # euc array
    euc_arr = np.empty(sub_sub_num)
    # single_param_distance
    dist_arr = np.empty((sub_sub_num, 2))
    # determinant array
    det_arr = np.empty(sub_sub_num)
    # NSE Array
    nse_arr = np.empty(sub_sub_num)
    # Probability Array
    prob_arr = np.empty(sub_sub_num)
    
    if stats_bool:                      
        # Stat Arr
        stat_arr = np.empty((sub_sub_num, len(stat_typ)))
    else:
        stat_arr = None

    for k in range(sub_sub_num): # sub_sub_num
        sub_sub = "{:02d}".format(k)

        # read in data space 
        y_sims, y_hat_plot = getData(sub, k, sbi_dir)
        y_sims = y_sims.T
        y_sims_mean = np.mean(y_sims,axis=1)

        # metric array
        temp_metric_arr = getMetricArray(y_sims_T=y_sims, y_hat_plot=y_hat_plot)
        
        nse = temp_metric_arr[1,:]

        # read in parameter space
        out_dir = getDataCov(sbi_dir, sub, k)

        # parameter space
        test_param, mean_param = out_dir['true param'], out_dir['mean param']
        euc_dist = norm(test_param-mean_param)
        determ = out_dir['determinant']
        
        # plotting up the distance from individual values
        dist_param1 = out_dir['mismatch, mu, param1']
        dist_param2 = out_dir['mismatch, mu, param2']
        
        # plotting up probability
        ## 'log_prob_true_thetas_ind' : log_prob_true_thetas_ind, 
        prob_val = out_dir['log_prob_true_thetas_all']

        # assigning
        param_arr[k,:] = test_param; euc_arr[k] = euc_dist; dist_arr[k,:] = [dist_param1, dist_param2]
        det_arr[k] = determ; nse_arr[k] = nse.mean(); prob_arr[k] = prob_val;
                          
        if stats_bool:
            stat_sim = getStatTypArray(stat_typ, y_hat_plot)
            stat_arr[k,:] = stat_sim
        
        del sub_sub, y_sims, y_hat_plot, y_sims_mean, temp_metric_arr, nse, out_dir, test_param, mean_param
        del euc_dist, determ, dist_param1, dist_param2, prob_val
    
    # assign dictionary
    out_dict = {'param_arr' : param_arr, 
                'euc_arr' : euc_arr,
                'dist_arr' : dist_arr,
                'det_arr' : det_arr,
                'nse_arr' : nse_arr,
                'prob_arr' : prob_arr,
                'stat_arr' : stat_arr
                }
    
    return out_dict
    
'''
Single SBI Visualization
'''

def plot_hydrograph(save_dir, sbi_dir, sub, sub_sub_num, dim1, dim2, 
                    trueflow=False, true_flow_idx=None, scaled_bool=True, save_bool=True, 
                    add_noise=False, f_noise=1e-02):
    '''
    Plots hydrograph for an entire SBI run with all observations
    But only one SBI run
    
    takes:
        save_dir
        subFunctions
        sub_sub_num
        div
        scaled_bool=True
        save_bool=True
        
    returns:
        None nb one day we might want to actually make this actually export the sims data
    
    '''
    if save_bool==True:
        # read in 
        try:
            os.mkdir(save_dir)
        except:
            print('warning: file exists')
            pass

        try:
            os.mkdir(f'{save_dir}{sub}/')
        except:
            print('warning: file exists')
            pass
    
    # set up plots (hard, sometimes chokes)
#     dim1, dim2 = int(sub_sub_num/div), int(sub_sub_num/div)
    fig, axs = plt.subplots(dim1, dim2, figsize=(30, 20))

    i, j = 0, 0
    for k in range(sub_sub_num): # sub_sub_num
        sub_sub = "{:02d}".format(k)
        if (k/dim1 == int(k/dim1)) and (k != 0):
            i = 0
            j = j + 1

        y_sims, y_hat_plot = getData(sub, k, sbi_dir, add_noise=add_noise, f_noise=f_noise)

        axs[i,j].plot(y_sims.T, color='blue', alpha=0.25) #, label='y_sims')
        axs[i,j].plot(np.mean(y_sims.T,axis=1), color='red', linewidth=1, label='mean')
        axs[i,j].plot(y_hat_plot, color='green', linewidth=1, label='y_hat')
        axs[i,j].set_title(f'{sub}_{sub_sub}')
        axs[i,j].set_ylim(0,1)
        axs[i,j].legend()
        if (trueflow==True) and (k >= true_flow_idx):
            axs[i,j].set_facecolor('gold')        

        i = i + 1
        
        del y_sims, y_hat_plot

    fig.suptitle(sbi_dir)
    if save_bool==True:
        fig.savefig(f'{save_dir}{sub}/'+'hydrograph.png')
        fig.savefig(f'{save_dir}{sub}/'+'hydrograph.eps', format='eps')
    plt.show()
    
    return None
    
    
'''
Data Grouping and Visualization
'''

def gen_2D_surface(z_name, z_arr, param_arr, save_dir, sub, trueflow=False, true_flow_idx=None, ngrid=50, levels=10, norm_bool=False, cmap="RdBu_r", save_bool=True):
    '''
    For generating a 2D surface
    Takes:
        z_name : string containing the name of the variable visualized
        z_arr : array (1D) containing the metric to plot the surface of
        param_arr : a parameter array 2xn, the x and y variables on the plot
        how coarse / fine to interpolate the grid
        save_dir : where to save
        ngrad : something to do with the spacing used to interpolate
        levels : something to do with how close or far apart to do the interpolation
    '''
    ngridx = ngrid
    ngridy = ngrid
    x = param_arr[:,0]
    y = param_arr[:,1]

    # HELP
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/irregulardatagrid.html
    fig, ax = plt.subplots(figsize=(10, 10))

    # -----------------------
    # Interpolation on a grid
    # -----------------------
    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.

    # Create grid values first.
    xi = np.linspace(-0.1,1.1, ngridx) # potentailly edit this range
    yi = np.linspace(-0.1,1.1, ngridy) # potentially edit this range

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z_arr)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    #from scipy.interpolate import griddata
    #zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

    if norm_bool:
        ax.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
        cntr1 = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, extend='both',
                           norm=colors.LogNorm(vmin=levels.min(), vmax=levels.max()))
    else:
        ax.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
        cntr1 = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, extend='both')

    fig.colorbar(cntr1, ax=ax)
    ax.plot(x, y, 'ko', ms=3)
    # plot 'true flows' if it is thrown, which just means symbologize the last few to be 'true flows'
    if trueflow:
        ax.plot(x[true_flow_idx:], y[true_flow_idx:],'*',ms=5,color='gold')
    
    ax.set(xlim=(0,1), ylim=(0, 1))
    ax.set_title(f'{z_name} grid and contour (%d points, %d grid points)' %
                  (len(z_arr), ngridx * ngridy))
    ax.set_aspect('equal')
    if save_bool:
        plt.savefig(f'{save_dir}{sub}/{z_name}_surf.png')
        plt.savefig(f'{save_dir}{sub}/{z_name}_surf.eps',format='eps')
    plt.show()
    plt.close()
    
    return None
    
def gen_post_all(sub_num, sub_sub_num, sbi_dir_list, label_list,
                 col_list, metrics_L, save_dir, save_bool=True, plot_bool=True):
    '''
    Takes:
        sub_num : the index of replicate posteriors
        sub_sub_num : the total number of truths to test
        
        sbi_dir_list : the list of directories from which to read data
        label_list : the labels to give each member of sbi_dir_list in order to make graphs more simple
        col_list : list of colors to be used in plotting
            **sbi_dir_list, label_list, col_list all same lengths
            
        metrics_L : a list of metrics to save
            
        save_dir : name of where to save this aggregation step
        save_bool : boolean of where to save things
    
    Returns:
        A set of figures, which is saved according to boolean
        out_arr : an array that returns the characteristics of interest for every parameter distribution listed here
    '''

    if save_bool==True: 
        try:
            os.mkdir(save_dir)
        except:
            print('warning: file exists')
            pass
        

    max_var_param1, max_var_param2 = 0,0
    max_theta_mu_diff_param1, max_theta_mu_diff_param2 = 0,0
    max_var_mu_norm_param1, max_var_mu_norm_param2 = 0,0

    out_arr = np.empty((sub_sub_num, sub_num, len(sbi_dir_list), len(metrics_L)))
                       
    if plot_bool==True:
        fig, axs = plt.subplots(1, 3, figsize=(20,6))
                       
    # loop through all the different model results
    for z in range(len(sbi_dir_list)):
        sbi_dir = sbi_dir_list[z]
        col = col_list[z]
        label = label_list[z]
        # loop through all repetitions of replicate posteriors
        for sub in range(sub_num):
            # loop through all truths
            for sub_sub_int in range(sub_sub_num):
                # the data read for specific iteration
                out_dir = getDataCov(sbi_dir, sub, sub_sub_int)
                    
                # extract paramsFunctions
                test_param, mean_param = out_dir['true param'], out_dir['mean param']

                # calculate distances...
                euc_dist = norm(test_param-mean_param)
                out_arr[sub_sub_int, sub, z, 0] = euc_dist

                # ------
                # calculate variance
                # ------
                var_param1, var_param2 = out_dir['dia var 11'], out_dir['dia var 22']
                out_arr[sub_sub_int, sub, z, 1:3] = np.array([var_param1, var_param2])

                # set minmax
                if var_param1.max() > max_var_param1:
                    max_var_param1 = var_param1.max()
                if var_param2.max() > max_var_param2:
                    max_var_param2 = var_param2.max()

                # ------
                # calculate theta difference 
                # ------
                theta_mu_diff_param1, theta_mu_diff_param2 = out_dir['mismatch, mu, param1'], out_dir['mismatch, mu, param2']
                out_arr[sub_sub_int, sub, z, 3:5] = np.array([theta_mu_diff_param1, theta_mu_diff_param2])

                # set minmax
                if theta_mu_diff_param1.max() > max_theta_mu_diff_param1:
                    max_theta_mu_diff_param1 = theta_mu_diff_param1.max()
                if theta_mu_diff_param2.max() > max_theta_mu_diff_param2:
                    max_theta_mu_diff_param2 = theta_mu_diff_param2.max()

                # ------
                # calculate normalized theta difference 
                # ------
                var_mu_norm_param1, var_mu_norm_param2 = out_dir['norm mismatch, mu, param1'], out_dir['norm mismatch, mu, param2']
                out_arr[sub_sub_int, sub, z, 5:7] = np.array([var_mu_norm_param1, var_mu_norm_param2])

                # set minmax
                if var_mu_norm_param1.max() > max_var_mu_norm_param1:
                    max_var_mu_norm_param1 = var_mu_norm_param1.max()
                if var_mu_norm_param2.max() > max_var_mu_norm_param2:
                    max_var_mu_norm_param2 = var_mu_norm_param2.max()

                # ------
                # determinant
                # ------
                det_sample = out_dir['determinant']
                out_arr[sub_sub_int, sub, z, 7] = det_sample
                       
                # ----- PLOT STUFF ------ 
                if plot_bool: 
                    # make an assertion about marker based on if parameter is in / out of sample
                    # if taken from 'out of sample', so to speak
                    if (max(test_param) == 1) or min(test_param) == 0:
        #                 print(test_param)
                        markerstyle = '+'
                    else:
                        markerstyle = 'o'
                    # ----- PLOT variance -------    
                    axs[0].scatter(var_param1, var_param2, color=col, label=label, marker=markerstyle, alpha=0.5)
                    # ----- PLOT difference -------    
                    axs[1].scatter(theta_mu_diff_param1, theta_mu_diff_param2, color=col, label=label, marker=markerstyle, alpha=0.5)          
                    # ----- PLOT Variance -------   
                    axs[2].scatter(var_mu_norm_param1, var_mu_norm_param2, color=col, label=label,marker=markerstyle, alpha=0.5)


    if plot_bool:
        axs[0].set_title('Variance (diagonals of cov)')
        axs[0].set_xlabel('variance in Hydraulic K')
        axs[0].set_ylabel('variance in Mannings')
        # axs[0].axis('equal')
        axs[0].set_xlim(1e-05, max(max_var_param1, max_var_param2))
        axs[0].set_ylim(1e-05, max(max_var_param1, max_var_param2))
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')

        axs[1].set_title('Difference between true and estimated theta')
        axs[1].set_xlabel('abs(theta_pred - theta*) in Hydraulic K')
        axs[1].set_ylabel('abs(theta_pred - theta*) in Mannings')
        axs[1].set_xlim(1e-04, 1)
        axs[1].set_ylim(1e-04, 1)
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')

        axs[2].set_title('Difference between true and estimated theta \n normalized by var')
        axs[2].set_xlabel('abs(theta_pred - theta*)/var_theta in Hydraulic K')
        axs[2].set_ylabel('abs(theta_pred - theta*)/var_theta in Mannings')
        axs[2].set_xlim(1e-02, max(max_var_mu_norm_param1, max_var_mu_norm_param1))
        axs[2].set_ylim(1e-02, max(max_var_mu_norm_param1, max_var_mu_norm_param1))
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')

        # fig.suptitle('summary : blue, MLP2 : red, MLP4 : green, full : orange \n *+* param 0,1')
        fig.savefig(f'{save_dir}'+'param_statistics_relation.png')
        # fig.savefig(f'{save_dir}'+'param_statistics_relation.eps', format='eps')
        plt.show()
        plt.close()
                       
    return out_arr

def plt_box_all(new_out_arr, label_list, save_dir, bar_name='', save_bool=True):
    '''
    Makes a box plot from the information in a processed out_arr
    
    returns:
        None
    '''
    fig, axs = plt.subplots(2, 4, figsize=(28,13))

    
    axs[0,0].set_title('Euclidean Distance to true Param')
    axs[0,0].boxplot(new_out_arr[:,:,0])

    axs[0,1].set_title('Determinant of covariance matrix')
    axs[0,1].boxplot(new_out_arr[:,:,-1])
    
    axs[0,2].set_title('Variance in param 1 (K)')
    axs[0,2].boxplot(new_out_arr[:,:,1])

    axs[0,3].set_title('variance in param 2 (Mannings)')
    axs[0,3].boxplot(new_out_arr[:,:,2])

    axs[1,0].set_title('difference normalized by variance (K)')
    axs[1,0].boxplot(new_out_arr[:,:,-2])

    axs[1,1].set_title('difference normalized by variance (M)')
    axs[1,1].boxplot(new_out_arr[:,:,-3])

    axs[1,2].set_title('abs theta difference (K)')
    axs[1,2].boxplot(new_out_arr[:,:,3])

    axs[1,3].set_title('abs theta difference (M)')
    axs[1,3].boxplot(new_out_arr[:,:,4])

    y_lim_arr = np.array([ [ [1e0, 1e-4], [1e-2, 1e-8], [1e0, 1e-5], [1e0, 1e-5] ],
                          [ [1e3, 1e-3], [1e3, 1e-3], [1e0, 1e-6], [1e0, 1e-6] ] ])

    for j in range(2):
        for k in range(4):
            y_lim_temp = y_lim_arr[j, k]
            axs[j, k].set_xticklabels(label_list, rotation=45)
            axs[j, k].set_yscale('log')
            axs[j, k].set_ylim(y_lim_temp[0], y_lim_temp[1])

    if save_bool:
        fig.savefig(f'{save_dir}bar_relation_{bar_name}.png')
        fig.savefig(f'{save_dir}bar_relation_{bar_name}.eps', format='eps')
    
    plt.title(bar_name)
    plt.show()
    plt.close()
    
    return None





