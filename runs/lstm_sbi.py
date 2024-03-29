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

from sbi import utils as utils
from sbi import analysis as analysis
from sbi import inference
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.types import Array, OneOrMore, ScalarFloat

import matplotlib.pyplot as plt


import sys
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from lstmutils import _sliding_windows, sliding_windows, delnull, randomidx, arrangeData, selectData, data_tensor, trainloader
from sbiutils import retStatTyp, parseListDf, parseUniqueParams
from summaryutils import summary
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/')
from lstm_build_utils import LSTM, runLSTM
from lstm_build import buildLSTM
from sbi_build_utils import RNN, MLP
from sbi_build import buildPosterior, simulate
from sbiutils import reshape_y, createYHatList
from posteriorsamputils import statTheta, genProbThetas, gen_Fit_Series_Wrapper

# LHS sampling - https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html
import skopt
from skopt.space import Space
from skopt.sampler import Lhs

'''
Build / Load Ensemble - Globals
'''
# -- Ensemble Globals
ensemble_name = '0819_01_mod2'
ensemble_path = f'/home/qh8373/SBI_TAYLOR/data/03_ensemble_out/_ensemble_{ensemble_name}/'

# -- LSTM Globals
Build_LSTM = False 
lstm_name = '02_09_lstm_C' # '02_06_lstm_C' # '02_05_lstm_A' # '11_09_log_mod' # 10_13_log_mod' (extremes reserved) # '09_13_log_mod' (totally random) # '10_13_log_mod_b' (from 10_13)
lstm_path = f'/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_name}/'

if Build_LSTM:
    random_flag = False # LSTM test-train selection method (if false, will remove 0-1 from test set)
    save_lstm = True
    shuffle_it_in = False
    num_members = 10
    
load_test_idx = True # if True, will load reserved test set from previous lstm, to be used to build LSTM
if load_test_idx:
    lstm_load = '10_13_log_mod'
    lstm_load_path = f'/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_load}/'
else:
    lstm_load = False
    lstm_load_path = None

# -- SBI Globals
Build_SBI = True
Sample_SBI = True
sbi_name = '0210_01' # '1013_lstm_stress_test' 

# -- Truth Type  (argumens below only for using surrogate truths)
truth_type = 'surrogate' # this is the type of truth_type = 'surrogate', 'parflow', 'observation'
if truth_type == 'surrogate':
    load_params = True # boolean of weather or not load params, and directory for loading (below)
    mix_load = True # if mix load, then loads params from lstm test data, too
    load_params_dir = '/home/qh8373/SBI_TAYLOR/data/05_sbi_out/0819_01_mod2_10_13_log_mod_1107_01_full_onelstm_0_raw/' # '/home/qh8373/SBI_TAYLOR/data/05_sbi_out/0819_01_mod2_10_13_log_mod_1107_01_full_onelstm_0_raw/' # '/home/qh8373/SBI_TAYLOR/data/05_sbi_out/0819_01_mod2_09_13_log_mod_1107_01_full_onelstm_1/'
    set_num_unique = 82 # number of unique 'truths' to test if truth_type == surrogate ONLY
else:
    load_params = None
    mix_load = None
    load_params_dir = None
    set_num_unique = None

# -- Ensemble
es_ensemble = True # if you want to use the full ensemble of possibilities set to true
idx_ensemble = 0 # some index to preferentially choose one single LSTM emulator as the 'simulator'

# -- Noise
add_noise = False # set to true to add some gaussian randomness, scaled by the factor f_noise below
f_noise = 1e-02 # scaling factor (multiplied times every point in the time series). Anything bigger than 1e-02 could overwhelm natural signal

# Statistics and stat typ
stat_method = 'full' #   stat_method = 'summary', 'full', 'embed'
stat_typ = None # np.array([4,12,13]) # np.array([1,12,13]) # np.array([9,10])  # np.array([1,3,4,5,7,9,10,11])   # np.array([9,10]) # np.array([9,10]) #   use arrays for multiple parameters np.array([9,10])
out_dim = None # 3 # 2 # number of dimensions for ML
embed_type = None # 'MLP' # 'MLP', 'CNN', 'RNN'
stat_typ = retStatTyp(stat_method, stat_typ=stat_typ, out_dim=out_dim, embed_type=embed_type)

# hyperparameters
L_sims = 10 # for l in L (number of sbi parameter spaces to create...)
num_dim = 2 # number of dimensions of parameters *NOTE - REEVALUATE THIS
chars = ['[', ',', ']'] # for scaling things (needs to be consistent with num_dims)
meth, model = 'SNPE', 'maf' # method, model for sbi
hidden_features = 10
num_transforms = 2
n_sims = 1000 # number of simulations for sbi this affects speed (100000 ~ 45 minutes, 10000 ~ 3 minutes, 1000 ~ 30 seconds)
n_samples = 5000 # number of samples for sbi this doesn't affect speed (more is better, though)

# prior meta variables - 'uniform' or 'lognormal'
prior_type = 'uniform'
prior_arg1 = 0. # this is min for uniform, loc for lognormal (LN: -3 is good for scalage between 0 and 1 over 4 orders of magnitude)
prior_arg2 = 1. # this is max for uniform, scale for lognormal (LN: 1 is good for scalage between 0 and 1 over 4 orders of magnitude)

#   brief textual description
desc = 'Creating Posteriors for Experiment 3 - Ensemble Modeling'
sbi_full_name = f'{ensemble_name}_{lstm_name}_{sbi_name}_{stat_typ}_{truth_type}'
sbi_dir = f'/home/qh8373/SBI_TAYLOR/data/05_sbi_out/{sbi_full_name}/'

try:
    os.mkdir(f'{sbi_dir}')
except:
    print('warning: file exists')
    pass

'''
If Build LSTM:
'''
if Build_LSTM:
    '''
    Set LSTM Build Variables
    '''
    
    '''
    build LSTM
    1. Define HyperParmaeters
    2. Arrange Data
    3. Save LSTM Build Information
    4. Train test routine for LSTMs
        1. Randomly set train / validation splits
        2. Train LSTM
    '''
    list_df_cond = buildLSTM(lstm_name, lstm_path, save_lstm, shuffle_it_in, 
                            num_members, ensemble_name, ensemble_path, 
                            random_flag=random_flag, load_test_idx=load_test_idx,
                            lstm_load_path=lstm_load_path)

else:
    '''
    load LSTM
    # stored as a list of dataframes containing all the data and models for each 
    # lstm ensemble run
    '''
    with open(lstm_path+'list_df_cond.pkl', 'rb') as fp:
        list_df_cond = pickle.load(fp)

'''
If Build SBI:
'''
if Build_SBI:
    '''
    parse list_df_cond [sbiutils]
        test params is used later on to test observations
    '''
    DataX_test, DataY_test, series_len, lstm_out_list = parseListDf(list_df_cond)
    test_params, num_params, num_unique, DataX_test = parseUniqueParams(DataX_test, series_len)
    del list_df_cond
    

    '''
    Make decisions about how to run inference based on if it is 
        parflow, surrogate, observation, something else
    '''
    # TODO move to different file for functions
    
    if truth_type == 'parflow':
        test_params = test_params
    elif truth_type == 'observation':
        None
    elif truth_type == 'surrogate':
        # handle parameters for surrogate model
        # if load_params = False:
            # 1. define length of truths / parameters
            # 2. choose a random params using latin hypercube sampling
        if load_params == False:
            # choose random params using lhc
            space = Space([(prior_arg1, prior_arg2), (prior_arg1, prior_arg2)])
            lhs = Lhs(criterion="maximin", iterations=10000)
            x = lhs.generate(space.dimensions, set_num_unique)
            x = torch.tensor(x)
        
            # reload the lstm truths
            if mix_load == True:
                test_params = torch.cat((x,test_params))
            else:
                # 
                test_params = x
    
            num_unique = len(test_params)
            del x, lhs, space

        # if load_params == True
            # 1. load in the random variables pickled earlier 
            # 2. define length of truths / parameters
        else:
            with open(f"{load_params_dir}test_params.pkl", "rb") as fp:
                test_params = pickle.load(fp)
            print(len(test_params))
            num_unique = len(test_params)
        
        # choose your lstm
        lstm_out = lstm_out_list[idx_ensemble]

        # forward simulation [days,1]
        DataY_test = torch.empty((num_unique*series_len,1))
        for u in range(num_unique): #num_unique
            theta = test_params[u, :]
            DataY_out = simulate(DataX=DataX_test, theta=theta, lstm=lstm_out)
            DataY_test[int(u*series_len):int((u+1)*series_len),:] = DataY_out
        del lstm_out, theta, u


    '''
    We have to make some decisions about inference based on if we run an ensemble
    '''
    if es_ensemble==False:
        lstm_out_list = [lstm_out_list[idx_ensemble]]
        
        
    '''
    create list of observations summarized by stat_typ [sbiutils]
        unique_series is used later on to test observations
    '''
    
    unique_series = createYHatList(DataY_test, series_len, num_unique,
                    stat_method, stat_typ=stat_typ, embed_type=embed_type)
    
    unique_series_full = createYHatList(DataY_test, series_len, num_unique,
                        stat_method='full', stat_typ=None, embed_type=None)
    
    
    '''
    create embedding net
    '''
    if embed_type == 'MLP':
        embedding_net = MLP(in_dim=series_len, out_dim=out_dim)
    elif embed_type == 'RNN':
        embedding_net = RNN(data_size=series_len, hidden_size=20, output_size=out_dim)
    else:
        embedding_net = None
    
    '''
    save inference 'build' information {sbi_dir}
    '''
    with open(f"{sbi_dir}lstm_out_list.pkl", "wb") as handle:
        pickle.dump(lstm_out_list, handle)
    
    with open(f"{sbi_dir}unique_series.pkl", "wb") as handle:
        pickle.dump(unique_series, handle)
    
    with open(f"{sbi_dir}unique_series_full.pkl", "wb") as handle:
        pickle.dump(unique_series_full, handle)

    with open(f"{sbi_dir}test_params.pkl", "wb") as handle:
        pickle.dump(test_params, handle)
    
    with open(f"{sbi_dir}DataX_test.pkl", "wb") as handle:
        pickle.dump(DataX_test, handle)
    
    with open(f"{sbi_dir}embedding_net.pkl", "wb") as handle:
        pickle.dump(embedding_net, handle)
        
    with open(f"{sbi_dir}num_unique.pkl", "wb") as handle:
        pickle.dump(num_unique, handle)
        
    with open(f"{sbi_dir}L_sims.pkl", "wb") as handle:
        pickle.dump(L_sims, handle)
        
    with open(f"{sbi_dir}series_len.pkl", "wb") as handle:
        pickle.dump(series_len, handle)
        
    with open(f"{sbi_dir}load_params_dir.pkl", "wb") as handle:
        pickle.dump(load_params_dir, handle)
    
    with open(f'{sbi_dir}/val.txt', 'w') as file:
        books = [f'Brief Textual Description: {desc}',
                 f'SBI Full Name: {sbi_full_name}',
                 f'Stat Method: {stat_method}',
                 f'Summary Statistic ID: {stat_typ}',
                 f'Output Dimension (embedding Net Only): {out_dim}',
                 f'Embed type: {embed_type}',
                 f'Instantion time is: {datetime.now()}',
                 f'Number of SBI Runs: {L_sims}',
                 f'Number of parameter Dimensions: {num_dim}',
                 f'Inference - method:{meth} - model:{model} - hidden_features:{hidden_features} - num_transforms:{num_transforms}',
                 f'Number of Simulations: {n_sims}',
                 f'Number of samples from posterior: {n_samples}',
                 f'Prior Function: {prior_type}',
                 f'Prior_Arg1:{prior_arg1} - Prior_Arg2:{prior_arg2}',
                 f'ParFlow Ensemble Name: {ensemble_name}',
                 f'LSTM model name: {lstm_name}',
                 f'truth type : {truth_type}',
                 f'mix load params : {mix_load}',
                 f'Is an ensemble? : {es_ensemble}',
                 f'Add Noise? What Noise Factor? : {add_noise}, {f_noise}',
                 f'What index lstm (for ensemble, and for surrgoate truth)? : {idx_ensemble}',
                 f'Load Params? : {load_params}',
                 f'Load Param Directory : {load_params_dir}',
                 f'*Test* Parameter Values : {test_params}'
                 ]
        file.writelines("% s\n" % data for data in books)

    '''
    loop through multiple sbi simulations
    '''
    for l in range(L_sims):
        '''
        save sbi dir sub
        '''
        sbi_dir_sub = f'/home/qh8373/SBI_TAYLOR/data/05_sbi_out/{sbi_full_name}/{l}/'
        try:
            os.mkdir(f'{sbi_dir_sub}')
        except:
            print('warning: file exists')
            pass

        '''
        build sbi
            create prior [sbi_build, helper]
            define simulator [sbi_build, helper]
            prepare simulator for sbi [sbi_build, helper]
            build posterior [sbi_build, helper]
        '''
        posterior, end_time = buildPosterior(prior_type, prior_arg1, prior_arg2, num_dim,
                            DataX=DataX_test, theta=None,
                            lstm_out_list=lstm_out_list, 
                            stat_method=stat_method, stat_typ=stat_typ,
                            meth=meth, model=model, hidden_features=hidden_features,
                            num_transforms=num_transforms, n_sims=n_sims, n_samples=n_samples,
                            embedding_net=embedding_net, add_noise=add_noise, f_noise=f_noise)
        '''
        save at child level 1
        '''
        with open(f"{sbi_dir_sub}posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
            
        with open(f'{sbi_dir_sub}/val.txt', 'w') as file:
            books = [f'Posterior Run Time: {end_time}']
            file.writelines("% s\n" % data for data in books)
            

else:
    '''
    load SBI posterior and relevant metadata
    '''
    # load back 
    with open(f"{sbi_dir}unique_series.pkl", "rb") as fp:
        unique_series = pickle.load(fp)
    
    with open(f"{sbi_dir}unique_series_full.pkl", "rb") as fp:
        unique_series_full = pickle.load(fp)

    with open(f"{sbi_dir}test_params.pkl", "rb") as fp:
        test_params = pickle.load(fp)
        
    with open(f"{sbi_dir}num_unique.pkl", "rb") as fp:
        num_unique = pickle.load(fp)
        
    with open(f"{sbi_dir}L_sims.pkl", "rb") as fp:
        L_sims = pickle.load(fp)
        
    with open(f"{sbi_dir}DataX_test.pkl", "rb") as fp:
        DataX_test = pickle.load(fp)
        
    with open(f"{sbi_dir}lstm_out_list.pkl", "rb") as fp:
        lstm_out_list = pickle.load(fp)
        
    with open(f"{sbi_dir}series_len.pkl", "rb") as fp:
        series_len = pickle.load(fp)
        
    # '''
    # ParFlow truth loads
    # lstm_df_cond should be loaded
    # to test ParFlow truths on models trained with LSTMs and previous tested with LSTM synthetic truths
    # '''
    
    # DataX_test, DataY_test, series_len, lstm_out_list = parseListDf(list_df_cond)
    # test_params, num_params, num_unique, DataX_test = parseUniqueParams(DataX_test, series_len)
    # unique_series_PF = createYHatList(DataY_test, series_len, num_unique,
    #                 stat_method, stat_typ=stat_typ, embed_type=embed_type)
    # unique_series_full_PF = createYHatList(DataY_test, series_len, num_unique,
    #                     stat_method='full', stat_typ=None, embed_type=None)
    
 
if Sample_SBI:
    '''
    Sample SBI at all observations
        (unique series), (unique_series_full)
    For every posterior simulation
        (in L_sims)
    
    bring back and set important values for df_post_samps
        test_params
        unique_series
        unique_series_full
    '''
    
    df_post_samps = pd.DataFrame(columns=['y_hat', 'y_hat_full', 'true_theta', 
                                'posterior_samples', 'log_probability',
                                'thetaTypeList', 'thetaStatsList', 'probStatsList',
                                ])
    

    df_post_samps['y_hat'] = unique_series
    df_post_samps['y_hat_full'] = unique_series_full
    df_post_samps['true_theta'] = test_params


    for l in range(L_sims):
        '''
        loop through posterior samples (load 'em up)
        '''
        sbi_dir_sub = f'/home/qh8373/SBI_TAYLOR/data/05_sbi_out/{sbi_full_name}/{l}/'
    
        with open(f"{sbi_dir_sub}posterior.pkl", "rb") as fp:
            posterior = pickle.load(fp)

        for idx in range(num_unique):
            '''
            save sbi dir sub sub
            '''
            idx_string = "{:02d}".format(idx)
            sbi_dir_sub_sub = f'/home/qh8373/SBI_TAYLOR/data/05_sbi_out/{sbi_full_name}/{l}/{idx_string}/'
            try:
                os.mkdir(f'{sbi_dir_sub_sub}')
            except:
                print('warning: file exists')
                pass
            '''
            sample observation (y_hat) and 'correct' parameter (true_theta)
                y_hat_full is the full timeseries (important for embed and summary stat method)
            '''
            y_hat = unique_series[idx]
            y_hat_full = unique_series_full[idx]
            true_theta = test_params[idx]
    
            '''
            sample posterior and create log probability
            '''
            # - given observation(s), sample posterior, evaluate probability, and plot
            posterior_samples = posterior.sample((n_samples,), x=y_hat) # type - tensor object
            # posterior_samples_simulations = bulk_simulate(posterior_samples, num_days, n_samples, ensemble_path, model_path)
            log_probability = posterior.log_prob(posterior_samples, x=y_hat).unsqueeze(1) # type - tensor object
            
            '''
            log probability of 'correct' parameters
            '''
            log_prob_true_thetas_ind, log_prob_true_thetas_all, log_prob_true_thetas_flag = genProbThetas(params_true_scaled=true_theta,
                                                                                            posterior_samples=posterior_samples.numpy(),
                                                                                            log_probability=log_probability.numpy()) # log probability of 'true' thetas
            print('log prob true', log_prob_true_thetas_all)
    
            # log_prob_true_thetas_ind, log_prob_true_thetas_all, log_prob_true_thetas_flag = None, None, None
    
    
            '''
            Interpretive Thetas
                Average, Median, Max Probability, Mode, Random Thetas Sampled from Posterior
                Order: ['mu', 'median', 'max', 'mode', 'random']
            '''
            thetaTypeList, thetaStatsList, probStatsList = statTheta(posterior_samples, log_probability)
    
            '''
            Generate full series of results (forward simulations 
                based off of inferred parameters)
                & 
                Fit Results
            '''
            # something wrong with this function (fitarr)
            seriesarr, fitarr = gen_Fit_Series_Wrapper(lstm_out_list, DataX_test, y_hat_full,
                                                true_theta, series_len,
                                                thetaTypeList, thetaStatsList)
                                                
            '''
            Pair Plot
            '''
            fig, axes = analysis.pairplot(posterior_samples, 
                                               points=torch.tensor(true_theta),
                                               limits=[[0, 1],[0, 1]], 
                                               figsize=(6,6)) #  
            fig.savefig(f'{sbi_dir_sub}pair_plot_{idx_string}.png')
            fig.savefig(f'{sbi_dir_sub_sub}pair_plot.png')
            fig.savefig(f'{sbi_dir_sub_sub}pair_plot.eps', format='eps')
            plt.close()
            '''
            Generate Plots
            '''
            
            '''
            Save at Child Level 2
            '''
            with open(f"{sbi_dir_sub_sub}y_hat.pkl", "wb") as handle:
                pickle.dump(y_hat, handle)
            
            with open(f"{sbi_dir_sub_sub}y_hat_full.pkl", "wb") as handle:
                pickle.dump(y_hat_full, handle)

            with open(f"{sbi_dir_sub_sub}true_theta.pkl", "wb") as handle:
                pickle.dump(true_theta, handle) 
                
            with open(f"{sbi_dir_sub_sub}posterior_samples.pkl", "wb") as handle:
                pickle.dump(posterior_samples, handle)       
            df_post_samps['posterior_samples'].iloc[idx] = posterior_samples
            
            with open(f"{sbi_dir_sub_sub}log_probability.pkl", "wb") as handle:
                pickle.dump(log_probability, handle)  
            df_post_samps['log_probability'].iloc[idx] = log_probability
    
            df_post_samps['thetaTypeList'].iloc[idx] = thetaTypeList
            df_post_samps['thetaStatsList'].iloc[idx] = thetaStatsList
            df_post_samps['probStatsList'].iloc[idx] = probStatsList
            
            with open(f"{sbi_dir_sub_sub}seriesarr.pkl", "wb") as handle:
                pickle.dump(seriesarr, handle)  
            
            with open(f"{sbi_dir_sub_sub}fitarr.pkl", "wb") as handle:
                pickle.dump(fitarr, handle)  
            
            with open(f'{sbi_dir_sub_sub}/val.txt', 'w') as file:
                books = [f'Observation: {y_hat}',
                         f'Theta (scaled): {true_theta}',
                         f'Stat Method: {stat_method}',
                         f'thetaTypeList: {thetaTypeList}',
                         f'thetaStatsList: {thetaStatsList}', 
                         f'probStatsList: {probStatsList}',
                         f'log probe true thetas indx: {log_prob_true_thetas_ind}',
                         f'log_prob_true_thetas_all: {log_prob_true_thetas_all}',
                         f'log_prob_true_thetas_flag: {log_prob_true_thetas_flag}']
                file.writelines("% s\n" % data for data in books)
    
# if nothing else
else:
    None

print('script complete')


