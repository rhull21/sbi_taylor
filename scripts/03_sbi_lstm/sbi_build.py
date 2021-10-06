# for SBI
from sbi import utils as utils
from sbi import analysis as analysis
from sbi import inference
from sbi.inference.base import infer
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.types import Array, OneOrMore, ScalarFloat

import sys
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/')
from sbi_build_utils import Box_LogNormal
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from sbiutils import setTheta, reshape_y
from summaryutils import setStatSim

import random
import os
import sys
import pickle
import torch
from datetime import datetime

def createPrior(prior_type, prior_arg1, prior_arg2, num_dim):
    '''
    Create the prior for SBI
    '''
    # - 1) a prior distribution that allows to sample parameter sets.
    #   - A prior is a distribution that allows us to sample parameter sets. Any class is allowed
    #       as long as it calls prior.sample()      

    # - uniform disbributrion between 0 and 1 for n dimensions
    if prior_type == 'uniform':
        low = torch.ones(num_dim)*prior_arg1
        high = torch.ones(num_dim)*prior_arg2
        prior = utils.BoxUniform(low=low, high=high)
    elif prior_type == 'lognormal':
        loc=torch.ones(num_dim)*prior_arg1
        scale=torch.ones(num_dim)*prior_arg2
        prior = Box_LogNormal(loc=loc, scale=scale)
    return prior
    
def simulate(DataX, theta, lstm):
    '''
    - 2) a simulator that takes parameter sets and produces simulation outputs.
      - A simulator is a callable that takes in a parameter set and outputs data with
          (at least some) degree of stoachsticity
    DataX : Data array where last len(theta) values are theta and need set
    theta : theta values to set
    lstm : lstm model
    '''
    
    DataX = setTheta(DataX, theta)
    lstm.eval()
    simulation = lstm(DataX)
    
    return simulation
    
    '''
    '''
    
def buildPosterior(prior_type, prior_arg1, prior_arg2, num_dim,
                    DataX, theta, lstm_out_list,
                    stat_method, stat_typ,
                    meth, model, hidden_features,
                    num_transforms, n_sims, n_samples,
                    embedding_net):
    '''
       build sbi
            create prior [sbi_build, helper]
            define simulator [sbi_build, helper]
            prepare simulator for sbi [sbi_build, helper]
            build posterior [sbi_build, helper]
            
        prior_type : 
        prior_arg1 : 
        prior_arg2 :
        num_dim :
        DataX : 
        theta : 
        lstm_out_list :
        stat_method :
        stat_typ : 
        meth : 
        model : 
        hidden_features : 
        num_transforms : 
        n_sims : 
        n_samples :
    '''
    start_time = datetime.now()
    
    # - 1) a prior distribution that allows to sample parameter sets.
    #   - A prior is a distribution that allows us to sample parameter sets. Any class is allowed
    #       as long as it calls prior.sample()  
    prior = createPrior(prior_type, prior_arg1, prior_arg2, num_dim)
    
    # - 2) a simulator that takes parameter sets and produces simulation outputs.
    #   - A simulator is a callable that takes in a parameter set and outputs data with
    #       (at least some) degree of stoachsticity
    
    def simulator(theta):
        '''
        a simple simulator that for the sake of example adds some Gaussian noise to the parameter set
        '''
        # # add 'stochasticity' (ca) (w/o ensembles..)
        # y_o = y_o + y_o * np.random.randn(y_o.shape[0]) * 0.05 

        # simulate, using either a randomly generate ensemble member or a series with random noise
        iterator = random.randint(0,len(lstm_out_list)-1)
        lstm_out = lstm_out_list[iterator]
        y_o = simulate(DataX=DataX, theta=theta, lstm=lstm_out).data.numpy()[:,0]

        # summary statics
        if stat_method == 'summary':
            stat_sim = setStatSim(y_o, stat_typ)
            result = torch.tensor(stat_sim)
        # full time series
        elif (stat_method == 'full') or (stat_method == 'embed'):
            result = reshape_y(y_o)
        return result
    
    # - 3) Prepare simulator and run inference
    # make a SBI-wrapper on the simulator object for compatibility
    simulator_wrapper, prior = prepare_for_sbi(simulator, prior)
    
    # instantiate the neural density estimator
    if stat_method == 'embed':
        neural_posterior = utils.posterior_nn(model=model, 
                                              embedding_net=embedding_net,
                                              hidden_features=hidden_features,
                                              num_transforms=num_transforms)
    else:
        neural_posterior = utils.posterior_nn(model=model, 
                                              hidden_features=hidden_features,
                                              num_transforms=num_transforms)
    
    # setup the inference procedure with the SNPE-C procedure
    if meth == 'SNPE':
        inference = SNPE(prior=prior, density_estimator=neural_posterior)
    
    # run the inference procedure on one round and n_sims simulated data points
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=n_sims)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    
    end_time = (datetime.now()-start_time)
    
    return posterior, end_time
    
    

    