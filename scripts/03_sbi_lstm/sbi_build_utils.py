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


import os
import sys
import pickle
import torch


'''
Embedding Nets
'''
class MLP(nn.Module):
    '''
    Multilayer Perceptron 
    Simplest use case of embedding_net
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 10),
          nn.ReLU(),
          nn.Linear(10, out_dim)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)   

class RNN(nn.Module):
    '''
    RNN to be used for embedding_net
    '''
    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output

class Box_LogNormal(Independent):
    def __init__(
        self,
        loc: ScalarFloat,
        scale: ScalarFloat,
        reinterpreted_batch_ndims: int = 1,
        device: str = "cpu",
    ):
        """Multidimensional lognormal distribution defined on a box.
        A `Uniform` distribution logNormal with e.g. a parameter vector loc or scale of
         length 3 will result in a /batch/ dimension of length 3. A log_prob evaluation
         will then output three numbers, one for each of the independent log_probs in
         the batch. Instead, a `Box_LogNormal` initialized in the same way has three
         /event/ dimensions, and returns a scalar log_prob corresponding to whether
         the evaluated point is in the box defined by loc and scale or outside.
        Refer to torch.distributions.LogNormal and torch.distributions.Independent for
         further documentation. 
        Args:
            loc: center of distribution (inclusive).
            scale: range of distribution (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
            device: device of the prior, defaults to "cpu", should match the training
                device when used in SBI.
        
        QH - 08102021 Code modified from original in sbi mckleab
        https://github.com/mackelab/sbi/blob/main/sbi/utils/torchutils.py
        
        lognormal documented here - https://pytorch.org/docs/stable/distributions.html#lognormal
        """

        super().__init__(
            LogNormal(
                loc=torch.as_tensor(loc, dtype=torch.float32, device=device),
                scale=torch.as_tensor(scale, dtype=torch.float32, device=device),
                validate_args=False,
            ),
            reinterpreted_batch_ndims,
        )





