# %%

from PIL import Image

import os
import os.path
import sys
import shutil
from pprint import pprint
from datetime import datetime
from copy import copy
from copy import deepcopy

from parflowio.pyParflowio import PFData
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from genutils import PFread

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

# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset # for refactoring x and y
# from torch.utils.data import DataLoader # for batch submission
# from torch.autograd import Variable
# from lstm_models import sliding_windows
# from lstm_models import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Path to the SandTank Repo
dev_path = '/home/SHARED/ML_TV/HydroGEN/modules/'
#Add Sand Tank path to the sys path
sys.path.append(dev_path)
from transform import float32_clamp_scaling

# user defined functions
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from genutils import PFread
from assessutils import compute_stats

    
''' forcing functions '''


def _assembleForcings(YEARS,label,SITE=0,path='../data_out'):
    '''
    returns a numpy array of forcings from site SITE for years YEARS of forcing label
    
    YEARS : list of years of record
    label : name of forcing (see labelist)
    path : path to forcings (by default '../data_out')
    SITE : index to site (see out_data.shape[0]) (by default, 0 because huc_name_list)
    
    *Updated 04/24/2021*
        * For files of name type `/UCRB_DSWR.1983.pfb`
        * You will see they are shape `out_data.shape` = `(5, 3, 366)`
        * Where `out_data.shape[0] = 5` -> number of stations in the dataset of this order `[9110000, 9329050, 9196500, 9107000, 9210500]`
        * Where `out_data.shape[1] = 3` -> scaler outputs of dynamic variables `0 = mean, 1 = std over tim, 2 = std over space`
        * Where `out_data.shape[2] = 366` -> number of days in a year (where the last is an extra just for leap years that is normally zero)
    
    Provenance: 
        `scripts/UCRB_figs.ipynb`
        `00_Principle_UCRB_Scripts/UCRB_LSTM_qh_mod_06132021.ipynb`
    '''
    for yr in YEARS: 
        # pull out data test
        out = PFData(path+'/UCRB_'+label+'.'+str(yr)+'.pfb')
        out.loadHeader()
        out.loadData()
        out_data = out.getDataAsArray()
        # get ride of random zeros in last row if not leap year
        # QH - this needs to get revised. Doesn't make sense to remove last date as water years
        if out_data[SITE,:,-1].all() == np.zeros((3)).all():
            out_data = out_data[SITE,:,:-1]
        else:
            out_data = out_data[SITE,:,:]

        # create array of placeholder values
        if yr == YEARS[0]:
            label_arr = out_data
        else:
            label_arr = np.concatenate([label_arr,out_data], axis=2)
            # print(label_arr.shape)

    return label_arr

def assembleAllForcings(labelist,YEARS,SITE,path_forcings,date0,extract_idx=0):
    '''
    labelist : list of the different forcing components
    YEARS : list of years (text)
    label : SITE (name of the site), format 9110000 (numeric)
    path_forcings : path to forcings, string
    date0 : string date of first date in time series
    extract_idx : index (0, 1, 2) respectively for scaler outputs of dynamic variables `0 = mean, 1 = std over tim, 2 = std over space`
        default = 0
        See `_assembleForcings for more documentation`
    '''

    for label in labelist:
        # Assemble Data
        label_arr_out = _assembleForcings(YEARS=YEARS,label=label,SITE=SITE,path=path_forcings)
        label_arr_out = deepcopy(label_arr_out) # not sure why this needs to be done
        
        # for indexing colorlist
        j = labelist.index(label)
        
        # Assemmble data frame
        if j == 0:
            df_forc = pd.DataFrame()
        # print(label_arr_out[0, :])
        df_forc[label] = label_arr_out[int(extract_idx),:] # 0 for mean 

        # if label == 'DLWR':
        #     print(label_arr)
        #     print(df_forc)
        
        del label_arr_out, j

        
    # Arrange Dates
    # set up date indexes
    df_forc['deltatime'] = pd.to_timedelta(df_forc.index, unit='D')
    df_forc.index = pd.Timestamp(date0) + df_forc['deltatime']
    df_forc.index.names = ['timestamp']
    df_forc.drop(columns=['deltatime'], inplace=True)
    
    return df_forc
    
''' streamflow functions '''


def convert_flow(inp,in_units='m3_hr',out_units='cfs'):
    '''
    converts from native parflow units (m^3 / hr) to human interpretable unit (cfs) [flexible]
    inp : an 'n' dimensional numpy array with units of in_units
    '''
    conversion = (1/0.0283168)*(1/60)*(1/60)
    if in_units == 'm3_hr':
        out = inp*conversion
        
    if in_units == 'cfs':
        out = inp*(1/conversion)
        
    return out
    
def _extract_sf_data(in_path,idx=2):
    '''
    Function returns a dataframe of simulated flow (for one event)
    in_path : pathway (including csv) for import data
    idx : note, data structure includes a partition for gage_flow and max_flow within watershed domain.
        Default 2 = gage_flow
        See Data Structure `(0 = index (1), 1 =  index (2), 2 = gage_flow, 3 = max_flow)
        
    '''
    out_data = pd.read_csv(in_path).to_numpy()
    if idx == 2:
        nm = 'gage_flow'
    elif idx == 3:
        nm = 'max_flow'
    df = pd.DataFrame(out_data[:,idx], columns = [nm])
    return df

def _parseSF(df, date0, member_name):
    '''
    Synthesizes a dataframe containing hourly streamflow (with no date idx) 
        and returns streamflow for converted units and daily
    Takes --
        df : a dataframe with raw parflow streamflow outputs
        date0 : a string containing the initial date of streamflow
        member_name : attribute from _parseMetaData()[0]
    Returns --
        df : a converted streamflow
    '''
    # condense to daily average
    df['deltatime'] = pd.to_timedelta(df.index, unit='h')
    df['timestamp'] = pd.Timestamp(date0) + df['deltatime']
    df = df.resample(rule='D',on='timestamp').mean()
    
    # convert from m^3 / hr to cfs
    ret = convert_flow(df.to_numpy())
    df['gage_flow'] = ret[:,0]
    df.rename(columns={'gage_flow': f'Flow_{member_name}'}, inplace=True)
    return df
    
def returnDF_ens(ens_list, date0, name_list):
    '''
    Function returns a datafrome of streamflow ensemble data
    Takes --
        ens_list : list of full file locations of streamflow
        date0 : date (string) at time 0
        name_list : list of full names of members (for plotting), could be any array for names, member_ens_list e.g.
    Returns --
        df_ens : dataframe of streamflow ensemble data
    '''
    idx = 0
    for test_path in ens_list:
        name = name_list[idx] # set name
        
        df_temp = _extract_sf_data(test_path) # extract sf data
        df_out = _parseSF(df_temp, date0, name) # manipulate to correct format
        
        # assemble new dataframe
        if idx == 0:
            df_ens = df_out
        else:
            df_ens[f'Flow_{name}'] = df_out[f'Flow_{name}']
        
        del df_out
        idx = idx + 1
    
    return df_ens
    
''' metadata functions '''
    
def lookUpIdx(AOC):
    '''
    set length of AOC value by looking for characters A-Z (big)
    AOC : a string that may or may not contain a character A-Z (ascii 65- 91)
    returns : a list of all the indexs
    '''
    asc_l = []
    for asc in range(65, 91):
        if AOC.find(chr(asc)) > -1:
            asc_l.append(AOC.find(chr(asc)))
    return asc_l

    
def _parseMetadata(in_path, ensemble_name):
    '''
    Function parses important metadata from the name
    in_path : pathway (including csv) for import data
    ensemble_name : 7 character ensemble name (like `0626_01`)
    returns --
    [0] member_name : the name of the member (lengthy and not often used)
    [1] yr_name : the name of the year in the particular member (int)
    [2] AOC : The attributes of concern (str), including denotation of type, for later use in parsing
    '''
    member_name = in_path[in_path.find(ensemble_name)+len(ensemble_name)+1:-13]
    yr_name = int(in_path[-25:-21])
    AOC = member_name[:member_name.find('_')]
    return member_name, yr_name, AOC
    
def parseAllMetadata(ens_list, ensemble_name):
    '''
    pass ens_list
    returns the metadata in three lists encoded in a tuple
    [0] member_name_ens : the name of the member (lengthy and not often used) inherited from member_name
    [1] yr_name_ens : the name of the year in the particular member (int) inherited from yr_name
    [2] AOC_ens : The attributes of concern (str), including denotation of type, for later use in parsing inherited from AOC
    '''
    member_name_ens = []
    yr_name_ens = []
    AOC_ens = []
    for test_path in ens_list:
        metadata = _parseMetadata(test_path, ensemble_name)
        member_name_ens.append(metadata[0])
        yr_name_ens.append(metadata[1])
        AOC_ens.append(metadata[2])
        
    return member_name_ens, yr_name_ens, AOC_ens

''' parsing AOC information '''
def _parseAOC(AOC):
    '''
    read in AOCs (attributes of concern) from member name
        if old naming scheme (< 06262021), then simple
        else if new naming scheme (> 06262021), then hard with multiple parameters
    returns ---  
    AOC_temp : parsed AOC type = list as values
    '''
    # old naming scheme
    if len(lookUpIdx(AOC)) == 0:
        AOC_temp = [float(AOC)]
    
    # new naming scheme
    else:
    
        le = 0
        AOC_temp = []
        fin = False

        # loop through all AOCs based on lists
        while fin == False:
            # reset AOC and remove the leading characters
            AOC = AOC[AOC.find('-')+1:]
            asc_l = lookUpIdx(AOC)
            # finish loop if there are no more   
            if len(asc_l) == 0:
                fin = True
                le = len(AOC)
            else:
                le = min(asc_l) 
            # slice string to include just the string / number
            # convert string to float
            AOC_temp.append(float(AOC[:le-1]))
            # update AOC
            AOC = AOC[le:]
            if fin:
                break
        
        del le, asc_l, fin
        
    return AOC_temp
    
def returnAOC_ens(AOC_ens):
    '''
    returns a multidimensional list of AOCs into a numpy array of shape...
    Takes --
        AOC_ens : a list of members, unparsed
    Returns --
        AOC_ens_l : a numpy array prased, of 
    '''
    AOC_ens_l = []
    for AOC in AOC_ens:
        AOC_out = _parseAOC(AOC)
        AOC_ens_l.append(AOC_out)
    
    AOC_ens_l = np.array(AOC_ens_l)
    
    return AOC_ens_l
    
    
''' misc functions '''

def assembleYears(yr_min,yr_max, day0=1, mon0=10):
    '''
    passess in yr_min and yr_max (optionally, edit the y0, day0, and mon0 if not start of water year)
    returns: 
        - a list of YEARS
        - date_0 
    '''

    YEARS = []
    for yr in range(yr_min,yr_max): # QH 1983, 2020 for all years
        YEARS.append(yr)
    # set instance of first day (assume Oct 1, YYYY[0]-1)
    if (day0 == 1) and (mon0 == 10):
        y0 = (yr_min-1)
    else:
        y0 = yr_min
    date0 = str(mon0)+'/'+str(day0)+'/'+str(y0)

    return YEARS, date0

            
# def GenScale(min_label_list, max_label_list, dist_range=[0, 1]):
#     '''
#     Automatically creates scales to be used to scale data
#     min_label_list : minimum values
#     max_label_list : max values
#     dist_range : acceptable range of values for scaling
    
#     returns - list of 
#     '''
    
    
#     if len(min_label_list) != len(max_label_list):
#         print('different number of minimums and maximums')

#     scale_l = []

#     for j in range(len(min_label_list)):
#         transf = float32_clamp_scaling(src_range=[min_label_list[j],max_label_list[j]], dst_range=[0, 1])
#         scale_l.append(transf)
        
#     return scale_l


''' min and max '''

def _ret_MinMax(labelist, df_l):
    '''
    identifies the min and max of all elements of a dataframe (df_l) containing 
        forcing and streamflow data.
    Takes ----
    labelist : label of members of structure [forcing1, forcing2, ..., 'Flow']
    df_l : dataframe containing unscaled forcing and flow data
    Returns ---
    min_label_list, max_label_list : lists of min and max in labelist (shared index)
    '''

    min_label_list = []
    max_label_list = []
    
    i = 0
    # loop through all forcings
    for elem in labelist[:-1]:
        # set minimums / maximums
        min_label_list.append(df_l[elem].min())
        max_label_list.append(df_l[elem].max())    
        i = i + 1
        
    min_label_list.append(1000000)
    max_label_list.append(-1000000)
    
    for elem in df_l.columns[i:]:
    
        # set minimum for element in flow dataset
        min_local = df_l[elem].min()
        max_local = df_l[elem].max()
            
        # set global minimum/maximum if it is the minimum/maximum overall
        if min_local < min_label_list[-1]:
            min_label_list[-1] = min_local
        if max_local > max_label_list[-1]:
            max_label_list[-1] = max_local
    
    return min_label_list, max_label_list
    
def _ret_AOCMinMax(AOC_ens_l):
    '''
    returns minimum maximum for AOCs
    '''
    # set min and max
    # set AOC lists
    min_AOC_list = []
    max_AOC_list = []
    # add min / max values to lists
    for j in range(AOC_ens_l.shape[1]):
        min_AOC_list.append(min(AOC_ens_l[:,j]))
        max_AOC_list.append(max(AOC_ens_l[:,j]))
        
    return min_AOC_list, max_AOC_list
    


