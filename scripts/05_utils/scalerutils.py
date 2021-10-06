import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import sys
# Path to the SandTank Repo
dev_path = '/home/SHARED/ML_TV/HydroGEN/modules/'
#Add Sand Tank path to the sys path
sys.path.append(dev_path)
from transform import float32_clamp_scaling


def _readScale(lines, slic, start_char, end_char):
    '''
    lines : lines read in using open(text file)
    slic : slicer (likely negative, such as -1, -2) for line to read from lines
    start_char : first character string to find
    end_char : second character string to find
    '''
    
    txt = lines[slic]
    txt = txt[txt.find(start_char)+1:]
    
    return float(txt[:txt.find(end_char)])
    
def setScalarArrayVal(min_val, max_val):
    '''
    Passes a minimum and maximum value and creates a numpy array ready for scaling  
    '''
    scale_range = np.array([min_val, max_val]).reshape(-1, 1)    

    return scale_range

def setScalerArray(lines, start_char, end_char):
    '''
    set up a scaler array (for one element)
    '''
    AOC_range = []
    for idx in range(-2,0):
        AOC_range.append(_readScale(lines, idx, start_char, end_char))
    
    AOC1_array = setScalarArrayVal(AOC_range[0], AOC_range[1])
    
    return AOC1_array
    
def setScaler(AOC1_array, dist_range=[0,1]):
    '''
    return scaled values 
    (must be for only one 'element' of numpy shape (-1,1) 
    '''
    scaler1 = MinMaxScaler(feature_range=dist_range)
    scaler1.fit(AOC1_array)
    
    return scaler1

def setInverseScale(scalar):
    '''
    turns a scalar (singleton) value into an array that can be ready sklearn
    '''
    return np.array(scalar).reshape(1, -1)

def scaled_SBI_Values(ensemble_dir, chars, poster_values, inverse=True):
    '''
    Context - SBI Workflow
    
    for converting from 0-1 to 'true' values during SBI workflow
    if inverse = False, then the opposite is attempted (converting from 'true' values to scaled)
    '''
    with open(ensemble_dir+'scale_info.txt') as f:
        lines = f.readlines()
        
    out_scaled = []
    for idx in range(len(chars)-1):
        
        start_char, end_char = chars[idx], chars[idx+1]
        outscaler = setScaler(lines, start_char, end_char)
        value = setInverseScale(poster_values[idx])
        if inverse:
            out_scaled.append(outscaler.inverse_transform(value)[0][0])
        else: 
            out_scaled.append(outscaler.transform(value)[0][0])
        
    return out_scaled
    
    
def scaled_ens_Values(min_label_list, max_label_list, dist_range=[0, 1]):
    '''
    Context - for scaling ensembles prior to LSTM training
    
    Automatically creates scales to be used to scale data
    min_label_list : minimum values
    max_label_list : max values
    dist_range : acceptable range of values for scaling
    
    08282021 - transition from homemade float32_clamp_scaling to sklearn MinMaxScalar
        via setScaler
    
    returns - list of sklearn scalers
    '''
    
    if len(min_label_list) != len(max_label_list):
        print('different number of minimums and maximums')

    scale_l = []
    trans_l = [] # dev only
    for j in range(len(min_label_list)):
        min_val, max_val = min_label_list[j], max_label_list[j]
        scale_range = setScalarArrayVal(min_val, max_val)
        scaler = setScaler(scale_range, dist_range=dist_range)
        scale_l.append(scaler)
        
        transf = float32_clamp_scaling(src_range=[min_val,max_val], dst_range=[0, 1])
        trans_l.append(transf)
        del min_val, max_val, scale_range, scaler, transf
        
    return scale_l, trans_l
    
    
def scaledForcingData(scale_l, df_l):
    '''
    Contect - for scaling data
    Input --
        scale_l : scale list
        df : df to scale
    Output --
    '''
    i = 0
    for elem in df_l.columns:
        # print(elem, i)
        df_2_scale = df_l[elem].to_numpy() 
        df_l[elem] = scale_l[i].transform(df_2_scale.reshape(-1,1))
        if elem.find('Flow') == -1: 
            i = i + 1
    
    return df_l
    
def scaledAOCData(scale_AOC, AOC_ens_l):
    '''
    scale things...
    '''
    AOC_ens_scale_l = np.empty(AOC_ens_l.shape)
    for j in range(AOC_ens_l.shape[1]):
        AOC_in = AOC_ens_l[:,j].reshape(-1,1)
        AOC_ens_scale_l[:,j] = scale_AOC[j].transform(AOC_in)[:,0] # use this for export
        
    return AOC_ens_scale_l

