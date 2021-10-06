# ---
    # 08242021 - This file taken from `/home/qh8373/UCRB/00_Principle_UCRB_Scripts/Taylor_LSTM_SBI_06252021.ipynb`
    
# --- To Do
    # 1. Concern: Leap Years
    # 2. ~~Scale K Attribute of Concern~~
    # 3. ~~Figure out how to make it more clear what attribute of concern is being used in training / testing~~
    # 4. Export Scalar Information
    # 5. Put Functions in Antoher File
    # 6. Turn into a Script
    # 7. ~~It looks like sorting can be dealt with,~~ but worth further investigation
    # 8. Divide script into LSTM and Data Assembly Pieces
    # 9. Remove manual references to scaling functions, replace with minmax_scale (or something more permanent)
        # (CHECK THIS OUT)
    # 10. Turn this into something that could be run like a function (if wanted)
    # 11. Beware of all the other functions involved
    # 12. Move forcings into the correct place.
    # 13. Put real streamflow somewhere
    # 14. Have a place to save and export this, save and export as vector art 
    # 15. Fix DLWR, DSWR input issue - why not registering?
# -- Modules:
from PIL import Image

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

# -- Globals:
# ensemble info
ensemble_name = '0819_01' # user defined ensemble name
mod_name = 'mod2' # mod name IMPORTANT FALSE if empty
remove_name_list = ['M-0.001', 'K-100'] # names to be removed IMPORTANT '' if empty
log_scale = True # renormalize parameters to log 10 scale

# paths
path_forcings = '/home/qh8373/SBI_TAYLOR/data/00_forcings/' # path to forcings
path_st = '/home/qh8373/SBI_TAYLOR/data/01_b_stream_true/0' # path to real streamflow
path_st_sim_ens = '/home/qh8373/SBI_TAYLOR/data/01_a_stream_sim/' # simulated ensemble data path
path_export = f'/home/qh8373/SBI_TAYLOR/data/03_ensemble_out/_ensemble_{ensemble_name}_{mod_name}/' # path for exporting

# ensemble_metadata
ens_list = glob.glob(f"{path_st_sim_ens}{ensemble_name}*.csv") # list of all ensembles names
ens_metadata = parseAllMetadata(ens_list, ensemble_name) # ensemble metadata
member_name_ens, yr_name_ens, AOC_ens = ens_metadata[0], ens_metadata[1], ens_metadata[2] # ensemble metadata

# modify original ensemble?
if mod_name is not False:
    i = 0
    while i < len(member_name_ens):
        # set member name to check
        member_name = member_name_ens[i]
        remove = False
        
        # loop through all the possible values to reserve
        for nm_reserve in remove_name_list:
            if member_name.find(nm_reserve) != -1:
                remove = True
                # print(member_name)

        # remove artificially large or small values (i.e. if remove = True)
        if remove:
            del ens_list[i], member_name_ens[i], yr_name_ens[i], AOC_ens[i]
        else:
            i = i + 1
        
# save toggle
save = True

# sites
huc_name_list = [9110000, 9329050, 9196500, 9107000, 9210500] # hucs available
SITE = 0  # sites index for fuc
gage_name = huc_name_list[SITE] # huc of interest

# year_range (make this dynamic)
yr_min, yr_max = min(yr_name_ens), max(yr_name_ens)+1 # 1995, 1996

# labels
labelist = ['DLWR', 'DSWR', 'Press', 'APCP', 'Temp', 'SPFH', 'UGRD', 'VGRD']
collist = ['blue','green','red', 'yellow', 'orange', 'purple', 'indigo','black']

# -- Writing
# Create List of Sites / Years [function]
YEARS, date0 = assembleYears(yr_min=yr_min,yr_max=yr_max) # ensemble information

# Create an ensemble directory 
try:
    os.mkdir(f'{path_export}')
except:
    print('warning: file exists')
    pass

# -- Data Assembly / Reading
# Assemble Forcings [functions]
df_forc = assembleAllForcings(labelist=labelist, YEARS=YEARS, SITE=SITE, path_forcings=path_forcings, date0=date0)


# Assemble real streamflow [functions]
labelist.append('Flow')
df_st = pd.read_csv(path_st+str(gage_name)+'.csv')
df_st.index = pd.to_datetime(df_st['date'])
df_st.drop(columns=['date', 'Unnamed: 0'], inplace=True)
df_st.rename(columns={'flow': 'Flow_Real'}, inplace=True)
df_st.index.names = ['timestamp']

# Assemble simulated streamflow and AOCs [functions]
df_ens = returnDF_ens(ens_list, date0, name_list=member_name_ens) # df of streamflow
AOC_ens_l = returnAOC_ens(AOC_ens) # np array of attributes of concern

# if log_scale (for parameters only) rescale using a log10 transform
if log_scale:
    # print(type(AOC_ens_l))
    # print(AOC_ens_l)
    AOC_ens_l = np.log10(AOC_ens_l)


# -- Merge together datasets, and plot
# Merge Streamflow
df_l = pd.merge(df_ens, df_st, how='left', on='timestamp')

# Plot Stuff
plt.rcParams.update({'font.size': 10})
### Climate Variables
plot_stuff(df_forc, same=False, title='Forcing Value at '+str(gage_name), 
           save=save,
           path=path_export)
### Compare 'real' and 'simulated' streamflow data
plot_stuff(df_l, same=True, ylabel='flow cfs', title='simulated flows at '+str(gage_name), 
           save=save,
           path=path_export)

# Merge Together streamflow and forcings datasets
df_l = pd.merge(df_forc, df_l, how='left', on='timestamp')

# -- Scale data
# Summarize min-max info [function]

# min-max of forcings / streamflow
min_label_list, max_label_list = _ret_MinMax(labelist, df_l)
scale_l, trans_l = scaled_ens_Values(min_label_list, max_label_list, dist_range=[0,1])
# min-max of AOCs 
min_AOC_list, max_AOC_list = _ret_AOCMinMax(AOC_ens_l)
scale_AOC, trans_AOC = scaled_ens_Values(min_AOC_list, max_AOC_list, dist_range=[0,1])

# transform based on min-max info [function]
# scale forcings / streamflow
df_l_scaled = scaledForcingData(scale_l, df_l)
AOC_ens_scale_l = scaledAOCData(scale_AOC, AOC_ens_l)


# -- final Plot Chack
# Plot scaled Data
plt.rcParams.update({'font.size': 10})
### Climate Variables
i = len(labelist)-1
plot_stuff(df_l_scaled.iloc[:,:i], ylabel='scaled forcing', same=True, title='Forcing Value at '+str(gage_name), 
           save=save,
           path=path_export)

### Compare 'real' and 'simulated' streamflow data
plot_stuff(df_l_scaled.iloc[:,i:], same=True, ylabel='scaled flow', title='simulated flows at '+str(gage_name), 
           save=save,
           path=path_export)

# # Plot real and simulated streamflow - all and separated
# ### Compare 'real' and 'simulated' streamflow data
# plot_stuff(df_l_scaled.iloc[:,i:], same=False, title='simulated flows at '+str(gage_name), 
#           save=save,
#           path=path_export)
           
#-- Save scaling information
if save:
    df_l_scaled.to_csv(f'{path_export}df_l_scaled.csv')
    df_l.to_csv(f'{path_export}df_l.csv')
    pd.DataFrame(AOC_ens).to_csv(f'{path_export}AOC_ens_scale_l.csv')

    #Pickling
    with open(path_export+"AOC_ens_scale_l.pkl", "wb") as fp:   
        pickle.dump(AOC_ens_scale_l, fp)
    
    file = open(path_export+"scale_info.txt", "w")
    file.write(f'Parameters and Targets: {labelist}\n')
    file.write(f'Minimum Value: {min_label_list}\n')
    file.write(f'Maximum Value: {max_label_list} \n')
    file.write(f'Minimum AOC Value: {min_AOC_list}\n')
    file.write(f'Maximum AOC Value: {max_AOC_list} \n')
    file.close()
    
    file = open(path_export+'metadata.txt', 'w')
    file.write(f'{ensemble_name}\n')
    file.write(f'path_forcings : {path_forcings}\n')
    file.write(f'path_st : {path_st}\n')
    file.write(f'path_st_sim_ens : {path_st_sim_ens}\n')
    file.write(f'path_export : {path_export}\n')
    file.write(f'{gage_name}\n')
    file.write(f'{min(yr_name_ens)} - {max(yr_name_ens)+1}\n')
    file.write(f'{date0}\n')
    file.close()
    
    file = open(path_export+'mods.txt', 'w')
    file.write(f'{ensemble_name}\n')
    file.write(f'mod_name : {mod_name}\n')
    file.write(f'remove_name_list : {remove_name_list}\n')
    file.write(f'log_10 scale? : {log_scale}\n')
    
    file.close()
    
    #Pickling
    with open(path_export+"scale_l.txt", "wb") as fp:   
        pickle.dump(scale_l, fp)

    with open(path_export+"scale_AOC.txt", "wb") as fp:
        pickle.dump(scale_AOC, fp)
        
    with open(path_export+"AOC_ens_scale_l.pkl", "wb") as fp:   
        pickle.dump(AOC_ens_scale_l, fp)
        
    with open(path_export+"labelist.pkl", "wb") as fp: 
        pickle.dump(labelist, fp)
        
    with open(path_export+'member_name_ens.pkl', 'wb') as fp:
        pickle.dump(member_name_ens, fp) 

    with open(path_export+'AOC_ens_l.pkl', 'wb') as fp:
        pickle.dump(AOC_ens_l, fp) 
        
