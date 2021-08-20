# -- This script for reading simulated streamflow for ParFow runs in {gage}
# -- gages are 9110000, 9329050, 9196500, 9107000, 9210500, where 9110000 is Taylor
# -- based off of two scripts: 
    # 1. `/home/qh8373/github/ucrb/example_reading.py`
    # 2. `/home/qh8373/UCRB/00_Principle_UCRB_Scripts/subset_UCRB_dynamic_df.py`
# -- open issue, what about pfidb?

# QH - 0614 - modified to dynamically read in the mannings value
# QH - 0706 - modified to be able to read in runs with multiple input parameters
# QH - 0820 - modifed from /home/qh8373/UCRB/00_Principle_UCRB_Scripts/

# ---- modules
import sys
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/')
from genutils import PFread
import numpy as np
from parflow import Run
from parflow.tools.settings import set_working_directory

from PIL import Image
from parflowio.pyParflowio import PFData
import matplotlib.pyplot as plt
from parflow.tools.hydrology import calculate_overland_flow_grid
from copy import copy
import pandas as pd

import os
import pandas as pd
from parflow import Run
from parflow.tools.fs import cp, mkdir
from pathlib import Path
import numpy as np
import sys
import shutil
import glob
import calendar
import itertools


# ---- globals
# set gage (list)
gages = [9110000]

# set name list, etc.. (list)
# ensemble_name (for the ensemble of outputs generated)
ensemble_name = '0819_01'

# (where it references)
path_folder = '/home/qh8373/SBI_TAYLOR/data/02_PARFLOW_OUT/'
# (where the supporting information is)
supporting_folder = '/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/00_supporting/'
# (where it saves)
save_folder = '/home/qh8373/SBI_TAYLOR/data/01_a_stream_sim/'


# year of simulation
year_run = 1995 
if calendar.isleap(year_run):
    no_day = 366
else:
    no_day = 365
N_hours = no_day*24
  

# Set Parameter of Concernm (POC) - 'K', 'M', 'KM' 
POC = 'KM'

# do multiple for different K / Mannings value
AOC_vals = []
for idx in range(len(POC)):
    POC_in = POC[idx]
    AOC_vals.append([])
    with open(f'{supporting_folder}{ensemble_name}_{POC_in}_{year_run}.txt', 'r') as AOC_lines:
        for line in AOC_lines:
            # print(float(line))
            AOC_vals[idx].append(float(line))
    AOC_lines.close()

# Set up number of unique permutations of lists in tuples
# https://cmsdk.com/python/all-possible-permutations-of-multiple-lists-and-sizes.html 
AOC_tuples = []
for i in list(itertools.product(*AOC_vals)):
    AOC_tuples.append(i)

# temporary
AOC_tuples = AOC_tuples[0:1]

# ---- set overland flow globals 
# # streamflow = calculate_overland_flow_grid(pressure, slopex, slopey, mannings, dx, dy, flow_method='OverlandFlow')
mannings = 0.0000024 # note, for POC = 'M' this value gets reset every iteration
dx = 1000
dy = 1000
slope_x = None # needs to be reset later in script
slope_y = None # needs to be reset later in script



# ---- loop through sites, 
for id_watershed in gages:
    # ---- generate a clipped mask
    #reading the mask of the chosen watershed and defining bounding box
    mask_watershed = np.array(Image.open(f'/hydrodata/PFCLM/UCRB/Domain_Files/Stream_Gauges/gauge_masks/{id_watershed}_WatershedMask.tif'))
    mask_watershed = np.flip(mask_watershed,0) #y in PF is S to N
    where = np.array(np.where(mask_watershed))
    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)
    
    # Clipping results to domain
    mask_clipped = mask_watershed[x1:x2, y1:y2]
    mask_clipped_simple = copy(mask_clipped)
    mask_clipped_simple[mask_clipped_simple == 2] = 1
    # print(mask_clipped.shape)
    
    
    # --- loop through all scenarios to generate streamflow for each ensemble member
    # Run ensemble
    for idx in range(len(AOC_tuples)):
        # make this as the AOC value
        AOC = AOC_tuples[idx]
        # print(AOC)
        
        # create the name run
        AOC_str = ''
        for AOC_idx in range(len(AOC)):
            AOC_str = AOC_str+str(POC[AOC_idx])+'-'+str(AOC[AOC_idx])
            if AOC_idx != len(AOC):
                AOC_str = AOC_str+'-'
        
        # set up directories for run (output) and read (input)
        name_run = f'{ensemble_name}_{AOC_str}_{year_run}' 
        read_dir = f'{path_folder}{name_run}'
        run_dir = save_folder

        
        #setting all file paths to copy required input files
        path_slope_x = f'{read_dir}/slope_x.pfb'
        path_slope_y = f'{read_dir}/slope_y.pfb'
        
        slope_x = PFread(PFData(path_slope_x))
        slope_y = PFread(PFData(path_slope_y))
        
        
        # resetting mannings (if POC includes the character 'M')
        if 'M' in POC:
            mannings = AOC[POC.find('M')]
        # print(type(mannings), mannings)
        
        # read streamflow
        # N_hours = 1 # manual, comment out
        # streamflow data [index, gaged flow, max flow]
        streamflow_data = np.zeros([N_hours, 3])
        for i in range(N_hours):
            # set index
            t0 = str(int(i)).rjust(5, '0')
            
            # set direectory location
            path_pressure = f'{read_dir}/Taylor_{year_run}.out.press.{t0}.pfb'
            
            # snag overland flow
            pressure = PFread(PFData(path_pressure))

            #### --- streamflow = calculate_overland_flow_grid(pressure, slopex, slopey, mannings, dx, dy, flow_method='OverlandFlow')
            streamflow = calculate_overland_flow_grid(pressure=pressure, slopex=slope_x, slopey=slope_y,
                                mannings=mannings, dx=dx, dy=dy, mask=np.expand_dims(mask_clipped, axis=0),
                                flow_method='OverlandFlow')# [mask_clipped==2]
            streamflow_gage = streamflow[mask_clipped==2]
            streamflow_max = streamflow.max()
            ## ---- check work
            # print(streamflow.shape)
            # print(i, 'streamflow is ', streamflow_gage)
            # print(np.where(streamflow == streamflow_gage))
            # print('max streamflow is ', streamflow_max)
            # print(np.where(streamflow == streamflow_max))
            
            ## ----- save points
            streamflow_data[i, 0] = i
            streamflow_data[i, 1] = streamflow_gage
            streamflow_data[i, 2] = streamflow_max
        
        # save work:
        # for saving streamflow time series
        runname_out = f'{run_dir}{name_run}_{id_watershed}'
        streamflow_df = pd.DataFrame(data=streamflow_data, columns=['idx', 'gage_flow', 'max_flow'])
        streamflow_df.to_csv(f'{runname_out}.out.flow.csv')
        print(f'{runname_out} complete')
        del streamflow_df, streamflow_data