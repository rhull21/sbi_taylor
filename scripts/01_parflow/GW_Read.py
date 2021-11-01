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

import matplotlib.pyplot as plt

# LHS sampling - https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html
import skopt
from skopt.space import Space
from skopt.sampler import Lhs

# QH: must edit each time
# ensemble_name (for the ensemble of outputs generated)
ensemble_name = '0819_01'

# modifications
ensemble_name_mod = '0819_01_mod2'
remove_list = [[100],[0.001]]

# (where it saves and references)
path_folder = '/scratch/taylor/ensembles_sbi/02_PARFLOW_OUT/02_PARFLOW_OUT/' # /home/qh8373/SBI_TAYLOR/data/02_PARFLOW_OUT/'
supporting_folder = '/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/00_supporting/'

#creating directory for the output
out_dir = f'/home/qh8373/SBI_TAYLOR/data/01_c_gw_sim/{ensemble_name_mod}/'
try:
  os.makedirs(out_dir)
except:
  'directory exists'

# QH Edit to include just the years you want, note that 1983 is not a good year
# QH Edit - note that we may in some cases want to run an additional year of simulations on top of what we have already done
new_Run = True # Set this to 'false' if not a new run 
year_run = 1995
if calendar.isleap(year_run):
  no_day = 366
else:
  no_day = 365
 
# Set Parameter of Concernm (POC) - 'K', 'M', 'KM' 
POC = 'KM'

# do multiple for different K / Mannings value
AOC_vals = []
for idx in range(len(POC)):
    remove_list_temp = remove_list[idx]
    POC_in = POC[idx]
    AOC_vals.append([])
    with open(f'{supporting_folder}{ensemble_name}_{POC_in}_{year_run}.txt', 'r') as AOC_lines:
        # print(AOC_lines)
        for line in AOC_lines:
            add_temp = float(line)
            if (add_temp in remove_list_temp):
                print(POC_in, add_temp, 'not included')
            else:
                AOC_vals[idx].append(float(line))
    AOC_lines.close()

print(AOC_vals)
# Set up number of unique permutations of lists in tuples
# https://cmsdk.com/python/all-possible-permutations-of-multiple-lists-and-sizes.html 
AOC_tuples = []
for i in list(itertools.product(*AOC_vals)):
    AOC_tuples.append(i)

print(AOC_tuples)
sys.exit()

# loc_idx -> for reading 
loc_idxs = np.array([[30, 24], [28, 11], [46, 15], [26, 41], [10, 27], [40, 17], [18, 19], [14, 15], [26, 7], [5, 17],
     [13, 20], [19, 35], [13, 12], [25, 23], [10, 2], [13, 23], [27, 42], [25, 18], [17, 28], [17, 20],
     [30, 13], [29, 11], [9, 7], [9, 28], [17, 37], [26, 36], [16, 13], [9, 32], [21, 7], [26, 34], [23, 23], 
     [23, 16], [21, 29], [39, 24], [24, 24], [32, 26], [25, 11], [38, 15], [19, 25], [39, 12], [4, 13],
     [14, 8], [19, 21], [8, 4], [44, 12], [16, 27], [29, 30], [27, 16], [11, 22], [7, 11], [32, 18], 
     [28, 29], [20, 40], [24, 27], [45, 16], [28, 9], [32, 16], [16, 12], [12, 17], [29, 41], [41, 14],
     [17, 30], [27, 19], [16, 24], [8, 13], [23, 35], [21, 9], [36, 20], [22, 37], [32, 14], [4, 16],
     [35, 26], [14, 19], [26, 23], [27, 14], [20, 21], [30, 30], [37, 19], [12, 35], [2, 8], [40, 18],
     [18, 17], [20, 39], [18, 15], [8, 5], [8, 6], [23, 26], [33, 13], [22, 32], [11, 30], [33, 23], [10, 22],
     [38, 27], [44, 20], [28, 8], [10, 17], [19, 33], [15, 32], [21, 25], [36, 19], [41, 10], [39, 28],
     [34, 28]])
     
loc_idxs_str = []
for loc in range(loc_idxs.shape[0]):
  loc_idxs_str.append(str(loc_idxs[loc]))


# Loop through ensemble
for idx in range(len(AOC_tuples)):# len(AOC_tuples)
  # make this as the AOC value
  AOC = AOC_tuples[idx]
  print(AOC)
  
  AOC_str = ''
  for AOC_idx in range(len(AOC)):
      AOC_str = AOC_str+str(POC[AOC_idx])+'-'+str(AOC[AOC_idx])
      if AOC_idx != len(AOC):
          AOC_str = AOC_str+'-'


  name_run = f'{ensemble_name}_{AOC_str}_{year_run}' 
  run_dir = f'{path_folder}{name_run}'
  path_PFdatabase = f'{run_dir}/Taylor_{year_run}.pfidb'
 
#   print(run_dir)
  
  # load the PF metadata and put it in the run data structure
  run = Run.from_definition(path_PFdatabase)
  
  # get dimensions of domain
  nx = run.ComputationalGrid.NX
  ny = run.ComputationalGrid.NY
  
  # get data
  data = run.data_accessor 
  
  
  # initialize holding array
  store_arr = np.empty((int(no_day*24)+1, len(loc_idxs)))
  
  # ---------
  # read dynamic PF-CLM Outputs
  # ---------
  for i in range(int(no_day*24)+1): # no_day+1
      print(i)
      data.time = i # time step for all Pf-cLM outputs

      # water table depth
      w = 'wtd'
      wtd_array = getattr(data,w)
      wtd_vals_temp = wtd_array[loc_idxs[:,0], loc_idxs[:,1]]
      store_arr[i,:] = wtd_vals_temp
      
    #   if (i==1) or (i==180) or (i==365):
    #       plt.imshow(wtd_array,aspect='equal',origin='lower')
    #       plt.scatter(loc_idxs[:,1], loc_idxs[:,0],color='black')
    #       plt.savefig(f'{out_dir}plots/wtd_{name_run}_{i}.png')
    #       plt.show()
    #       plt.close()

      del wtd_vals_temp, wtd_array, w
      
 
  store_df = pd.DataFrame(store_arr, columns=loc_idxs_str)
  print(store_df)
  store_df.to_csv(f'{out_dir}{name_run}.csv')
  
  del store_df, store_arr, run, data
  
      
      

