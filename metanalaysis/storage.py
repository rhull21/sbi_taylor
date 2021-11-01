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
out_dir = f'/home/qh8373/SBI_TAYLOR/data/07_metadata_out/{ensemble_name_mod}/'
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

# Set up number of unique permutations of lists in tuples
# https://cmsdk.com/python/all-possible-permutations-of-multiple-lists-and-sizes.html 
AOC_tuples = []
for i in list(itertools.product(*AOC_vals)):
    AOC_tuples.append(i)

# # # temporary
# AOC_tuples = AOC_tuples[200:201]
# idx = 0
# for AO in AOC_tuples:
#     print(idx, AO)
#     idx = idx + 1
# print(AOC_tuples)
# del idx

# for tracking the delta storage, for later
dS_arr = np.empty((len(AOC_tuples), 3))


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
  store_arr = np.empty((no_day+1, 3))
  
  # ---------
  # read dynamic PF-CLM Outputs
  # ---------
  for i in range(no_day+1):
      data.time = i # time step for all Pf-cLM outputs
    #   data.forcing_time = 0 # time step for all forcings
      
      # storages
      w = 'subsurface_storage'
      sub_array = getattr(data,w)
      sub_sum = np.round(sub_array.sum(), 0)
      
      w = 'surface_storage'
      surf_array = getattr(data,w)
      surf_sum = np.round(surf_array.sum(), 0)
     
      tot_sum = np.round(sub_sum + surf_sum, 0)
     
      # append
      store_arr[i, :] = [sub_sum, surf_sum, tot_sum] 
     
      del sub_array, sub_sum, surf_array, surf_sum, tot_sum
  
  fig, ax = plt.subplots()
  
  ax.plot(store_arr[:,1], label='surface', color='red')
  ax.set_xlabel('time (days)')
  ax.set_ylabel('total storage (units?)')
#   ax.set_ylim(1e10, 1e11)
  
#   print((store_arr[:,0]/store_arr[:,2]))
  
#   ax2 = ax.twinx()
#   ax2.plot((store_arr[:,0]/store_arr[:,2]), label='percent subsurface', color='red')
#   ax2.set_ylabel('perc subsurface (decimal)')
#   ax2.set_ylim(0, 1.1)

  ax.set_title(f'{name_run} surface storage unscaled')
  fig.legend()
  fig.savefig(f'{out_dir}{name_run}_surface_storage_unscaled.png')
  plt.show()
  plt.close()
  
  
  fig, ax = plt.subplots()
  
  ax.plot(store_arr[:,2], label='total')
  ax.set_xlabel('time (days)')
  ax.set_ylabel('total storage (units?)')
#   ax.set_ylim(1e10, 1e11)
  
#   print((store_arr[:,0]/store_arr[:,2]))
  
#   ax2 = ax.twinx()
#   ax2.plot((store_arr[:,0]/store_arr[:,2]), label='percent subsurface', color='red')
#   ax2.set_ylabel('perc subsurface (decimal)')
#   ax2.set_ylim(0, 1.1)

  ax.set_title(f'{name_run} storage unscaled')
  fig.legend()
  fig.savefig(f'{out_dir}{name_run}_storage_unscaled.png')
  plt.show()
  plt.close()
  
  # calculate net storage change and update
  for j in range(3):
    out = abs(store_arr[0,j] - store_arr[365,j]) / store_arr[0,j]
    # print(out)
    dS_arr[idx, j] = out
    del out
    
#   print(dS_arr[idx, :])

  
# ------
# compare change in storage for all models
# ------
fig, ax = plt.subplots(1,3, sharey=True)

nmlist = ['subsurface', 'surface', 'total']
for l in range(3):
    ax[l].boxplot(dS_arr[:, l]) # , labels=nmlist[l])
    ax[l].set_ylim(0.0000001,10)
    ax[l].set_yscale('log')
    ax[l].set_title(f'{nmlist[l]} storage')
    if l == 0:
        ax[l].set_ylabel('perc change storage')

fig.legend()
fig.savefig(f'{out_dir}00_all_deltastorage_log.png')
fig.savefig(f'{out_dir}00_all_deltastorage_log.eps', type='eps')
plt.show()
plt.close()


  
  

  
 
  







    

