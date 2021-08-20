# --- From Elena 06/10/2021
# to be used to run scripts
# original located in `/home/qh8373/github/sbi_ucrb/04_scripts/..` *need to pull and update*
# 06/22/2021
    # - Updated to include notes about Mannings
# 06/25/2021
    # - Updated to be able to run Mannings and K at the same time (or any combination of elements)
# 07/21/2021
    # - First attempt at feeding PARFLOW forcings back in
# 08/19/2021
    # - Git Repo created and added from original parent directory (see readme)
    

# %%
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

# %%
# QH: must edit each time
# ensemble_name (for the ensemble of outputs generated)
ensemble_name = '0819_01'


# (where it saves and references)
path_folder = '/home/qh8373/SBI_TAYLOR/data/02_PARFLOW_OUT/'
supporting_folder = '/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/00_supporting/'

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

# # temporary
AOC_tuples = AOC_tuples[200:210]
print(len(AOC_tuples))

# Run ensemble
for idx in range(len(AOC_tuples)):
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
 
  print(run_dir)
 
  #creating directory for the run
  try:
    os.makedirs(run_dir)
  except:
    'directory exists'

  #cd into the run directory
  os.chdir(run_dir)
  
  #setting all file paths to copy required input files
  path_slope_x = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/slope_x.pfb'
  path_slope_y = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/slope_y.pfb'
  path_drv_clmin = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/drv_clmin.dat'
  path_drv_vegm = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/drv_vegm_v2.Taylor.dat'
  path_drv_vegp = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/drv_vegp.dat'
  path_indicator = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/Taylor.IndicatorFile_v2.pfb'
  path_pfsol = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/Taylor.pfsol'
  
  indicator = f'Taylor.IndicatorFile_v2.pfb'
  
  #copy all the input files in current directory
  shutil.copy(f'{path_slope_x}', run_dir)
  shutil.copy(f'{path_slope_y}', run_dir)
  shutil.copy(f'{path_drv_clmin}', run_dir)
  shutil.copy(f'{path_drv_vegm}', run_dir)
  shutil.copy(f'{path_drv_vegp}', run_dir)
  shutil.copy(f'{path_pfsol}', run_dir)
  shutil.copy(f'{path_indicator}', run_dir)
    
  #extra step to change the drv_clmin so that it has the correct start and end date:
  a_file = open(f'drv_clmin.dat', "r")
  list_of_lines = a_file.readlines()
  #changing syr and eyr
  list_of_lines[35]=f'syr {year_run-1} Starting Year\n'
  list_of_lines[42]=f'eyr {year_run} Ending Year\n'
  #changing startcodes: starting new run from scratch
  list_of_lines[29]=f'startcode      2\n'
  list_of_lines[46]=f'clm_ic         2\n'
  a_file = open(f'drv_clmin.dat', "w")
  a_file.writelines(list_of_lines)
  a_file.close()
  
  #copying initial pressure
  if new_Run:
    ip0 = f'/hydrodata/PFCLM/Taylor/Simulations/{year_run}/Taylor_{year_run}.out.press.00000.pfb'
  else:
      # will by default look for previous year run of similar name previous year
      # *QH NOTE 06262021 - this section needs to be edited if using multiple years...
      prev_year_run = year_run-1
      prev_name_run = f'{ensemble_name}_{AOC}_{prev_year_run}'
      prev_run_dir = f'{path_folder}{prev_name_run}'
      if calendar.isleap(prev_year_run):
        prev_hrs = 366*24
      else:
        prev_hrs = 365*24
      ip0 = f'{prev_run_dir}/Taylor_{prev_year_run}.out.press.0{prev_hrs}.pfb'
      del prev_year_run, prev_name_run, prev_run_dir
  print(ip0)
  ip = 'initial_pressure.pfb'
  shutil.copy(ip0, f'{run_dir}/{ip}')
  
  met_path = f'{run_dir}/NLDAS/'

  try:  
    os.mkdir(met_path)
  except:
    'directory exists'
    
  #copy the correct forcing
  met_path_to_copy = f'/hydrodata/PFCLM/Taylor/Simulations/{year_run}/NLDAS/'
   
  
  for filename in glob.glob(os.path.join(met_path_to_copy, '*.pfb')):
    shutil.copy(filename, met_path)
    
  reference_run_path = '/home/qh8373/UCRB/00_Principle_UCRB_Scripts/ParFlow/'
  reference_run_name = 'Reference.yaml'
  
  shutil.copy(f'{reference_run_path}{reference_run_name}', run_dir)
  #Read in the run
  run = Run.from_definition(f'{reference_run_name}')
  run.set_name(f'Taylor_{year_run}') # don't put the K value in here.
  
  #updating the directory where the forcing is
  run.Solver.CLM.MetFilePath = met_path
  # = number of time steps we want ParFlow to run for 
  run.TimingInfo.StopTime = 24*no_day # testing -> 10 # max -> 24*no_day
  
  print(f'running from {run.TimingInfo.StartCount} or {run.TimingInfo.StartTime} to {run.TimingInfo.StopTime}')
  
  if 'K' in POC:
      K = AOC[POC.find('K')]
      print('K', K)
      
      K_list = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 'g1', 'g2', 'g3', 'g4',
      'g5', 'g6', 'g7', 'g8', 'b1', 'b2']
      # better way to do this using exec? 
      run.Geom.s1.Perm.Value = run.Geom.s1.Perm.Value*K
      run.Geom.s2.Perm.Value = run.Geom.s2.Perm.Value*K
      run.Geom.s3.Perm.Value = run.Geom.s3.Perm.Value*K
      run.Geom.s4.Perm.Value = run.Geom.s4.Perm.Value*K
      run.Geom.s5.Perm.Value = run.Geom.s5.Perm.Value*K
      run.Geom.s6.Perm.Value = run.Geom.s6.Perm.Value*K
      run.Geom.s7.Perm.Value = run.Geom.s7.Perm.Value*K
      run.Geom.s8.Perm.Value = run.Geom.s8.Perm.Value*K
      run.Geom.s9.Perm.Value = run.Geom.s9.Perm.Value*K
      run.Geom.s10.Perm.Value = run.Geom.s10.Perm.Value*K
      run.Geom.s11.Perm.Value = run.Geom.s11.Perm.Value*K
      run.Geom.s12.Perm.Value = run.Geom.s12.Perm.Value*K
      run.Geom.s13.Perm.Value = run.Geom.s13.Perm.Value*K
      run.Geom.g1.Perm.Value = run.Geom.g1.Perm.Value*K
      run.Geom.g2.Perm.Value = run.Geom.g2.Perm.Value*K
      run.Geom.g3.Perm.Value = run.Geom.g3.Perm.Value*K
      run.Geom.g4.Perm.Value = run.Geom.g4.Perm.Value*K
      run.Geom.g5.Perm.Value = run.Geom.g5.Perm.Value*K
      run.Geom.g6.Perm.Value = run.Geom.g6.Perm.Value*K
      run.Geom.g7.Perm.Value = run.Geom.g7.Perm.Value*K
      run.Geom.g8.Perm.Value = run.Geom.g8.Perm.Value*K
      run.Geom.b1.Perm.Value = run.Geom.b1.Perm.Value*K
      run.Geom.b2.Perm.Value = run.Geom.b2.Perm.Value*K
      
  if 'M' in POC:
      mannings = AOC[POC.find('M')]
      print('mannings', mannings)
      # This is the command to change Mannings
      run.Mannings.Type = 'Constant'
      run.Mannings.GeomNames = 'domain'
      run.Mannings.Geom.domain.Value = mannings
  
  print("Starting Distribution Inputs/Forcing")
  run.dist('slope_x.pfb')
  run.dist('slope_y.pfb')
  ## - COMMENT:
  ## # Explanation - Multiple Processor to split inputs
  ## # Will not work for 1983 (already distributed)
  for filename_forcing in os.listdir(met_path):
    run.dist(f'{met_path}{filename_forcing}')
  print('Done distributing forcing')
  
  run.dist(indicator)
  
  run.dist(ip)
  
  print(f'Starting run')
  run.write()
  run.write(file_format='yaml')
  run.write(file_format='json')
  run.run()

  sys.exit()

