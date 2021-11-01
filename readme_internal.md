* Updated: 08/24/2021
* Note - this workflow copied and modified from `/home/qh8373/UCRB/00_Principle_UCRB_Scripts/` 

1. Run ParFlow
    * Run Scripts:
        * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/01_parflow/..`
        * `PARFLOW_Taylor_run.py`
        * `PARFLOW_Taylor_run_slurm.sh`
    * Supporting Information:
        * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/00_supporting`
    * Naming convention:
        * `{0721_01}_{K}_{1995}.txt`
    * Storage: 
        * (Long-term) `/scratch/taylor/ensembles_sbi/02_PARFLOW_OUT/02_PARFLOW_OUT`
            * Issues with submitting jobs via sbatch directly into this directory
        * (Short-term) `/home/qh8373/SBI_TAYLOR/data/02_PARFLOW_OUT/`
            * Note, must move from short- to long-term using the script `/home/qh8373/SBI_TAYLOR/data/02_PARFLOW_OUT/mv-all.py` (without submitting job) 
        * Naming convention:
            * `{0626}_{01}_{K}-{0.1}-{M}-{0.0001}-_{1995}` -> `{date}_{iteration}_{param_1_name}-{param_1_value}-{param_2_name}-{param_2_value}-_{year}`

2. Extract Streamflow from Simulation
    * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/01_parflow/PF_Read_Gage_Dynamic.py`
    * Supporting Information:
        * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/00_supporting`
    * Supporting Utilities:
        * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/genutils.py`
    * Naming Convention:
        * `{0626}_{01}_{K}-{0.1}-{M}-{0.0001}-_{1995}_{9110000}.out.flow.csv` -> `{date}_{iteration}_{param_1_name}-{param_1_value}-{param_2_name}-{param_2_value}-_{year}_{gage}.out.flow.csv`
    * Storage:  
        * `/home/qh8373/SBI_TAYLOR/data/01_a_stream_sim/`
    
3. Read, Visualize, Aggregate Data
    * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/02_ensemble/ensemble_agg.py`
    * Supporting Information:
        * forcings: `/home/qh8373/SBI_TAYLOR/data/00_forcings/`
        * streamflow: `/home/qh8373/SBI_TAYLOR/data/01_a_stream_sim/`
    * Supporting Utilities:
        * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/ensembleutils.py`
        * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/scalerutils.py`
    * Naming Convention:
        * ensemble name = `_ensemble_{date}_{iterate}`
    * Storage:
        * `/home/qh8373/SBI_TAYLOR/data/03_ensemble_out/{ensemble name}/`
            * `AOC_ens_l.pkl` - AOC ensemble members preserved by index #numpy
            * `AOC_ens_scale_l.pkl` - AOC ensemble member values, scaled #numpy
            * `labelist.pkl` - list of members of df_l #list
            * `member_name_ens.pkl` - list of members names in ensemble (also `name_ens_l`) #list
            * `metadata.txt` - metadata about where source documents are located
            * `scale_info.txt` - text used to create scaler for flow, AOCs (read: parameters), and forcings #string
            * `scale_AOC.txt` - list containing scalers for each AOC #list, sklearn scaler
            * `scale_l.txt` - list containing scalers for each forcing and flow #list, sklearn scaler
            * --------
            * many pngs/eps of forcings and streamflow
            * `df_l.csv` - dataframe of forcings + streamflow, unscaled
            * `df_l_scaled.csv` - dataframe of forcings + streamflow, scaled
            
4. LSTM-SBI
    * `/home/qh8373/SBI_TAYLOR/sbi_taylor/runs/`
    * `lstm_sbi.py`
        * core 'run' workflow that chooses to:
            1. train or load a LSTM
            2. build or load sbi  
    1. Train / Load LSTM
        * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/lstm_build.py`
        * Supporting Information:
            * `/home/qh8373/SBI_TAYLOR/data/03_ensemble_out/{ensemble name}/`
        * Supporting Utilities:
            * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/ensembleutils.py`
            * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/05_utils/lstmutils.py` # support for lstm routines
            * `/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/lstm_build_utils.py` # runs lstm train-test routine
        * Naming Convention:
            * lstm name = `{date}_{iterate}_{comment}`
        * Storage:
            * `/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_name}/` # general outputs
                * `AOC_ens_l_idx_test.pkl` # test AOC ensemble members values preserved by index #numpy
                * `AOC_ens_l_idx_train_val.pkl` # train_val AOC ensemble members values preserved by index #numpy
                * `AOC_ens_scale_l_idx_test.pkl` # test AOC ensemble member values preserved by index, scaled #numpy
                * `AOC_ens_scale_l_idx_train_val.pkl` # train_val AOC ensemble member values preserved by index, scaled #numpy
                * `dataX_test.pkl` # tensor of array of test values organized for LSTM # pytorch tensor
                * `dataY_test.pkl` # tensor of array of test values (y) organized for LSTM # pytorch tensor
                * `data_out_test.pkl` # df of test data preprocessed # pandas dataframe
                * `data_out_train_val.pkl` # df of train_val data preprocessed # pandas dataframe
                * `member_name_list_test.pkl` # member_names selected for test fraction # list
                * `member_name_list_train_val.pkl` # member_names reserved for train validation fraction # list
                * `name_ens_l_idx_test.pkl` # name of enseble members for test fraction # list (verbose)
                * `name_ens_l_idx_train_val.pkl` # name of enseble members reserved for train validation fraction # list (verbose)
                * `params.txt` # - hyperparameters etc... relevant to lstm model # text
                * `params_2.txt` # - hyperparameters etc... relevant to structure of train test split # text
                * `t_bool_test.pkl` # - numpy array of boolean values (if na, for example) # numpy
                * `test_idx.pkl` # - indices of test fraction # list
                * `train_idx.pkl` # - antiquated (I think) # list
                * `train_val_idx.pkl`# - indeices of train_validation fraction # list
                * `x_test.pkl` # - array of test values organized for LSTM # np array
                * `y_test.pkl` # - array of test values (y) organized for LSTM # np array
            * `/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_name}/{member_name}` # by member outputs
                * PLACEHOLDERS
                * # lstm model
                * # train-val split 
                * # loss plot
                * # plots 
                * # fit
    2. Run / Load Inference
        * TBD



5. Aggregated and Interpret SBI results
6. 



# ---------
Some notes from 'spinning up' CLM: 
    * 




