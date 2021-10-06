Updated 09132021

PseudoCode:

1. Run Ensemble of ParFlow Simulations
    * theta.shape[0] = 2; theta.shape[1] ~ 14; `num_simulations` ~ 200; `num_years` = 1 (1995) 
2. Harvest overland flow (streamflow) at gaged location for each ParFlow Simulation 
    * series.shape[0] = 8760 [hrs]
3. Aggregate, transform, and scale streamflow+forcing data for emulator training
    * series.shape[0] = 365 [days]
    * theta_scaled = transform log10(theta), scaled between 0 and 1
4. Build LSTM emulator of ParFlow streamflow
    * `num_members` = 10 ('ensemble' of LSTM emulators)
    * `train_frac`, `val_frac`, `test_frac` = 0.7, 0.2, 0.1
        * random mixture of train, validation set used to train all 10 emulators
        * test set reserved for SBI (synthetic truth(s))
5. Build SBI Posterior Response Space (`inference.build_posterior`)
    * Define Hyperparameters:
        * Observation type (`summary`, `full`, `embed`)
        * Number of SBI Builds (`L_sims`)
        * NDE hyperparameters (`meth`='SNPE', `model`='maf', `hidden_transforms`, `num_transforms`, etc...)
        * Prior Type (`uniform`, `lognormal`)
    * For L in L_sims:
        1. Create Prior
        2. Define Simulator (flexibly)
            * Select Observation Type (`summary`, `full`, `embed`)
            * Randomly Select LSTM (n = 10) - source of stochsticity -
            * Simulate (given theta)
            * return result
        3. Prepare Simulator (verbose) (`n_simulations` = 1000)
        4. Build Posterior
6. Sample SBI Posterior
    * For L in L_sims: [`unique SBI posteriors`]
        * For `y_hat` in `all_y_hat`: [`unique observations (from test fraction)`]
.           * sample posterior at observation `y_hat` (`n_samples` = 5000)
            * log probability at sampled posterior
            
            * find log probability of 'true' theta
                * Loop through each 'true' theta
                * rounds theta values to a certain precicion, and adds bracketing window
                * Find index of true tetas within bracketing window
                * `log_prob_true_thetas` = average of log probability of all indexed results 
            * Generate Interpretative Thetas
                * Average, Median, Max Probability, Mode, Random Thetas Sampled from Posterior
            * Gen Series & Fit
                * Generate full time series (based off of interpretive thetas)
                   * Complications, multiple interpretive thetas, multiple lstms
                        (timeseries, len(`lstm_out_list`), len(thetaStatsList)
                        (350,        10,          5)
                * Compare fit of Results
                    * RMSE, NSE, KGE
                        (timeseries, len(`lstm_out_list`), len(thetaStatsList)
                        (3,        10,          5) 
            * Plot Pair Plots
    
7. Interpretive Plots
            

run_path: [/home/qh8373/SBI_TAYLOR/sbi_taylor/runs/]
LSTM-SBI Inference Workflow: [lstm_sbi.py]

  script_path: ['/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/']
  Build / Load LSTM Ensemble:
  
    If Build [lstm_build.py]: 
      Pass:
        from [/home/qh8373/SBI_TAYLOR/data/03_ensemble_out/{ensemble_name}/]
        Processed & Scaled DataFrame Containing Streamflow, AOCs, and Metadata
        
      Set: [../utils/lstm_utils.py]
        LSTM Ensmble Name
        LSTM HyperParameters 
        Define LSTM, optimizer, loss function 
        number of LSTM ensemble member n
        
      Arrange Data: [../utils/lstm_utils.py]
        Randomly set 'test' (10%) and 'train_val' (90%) split
          Sliding Windows and arrange 'test' dataset
          Store 'train_val' dataset for later
        
      Save **Build** Information:
        to [/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_name}/]
        Hyperparameters, arranged test data, train_val-test split, etc...
        
      Train / Test routine: 
        For n in LSTM Ensemble: 
          Randomly set 'train' (70%) and 'val' (20%) split from 'train_val' [../utils/lstm_utils.py]
          
          Train LSTM routine: [lstm_build_utils.py]
            opt: using 'val' set for early stopping

          Evaluate Trained LSTM Ensemble Member: 
            For Train, Val, and Test Data: 
              Plot Scatter [lstm_build_utils.py]
              R^2, NSE, KGE [lstm_build_utils.py] [../utils/gen_utils.py]
              Plot Series Simulated v Real [lstm_build_utils.py]
            
          Save Ensemble Member Build Information:
            nb: to a nested 'member' folder until LSTM ensemble name
              to [/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_name}/{member_name}]
            lstm model
            train-val split 
            loss plot
            plots 
            fit

  Elif Load:
    Pass:
      from [/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{lstm_name}/]
      all of the above LSTM stuff for SBI workflow [../utils/sbi_utils.py]
      including test data

    script_path: ['/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/']
    Build / Load SBI Ensemble:
    
      If Build: [sbi_build.py]
        Set: [../utils/sbi_utils.py]
          SBI Ensemble Name
          Stat_method (summary, full, embed)
          Data for Inference
          simTruth tag
          Description
          SBI Globals: 
            NDE hyperparameters
            prior
          Simulator
          Prior
          Number of SBI ensemble members M
 
        Save Inference **Build** Information:
          to [/home/qh8373/SBI_TAYLOR/data/04_sbi_out/{sbi_name}/]
          
        Inference Routine: [../utils/sbi_utils.py] [sbi_build_utils.py]
            prepare simulator for inference
    
            For m in M (SBI ensemble members):
                Run Inference 
                Generate Posterior and Log Porbability
                Output Interpretative Statistics
                Make plots 
    
                Save Inference **Member** Information:
                  to [/home/qh8373/SBI_TAYLOR/data/04_sbi_out/{sbi_name}/{sbi_member_name}]

      Elif Load: [sbi_build.py]
        Pass:
          from [/home/qh8373/SBI_TAYLOR/data/04_sbi_out/{sbi_name}/]
          all of the above sbi stuff for Inference Routine [../utils/sbi_utils.py]
          including each member
          
          For m in M (SBI ensemble members):
              For i in test_data:
                define 'observation' based on test data
                Do some stuff
                
                Save Inference **Member** Information:
                  to [/home/qh8373/SBI_TAYLOR/data/04_sbi_out/{sbi_name}/{sbi_member_name}]
            

  Interpretation:
            

          
        
    
    

        
    


  