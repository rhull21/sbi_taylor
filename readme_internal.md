* Updated: 08/19/2021
* Note - this workflow copied and modified from `/home/qh8373/UCRB/00_Principle_UCRB_Scripts/` 08/19/2021

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
        * (Discontinued) ~~`/scratch/taylor/ensembles_sbi/02_PARFLOW_OUT/..`~~ 
        * (Because of issues with SLURM) `/home/qh8373/SBI_TAYLOR/data/02_PARFLOW_OUT/`
        *  `{0626_01}_{K-0.0001}-{M-0.0001}-_{1995}`
        * Naming convention:
            * `{0626}_{01}_{K}-{0.1}-{M}-{0.0001}-_{1995}` -> `{date}_{iteration}_{param_1_name}-{param_1_value}-{param_2_name}-{param_2_value}-_{year}`
2. 