# For generating the statistics to evalute

import pickle
import sys
sys.path.append('/home/qh8373/SBI_TAYLOR/sbi_taylor/scripts/03_sbi_lstm/')
from lstm_build_utils import LSTM, runLSTM, evallstm

name_run = '10_13_log_mod/10_13_log_mod_00/'
in_dir = f'/home/qh8373/SBI_TAYLOR/data/04_lstm_out/{name_run}'
temp_df = pickle.load(open(f'{in_dir}cond_df.pkl', 'rb'))


for idx in ['test', 'train', 'val']:
    # I think there might be an error here, not sure how...
    r2_out, stats_out = evallstm(lstm=temp_df['lstm_out'][idx], dataY=temp_df['DataY'][idx], dataX=temp_df['DataX'][idx],
                        cond=idx, save_path_out=temp_df['save_path_out'][idx],
                        member_name_l=temp_df['member_name'][idx], series_len=temp_df['series_len'][idx],
                        shuffle_it_in=temp_df['shuffle_it_in'][idx], save=False)
    print(idx)
    print(r2_out)
    print(stats_out)







# Index(['DataX', 'DataY', 't_bool', 'member_name', 'r2', 'rmse-kge-nse',
#       'lstm_out', 'series_len', 'save_path_out', 'shuffle_it_in',
#       'lstm_member_name'],

# >>> play_list['rmse-kge-nse']
# test     NaN
# train    NaN
# val      NaN
# Name: rmse-kge-nse, dtype: object
# >>> play_list['r2']
# test     NaN
# train    NaN
# val      NaN
# Name: r2, dtype: object
