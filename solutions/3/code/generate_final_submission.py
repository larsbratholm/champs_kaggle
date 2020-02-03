import pandas as pd
import glob


sub_files = glob.glob('./output/*.csv')

sub_df_list = [pd.read_csv(file) for file in sub_files]

for i in range(1, len(sub_df_list)):
    sub_df_list[0]['scalar_coupling_constant'] += sub_df_list[i]['scalar_coupling_constant']

sub_df_list[0]['scalar_coupling_constant'] /= len(sub_df_list)

sub_df_list[0].to_csv('final_submission.csv', index=False)
