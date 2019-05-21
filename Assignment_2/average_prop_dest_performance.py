#%%
import pandas as pd 
import numpy as np



df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')
newdf = df.head(2000).copy()
del df
#%% 
prop_dest_avg_corrected_pos = dict()

gb = newdf.groupby(['prop_id', 'srch_destination_id']).mean()

#%%

gb
#%%
def create_prop_dest_mean_performance(df, gb=None, col_names):
        if groupby == None:
                gb = df.groupby(['prop_id', 'srch_destination_id']).mean()
        newdf = df.merge(gb[col_names], how='left', on=['prop_id', 'srch_destination_id'])
        return newdf