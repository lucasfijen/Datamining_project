#%%
import pandas as pd 
import numpy as np




#%%
def create_prop_dest_mean_performance(df, col_names, gb=None):
    '''
        Takes a df and optionally a df/groupby object with multiindex
        col_names should be a list of columns to merge
        information is added as a new column with suffix _mean
        Returns:
        - new dataframe with merged columns
        - used groupby object
    '''
    if gb is None:
        gb = df.groupby(['prop_id', 'srch_destination_id']).mean()

    newdf = df.merge(gb[col_names], 
                     how='left', 
                     on=['prop_id', 'srch_destination_id'],
                     suffixes=(None,'_mean'))
    
    # print(newdf.isna().sum())
    newdf = newdf.fillna(0)
    # print(newdf.isna().sum())
    
    return newdf, gb

#
#EXAMPLE:
# df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')
# df2, gb = create_prop_dest_mean_performance(df, col_names=['position'])
