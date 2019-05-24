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
        gb1 = df.groupby(['prop_id', 'srch_destination_id']).mean()
        gb1 = gb1[col_names]
        gb1.columns = [str(col) + '_mean' for col in gb1.columns]
        
        gb2 = df.groupby(['prop_id', 'srch_destination_id']).std()
        gb2 = gb2[col_names]
        gb2.columns = [str(col) + '_std' for col in gb2.columns]
        gb = pd.concat([gb1,gb2], axis=1)

    newdf = df.merge(gb, 
                     how='left', 
                     on=['prop_id', 'srch_destination_id'],
                     suffixes=('_already_existed','_newlyadded'))
    
    # print(newdf.isna().sum())
    newdf = newdf.fillna(0)
    # print(newdf.isna().sum())
    
    return newdf, gb

#
#EXAMPLE:
# df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')
# df2, gb = create_prop_dest_mean_performance(df, col_names=['position'])


#%%
# df2.columns

#%%
