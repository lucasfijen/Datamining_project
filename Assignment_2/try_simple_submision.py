#%%

import os
try:    
    os.chdir('Assignment_2')
except:
    print("youre already in assignment 2")
import pandas as pd 
import numpy as np
from average_prop_dest_performance import *

# #%%

# df = pd.read_csv('data/training_set_VU_DM.csv')
# df_test = pd.read_csv('data/test_set_VU_DM.csv')

# #%% create gain and a gb with tupels

# df['gain'] = df['click_bool'] + (5 * df['booking_bool'])
# df, gb = create_prop_dest_mean_performance(df, 'gain')

# #%%
# testdf, gb = create_prop_dest_mean_performance(df_test, ['gain'], gb)

test_prediction = pd.read_pickle('test_prediction_pos.pickle')
test_prediction.head()
#%%
from ndcg import make_submision
#%%
make_submision(test_prediction, 'prediction', ascending=True, filename='data/testsub.csv')
#%%
