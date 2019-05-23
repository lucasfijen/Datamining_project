#%% Imports
import os
try:    
    os.chdir('Assignment_2')
except:
    print("youre already in assignment 2")

import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")
from position_bias import *
from ndcg import *

#%%

df_train = pd.read_csv('prepped_df_train.csv')
df_test = pd.read_csv('prepped_df_test.csv')
df_val = pd.read_csv('prepped_df_val.csv')

#%% SHAPE & DIFFERENCES
print('df_train.shape', df_train.shape)
print('df_val.shape', df_val.shape)
print('df_test.shape', df_test.shape)
print(set(df_train.columns) - set(df_test.columns))
print(set(df_val.columns) - set(df_test.columns))
print(set(df_train.columns) - set(df_val.columns))

#%% STEP 1: POSTION BASED MODEL
# Since position based, only take non-random to train on!
df_train = df_train[df_train['random_bool'] == 0]
df_val = df_val[df_val['random_bool'] == 0]
df_test = df_test[df_test['random_bool'] == 0]

#%% Index = srch_id, prop_id
index_train = df_train[['srch_id', 'prop_id']]
index_val = df_val[['srch_id', 'prop_id']]
index_test = df_test[['srch_id', 'prop_id']]

#%% Target = position (position based model)
target_train = df_train[['corrected_position']]
target_val = df_val[['corrected_position']]


#%%
exclude = ['srch_id', 
            'prop_id'
            ]
train_exclude = ['total_non_corrected_gain', 
            'corrected_click_gain',
            'corrected_book_gain',
            'total_corrected_gainNone',
            'corrected_position',
            #'total_corrected_gain_mean',
            'position',
            'click_bool',
            'gross_bookings_usd',
            'booking_bool']

X_test = df_test.drop(exclude, axis=1)
X_train = df_train.drop(exclude+train_exclude, axis=1)
X_val = df_val.drop(exclude+train_exclude, axis=1)

#%%
# print(sum(np.isnan(X_val.values)))
print(len(X_val))
X_val.isna().sum()
#%%
print('df_train.shape', X_train.shape)
print('df_val.shape', X_val.shape)
print('df_test.shape', X_test.shape)
print(set(X_train.columns) - set(X_test.columns))
print(set(X_val.columns) - set(X_test.columns))
print(set(X_train.columns) - set(X_val.columns))

#%% GBM imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#%% GBM 
# Position model
GBR_position = GradientBoostingRegressor()
GBR_position.fit(X_train.values, target_train.values.reshape(-1,))

#%%
valid_position_prediction = GBR_position.predict(X_val.values)
MSE = mean_squared_error(target_val.values, valid_position_prediction)
print("MSE: %.4f" % MSE)

#%% STEP 2: QUALITY / GAIN MODEL
df_train = pd.read_csv('prepped_df_train.csv')
df_test = pd.read_csv('prepped_df_test.csv')
df_val = pd.read_csv('prepped_df_val.csv')

df_train = df_train[df_train['random_bool'] == 0]
df_val = df_val[df_val['random_bool'] == 0]
df_test = df_test[df_test['random_bool'] == 0]


#%% Target = position (position based model)
target_train_gain = df_train[['total_corrected_gainNone']]
target_valid_gain = df_val[['total_corrected_gainNone']]

X_test_gain = df_test.drop(exclude, axis=1)
X_train_gain = df_train.drop(exclude+train_exclude, axis=1)
X_val_gain = df_val.drop(exclude+train_exclude, axis=1)

#%%
print(X_test.columns)

#%%
# Gain model
GBR_gain = GradientBoostingRegressor()
GBR_gain.fit(X_train_gain.values, target_train_gain.values.reshape(-1,))

#%%
valid_gain_prediction = GBR_gain.predict(X_val_gain.values)
MSE = mean_squared_error(target_valid_gain.values, valid_gain_prediction)
print("MSE: %.4f" % MSE)

#%%
valid_position_prediction = (1. - valid_position_prediction) * 6
#%%
total_prediction = (0.5 * valid_position_prediction) + (0.5 * valid_gain_prediction)

#%%
total_df = pd.DataFrame(total_prediction, columns=['prediction'])
total_df = pd.concat([total_df, df_val['total_non_corrected_gain'].rename('gain'), index_val], axis=1)

#%%
total_df.columns

#%%
print(perform_new_ndcg(total_df, 'prediction', 'gain'))

#%% RANDOM SCORE 
# random_prediction = np.random.normal(size=(len(total_df,)))
# random_df = pd.DataFrame(random_prediction, columns=['prediction'])
# random_df = pd.concat([random_df, df_val['total_non_corrected_gain'].rename('gain'), index_val], axis=1)
# print(perform_new_ndcg(random_df, 'prediction', 'gain'))

#%%
#%% SCORE ON JUST <PROP_ID, DESTINATION_ID> CORRECTED GAIN
# df_train = pd.read_csv('prepped_df_train.csv')
# df_test = pd.read_csv('prepped_df_test.csv')
# df_val = pd.read_csv('prepped_df_val.csv')

# for c in df_train.columns:
#     print(c)

#%%
# GBR_corrected_position = GradientBoostingRegressor()
# X_train_corrected_position = X_train['corrected_position']
# GBR_corrected_position.fit(X_train_gain.values, target_train_gain.values.reshape(-1,))


#%%
