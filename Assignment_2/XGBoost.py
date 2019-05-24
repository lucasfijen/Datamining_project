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
import xgboost as xgb
from xgboost import DMatrix
from itertools import groupby

#%% LOAD
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
X_train_pos = df_train[df_train['random_bool'] == 0]
X_val_pos = df_val #[df_val['random_bool'] == 0]
X_test_pos = df_test #[df_test['random_bool'] == 0]

#%% Index = srch_id, prop_id
index_train = X_train_pos[['srch_id', 'prop_id']].copy().values
index_val = X_val_pos[['srch_id', 'prop_id']].copy().values
index_test = X_test_pos[['srch_id', 'prop_id']].copy().values

# Group sizes
train_group_sizes = X_train_pos.groupby(['srch_id']).size().values
val_group_sizes = X_val_pos.groupby(['srch_id']).size().values
test_group_sizes = X_test_pos.groupby(['srch_id']).size().values

#%% Target = position (position based model)
y_train_pos = X_train_pos[['corrected_position']].values
y_val_pos = X_val_pos[['corrected_position']].values

#%% DELETE, most importantly: 'total_corrected_gain_mean' & 'total_corrected_gain_std'
exclude = ['srch_id', 
            'prop_id',
            'total_corrected_gain_mean', 
            'total_corrected_gain_std'
            ]
train_exclude = ['total_corrected_gain', 'corrected_click_gain', 
                 'total_non_corrected_gain', 'corrected_position', 
                 'booking_bool', 'click_bool', 'position', 
                 'gross_bookings_usd', 'corrected_book_gain'
                 ]

X_train_pos = X_train_pos.drop(exclude+train_exclude, axis=1)
X_val_pos = X_val_pos.drop(exclude+train_exclude, axis=1)
X_test_pos = X_test_pos.drop(exclude, axis=1)

#%% FINAL CHECK
print('X_train_pos.shape', X_train_pos.shape)
print('X_val_pos.shape', X_val_pos.shape)
print('X_test_pos.shape', X_test_pos.shape)
print('y_train_pos.shape', y_train_pos.shape)
print('y_val_pos.shape', y_val_pos.shape)
print(set(X_train_pos.columns) - set(X_test_pos.columns))
print(set(X_val_pos.columns) - set(X_test_pos.columns))
print(set(X_train_pos.columns) - set(X_val_pos.columns))

#%% Prepare matrices
train_dmatrix_pos = DMatrix(X_train_pos.values, y_train_pos)
valid_dmatrix_pos = DMatrix(X_val_pos.values, y_val_pos)
test_dmatrix_pos = DMatrix(X_test_pos.values)

train_dmatrix_pos.set_group(train_group_sizes)
valid_dmatrix_pos.set_group(val_group_sizes)
test_dmatrix_pos.set_group(test_group_sizes)

#%% XGBOOST MODEL: POSITION-BASED
#XGB model using efficient data structure Dmatrix
params_pos = {'max_depth':3,
'min_child_weight':10,
'learning_rate':0.3,
'subsample':0.5,
'colsample_bytree':0.6,
'objective':'rank:ndcg',
'n_estimators':10000,
'verbose':'True'}

# TRAIN
xgb_model_pos = xgb.train(params_pos, train_dmatrix_pos, evals=[(valid_dmatrix_pos, 'validation')])

# VALIDATE
predicted_val_pos = xgb_model_pos.predict(valid_dmatrix_pos)
#%%
print(predicted_val_pos.shape)

#%%
prediction_df = pd.DataFrame(predicted_val_pos, columns=['prediction'])
corrected_pos_df = pd.DataFrame(y_val_pos, columns=['gain'])
index_pos_df = pd.DataFrame(index_val, columns=['srch_id', 'prop_id'])
total_df = pd.concat([prediction_df, corrected_pos_df, index_pos_df], axis=1)

total_df.head()
#%%
print("Validation NDCG:", perform_new_ndcg(total_df, 'prediction', 'gain', ascending=True))
#%% TEST
predicted_test_pos = xgb_model_pos.predict(test_dmatrix_pos)
#%%
prediction_df = pd.DataFrame(predicted_test_pos, columns=['prediction'])
index_pos_df = pd.DataFrame(index_test, columns=['srch_id', 'prop_id'])
total_df = pd.concat([prediction_df, index_pos_df], axis=1)

total_df.to_pickle('test_prediction_pos.pickle')
#%%
# print(perform_new_ndcg(total_df, 'prediction', 'gain', ascending=True))
