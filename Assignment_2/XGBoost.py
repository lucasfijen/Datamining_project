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

#%%
# POSITION 'pos' BASED OR GAIN 'gain
model_type = 'gain'

#%% LOAD
df_train = pd.read_csv('data/prepped_df_train.csv')
df_test = pd.read_csv('data/prepped_df_test.csv')
df_val = pd.read_csv('data/prepped_df_val.csv')


#%% SHAPE & DIFFERENCES
print('df_train.shape', df_train.shape)
print('df_val.shape', df_val.shape)
print('df_test.shape', df_test.shape)
print(set(df_train.columns) - set(df_test.columns))
print(set(df_val.columns) - set(df_test.columns))
print(set(df_train.columns) - set(df_val.columns))

#%%
# If position based, only take non-random to train on!
if model_type == 'pos':
    X_train = df_train[df_train['random_bool'] == 0]
else:
    X_train = df_train
X_val = df_val #[df_val['random_bool'] == 0]
X_test = df_test #[df_test['random_bool'] == 0]

#%% Index = srch_id, prop_id
index_train = X_train[['srch_id', 'prop_id']].values
index_val = X_val[['srch_id', 'prop_id']].values
index_test = X_test[['srch_id', 'prop_id']].values

# Group sizes
train_group_sizes = X_train.groupby(['srch_id']).size().values
val_group_sizes = X_val.groupby(['srch_id']).size().values
test_group_sizes = X_test.groupby(['srch_id']).size().values

#%% Target = position (position based model)
if model_type == 'pos':
    y_train = X_train[['corrected_position']].values
    # y_val = X_val[['corrected_position']].values
else:
    y_train = X_train[['total_corrected_gain']].values
    # y_val = X_val[['total_corrected_gain']].values
y_val = X_val[['total_non_corrected_gain']].values
#%% DELETE, most importantly: 'total_corrected_gain_mean' & 'total_corrected_gain_std'
exclude = ['srch_id', 
            'prop_id',
            ]
if model_type == 'pos':
    model_specific = ['total_corrected_gain_mean', 
                'total_corrected_gain_std']
else:
    model_specific = ['corrected_position_mean', 
                'corrected_position_std']
train_exclude = ['total_corrected_gain', 'corrected_click_gain', 
                 'total_non_corrected_gain', 'corrected_position', 
                 'booking_bool', 'click_bool', 'position', 
                 'gross_bookings_usd', 'corrected_book_gain'
                 ]

X_train = X_train.drop(model_specific+exclude+train_exclude, axis=1)
X_val = X_val.drop(model_specific+exclude+train_exclude, axis=1)
X_test = X_test.drop(model_specific+exclude, axis=1)

#%% FINAL CHECK
print('X_train.shape', X_train.shape)
print('X_val.shape', X_val.shape)
print('X_test.shape', X_test.shape)
print('y_train.shape', y_train.shape)
print('y_val.shape', y_val.shape)
print(set(X_train.columns) - set(X_test.columns))
print(set(X_val.columns) - set(X_test.columns))
print(set(X_train.columns) - set(X_val.columns))

#%% Prepare matrices
train_dmatrix = DMatrix(X_train.values, y_train)
valid_dmatrix = DMatrix(X_val.values, y_val)
test_dmatrix = DMatrix(X_test.values)

train_dmatrix.set_group(train_group_sizes)
valid_dmatrix.set_group(val_group_sizes)
test_dmatrix.set_group(test_group_sizes)

#%% XGBOOST MODEL: POSITION-BASED
#XGB model using efficient data structure Dmatrix
params = {
    'max_depth':3,
    'min_child_weight':10,
    'learning_rate':0.3,
    'subsample':0.5,
    'colsample_bytree':0.6,
    'objective':'rank:ndcg',
    'eval_metric':'ndcg@5',
    'n_estimators':10000,
    'verbose':'True',
    'nthread': 5,
}

# TRAIN
xgb_model = xgb.train(params, train_dmatrix, evals=[(valid_dmatrix, 'validation')])

# VALIDATE
predicted_val = xgb_model.predict(valid_dmatrix)

#%%
prediction_df = pd.DataFrame(predicted_val, columns=['prediction'])
corrected_df = pd.DataFrame(y_val, columns=['gain'])
index_df = pd.DataFrame(index_val, columns=['srch_id', 'prop_id'])
total_df = pd.concat([prediction_df, corrected_df, index_df], axis=1)

# total_df.head()
#%%
if model_type == 'pos':
    asc = True
else:
    asc = False

print("Validation NDCG:", perform_new_ndcg(total_df, 'prediction', 'gain', ascending=asc))
#%% TEST
predicted_test = xgb_model.predict(test_dmatrix)
#%%
prediction_df = pd.DataFrame(predicted_test, columns=['prediction'])
index_df = pd.DataFrame(index_test, columns=['srch_id', 'prop_id'])
total_df = pd.concat([prediction_df, index_df], axis=1)
total_df.to_pickle('test_prediction' + model_type + '.pickle')