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

#%% Index = srch_id, prop_id
index_train = df_train[['srch_id', 'prop_id']]
index_val = df_val[['srch_id', 'prop_id']]
index_test = df_test[['srch_id', 'prop_id']]

# Group sizes
train_group_sizes = df_train.groupby(['srch_id']).size().values
val_group_sizes = df_val.groupby(['srch_id']).size().values
test_group_sizes = df_test.groupby(['srch_id']).size().values
#%% STEP 1: POSTION BASED MODEL
# Since position based, only take non-random to train on!
X_train_pos = df_train[df_train['random_bool'] == 0]
X_val_pos = df_val[df_val['random_bool'] == 0]
X_test_pos = df_test[df_test['random_bool'] == 0]

# Remove ....

#%% Target = position (position based model)
y_train_pos = df_train[['corrected_position']].values
y_val_pos = df_val[['corrected_position']].values

train_dmatrix_pos = DMatrix(X_train_pos, y_train_pos)
valid_dmatrix_pos = DMatrix(X_val_pos, y_val_pos)
test_dmatrix_pos = DMatrix(X_test_pos)

train_dmatrix_pos.set_group(train_group_sizes)
valid_dmatrix_pos.set_group(val_group_sizes)
test_dmatrix_pos.set_group(test_group_sizes)
#%% XGBOOST MODEL: POSITION-BASED
#XGB model using efficient data structure Dmatrix
params = {'max_depth':3,
'min_child_weight':10,
'learning_rate':0.3,
'subsample':0.5,
'colsample_bytree':0.6,
'objective':'rank:pairwise',
'n_estimators':10000,
'eta':0.3,
'verbose':'True'}

# train model
xgb_model = xgb.train(params, train_dmatrix, evals=[(valid_dmatrix, 'validation')])

# make predictions
predict_xgb = xgb_model.predict(test_dmatrix)