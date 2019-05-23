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
target_valid = df_val[['corrected_position']]

#%%
exclude = ['srch_id', 
            'prop_id'
            ]
train_exclude = ['total_non_corrected_gain', 
            'corrected_click_gain',
            'corrected_book_gain',
            'total_corrected_gainNone',
            'corrected_position',
            'total_corrected_gain_mean',
            'position',
            'click_bool',
            'gross_bookings_usd',
            'booking_bool']

X_test = df_test.drop(exclude, axis=1)
X_train = df_train.drop(exclude+train_exclude, axis=1)
X_val = df_val.drop(exclude+train_exclude, axis=1)

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
X_train_np = X_test.values()
y_train_np = target_train.values()
GBR = GradientBoostingRegressor()
GBR.fit(X_train_np, y_train_np)

MSE = mean_squared_error(y[-100:], GBR.predict(X[-100:]))
print("MSE: %.4f" % MSE)