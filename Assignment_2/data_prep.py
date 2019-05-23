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
from categorical_processing import *
from pathlib import Path
from average_prop_dest_performance import *

#%% Reading in db

df = pd.read_csv('data/training_set_VU_DM.csv', nrows=20000)
df_test = pd.read_csv('data/test_set_VU_DM.csv', nrows=20000)

print('df.shape', df.shape)
print('df_test.shape', df_test.shape)
print(set(df.columns) - set(df_test.columns))
#%% Numerical features, 
# over prop_id (std & median)
df, df_test = add_descriptors(df, df_test, 'prop_id')

print('df.shape', df.shape)
print('df_test.shape', df_test.shape)
print(set(df.columns) - set(df_test.columns))

# for c in df.columns:
#     print(c)
# over prop_country_id
# df, df_test = add_descriptors(df, df_test, 'prop_country_id')

# over 'srch_destination_id'
# df, df_test = add_descriptors(df, df_test, 'prop_country_id')


#%% fill nans

df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(10) # values are logs of probabilaty, all negtive
df = df.fillna(-10)

df_test['srch_query_affinity_score'] = df_test['srch_query_affinity_score'].fillna(10) # values are logs of probabilaty, all negtive
df_test = df_test.fillna(-10)

#%% DEDUCT MONTH as categorical variable
def add_target_month(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['target_month'] = (df['date_time'] + df['srch_booking_window'].astype('timedelta64[D]')).dt.month
    return df

df = add_target_month(df)
df_test = add_target_month(df_test)

#%% One hot encoding van site_id, visitor_location_country, prop_country_id, target_month
# Not: 'prop_id', 'srch_destination_id', 
# 'visitor_location_country_id', 'prop_country_id', drop also 'site_id', because not the same in sets

print('df.shape', df.shape)
print('df_test.shape', df_test.shape)
print(set(df.columns) - set(df_test.columns))

def onehot(df):
    onehot = pd.get_dummies(df, columns=['target_month'])
    # print(onehot.shape)
    return onehot


month_OH_train = onehot(pd.DataFrame(df['target_month']))
print(month_OH_train.shape)
month_OH_test = onehot(pd.DataFrame(df_test['target_month']))

df = pd.concat([df, month_OH_train], axis=1)
df_test = pd.concat([df_test, month_OH_test], axis=1)
df = df.drop(['target_month', 'date_time'], axis=1)
df_test = df_test.drop(['target_month', 'date_time'], axis=1)

print('df.shape', df.shape)
print('df_test.shape', df_test.shape)
print(set(df.columns) - set(df_test.columns))

#%% Perform Train / valid split
def split_dataset(df):
    np.random.seed(10)
    unique_srch_ids = df.srch_id.unique()
    np.random.shuffle(unique_srch_ids)
    selection = unique_srch_ids[:int(unique_srch_ids.size * 0.8)]
    msk = df.srch_id.isin(selection)
    training = df[msk]
    val = df[~msk]
    return training, val


df_train, df_val = split_dataset(df)

#%% CORRECTION for position bias (happens in position_bias.py)
# No knowledge of train may go into val and test!
pb_correction_train, corrected_gain_train = get_corrected_gain(df_train, None)
df_train = pd.concat([df_train, corrected_gain_train], axis=1)

_, corrected_gain_valid = get_corrected_gain(df_val, pb_correction_train)
df_val = pd.concat([df_val, corrected_gain_valid], axis=1)

#%%

df_train[['position', 'click_bool', 'booking_bool', 'corrected_click_gain', 'corrected_book_gain']].iloc[300:400, :]

# _, corrected_gain_test = get_corrected_gain(df_test, pb_correction_train)
# df_test = pd.concat([df_test, corrected_gain_test], axis=1)

#%% Normalized position
def normalize_pos(df):
    df = df.sort_values(['srch_id', 'position'])
    df['corrected_position'] = df.groupby(['srch_id']).cumcount()+1
    df['corrected_position'] = df.corrected_position / df.groupby('srch_id').corrected_position.transform(np.max) 
    return df

df_train = normalize_pos(df_train)
df_val = normalize_pos(df_val)

#%%
df_train[df_train['srch_id']==4][['position', 'corrected_position']]

#%%
#%% <PROP_ID, DESTINATION_ID> performance in terms of POSITION & CORRECTED GAIN

# print("CHECK!!!!!!!!")
# print(df_train.columns)
# print(df_val.columns)
# print(df_test.columns)
# exit()
df_train, gb_train = create_prop_dest_mean_performance(df_train, ['total_corrected_gain'], None)
df_val, _ = create_prop_dest_mean_performance(df_val, ['total_corrected_gain'], gb_train)
# DOES NOT WORK FOR TEST
df_test, _ = create_prop_dest_mean_performance(df_test, ['total_corrected_gain'], gb_train)

# df_train.rename(columns={'total_corrected_gain_mean':'XXX_GAIN_MEAN_XXX'}, inplace=True)
# df_val.rename(columns={'total_corrected_gain_mean':'XXX_GAIN_MEAN_XXX'}, inplace=True)
# df_test.rename(columns={'total_corrected_gain_mean':'XXX_GAIN_MEAN_XXX'}, inplace=True)

#%% ALSO DROP some stuff: 
def drop(df):
    df = df.drop(['site_id', 'visitor_location_country_id', 'prop_country_id'], axis=1)
    return df

df_test = drop(df_test)
df_train = drop(df_train)
df_val = drop(df_val)

#%% SANITY CHECK
print('df_train.shape', df_train.shape)
print('df_val.shape', df_val.shape)
print('df_test.shape', df_test.shape)
print(set(df_train.columns) - set(df_test.columns))
print(set(df_val.columns) - set(df_test.columns))
print(set(df_train.columns) - set(df_val.columns))

#%%
print('writing to file')
df_train.to_csv('prepped_df_train.csv')
df_test.to_csv('prepped_df_test.csv')
df_val.to_csv('prepped_df_val.csv')
#%%
