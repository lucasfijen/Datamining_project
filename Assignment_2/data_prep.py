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

df = pd.read_csv('data/training_set_VU_DM.csv',nrows=1000)
df_test = pd.read_csv('data/test_set_VU_DM.csv',nrows=1000)

#%% Numerical features, average over prop_id (std & median)
df, df_test = add_descriptors(df, df_test, 'prop_id')

df = pd.read_csv('data/training_set_VU_DM.csv')
df_test = pd.read_csv('data/test_set_VU_DM.csv')

#%% DEDUCT MONTH as categorical variable
def add_target_month(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['target_month'] = (df['date_time'] + df['srch_booking_window'].astype('timedelta64[D]')).dt.month
    return df

df = add_target_month(df)
df_test = add_target_month(df_test)
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
df_train.columns
#%% <PROP_ID, DESTINATION_ID> performance in terms of POSITION & CORRECTED GAIN
df_train, gb_train = create_prop_dest_mean_performance(df_train, ['total_corrected_gain'], None)
df_val, _ = create_prop_dest_mean_performance(df_val, ['total_corrected_gain'], gb_train)

#%% One hot encoding van site_id, visitor_location_country, prop_country_id, target_month
# Not: 'prop_id', 'srch_destination_id', 
def onehot(df):
    return pd.get_dummies(df, columns=['site_id', 'visitor_location_country_id', 'prop_country_id', 'target_month'], \
                    prefix=['site_id', 'visitor_location_country_id', 'prop_country_id', 'target_month'])

df_train = onehot(df_train)
df_val = onehot(df_val)
df_test = onehot(df_test)

#%%
df_train.to_pickle('prepped_df_train.pkl')
df_test.to_pickle('prepped_df_test.pkl')
df_val.to_pickle('prepped_df_val.pkl')

#%%
