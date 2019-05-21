#%% Imports
import os
os.chdir('Assignment_2')

import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")
from position_bias import *
from categorical_processing import *
from pathlib import Path


#%% Reading in db
df = pd.read_csv('data/training_set_VU_DM.csv',nrows=1000)
df_test = pd.read_csv('data/test_set_VU_DM.csv',nrows=1000)

#%%
df, df_test = add_descriptors(df, df_test, 'prop_id')
print(df)

#%% DEDUCT MONTH as categorical variable
df['date_time'] = pd.to_datetime(df['date_time'])
df['target_month'] = (df['date_time'] + df['srch_booking_window'].astype('timedelta64[D]')).dt.month

#%% Perform split
def split_dataset(df):
    np.random.seed(10)
    unique_srch_ids = df.srch_id.unique()
    np.random.shuffle(unique_srch_ids)
    selection = unique_srch_ids[:int(unique_srch_ids.size * 0.8)]
    msk = df.srch_id.isin(selection)
    training = df[msk]
    val = df[~msk]
    return training, val

if Path('data/valset.pkl').is_file():
    print("Reading split from file")
    val_df = pd.read_pickle('data/valset.pkl')
    train_df = pd.read_pickle('data/trainingset.pkl')
else:
    print("Making split")
    train_df, val_df = split_dataset(df)
    val_df.to_pickle('data/valset.pkl')
    train_df.to_pickle('data/trainingset.pkl')

#%% CORRECTION for position bias (happens in position_bias.py)
correction_df_train, corrected_gain_train = get_corrected_gain(train_df, None)
train_df = pd.concat([train_df, corrected_gain_train], dim=1)

_, corrected_gain_valid = get_corrected_gain(val_df, correction_df_train)
val_df = pd.concat([val_df, corrected_gain_valid], dim=1)
#%% Numerical features, average over prop_id (std & median)
all_groupby = all_numeric.groupby('prop_id',sort=True).agg([np.median, np.mean, np.std])


#%% <PROP_ID, DESTINATION_ID> performance in terms of POSITION & CORRECTED GAIN

prop_dest_avg_corrected_pos = dict()

for prop in df['prop_id'].unique():
    for dest in df['srch_destination_id'].unique():
        prop_dest_set = df.loc[(df['prop_id'] == prop) & df['srch_destination_id'].isin([dest])]
        prop_dest_avg_corrected_pos[prop+dest]= np.average(prop_dest_set['corrected_position'])

prop_dest_avg_gain = dict()

for prop in df['prop_id'].unique():
    for dest in df['srch_destination_id'].unique():
        prop_dest_set = df.loc[(df['prop_id'] == prop) & df['srch_destination_id'].isin([dest])]
        prop_dest_avg_gain[prop+dest]= np.average(prop_dest_set['non_corrected_total'])
