#%% Imports
import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")

#%% Reading in db
try:
    df = pd.read_csv('data/training_set_VU_DM.csv')
except:
    df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')


#%% Calculate the month for which people are looking
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
trainingset, valset = split_dataset(df)

#%% Print some stats
print('split devision')
print(len(trainingset) / (len(trainingset) + len(valset)))
print('')
print('Devision random - not random')
print('training')
print(valset.random_bool.value_counts())
print('val')
print(trainingset.random_bool.value_counts())
print('')
print('Devision clicks')
print('training')
print(valset.click_bool.value_counts())
print('val')
print(trainingset.click_bool.value_counts())
print('')
print('Devision of months')
print('training')
print(trainingset.target_month.value_counts())
print('val')
print(valset.target_month.value_counts())

#%% Write to files
valset.to_pickle('Assignment_2/data/valset.pkl')
trainingset.to_pickle('Assignment_2/data/trainingset.pkl')

#%%
