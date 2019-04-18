#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Loading the database from a folder 1 hierarchy higher
df = pd.read_csv('../dataset_mood_smartphone.csv', index_col=0)
df['time'] = pd.to_datetime(df['time'])
df = df.drop_duplicates()

# Help function to split days
def split_to_days(x, splitvalue):
    if x.hour > splitvalue:
        return x
    else:
        return (x - datetime.timedelta(days = 1))

df['variable'] = df.variable.str.replace(" ","")

# Splitting the days on a specific hour, in this case 03:00
hourdaystarts = 3
df['subtractedtime'] = df.time.apply(lambda x: split_to_days(x, hourdaystarts))
df['date'] = df['subtractedtime'].dt.date
df['day'] = df['subtractedtime'].dt.weekday

# NORMALISATION
# print(df.variable.unique())
# print(df[df.variable.isin(['circumplex.arousal', 'circumplex.valence'])].value * 100)

# df[df.variable.isin(['circumplex.arousal', 'circumplex.valence'])].value = df[df.variable.isin(['circumplex.arousal', 'circumplex.valence'])].value * 1000
# print(df[df.variable.isin(['circumplex.arousal', 'circumplex.valence'])].value)

# SUM FOR ALL OTHERS
sumdf = df[~df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence', 'activity'])].groupby(['id', 'date', 'variable'])['value'].sum().unstack()
sumdf.columns = 'sum_' + sumdf.columns

# MEAN FOR MOOD, AROUSAL, VALENCE & ACTIVITY
meandf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence', 'activity'])].groupby(['id', 'date', 'variable'])['value'].mean().unstack()
meandf.columns = 'mean_' + meandf.columns

# STD FOR MOOD, AROUSAL, VALENCE & ACTIVITY
stddf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence', 'activity'])].groupby(['id', 'date', 'variable'])['value'].std().unstack()
stddf.columns = 'std_' + stddf.columns

# CONCAT IN FINAL DF
final = pd.concat([meandf, stddf, sumdf], axis=1, sort=True)

#Filling na with 0 for all sum columns
final.loc[:, ~final.columns.isin(['mood', 'mean_activity', 'mean_circumplex.arousal', 'mean_circumplex.valence',
       'mean_mood', 'std_activity', 'std_circumplex.arousal',
       'std_circumplex.valence', 'std_mood'])] = final.loc[:, ~final.columns.isin(['mood', 'mean_activity', 'mean_circumplex.arousal', 'mean_circumplex.valence',
       'mean_mood', 'std_activity', 'std_circumplex.arousal',
       'std_circumplex.valence', 'std_mood'])].fillna(0)



# INTERPOLATE MEAN MOOD
final['interpolate_mood_bool'] = final['mean_mood'].isnull().astype(int)
final['mean_mood'] = final['mean_mood'].groupby(['id']).fillna(method='ffill')

final['interpolate_arousal_bool'] = final['mean_circumplex.arousal'].isnull().astype(int)
final['mean_circumplex.arousal'] = final['mean_circumplex.arousal'].groupby(['id']).fillna(method='ffill')

final['interpolate_valence_bool'] = final['mean_circumplex.valence'].isnull().astype(int)
final['mean_circumplex.valence'] = final['mean_circumplex.valence'].groupby(['id']).fillna(method='ffill')

final['interpolate_activity_bool'] = final['mean_activity'].isnull().astype(int)
final['mean_activity'] = final['mean_activity'].groupby(['id']).fillna(method='ffill')

# SHIFT INTERPOLATED MEAN MOOD FOR TARGET COLUMN
final['shifted_target_mood_bool'] = final.groupby(['id'])['interpolate_mood_bool'].transform(lambda x:x.shift(-1))
final['target_mood'] = final.groupby(['id'])['mean_mood'].transform(lambda x:x.shift(-1))

# DROPPING THE NA's THAT STILL OCCUR, ARE THE DATA BEFORE THE FIRST 
# OCCURANCE OF THE VALUE, THUS THE ONES WE CANT INTERPOLATE
final.dropna(axis=0, how='any', inplace=True)

#%%
final.to_pickle('database_basic.pkl')
print('Saved in database_basic.pkl')

#%%
final['mean_circumplex.arousal']

#%%
