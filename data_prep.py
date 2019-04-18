#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Loading the database from a folder 1 hierarchy higher
df = pd.read_csv('../dataset_mood_smartphone.csv', index_col=0)
df['time'] = pd.to_datetime(df['time'])

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

# INTERPOLATE MEAN MOOD
final['interpolate_bool'] = final['mean_mood'].isnull().astype(int)
final['interpol_mean_mood'] = final['mean_mood'].groupby(['id']).fillna(method='ffill')

# SHIFT INTERPOLATED MEAN MOOD FOR TARGET COLUMN
final['shifted_interpol_mean_mood'] = final.groupby(['id'])['interpol_mean_mood'].transform(lambda x:x.shift(-1))

print(final[['interpolate_bool', 'interpol_mean_mood', 'mean_mood']])

sumdf#%%
final.to_pickle('database_basic.pkl')
print('Saved in database_basic.pkl')

#%%
