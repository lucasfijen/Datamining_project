#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Loading the database from a folder 1 hierarchy higher
df = pd.read_csv('/Users/janzuiderveld/Documents/GitHub/dataset_mood_smartphone.csv', index_col=0)
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

# Replacing every value with amount of times on apps or screen that are negative,
# Considered as invallid values, thus set to 0
df.loc[(df.variable.isin(['appCat.builtin',
                         'appCat.communication', 
                         'appCat.entertainment', 
                         'appCat.finance', 
                         'appCat.game', 
                         'appCat.office', 
                         'appCat.other', 
                         'appCat.social', 
                         'appCat.travel', 
                         'appCat.unknown', 
                         'appCat.utilities', 
                         'appCat.weather',
                         'screen'])) & (df.value < 0), 'value'] = 0

# NORMALISATION all values to range [0-1]
#StandardScaling
Scaler = StandardScaler()
Scaler.fit(df.loc[df.variable.isin(['circumplex.valence']), 'value'].values.reshape(-1,1))
df.loc[df.variable.isin(['circumplex.valence']), 'value'] = Scaler.transform(df.loc[df.variable.isin(['circumplex.valence']), 'value'].values.reshape(-1,1))

Scaler = StandardScaler()
Scaler.fit(df.loc[df.variable.isin(['circumplex.arousal']), 'value'].values.reshape(-1,1))
df.loc[df.variable.isin(['circumplex.arousal']), 'value'] = Scaler.transform(df.loc[df.variable.isin(['circumplex.arousal']), 'value'].values.reshape(-1,1))

#Normalization
# df.loc[df.variable.isin(['circumplex.arousal', 'circumplex.valence']), 'value'] += 2
# df.loc[df.variable.isin(['circumplex.arousal', 'circumplex.valence']), 'value'] /= 4

#%% Moods are now from 1-10, should be [0-1]
#StandardScaling
Scaler = StandardScaler()
Scaler.fit(df.loc[df.variable.isin(['mood']), 'value'].values.reshape(-1,1))
df.loc[df.variable.isin(['mood']), 'value'] = Scaler.transform(df.loc[df.variable.isin(['mood']), 'value'].values.reshape(-1,1))

#Normalization
# df.loc[df.variable.isin(['mood']), 'value'] -= 1
# df.loc[df.variable.isin(['mood']), 'value'] /= 9

# Normalise from seconds to range [0-1]
# df.loc[df.variable.isin(['appCat.builtin',
#                          'appCat.communication', 
#                          'appCat.entertainment', 
#                          'appCat.finance', 
#                          'appCat.game', 
#                          'appCat.office', 
#                          'appCat.other', 
#                          'appCat.social', 
#                          'appCat.travel', 
#                          'appCat.unknown', 
#                          'appCat.utilities', 
#                          'appCat.weather',
#                          'screen']), 'value'] /= 216000

# NORMALISATION OF THE COUNTS OF SMS AND CALLS
# There is no maximum amount of calls per day, but we chose a high value
# which is way higher than the highest observed value in the trainingdata
# df.loc[df.variable.isin(['sms', 'call']), 'value'] /= 50

# SUM FOR ALL OTHERS
sumdf = df[~df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence', 'activity'])].groupby(['id', 'date', 'variable'])['value'].sum().unstack()
sumdf.columns = 'sum_' + sumdf.columns

# Standardize
Scaler = StandardScaler()
Scaler.fit(sumdf.values)

# sumdf.loc[~sumdf.variable.isin(['lel']), 'value'] = Scaler.transform(sumdf.loc[~df.variable.isin(['lel']), 'value'])

sumdfScaled = pd.DataFrame(Scaler.transform((sumdf.values)))
sumdfScaled.columns = 'sum_' + sumdf.columns
sumdfScaled.index = sumdf.index.copy()

# MEAN FOR MOOD, AROUSAL, VALENCE & ACTIVITY
meandf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence', 'activity'])].groupby(['id', 'date', 'variable'])['value'].mean().unstack()
meandf.columns = 'mean_' + meandf.columns

# STD FOR MOOD, AROUSAL, VALENCE & ACTIVITY
stddf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence', 'activity'])].groupby(['id', 'date', 'variable'])['value'].std().unstack()
stddf.columns = 'std_' + stddf.columns

# CONCAT IN FINAL DF
final = pd.concat([meandf, stddf, sumdfScaled], axis=1, sort=True)

#Filling na with 0 for all sum columns
final.loc[:, ~final.columns.isin(['mood', 'mean_activity', 'mean_circumplex.arousal', 'mean_circumplex.valence',
       'mean_mood', 'std_activity', 'std_circumplex.arousal',
       'std_circumplex.valence', 'std_mood'])] = final.loc[:, ~final.columns.isin(['mood', 'mean_activity', 'mean_circumplex.arousal', 'mean_circumplex.valence',
       'mean_mood', 'std_activity', 'std_circumplex.arousal',
       'std_circumplex.valence', 'std_mood'])].fillna(0)


# final.to_csv('final.csv')

# INTERPOLATE MEAN MOOD
final['interpolate_mood_bool'] = final['mean_mood'].isnull().astype(int)
final[['mean_mood', 'std_mood']] = final[['mean_mood', 'std_mood']].groupby(['id']).fillna(method='ffill')

final['interpolate_arousal_bool'] = final['mean_circumplex.arousal'].isnull().astype(int)
final[['mean_circumplex.arousal', 'std_circumplex.arousal']] = final[['mean_circumplex.arousal', 'std_circumplex.arousal']].groupby(['id']).fillna(method='ffill')

final['interpolate_valence_bool'] = final['mean_circumplex.valence'].isnull().astype(int)
final[['mean_circumplex.valence', 'std_circumplex.valence']] = final[['mean_circumplex.valence', 'std_circumplex.valence']].groupby(['id']).fillna(method='ffill')

final['interpolate_activity_bool'] = final['mean_activity'].isnull().astype(int)
final[['mean_activity', 'std_activity']] = final[['mean_activity', 'std_activity']].groupby(['id']).fillna(method='ffill')

# SHIFT INTERPOLATED MEAN MOOD FOR TARGET COLUMN
final['shifted_target_mood_bool'] = final.groupby(['id'])['interpolate_mood_bool'].transform(lambda x:x.shift(-1))
final['target_mood'] = final['mean_mood'].groupby(['id']).transform(lambda x:x.shift(-1))
# final.to_csv('final.csv')
# DROPPING THE NA's THAT STILL OCCUR, ARE THE DATA BEFORE THE FIRST 
# OCCURANCE OF THE VALUE, THUS THE ONES WE CANT INTERPOLATE
final.dropna(axis=0, how='any', inplace=True)

#%%
final.to_pickle('/Users/janzuiderveld/Documents/GitHub/database_basic.pkl')
print('Saved in database_basic.pkl')
final.to_csv('/Users/janzuiderveld/Documents/GitHub/database_basic.csv')
#%%
