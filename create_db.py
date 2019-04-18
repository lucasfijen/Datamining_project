#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Loading the database from a folder 1 hierarchy higher
df = pd.read_csv('../dataset_mood_smartphone.csv', index_col=0)
df['time'] = pd.to_datetime(df['time'])

df.head()

#%%

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

#%% ADDING COLUMNS
sumdf = df[~df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].sum().unstack()
# sumdf

# Calculate mean values per id per date per unique variable
meandf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].mean().unstack()
meandf.columns = 'mean' + meandf.columns

# Calculate std values per id per date per unique variable
stddf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].std().unstack()
stddf.columns = 'std' + stddf.columns

# Calculate median values per id per date per unique variable
mediandf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].median().unstack()
mediandf.columns = 'median' + mediandf.columns

targetdf = df[df.variable == 'mood'].copy()
del targetdf['date']
targetdf['date'] = df.subtractedtime.apply(lambda x: x - datetime.timedelta(days = 1)).dt.date
# # Create a df with the mean values of a mood per day
targetdf = targetdf.groupby(['id', 'date', 'variable'])['value'].mean().unstack()
targetdf.columns = 'target' + targetdf.columns

# Concatenates these dfs into one
finaldf = pd.concat([targetdf, meandf, stddf, mediandf, sumdf], axis=1, sort=True)

finaldf.to_pickle('../database.pkl')
print('Saved in database.pkl')

finaldf.head()

#%%

finaldf.columns


#%%
