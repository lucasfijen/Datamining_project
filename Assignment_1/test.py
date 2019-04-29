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

mooddf = df[df.variable == 'mood']
# Create a df with the mean values of a mood per day
meanmooddf = mooddf.groupby(['id', 'date']).mean().reset_index()
# meanmooddf


#%%
df.variable.value_counts()
# df.columns

#%%
# Plotting the moods per person averaged over days

for i in meanmooddf.id.unique():
    values = meanmooddf[meanmooddf.id == i].value
    plt.plot([i for i in range(values.size)], values)
    plt.title(i)
    plt.ylabel('Mood')
    plt.xlabel('Day')
    plt.show()

#%%
# Boxplot of moods per weekday
mooddf.boxplot(grid=False,column=['value'], by='day')
plt.show()

#%%
# Plotting moods per day per user
for i in mooddf.id.unique():
    mooddf[mooddf.id == i].boxplot(grid=False, column=['value'], by='day')
    plt.title(i)

    plt.show()

#%%
mooddf.boxplot(grid=False, rot=90, column=['value'], by='id')
plt.show()

#%%
# This one seems a bit odd, 144 time 7, but every now and then
# a 6 or 8, however boxplot shows clearly only the 7s
print(mooddf[mooddf.id == 'AS14.31'].value.value_counts())

#%%
# Get sum values per id, date and unique variable value
# Then converts them into columns
sumdf = df[~df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].sum().unstack()
sumdf

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
# # print(targetdf.shape)
del targetdf['date']
targetdf['date'] = df.subtractedtime.apply(lambda x: x - datetime.timedelta(days = 1)).dt.date
# # Create a df with the mean values of a mood per day
targetdf = targetdf.groupby(['id', 'date', 'variable'])['value'].mean().unstack()
targetdf.columns = 'target' + targetdf.columns
# print(targetdf2.shape)

# Concatenates these dfs into one
finaldf = pd.concat([targetdf, meandf, stddf, mediandf, sumdf], axis=1, sort=True)
finaldf


#%%
filtereddf = finaldf.dropna(subset=['targetmood']).fillna(0)


#%%
filtereddf

#%%
# Splitting the data in the target values y and the predictors X
# Excluded some categories that are not known beforehand
y = filtereddf.targetmood.values
X = filtereddf.loc[:, ~filtereddf.columns.isin(['targetmood'])].values
y = y.reshape((len(y), 1))
#%%
# Adding a bias to X
X = np.hstack((np.ones(y.shape), X))

#%%
from numpy.linalg import lstsq

# Performing linear regression
betas = lstsq(X, y, rcond=None)[0]

#Calculating predicted ys and their MSE and r
y_hat = X.dot(betas)
mse = np.mean((y - y_hat)**2)
N = y.size
P = X.shape[1]
r = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
print(betas)
print(filtereddf.columns)
print('mse: %.3f' % mse)
print('r value: %.3f' % r)
#%%


#%%
for i in df.variable.unique():
    df[df.variable == i].value.hist(bins=100)
    plt.title(i)
    plt.show()

#%%

for i in df.id.unique():
    mooddf = df[(df.variable == 'mood') & (df.id == i)].sort_values(by=['date'])
    plt.plot(mooddf['date'], mooddf['value'])
    plt.title(i)
    plt.show()

#%%
for i in sumdf.columns:
    sumdf[i].hist()
    plt.title(i)
    plt.show()

#%%
sumdf.min()

#%%
df[df.variable == 'activity'].value.hist()

#%%
#%%
sumdf.isna().sum(axis=0)
# finaldf.count(axis=0) + finaldf.isna().sum(axis=0)
# sumdf.shape

#%%
