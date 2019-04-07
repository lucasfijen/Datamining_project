#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
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
meanmooddf


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
#%%
# Calculate mean values per id per date per unique variable

meandf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].mean().unstack()
meandf.columns = 'mean' + meandf.columns
#%%
# Calculate std values per id per date per unique variable
stddf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].std().unstack()
stddf.columns = 'std' + stddf.columns
#%%
# Calculate std values per id per date per unique variable
mediandf = df[df.variable.isin(['mood', 'circumplex.arousal', 'circumplex.valence'])].groupby(['id', 'date', 'variable'])['value'].median().unstack()
mediandf.columns = 'median' + mediandf.columns

#%%
finaldf = pd.concat([meandf, stddf, mediandf, sumdf], axis=1, sort=False)
finaldf


#%%
