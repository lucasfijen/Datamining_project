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
