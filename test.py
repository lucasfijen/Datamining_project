#%%
import pandas as pd
import datetime
df = pd.read_csv('../dataset_mood_smartphone.csv', index_col=0)
df['time'] = pd.to_datetime(df['time'])

#Substracted 1 second from the time so 00:00 will be the last day)
df['time'] -= datetime.timedelta(seconds=1)


mooddf = df[df.variable == 'mood']
# Create a df with the mean values of a mood per day
meanmooddf = mooddf.groupby(['id',mooddf.time.dt.date]).mean().reset_index()
print(meanmooddf)


#%%
