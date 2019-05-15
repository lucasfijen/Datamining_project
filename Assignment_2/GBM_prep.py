#%% Prep for GBM test
# Imports
import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")

#%% Reading in db NOT ALL OF IT, JUST 1000 rows for testing
try:
    df = pd.read_csv('data/training_set_VU_DM.csv', nrows=1000)
except:
    df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')

#%% Calculate the month for which people are looking
df['date_time'] = pd.to_datetime(df['date_time'])
df['target_month'] = (df['date_time'] + df['srch_booking_window'].astype('timedelta64[D]')).dt.month

#%% Make a new column which is the position divided by the max position
# of that srch_id. Also, first close the missing positions.

# THIS IS THE TARGET VARIABLE y

df = df.sort_values(['srch_id', 'position'])
df['corrected_position'] = df.groupby(['srch_id']).cumcount()+1
df['corrected_position'] = df.corrected_position / df.groupby('srch_id').corrected_position.transform(np.max) 

#%% If we are predicting position, we should only select NON-random positioning

df = df[df['random_bool'] == 0]
#%%
print(df.shape)

#%% Lets see what we have left
print('The datatypes:', df.dtypes)

#%%
drop_columns = ['click_bool', 'gross_bookings_usd', 'booking_bool', 'date_time', 'position']
df = df.drop(drop_columns, axis=1)
print('The datatypes:', df.dtypes)

#%% ONE HOT
df = pd.get_dummies(df, columns=['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id', 'target_month'], \
                    prefix=['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'srch_destination_id', 'target_month'])

#%%
y = df['corrected_position'].values
df = df.drop('corrected_position', axis=1)
X = df.values

#%%
print(X.shape)
print(y.shape)


#%%
