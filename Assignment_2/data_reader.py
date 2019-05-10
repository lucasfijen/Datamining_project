#%% Imports
import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

#%% Reading in db
try:
    df = pd.read_csv('data/training_set_VU_DM.csv')
except:
    df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')
#%% Basis featurs of the dataset
print('Number of datapoints:', df.shape[0], 'Number of initial features:', df.shape[1], '\n')
print('The features:', df.columns.values)

#%% Position bias relative chance of positions beeing clicked calculated
random = df[df['random_bool'] == 1]
non_random = df[df['random_bool'] == 0]

random_clicks = random.groupby('position')['click_bool'].mean()
random_bookings = random.groupby('position')['booking_bool'].mean()
random_ = pd.concat([random_clicks.rename('clicks'), random_bookings.rename('bookings')], axis=1)

nonrandom_clicks = non_random.groupby('position')['click_bool'].mean()
nonrandom_bookings = non_random.groupby('position')['booking_bool'].mean()
nonrandom_ = pd.concat([nonrandom_clicks.rename('clicks'), nonrandom_bookings.rename('bookings')], axis=1)

#%% Calculate the relative chances of a click being performed due to click bias
clicks_correction = random_clicks / nonrandom_clicks
bookings_correction = random_bookings / nonrandom_bookings
correction_df = pd.concat([random_clicks.rename('random clicks'), nonrandom_clicks.rename('non_random_clicks'), random_bookings.rename('random bookings'), nonrandom_bookings.rename('nonrandom bookings'), clicks_correction.rename('random/nonrandom clicks'), bookings_correction.rename('random/nonrandom bookings')], axis=1)
correction_df.reset_index(inplace=True)
correction_df.to_csv('position_bias_correction_df.csv')

#%% Perform merge to other dataframe
temp_df = correction_df[['position', 'random/nonrandom clicks', 'random/nonrandom bookings']]
newdf = df.merge(temp_df, on='position')
print(newdf.head(20))
#%% add the scorings and compensated scorings based on click bias
newdf['corrected_click_gain'] = newdf.click_bool * (1 - newdf['random/nonrandom clicks'])
newdf['corrected_book_gain'] = newdf.booking_bool * (1 - newdf['random/nonrandom bookings']) * 5
newdf['corrected_total'] = newdf['corrected_click_gain'] + newdf['corrected_book_gain']
newdf['non_corrected_total'] = newdf['click_bool'] + (5 * newdf['booking_bool'])
print(newdf[(newdf.booking_bool == 1) & (newdf.position ==1) & (newdf.click_bool ==1)]['non_corrected_total'])
print(df.shape)
print(newdf.shape)

#%% PLOT RANDOM VERSUS NON-RANDOM FRACTION OF CLICKS & BOOKINGS PER POSITION
all = pd.concat([random_clicks.rename('random clicks'), nonrandom_clicks.rename('non random clicks'), random_bookings.rename('random bookings'), nonrandom_bookings.rename('non random_bookings')], axis=1)
all_plot = all.plot.bar(figsize=[14, 10], width=1, color=sns.color_palette("Paired")[:4], linewidth=0)
all_plot.set_ylabel('fraction', fontsize=14)
all_plot.set_xlabel('position', fontsize=14)
all_plot.set_title('Fraction of clicks and bookings at positions of the result page', fontsize=20)
all_fig = all_plot.get_figure()
all_fig.savefig('random_vs_nonrandom.png', dpi=200, facecolor='white')

#%%
# random_plot = random_.plot.bar(width=1, figsize=[14, 8], title='Random positioning')
# random_fig = random_plot.get_figure()
# random_fig.savefig('random_position.png', dpi=200, facecolor='white')
#%%
# nonrandom_fig = nonrandom_plot.get_figure()
# nonrandom_plot = nonrandom_.plot.bar(width=1, figsize=[14, 8], title='Non-random positioning')
# nonrandom_fig.savefig('nonrandom_position.png', dpi=200, facecolor='white')
