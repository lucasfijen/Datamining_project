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

#%%
random = df[df['random_bool'] == 1]
non_random = df[df['random_bool'] == 0]

random_clicks = random.groupby('position')['click_bool'].mean()
random_bookings = random.groupby('position')['booking_bool'].mean()
random_ = pd.concat([random_clicks.rename('clicks'), random_bookings.rename('bookings')], axis=1)

nonrandom_clicks = non_random.groupby('position')['click_bool'].mean()
nonrandom_bookings = non_random.groupby('position')['booking_bool'].mean()
nonrandom_ = pd.concat([nonrandom_clicks.rename('clicks'), nonrandom_bookings.rename('bookings')], axis=1)

#%%
clicks_correction = random_clicks / nonrandom_clicks
bookings_correction = random_bookings / nonrandom_bookings
correction_df = pd.concat([random_clicks.rename('random clicks'), nonrandom_clicks.rename('non_random_clicks'), random_bookings.rename('random bookings'), nonrandom_bookings.rename('nonrandom bookings'), clicks_correction.rename('random/nonrandom clicks'), bookings_correction.rename('random/nonrandom bookings')], axis=1)
correction_df.to_csv('position_bias_correction_df.csv')

#%% PLOT RANDOM VERSUS NON-RANDOM FRACTION OF CLICKS & BOOKINGS PER POSITION
all = pd.concat([random_clicks.rename('random clicks'), nonrandom_clicks.rename('non random clicks'), random_bookings.rename('random bookings'), nonrandom_bookings.rename('non random_bookings')], axis=1)
all_plot = all.plot.bar(figsize=[14, 10], width=1, color=sns.color_palette("Paired")[:4], linewidth=0)
all_plot.set_ylabel('fraction', fontsize=14)
all_plot.set_xlabel('position', fontsize=14)
all_plot.set_title('Fraction of clicks and bookings at positions of the result page', fontsize=20)
all_fig = all_plot.get_figure()
all_fig.savefig('random_vs_nonrandom.png', dpi=200, facecolor='white')
#%%
# random_nonrandom_bookings = nonrandom_bookings - random_bookings
# random_nonrandom_bookings.plot.bar(color='skyblue')
#%%
# random_nonrandom_clicks = nonrandom_clicks - random_clicks
# random_nonrandom_clicks.plot.bar(color='skyblue')
#%%
# random_plot = random_.plot.bar(width=1, figsize=[14, 8], title='Random positioning')
# random_fig = random_plot.get_figure()
# random_fig.savefig('random_position.png', dpi=200, facecolor='white')
#%%
# nonrandom_fig = nonrandom_plot.get_figure()
# nonrandom_plot = nonrandom_.plot.bar(width=1, figsize=[14, 8], title='Non-random positioning')
# nonrandom_fig.savefig('nonrandom_position.png', dpi=200, facecolor='white')
#%% Take a small subset because the dataset is BIG
smalldf = df.head(10000)

#%% The number of times a property occurs in the dataset
df.prop_id.value_counts()

#%% The number of countries is 172
df.prop_country_id.value_counts()

#%% 

#%%
# One srch_id corresponds to one result page of a search and
# the variable that is to be predicted is the booking_bool (and click_bool?)

# One result page of datapoints:
# print(smalldf[smalldf['srch_id'] == 1])

# The result pages are NOT of equal length:
RP_lengths = []
clicked = []
booked = []
clicked_and_booked = []
for search_id in np.unique(smalldf['srch_id'].values):
    RP_lengths.append(smalldf[smalldf['srch_id'] == search_id].shape[0])
    clicked.append(len(smalldf[(smalldf['srch_id'] == search_id) & (smalldf['click_bool'] == 1)]))
    booked.append(len(smalldf[(smalldf['srch_id'] == search_id) & (smalldf['booking_bool'] == 1)]))
    clicked_and_booked.append(len(smalldf[(smalldf['srch_id'] == search_id) & (smalldf['booking_bool'] == 1) & (smalldf['click_bool'] == 1)]))

#%% RESULT PAGE LENGTH (how many properties are shown for a user search)
plt.hist(RP_lengths, bins=max(RP_lengths), density=True)
plt.xlabel('Length of the result page (number of properties)')
plt.ylabel('Percentage of searches')
plt.show()

#%% Per user search 1 or 0 hotels are booked (never more)
print(np.unique(booked))
plt.bar([0, 1], [len(booked) - np.count_nonzero(booked), np.count_nonzero(booked)])
plt.show()

#%% 
print('Per user search :', np.unique(clicked), 'nr of times clicked, so never 0!')
# plt.bar([0, 1], [len(clicked) - np.count_nonzero(clicked), np.count_nonzero(clicked)])
plt.hist(clicked, density=True)
plt.show()

#%% BOOKED = CLICKED it seems (which makes sense)
labels = ['booked', 'clicked', 'clicked_and_booked']
plt.hist([booked, clicked, clicked_and_booked], density=True, label=labels)
plt.legend()
plt.show()