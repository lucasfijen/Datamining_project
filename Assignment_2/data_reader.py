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

#%% Determine number of destination_ids per prop_id
prop_id_srch_dest = []
prop_dest_dict = dict()
# prop_dest_count = dict({key: 0 for key in np.unique(smalldf['prop_id'].values)})     # len(np.unique(smalldf['prop_id'].values))
prop_in_search_count = dict({key: 0 for key in np.unique(df['prop_id'].values)})

#Iterate over searches
for search_id in np.unique(df['srch_id'].values):
    search_items = df[(df['srch_id'] == search_id)]
    dest_id = np.unique(search_items['srch_destination_id'])
    prop_ids = search_items['prop_id']
    #iterate over properties in search
    for prop_id in prop_ids:
        prop_in_search_count[prop_id] += 1
        # add destination to value string if prop in dict
        if prop_id in prop_dest_dict:
            if str(dest_id) not in str(prop_dest_dict[prop_id]):
                prop_dest_dict[prop_id] = str(prop_dest_dict[prop_id]) + "," + str(dest_id)
        
        # else, create new prop key with destination value
        else:
            prop_dest_dict[prop_id] = str(dest_id) 


#%%
prop_dest_count = []
prop_dest_count_normalized = []
for prop_id in np.unique(df['prop_id'].values):
    # bracket in value str signals dest_id, count brackets, such innovative
    amount_of_dests = prop_dest_dict[prop_id].count('[')
    amount_appearances_in_search = prop_in_search_count[prop_id]
    prop_dest_count.append((amount_of_dests))
    prop_dest_count_normalized.append(amount_of_dests/amount_appearances_in_search)


# bar plot of amount of different dest id per prop
plt.hist(prop_dest_count, bins =50)
plt.xlabel('amount of associated destinations IDs for one property')
plt.ylabel('amount of properties')
plt.savefig("dest_ids_prop")
plt.show()
# bar plot of amount of different dest id per prop normalized
plt.hist(prop_dest_count_normalized, bins =5)
plt.xlabel('amount of associated destinations IDs for one property normalized by amount of search appearances')
plt.ylabel('amount of properties')
plt.savefig("(dest_ids_prop)_normalized")
plt.show()
#%% WORK IN PROGRESS
def NDCG(ranking_df):
    '''
    This function expects a ranking_df, which is a Dataframe 
    containing the following columns and corresponding 1 search_id:
    prop_id, predicted_val,  click_bool, booking_bool
    '''
    count_booked = 1
    count_clicked = 6
    count_nothing = 5
    total_length = count_booked + count_clicked + count_nothing
    for i in range(total_length)
    ideal_dcg = for i in range(count_booked

# nonrandom_fig = nonrandom_plot.get_figure()
# nonrandom_plot = nonrandom_.plot.bar(width=1, figsize=[14, 8], title='Non-random positioning')
# nonrandom_fig.savefig('nonrandom_position.png', dpi=200, facecolor='white')

