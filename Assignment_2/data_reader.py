#%% Imports
import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")

#%% Reading in db
df = pd.read_csv('data/training_set_VU_DM.csv')

#%% Basis featurs of the dataset
print('Number of properties:', df.shape[0], 'Number of initial features:', df.shape[1])
print('The features:', df.columns.values)

#%% Take a small subset because the dataset is BIG
smalldf = df.head(10000)

#%% The number of times a property occurs in the dataset
df.prop_id.value_counts()

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

#%% WORK IN PROCES
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
