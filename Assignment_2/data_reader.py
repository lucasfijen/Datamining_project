#%% Imports
import pandas as pd 
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.set_palette("Paired")
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
# from sklearn.impute import SimpleImputer
import missingno as msno

#%% Reading in db
try:
    df = pd.read_csv('data/training_set_VU_DM.csv')
except:
    df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')

try:
    df_test = pd.read_csv('data/test_set_VU_DM.csv')
except:
    df_test = pd.read_csv('Assignment_2/data/test_set_VU_DM.csv')

#%%
all_train = len(df)
random_train = len(df[df['random_bool'] == 1])
print(random_train/all_train)
#%%
all_test = len(df_test)
random_test = len(df_test[df_test['random_bool'] == 1])
print(random_test/all_test)
#%%
missingdata_df = df.columns[df.isnull().any()].tolist()
plot = msno.matrix(df[missingdata_df])
#%%
# plt.subplots_adjust(top=0.88)
fig = plot.get_figure()
fig.savefig("missing.png", bbox_inches="tight")

# plt.savefig('missing_data.png', dpi=200)
#%% Print some stats
print('split devision')
print(len(trainingset) / (len(trainingset) + len(valset)))
print('')
print('Devision random - not random')
print('training')
print(valset.random_bool.value_counts())
print('val')
print(trainingset.random_bool.value_counts())
print('')
print('Devision clicks')
print('training')
print(valset.click_bool.value_counts())
print('val')
print(trainingset.click_bool.value_counts())
print('')
print('Devision of months')
print('training')
print(trainingset.target_month.value_counts())
print('val')
print(valset.target_month.value_counts())

#%% Basis featurs of the dataset
print('Number of datapoints:', df.shape[0], 'Number of initial features:', df.shape[1], '\n')
# print('The features:', df.columns.values)
print('The datatypes:', df.dtypes)
# df.head()

#%% Calculate the month for which people are looking
df['date_time'] = pd.to_datetime(df['date_time'])
df['target_month'] = (df['date_time'] + df['srch_booking_window'].astype('timedelta64[D]')).dt.month


#%% Make a new column which is the position divided by the max position
# of that srch_id. Also, first close the missing positions.

df = df.sort_values(['srch_id', 'position'])
df['corrected_position'] = df.groupby(['srch_id']).cumcount()+1
df['corrected_position'] = df.corrected_position / df.groupby('srch_id').corrected_position.transform(np.max) 

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

correction_df

#%%
# smalldf = df.head(10000)
def correct_click(row):
    position = row['position']
    correction = correction_df[correction_df['position'] == position]['random/nonrandom clicks'].values[0]  
    return row['click_bool'] * correction

def correct_booking(row):
    position = row['position']
    correction = correction_df[correction_df['position'] == position]['random/nonrandom bookings'].values[0]  
    return row['booking_bool'] * 5 * correction

df['corrected_click_gain'] = df.apply(correct_click, axis=1)
df['corrected_book_gain'] = df.apply(correct_booking, axis=1)
df['corrected_total'] = df['corrected_click_gain'] + df['corrected_book_gain']

df['non_corrected_total'] = df['click_bool'] + (5 * df['booking_bool'])

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

#%% Add column with corrected position
df = df.sort_values(['srch_id', 'position'])
df['corrected_position'] = df.groupby(['srch_id']).cumcount()+1
df['corrected_position'] = df.corrected_position / df.groupby('srch_id').corrected_position.transform(np.max) 

#%% Create dict with props as keys, all search dests it appeared in as values
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


#%% plot prop search dest amounts + normalized for number of searches it appeared in
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

#%% Measuring overlap of srch destination IDs between train and test set

dest_ids_test = df_test['srch_destination_id'].unique().tolist()
dest_ids_train = df['srch_destination_id'].unique().tolist()

overlap = np.sum(np.isin(dest_ids_test, dest_ids_train))/len(dest_ids_test)

print('srch_dest overlap', overlap) # 69 percent of test dest IDs in train dest IDs

#%% Measuring overlap of prop IDs between train and test set

prop_ids_test = df_test['prop_id'].unique().tolist()
prop_ids_train = df['prop_id'].unique().tolist()

overlap = np.sum(np.isin(prop_ids_test, prop_ids_train))/len(prop_ids_test)
print('prop_id overlap', overlap) # 94 percent of test prop IDs in train prop IDs

#%% Create dicts with prop+dest as keys, average corrected position AND average gain as values

prop_dest_avg_corrected_pos = dict()

for prop in df['prop_id'].unique():
    for dest in df['srch_destination_id'].unique():
        prop_dest_set = df.loc[(df['prop_id'] == prop) & df['srch_destination_id'].isin([dest])]
        prop_dest_avg_corrected_pos[prop+dest]= np.average(prop_dest_set['corrected_position'])

prop_dest_avg_gain = dict()

for prop in df['prop_id'].unique():
    for dest in df['srch_destination_id'].unique():
        prop_dest_set = df.loc[(df['prop_id'] == prop) & df['srch_destination_id'].isin([dest])]
        prop_dest_avg_gain[prop+dest]= np.average(prop_dest_set['non_corrected_total'])

#%%
def prep_categorical_numerical_descriptors(data, train = 1):
    to_delete = [
        'date_time',
        'site_id',
        'visitor_location_country_id',
        'prop_country_id',
        'prop_brand_bool',
        'promotion_flag',
        'srch_destination_id',
        'random_bool',
        'srch_id'
    ]

    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        #diff = 'comp' + str(i) + '_rate_percent_diff'
        to_delete.extend([rate,inv])

    if train == 1:
        to_delete.extend(['position','gross_bookings_usd', 'click_bool', 'booking_bool'])

    data.drop(to_delete, axis=1, inplace=True)

#%% 
def fill_nan(data):
    data['srch_query_affinity_score'].fillna(10) # values are logs of probabilaty, all negtive
    data.fillna(-10)

#%% create prop_id numerical values frame
df_test_copy = df_test.copy()
df_train_copy = df.copy()
remove_un_numerical(df_train_copy)
remove_un_numerical(df_test_copy, 0)
all_numeric = pd.concat([df_train_copy,df_test_copy],axis=0)

all_groupby = all_numeric.groupby('prop_id',sort=True).agg([np.nanmedian, np.nanmean, np.nanstd])
all_groupby_reset_index = all_groupby.reset_index()
all_groupby_reset_index.columns = ['_'.join(col).strip() for col in all_groupby_reset_index.columns.values]
all_groupby_reset_index.fillna(0,inplace=True)

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

#%%
