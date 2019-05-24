#%%
import pandas as pd
import os
try:    
    os.chdir('Assignment_2')
except:
    print("youre already in assignment 2")

dropcols = ['srch_children_count_std', 'srch_saturday_night_bool', 'comp3_inv', 'srch_saturday_night_bool_mean', 'target_month_9', 'srch_children_count', 'comp2_rate_percent_diff', 'srch_query_affinity_score', 'comp5_rate', 'comp2_inv', 'target_month_8', 'target_month_10', 'comp2_rate', 'target_month_7', 'visitor_hist_adr_usd_mean', 'comp8_inv', 'comp8_rate', 'srch_adults_count', 'target_month_11', 'prop_starrating_std', 'prop_location_score1_std', 'prop_log_historical_price_mean', 'srch_room_count_std', 'comp5_inv', 'orig_destination_distance_std', 'srch_adults_count_mean', 'srch_saturday_night_bool_std', 'prop_log_historical_price_std', 'prop_location_score1_mean', 'orig_destination_distance', 'visitor_hist_starrating_std', 'target_month_6', 'srch_adults_count_std', 'target_month_5', 'srch_query_affinity_score_std', 'target_month_2', 'target_month_1', 'visitor_hist_adr_usd_std', 'price_usd_std', 'target_month_12', 'target_month_3', 'Unnamed: 0', 'srch_destination_id', 'price_usd_mean', 'comp5_rate_percent_diff', 'comp8_rate_percent_diff', 'visitor_hist_starrating_mean', 'comp3_rate_percent_diff', 'target_month_4', 'orig_destination_distance_mean', 'price_usd', 'prop_review_score_std']

folder = 'data'
for filename in ['prepped_df_train', 'prepped_df_test', 'prepped_df_val']:
    print('loading file: ', filename)
    df = pd.read_csv(folder + '/' + filename + '.csv')
    print(df.shape)
    print('dropping cols')
    newdf = df.drop(dropcols, axis=1)
    del df
    print('creating pickle')
    newdf.to_pickle(folder + '/' + filename + '.pkl')
    del newdf