#%% imports
import pandas as pd 
import numpy as np

def add_descriptors(train_data, test_data, variable):
    df_test_copy = prep_categorical_numerical_descriptors(train_data)
    df_train_copy = prep_categorical_numerical_descriptors(test_data, 0)
    all_numeric = pd.concat([df_train_copy, df_test_copy],axis=0)
    del df_test_copy
    del df_train_copy
    # print(all_numeric.columns)
    print('Doing groupby')
    all_groupby = all_numeric.groupby(variable, sort=True).agg(['mean', 'std']) #'median', 
    all_groupby_reset_index = all_groupby
    del all_numeric
    # print(all_groupby_reset_index.head())
    all_groupby_reset_index.columns = ['_'.join(col).strip() for col in all_groupby_reset_index.columns.values]
    all_groupby_reset_index.fillna(-10, inplace=True)
    # print(all_groupby_reset_index.head())
    print('merging')

    newdf_train = train_data.merge(all_groupby_reset_index, 
                    how='left', 
                    on=[variable])
    newdf_test = test_data.merge(all_groupby_reset_index, 
                    how='left', 
                    on=[variable])
                    
    del all_groupby_reset_index
    # newdf_train.drop([variable], axis=1, inplace=True)
    # newdf_test.drop([variable], axis=1, inplace=True)

    dont_delete = []
    for i in [2,3,5,8]:
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        dont_delete.extend([rate,inv, diff])

    to_delete = []
    for c in newdf_train:
        if ('prop_id_' in c)  or ('comp' in c):
            if not c in dont_delete:
                to_delete.append(c)

    newdf_test.drop(to_delete, axis=1, inplace=True)
    newdf_train.drop(to_delete, axis=1, inplace=True)

    to_delete = [  'prop_country_id',
                'prop_starrating',
                'prop_review_score',
                'prop_brand_bool',
                'prop_location_score1',
                'prop_location_score2',
                'prop_log_historical_price']

    newdf_test.drop(to_delete, axis=1, inplace=True)
    newdf_train.drop(to_delete, axis=1, inplace=True)

    return newdf_train, newdf_test


def prep_categorical_numerical_descriptors(data, train = 1):

    # remove describing cols if there
    # data.drop(list(data.filter(regex='std')), axis=1, inplace =True)
    # data.drop(list(data.filter(regex='mean')), axis=1, inplace =True)
    # data.drop(list(data.filter(regex='median')), axis=1, inplace =True)
    # print(data.columns)

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
        diff = 'comp' + str(i) + '_rate_percent_diff'
        to_delete.extend([rate,inv, diff])

    if train == 1:
        to_delete.extend(['position','gross_bookings_usd', 'click_bool', 'booking_bool'])

    # if 'comp8_rate_percent_diff_mean_std' in data.columns:
    #     for i in range(1,9):
    #         a = 'comp' + str(i) + '_rate_percent_diff_median'
    #         b = 'comp' + str(i) + '_rate_percent_diff_std'
    #         c = 'comp' + str(i) + '_rate_percent_diff_mean'
    #         to_delete.extend([diff])

    return data.drop(to_delete, axis=1).copy()



#%%
# df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv',nrows=1000)
# df_test = pd.read_csv('Assignment_2/data/test_set_VU_DM.csv', nrows=1000)

# a, b = add_descriptors(df, df_test, 'prop_id')
# print(a.columns)

# #%%
# print(a.columns)

# #%%


#%%
# print(a)

#%%
