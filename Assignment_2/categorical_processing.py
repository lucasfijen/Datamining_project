#%% imports
import pandas as pd 
import numpy as np

def add_descriptors(train_data, test_data, variable):
    df_test_copy = test_data.copy()
    df_train_copy = train_data.copy()
    prep_categorical_numerical_descriptors(df_train_copy)
    prep_categorical_numerical_descriptors(df_test_copy, 0)
    all_numeric = pd.concat([df_train_copy,df_test_copy],axis=0)

    all_groupby = all_numeric.groupby(variable,sort=True).agg([np.median, np.mean, np.std])
    all_groupby_reset_index = all_groupby.reset_index()
    all_groupby_reset_index.columns = ['_'.join(col).strip() for col in all_groupby_reset_index.columns.values]
    all_groupby_reset_index.fillna(-10,inplace=True)
    print(all_groupby_reset_index.columns)
    newdf_train = train_data.merge(all_groupby_reset_index, 
                    how='left', 
                    left_on=[variable],
                    right_on=[variable+'_'],
                    suffixes=(None, '_'+variable))
    newdf_test = test_data.merge(all_groupby_reset_index, 
                    how='left', 
                    left_on=[variable],
                    right_on=[variable+'_'],
                    suffixes=[None, '_'+variable])
    
    return(newdf_train, newdf_test)


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



# #%%
# df = pd.read_csv('data/training_set_VU_DM.csv',nrows=1000)
# df_test = pd.read_csv('data/test_set_VU_DM.csv', nrows=1000)

# a, b = add_descriptors(df, df_test, 'prop_id')
# print(a.columns)

# #%%
# print(a.columns)

# #%%
