#%%
import pandas as pd
import numpy as np

import sys

# sys.path.append("Assignment_2")

def perform_new_ndcg(df, ranking_col, gain_col, ascending=False, cutoff=5):
    ''' Calculates the mean ndcg of the dataframe
        Requires:
        - the column that describes the actual gain,
        - a column on which the database should be sorted
        - whether to sort this last column
          in ascending or descending way.
        Returns: Mean ndcg
    '''
    ranks = np.array([i+1 for i in range(cutoff)])
    ranks = np.where(ranks == 1, ranks, np.log2(ranks))
    results = np.zeros(len(df['srch_id'].unique()))

    actual_ranking = df.sort_values(by=ranking_col, ascending=ascending).groupby('srch_id', sort=True)
    ideal_ranking = df.sort_values(by=gain_col, ascending=False).groupby('srch_id', sort=True)
    for i, ((_, actual_group), (_, ideal_group)) in enumerate(zip(actual_ranking, ideal_ranking)):
        dcg = (actual_group['gain'].head(cutoff) / ranks).sum()
        idcg = (ideal_group['gain'].head(cutoff) / ranks).sum()
        results[i] = dcg/idcg
    return results.mean()

def make_submision(df, ranking_col, ascending=False, filename='Assignment_2/data/submision.csv'):
    '''
        Creates output file
        from df,
        given the column on which it should sort,
        whether it should sort in ascending order or not
    '''

    sorted_df = df.sort_values(by=['srch_id', ranking_col], ascending=[True, ascending])
    sorted_df = sorted_df[['srch_id', 'prop_id']]
    sorted_df.to_csv(filename, index=False)


#%%
# df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')
# df = pd.read_csv('Assignment_2/toy_example_ndcg.csv', sep=';')

# # perform_ndcg(df, 'predicted', 'gain', False)

#%%
# df['gain'] = df['click_bool'] + (5 * df['booking_bool'])

# #%%
# selected_df = df[['gain', 'prop_id', 'srch_id', 'position', 'random_bool']]

# #%%
# rundf = selected_df[selected_df.random_bool == 1]
# print('NDCG of random set:', perform_new_ndcg(rundf, 'position', 'gain', True))
# #%%
# rundf = selected_df[selected_df.random_bool == 0]
# print('NDCG of non-random set:', perform_new_ndcg(rundf, 'position', 'gain', True))
# #%%
# rundf = selected_df
# print('NDCG of whole set:', perform_new_ndcg(rundf, 'position', 'gain', True))

# #%%
# selection_df = df.head(1200)
# make_submision(selection_df, 'position', True)
