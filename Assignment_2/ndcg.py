#%%
import pandas as pd
import numpy as np

def perform_ndcg(df, ranking_col, gain_col, ascending=False, cutoff=5):
    ''' Calculates the mean ndcg of the dataframe
        Requires:
        - the column that describes the actual gain,
        - a column on which the database should be sorted
        - whether to sort this last column
          in ascending or descending way.
        Returns: Mean
    '''
    results = []
    for _, group in df.groupby(by='srch_id'):
        results.append(helper_ndcg(group, ranking_col, gain_col, ascending))

    return np.mean(results)

def helper_ndcg(group, ranking_col, gain_col, ascending, cutoff=5):
    ''' Helper function for the perform_ndcg

    '''
    sorted_df = group.sort_values(by=ranking_col, ascending=ascending).reset_index(drop=True)
    ranks = sorted_df.index.values + 1
    rels = sorted_df[gain_col]
    dcg = rels / np.where(ranks == 1, ranks, np.log2(ranks))
    if len(dcg) > cutoff:
        dcg = dcg[:cutoff]
    dcg = dcg.sum()

    sorted_df = group.sort_values(by=gain_col, ascending=False).reset_index(drop=True)
    ranks = sorted_df.index.values + 1
    rels = sorted_df[gain_col]
    idcg = rels / np.where(ranks == 1, ranks, np.log2(ranks))
    if len(idcg) > cutoff:
        idcg = idcg[:cutoff]
    idcg = idcg.sum()
    return dcg / idcg


def make_scoring(df, ranking_col, ascending=False):
    sorted_df = df.sorted_values(by=['srch_id', ranking_col], ascending=[True, ascending])
    sorted_df = sorted_df['srch_id', 'prop_id']
    sorted_df.to_csv('Assignement_2/data/submision.csv', index=False)


#%%
df = pd.read_csv('Assignment_2/data/training_set_VU_DM.csv')
# df = pd.read_csv('Assignment_2/toy_example_ndcg.csv', sep=';')

# perform_ndcg(df, 'predicted', 'gain', False)

#%%
df['gain'] = df['click_bool'] + (5 * df['booking_bool'])

#%%
selected_df = df[['gain', 'prop_id', 'srch_id', 'position', 'random_bool']]
#%%
rundf = selected_df[selected_df.random_bool == 0]
perform_ndcg(rundf, 'position', 'gain', True)

#%%
