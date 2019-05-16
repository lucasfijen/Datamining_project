#%%
import pandas as pd
import numpy as np

def perform_ndcg(df, ranking_col, gain_col, ascending=False):
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

def helper_ndcg(group, ranking_col, gain_col, ascending):
    ''' Helper function for the perform_ndcg

    '''
    sorted_df = group.sort_values(by=ranking_col, ascending=ascending).reset_index(drop=True)
    ranks = sorted_df.index.values + 1
    rels = sorted_df[gain_col]
    dcg = rels / np.where(ranks == 1, ranks, np.log2(ranks))
    print(np.where(ranks == 1, ranks, np.log2(ranks)))
    dcg = dcg.sum()

    sorted_df = group.sort_values(by=gain_col, ascending=False).reset_index(drop=True)
    ranks = sorted_df.index.values + 1
    rels = sorted_df[gain_col]
    idcg = rels / np.where(ranks == 1, ranks, np.log2(ranks))
    idcg = idcg.sum()
    return dcg / idcg


def make_scoring(df, ranking_col, ascending=True):
    print('lol')


# df = pd.read_csv('Assignment_2/toy_example_ndcg.csv', sep=';')

# perform_ndcg(df, 'predicted', 'gain', False)