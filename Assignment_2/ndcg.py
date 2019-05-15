import pandas as pd
import numpy as numpy

def perform_ndcg(df, ranking_col, ascending=True):
    results = []
    for i, group in df.groupby(by='srch_id'):
        ndcg = 

def ndcg(group, ranking_col, ascending=True):
    group.sort_values(by= ranking_col, ascending= ascending)