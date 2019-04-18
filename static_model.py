#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

database = pd.read_pickle('database_basic.pkl')
database.head()

#%%
# Database that looks like:
# [Original row + History row of N]
def add_history_of_N_days(database, N, method='mean'):
    history_data = []
    for row_id in range(database.shape[0]):
        if row_id > 5: 
            history = database.iloc[row_id-5:row_id, :]
        elif row_id == 0:
            mean_history = np.zeros((1, database.shape[1]))
        else:
            history = database.iloc[0:row_id, :]
        if method == 'mean' and (row_id != 0):
            mean_history = history.values.mean(axis=0)
        history_data.append(mean_history)
    return np.vstack(history_data)

history = add_history_of_N_days(database.iloc[:, :], 5, method='mean')
history = pd.DataFrame(history, dtype=float)
# remove boolean from history

df=df[cols]
history.columns = 'history_' + database.columns
new = pd.concat([database.iloc[:, :].reset_index(), history], axis=1)

