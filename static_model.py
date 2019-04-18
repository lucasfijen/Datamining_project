#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
import numpy as np

database = pd.read_pickle('database_basic.pkl')
database.head()

# Database that looks like:
# [Original row + History row of N]
def add_history_of_N_days(database, N, method='mean'):
    history_data = []
    for _, group in database.groupby(['id']):
        for row_id in range(group.shape[0]):
            if row_id > N: 
                history = database.iloc[row_id-N:row_id, :]
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
history.columns = 'history_' + database.columns
database_with_history = pd.concat([database.iloc[:, :].reset_index(), history], axis=1)

# print(database_with_history['id'])

TRAIN_P = 0.6
VALID_P = 0.2
TEST_P = 0.2

X_train = []
X_valid = []
X_test = []
y_train = []
y_valid = []
y_test = []
for person in database_with_history['id'].unique():
    person_db = database_with_history[database_with_history['id'] == person]
    length = person_db.shape[0]
    train_valid_split = int(length * TRAIN_P)
    valid_test_split = int(length * (1 - TEST_P))
    y_train.append(person_db.iloc[:train_valid_split]['target_mood'].values.reshape(-1, 1))
    y_valid.append(person_db.iloc[train_valid_split:valid_test_split]['target_mood'].values.reshape(-1, 1))
    y_test.append(person_db.iloc[valid_test_split:]['target_mood'].values.reshape(-1, 1))
    cols = [c for c in person_db.columns if ('bool' not in c) and ('target' not in c) and ('date' not in c) and ('id' not in c)]
    person_db = person_db[cols]
    X_train.append(person_db.iloc[:train_valid_split].values)
    X_valid.append(person_db.iloc[train_valid_split:valid_test_split].values)
    X_test.append(person_db.iloc[valid_test_split:].values)

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
X_valid = np.vstack(X_valid)
y_train = np.vstack(y_train)
y_test = np.vstack(y_test)
y_valid = np.vstack(y_valid)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)

#%% FIT SVM REGRESSOR
clf = SVR()
clf.fit(X_train, y_train)

#%%
y_valid_predict = clf.predict(X_valid)
y_test_predict = clf.predict(X_test)

mse_valid = (abs(y_valid_predict.reshape(-1, 1) - y_valid)).mean(axis=0)
print(mse_valid)
mse_test = (abs(y_test_predict.reshape(-1, 1) - y_test)).mean(axis=0)
print(mse_test)

#%%
