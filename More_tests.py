#%% init

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# LOAD DATA
# df = pd.read_pickle('/Users/janzuiderveld/Documents/GitHub/database_basic.pkl')
# df = pd.read_csv('database_new_standardisation.csv', index_col=0)
# df = df.reset_index()

database_standard = pd.read_csv('database_basic_stand.csv', index_col=0)
database_standard = database_standard.reset_index()
# database_standard = database_standard.set_index(['id', 'date'])
database_norm = pd.read_csv('database_basic_norm.csv', index_col=0)
database_norm = database_norm.reset_index()
# database_norm = database_norm.set_index(['id', 'date'])

df = database_standard

#%%

TRAIN_P = 0.6
VALID_P = 0.2
TEST_P = 0.2

X_train = []
X_valid = []
X_test = []
y_train = []
y_valid = []
y_test = []
y_train_bool = []
y_valid_bool = []
y_test_bool = []
for person in df['id'].unique():
    person_db = df[df['id'] == person]
    length = person_db.shape[0]
    train_valid_split = int(length * TRAIN_P)
    valid_test_split = int(length * (1 - TEST_P))
    
    y_train_bool.append(1 - person_db.iloc[:train_valid_split]['shifted_target_mood_bool'].values.reshape(-1, 1))
    y_valid_bool.append(1 - person_db.iloc[train_valid_split:valid_test_split]['shifted_target_mood_bool'].values.reshape(-1, 1))
    y_test_bool.append(1 - person_db.iloc[valid_test_split:]['shifted_target_mood_bool'].values.reshape(-1, 1))

    y_train.append(person_db.iloc[:train_valid_split]['target_mood'].values.reshape(-1, 1))
    y_valid.append(person_db.iloc[train_valid_split:valid_test_split]['target_mood'].values.reshape(-1, 1))
    y_test.append(person_db.iloc[valid_test_split:]['target_mood'].values.reshape(-1, 1))
    cols = [c for c in person_db.columns if ('bool' not in c) and ('target' not in c) and ('date' not in c) and ('id' not in c)]
    person_db = person_db[cols]
    X_train.append(person_db.iloc[:train_valid_split]['mean_mood'].values.reshape(-1, 1))
    X_valid.append(person_db.iloc[train_valid_split:valid_test_split]['mean_mood'].values.reshape(-1, 1))
    X_test.append(person_db.iloc[valid_test_split:]['mean_mood'].values.reshape(-1, 1))

# BENCHMARK
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
X_valid = np.vstack(X_valid)
y_train = np.squeeze(np.vstack(y_train))
y_test = np.squeeze(np.vstack(y_test))
y_valid = np.squeeze(np.vstack(y_valid))
y_train_bool = np.vstack(y_train_bool)
y_test_bool = np.vstack(y_test_bool)
y_valid_bool = np.vstack(y_valid_bool)
y_bool_all = np.vstack([y_train_bool, y_valid_bool, y_test_bool])

train_n = np.count_nonzero(y_train_bool)
valid_n = np.count_nonzero(y_valid_bool)
test_n = np.count_nonzero(y_test_bool)
all_n = np.count_nonzero(y_bool_all)

X_train = np.squeeze(np.vstack(X_train)) * np.squeeze(y_train_bool)
X_test = np.squeeze(np.vstack(X_test)) * np.squeeze(y_test_bool)
X_valid = np.squeeze(np.vstack(X_valid)) * np.squeeze(y_valid_bool)
y_train = np.squeeze(np.vstack(y_train)) * np.squeeze(y_train_bool)
y_test = np.squeeze(np.vstack(y_test)) * np.squeeze(y_test_bool)
y_valid = np.squeeze(np.vstack(y_valid)) * np.squeeze(y_valid_bool)

print(X_train.shape, X_valid.shape, X_test.shape)

X_all = np.squeeze(np.concatenate([X_train, X_valid, X_test])) * np.squeeze(y_bool_all)
y_all = np.squeeze(np.concatenate([y_train, y_valid, y_test])) * np.squeeze(y_bool_all)

mse_train = ((X_train - y_train)**4).sum() / train_n
mse_valid = ((X_valid - y_valid)**4).sum() / valid_n
mse_test = ((X_test - y_test)**4).sum() / test_n
mse_all = ((X_all - y_all)**4).sum() / all_n

print('standardised')
print(list(X_test))

print('mse_train', mse_train)
print('mse_valid', mse_valid)
print('mse_test', mse_test)
print('mse_all', mse_all)

# #%% T-Test
# from scipy import stats
# >>> pts = 1000
# >>> np.random.seed(28041990)
# >>> a = np.random.normal(0, 1, size=pts)
# >>> b = np.random.normal(2, 1, size=pts)
# >>> x = moods
# >>> k2, p = stats.normaltest(x)
# >>> alpha = 1e-3
# >>> print("p = {:g}".format(p))

# if p < alpha:  # null hypothesis: x comes from a normal distribution
#     print("The null hypothesis can be rejected")
# else:
#     print("The null hypothesis cannot be rejected")


# #%% Wilcoxon 
# stats.wilcoxon(moods[1:100], moods[101:200])
# #%% PCA

# X,y =convert_to_X_y(df)
# print(X.shape)
# pca = decomposition.PCA(n_components=28)
# pca.fit(X)
# X = pca.transform(X)
# print(X.shape)
# print(X)



# #%%
# print(df.target_mood)

# #%%


#%%
