#%% init

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn import decomposition
from sklearn import datasets


from sklearn.preprocessing import StandardScaler
df = pd.read_pickle('/Users/janzuiderveld/Documents/GitHub/database_basic.pkl')
df.columns

# LOAD DATA|
df.head()



def convert_to_X_y(df):
    ''' 
    Splits df into X and y shape as numpy arrays
    '''
    X = df.loc[:, ~df.columns.isin(['target_mood'])].values
    y = df.target_mood.values
    y = y.reshape((len(y), 1))
    return X, y

TRAIN_P = 0.6
VALID_P = 0.2
TEST_P = 0.2


X_train = []
X_valid = []
X_test = []
y_train = []
y_valid = []
y_test = []

train_lengths = []
# length train + validation
valid_lengths = []
for _, group in df.groupby(['id']):
    length = group.shape[0]
    train_valid_split = int(length * TRAIN_P)
    valid_test_split = int(length * (1 - TEST_P))

    train_lengths.append(train_valid_split)
    valid_lengths.append(valid_test_split)
    y_train.append(group.iloc[:train_valid_split]['target_mood'].values)
    y_valid.append(group.iloc[train_valid_split:valid_test_split]['target_mood'].values)
    y_test.append(group.iloc[valid_test_split:]['target_mood'].values)

    cols = [c for c in group.columns if ('bool' not in c) and ('target' not in c) and ('date' not in c) and ('id' not in c)]
    group = group[cols]
    
    X_train.append(group.iloc[:train_valid_split]['mean_mood'].values)
    X_valid.append(group.iloc[train_valid_split:valid_test_split]['mean_mood'].values)
    X_test.append(group.iloc[valid_test_split:]['mean_mood'].values)

#%% Benchmark 
moods = df.mean_mood[df.interpolate_mood_bool == 0]

lenMoods =(len(moods))
# moodsTest = moods[-0.2 * lenMoods, -1]


targetmoods = df.target_mood[df.interpolate_mood_bool == 0]
mse = ((moods - targetmoods)**2).mean(axis=0)
mse = ((np.array(X_test) - np.array(y_test))**2).mean(axis=0)
se = (moods - targetmoods)**2
print(mse)

#%% T-Test
from scipy import stats
>>> pts = 1000
>>> np.random.seed(28041990)
>>> a = np.random.normal(0, 1, size=pts)
>>> b = np.random.normal(2, 1, size=pts)
>>> x = moods
>>> k2, p = stats.normaltest(x)
>>> alpha = 1e-3
>>> print("p = {:g}".format(p))

if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")


#%% Wilcoxon 
stats.wilcoxon(moods[1:100], moods[101:200])
#%% PCA

X,y =convert_to_X_y(df)
print(X.shape)
pca = decomposition.PCA(n_components=28)
pca.fit(X)
X = pca.transform(X)
print(X.shape)
print(X)



#%%
print(df.target_mood)

#%%
