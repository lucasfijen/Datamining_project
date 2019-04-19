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

# LOAD DATA
df.head()



def convert_to_X_y(df):
    ''' 
    Splits df into X and y shape as numpy arrays
    '''
    X = df.loc[:, ~df.columns.isin(['target_mood'])].values
    y = df.target_mood.values
    y = y.reshape((len(y), 1))
    return X, y



#%% Benchmark
moods = df.mean_mood[df.interpolate_mood_bool == 0]
targetmoods = df.target_mood[df.interpolate_mood_bool == 0]
mse = ((moods - targetmoods)**2).mean(axis=0)
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
pca = decomposition.PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)
print(X.shape)
print(X)



#%%
print(df.target_mood)

#%%
