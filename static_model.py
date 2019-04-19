#%%
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
import numpy as np
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression

TRAIN_P = 0.6
VALID_P = 0.2
TEST_P = 0.2

# OLD
# database = pd.read_pickle('database_basic_standardised.pkl')
# NEW
database_standard = pd.read_pickle('../database_basic_stand')
database_standard = database_standard.reset_index()
database_standard = database_standard.set_index(['id', 'date'])
database_norm = pd.read_pickle('../database_basic_norm.pkl')
database_norm = database_norm.reset_index()
database_norm = database_norm.set_index(['id', 'date'])
# database.head()

#%%

# Database that looks like:
# [Original row + History row of N]
def add_history_of_N_days(database, N, method='mean'):
    history_data = []
    database = database[database.columns[:-7]]
    for _, group in database.groupby(['id']):
        for row_id in range(group.shape[0]):
            if row_id > N: 
                history = group.iloc[row_id-N:row_id, :]
            elif row_id == 0:
                # mean_history = database.iloc[0, :]
                mean_history = np.zeros((1, group.shape[1]))
            else:
                history = group.iloc[0:row_id, :]
            if method == 'mean' and (row_id != 0):
                mean_history = history.values.mean(axis=0)
            history_data.append(mean_history)
    return np.vstack(history_data)

def do_tests(database):
    svm_results_valid = []
    svm_results_test = []
    svm_error_valid = []
    svm_error_test = []
    linr_results_valid = []
    linr_results_test = []
    linr_error_valid = []
    linr_error_test = []
    for i in range(0, 30):
        if i != 0:
            history = add_history_of_N_days(database.iloc[:, :], i, method='mean')
            history = pd.DataFrame(history, dtype=float)
            history.columns = 'history_' + database.columns[:-7]
            database_with_history = pd.concat([database.iloc[:, :].reset_index(), history], axis=1)
        else:
            database_with_history = database.reset_index()
        # database_with_history = database.reset_index()
        
        # if i == 4:
        #     print(database_with_history[['id', 'date', 'history_target_mood', 'target_mood']])
        #     return
        X_train = []
        X_valid = []
        X_test = []
        y_train = []
        y_valid = []
        y_test = []
        y_train_bool = []
        y_valid_bool = []
        y_test_bool = []
        for person in database_with_history['id'].unique():
            person_db = database_with_history[database_with_history['id'] == person]
            length = person_db.shape[0]
            train_valid_split = int(length * TRAIN_P)
            valid_test_split = int(length * (1 - TEST_P))
            
            y_train.append(person_db.iloc[:train_valid_split]['target_mood'].values.reshape(-1, 1))
            y_valid.append(person_db.iloc[train_valid_split:valid_test_split]['target_mood'].values.reshape(-1, 1))
            y_test.append(person_db.iloc[valid_test_split:]['target_mood'].values.reshape(-1, 1))
            
            y_train_bool.append(person_db.iloc[valid_test_split:]['shifted_target_mood_bool'].values.reshape(-1, 1))
            y_valid_bool.append(person_db.iloc[train_valid_split:valid_test_split]['shifted_target_mood_bool'].values.reshape(-1, 1))
            y_test_bool.append(person_db.iloc[valid_test_split:]['shifted_target_mood_bool'].values.reshape(-1, 1))

            cols = [c for c in person_db.columns if ('bool' not in c) and ('target' not in c) and ('date' not in c) and ('id' not in c)]
            person_db = person_db[cols]
            X_train.append(person_db.iloc[:train_valid_split].values)
            X_valid.append(person_db.iloc[train_valid_split:valid_test_split].values)
            X_test.append(person_db.iloc[valid_test_split:].values)

        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)
        X_valid = np.vstack(X_valid)
        y_train = np.squeeze(np.vstack(y_train))
        y_test = np.squeeze(np.vstack(y_test))
        y_valid = np.squeeze(np.vstack(y_valid))
        y_train_bool = np.vstack(y_train_bool)
        y_test_bool = np.vstack(y_test_bool)
        y_valid_bool = np.vstack(y_valid_bool)

        # print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)
        y_test_bool = np.squeeze(1 - y_test_bool)
        y_valid_bool = np.squeeze(1 - y_valid_bool)

        #%% FIT SVM REGRESSOR
        clf = SVR()
        # temp = np.zeros(X_train.shape)
        clf.fit(X_train, y_train)

        y_valid_predict = clf.predict(X_valid)
        y_test_predict = clf.predict(X_test)

        svm_y_valid_predict_corrected = y_valid_predict * y_valid_bool
        svm_y_test_predict_corrected = y_test_predict * y_test_bool
        y_valid_corrected = y_valid * y_valid_bool
        y_test_corrected = y_test * y_test_bool 

        svm_results_valid.append(svm_y_valid_predict_corrected)
        svm_results_test.append(svm_y_test_predict_corrected)

        valid_nr_not_interpolated = np.count_nonzero(y_valid_bool)
        test_nr_not_interpolated = np.count_nonzero(y_test_bool)

        svm_mse_valid = (((svm_y_valid_predict_corrected - y_valid_corrected)**4).sum()) / valid_nr_not_interpolated
        print(i, 'SVM valid', svm_mse_valid)
        svm_error_valid.append(svm_mse_valid)
        svm_mse_test = (((svm_y_test_predict_corrected - y_test_corrected)**4).sum()) / test_nr_not_interpolated
        print(i, 'SVM test', svm_mse_test)
        svm_error_test.append(svm_mse_test)

        #FIT LINEAR REGRESSION

        # temp = np.zeros(X_train.shape)
        reg = LinearRegression().fit(X_train, y_train)
        print(reg.intercept_)

        y_valid_predict = reg.predict(X_valid)
        y_test_predict = reg.predict(X_test)

        lin_y_valid_predict_corrected = y_valid_predict * y_valid_bool
        lin_y_test_predict_corrected = y_test_predict * y_test_bool
        y_valid_corrected = y_valid * y_valid_bool
        y_test_corrected = y_test * y_test_bool 

        linr_results_valid.append(lin_y_valid_predict_corrected)
        linr_results_test.append(lin_y_test_predict_corrected)

        valid_nr_not_interpolated = np.count_nonzero(y_valid_bool)
        test_nr_not_interpolated = np.count_nonzero(y_test_bool)

        linr_mse_valid = (((lin_y_valid_predict_corrected - y_valid_corrected)**4).sum()) / valid_nr_not_interpolated
        print(i, 'LinReg valid', linr_mse_valid)
        linr_mse_test = (((lin_y_test_predict_corrected - y_test_corrected)**4).sum()) / test_nr_not_interpolated
        print(i, 'LinReg test', linr_mse_test)

        linr_error_test.append(linr_mse_test)
        linr_error_valid.append(linr_mse_valid)


    return svm_results_valid, svm_results_test, svm_error_valid, svm_error_test, \
            linr_results_valid, linr_results_test, linr_error_valid, linr_error_test


svm_results_valid_norm, svm_results_test_norm, svm_error_valid_norm, svm_error_test_norm, linr_results_valid_norm, linr_results_test_norm, linr_error_valid_norm, linr_error_test_norm = do_tests(database_norm)
svm_results_valid_std, svm_results_test_std, svm_error_valid_std, svm_error_test_std, linr_results_valid_std, linr_results_test_std, linr_error_valid_std, linr_error_test_std = do_tests(database_standard)

#%%
print(svm_error_valid_norm.index(min(svm_error_valid_norm)) + 1)
print(svm_error_test_norm.index(min(svm_error_test_norm)) + 1)
print(linr_error_valid_norm.index(min(linr_error_valid_norm)) + 1)
print(linr_error_test_norm.index(min(linr_error_test_norm)) + 1)
#%%
from scipy import stats

for s1 in [svm_results_test_norm[0], svm_results_test_std[0], linr_results_test_norm[0], linr_results_test_std[0]]:
    for s2 in [svm_results_test_norm[0], svm_results_test_std[0], linr_results_test_norm[0], linr_results_test_std[0]]:
        print(stats.wilcoxon(s1, s2))


plt.plot(svm_error_test_norm,'--', label='svm norm')
# plt.plot(svm_error_test_std, '--', label='svm std')
plt.plot(svm_error_valid_norm, label='svm norm')
# plt.plot(svm_error_valid_std, label='svm std')
plt.plot(linr_error_test_norm, '--', label='linr norm')
# plt.plot(linr_error_test_std, '--', label='linr std')
plt.plot(linr_error_valid_norm, label='linr norm')
# plt.plot(linr_error_valid_std, label='linr std')
plt.title('Normalised data')
plt.legend()
plt.show()


print(svm_results_test_norm[0])
#%%

_, p1 = stats.wilcoxon(svm_y_valid_predict_corrected, lin_y_valid_predict_corrected)
_, p2 = stats.wilcoxon(svm_y_test_predict_corrected, lin_y_test_predict_corrected)

alpha = 0.05

print(p1, p2)

# if p1 < alpha:  # null hypothesis: x comes from a normal distribution
#     print("The null hypothesis can be rejected")
# else:
#     print("The null hypothesis cannot be rejected")

#%%
# cols = [c for c in database_with_history.columns if ('bool' not in c) and ('target' not in c) and ('date' not in c) and ('id' not in c)]
# df_cor = database_with_history[cols].corr()
# # print(df_cor)
# # df_cor = df.corr()
# # pd.DataFrame(np.linalg.inv(df_cor.values), index = df_cor.index, columns=df_cor.columns)
# print(pd.Series(np.linalg.inv(df_cor.values).diagonal(), index=df_cor.index).round(2))

# database_with_history = history.reset_index()
# database_with_history.head()
# cols = [c for c in database_with_history.columns if (('history' in c) or ('id' in c) or ('date' in c) or ('target_mood' in c))]
# database_with_history = database_with_history[cols]
# database_with_history.head()

# database_with_history = database_with_history[cols]
# database_with_history = database_with_history.reset_index()
# database_with_history.head()
