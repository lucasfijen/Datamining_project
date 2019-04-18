#%%
# IMPORTS AND GLOBALS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from lstm_model import LSTM
import numpy as np

TRAIN_PERC = 0.7
VALID_PERC = 0.2
TEST_PERC = 0.1

#%%
# LOAD DATA
all_data = pd.read_pickle('../database.pkl')
all_data.head()

#%%
def convert_to_X_y(df):
    ''' 
    Splits df into X and y shape as numpy arrays
    '''
    X = df.loc[:, ~df.columns.isin(['targetmood'])].values
    y = df.targetmood.values
    y = y.reshape((len(y), 1))
    return X, y

#%%
person_ids = all_data.index.levels[0]

train, valid, test = [], [], []

all_data = all_data.fillna(0)

# 80% of MOOD days into train set (so not just 80% of days)
for person, days_group in all_data.groupby('id'):
    # days_group = days_group.reset_index()
    mood_df = days_group.dropna(subset=['targetmood']).reset_index()
    first_mood_day = mood_df['date'].iloc[0]
    # print('first_mood_day', first_mood_day)
    last_mood_day = mood_df['date'].iloc[-1]
    # print('last_mood_day', last_mood_day)
    mood_interval = last_mood_day - first_mood_day
    # print('mood_interval', mood_interval)
    split_day_train_valid = first_mood_day + (mood_interval * TRAIN_PERC)
    # print('split_day_train_valid',split_day_train_valid)
    split_day_valid_test = first_mood_day + (mood_interval * (TRAIN_PERC + VALID_PERC))
    # print('split_day_valid_test', split_day_valid_test)
    idx = pd.IndexSlice
    X, y = convert_to_X_y(days_group.loc[idx[:, :split_day_train_valid], :])
    # y = days_group['targetmood'].loc[idx[:, :split_day_train_valid], :]
    # t = days_group.loc[idx[:, :split_day_train_valid], :-1].values.reshape(-1, 1, 25)
    # print(t.shape)
    X = X.reshape(-1, 1, 25)
    train.append(days_group.loc[idx[:, :split_day_train_valid], :])
    valid.append(days_group.loc[idx[:, split_day_train_valid:split_day_valid_test], :])
    test.append(days_group.loc[idx[:, split_day_valid_test:], :])

train = pd.concat(train, axis=0, join='outer', copy=True)
valid = pd.concat(valid, axis=0, join='outer', copy=True)
test = pd.concat(test, axis=0, join='outer', copy=True)

X_train, y_train = convert_to_X_y(train)
X_valid, y_valid = convert_to_X_y(valid)
X_test, y_test = convert_to_X_y(test)

print(X_train.shape)

#%%

X_train_t = torch.from_numpy(X_train) #[:, :, np.newaxis]
X_test_t = torch.from_numpy(X_test[:, :, np.newaxis])
X_valid_t = torch.from_numpy(X_valid[:, :, np.newaxis])

y_train_t = torch.from_numpy(y_train)
y_test_t = torch.from_numpy(y_test)
y_valid_t = torch.from_numpy(y_valid)

lstm_input_size = train.shape[1]
num_train = 1
hidden = 30
output_dim = 1
num_layers = 1
learning_rate = 0.001
num_epochs = 50

model = LSTM(25, hidden, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(reduction='sum')

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%

# input_seq = torch.randn(20, 1, 25)

input_seq = torch.from_numpy(X).float()
# input_seq = input_seq.flo

model.hidden = model.init_hidden()
    
# Forward pass
y_pred = model(input_seq)
print(y_pred)
# loss = loss_fn(y_pred, 1)

#%%
#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Clear stored gradient
    model.zero_grad()
    
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()
    
    # Forward pass
    y_pred = model(X_train_t)

    loss = loss_fn(y_pred, y_train_t)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

#%%