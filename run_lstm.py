#%%
# IMPORTS AND GLOBALS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
# from lstm_model import LSTM
import numpy as np
import matplotlib.pyplot as plt

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        # print("HELLO")
        self.input_dim = input_dim
        # print('test',self.input_dim,'test')
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # a = hidden, b = cellstate
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        # print(input.size())
        # seq_len, batch_size, embedding_size
        # print(input.view(len(input), self.batch_size, 25))
        # print('test2', len(input), self.batch_size, self.input_dim, 'test2')
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, self.input_dim))
        
        # print(lstm_out.size())

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)
        
        return y_pred.view(-1)

#%%

TRAIN_P = 0.6
VALID_P = 0.2
TEST_P = 0.2

database = pd.read_pickle('database_basic.pkl')
database.head()

X_train = []
X_valid = []
X_test = []
y_train = []
y_valid = []
y_test = []
for _, group in database.groupby(['id']):
    length = group.shape[0]
    train_valid_split = int(length * TRAIN_P)
    valid_test_split = int(length * (1 - TEST_P))
    y_train.append(group.iloc[:train_valid_split]['target_mood'].values)
    y_valid.append(group.iloc[train_valid_split:valid_test_split]['target_mood'].values)
    y_test.append(group.iloc[valid_test_split:]['target_mood'].values)
    cols = [c for c in group.columns if ('bool' not in c) and ('target' not in c) and ('date' not in c) and ('id' not in c)]
    group = group[cols]
    X_train.append(group.iloc[:train_valid_split].values)
    X_valid.append(group.iloc[train_valid_split:valid_test_split].values)
    X_test.append(group.iloc[valid_test_split:].values)

#%% TRAIN

lstm_input_size = X_train[0].shape[1]
num_train = 1
hidden = 30
output_dim = 1
num_layers = 1
learning_rate = 0.001
num_epochs = 50

lstm_model = LSTM(lstm_input_size, hidden, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
lstm_model.hidden = lstm_model.init_hidden()
optimiser = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

train_losses = []
valid_losses = []
for e in range(100):
    for i, sequence in enumerate(X_train):
        # print(e, i, '/', len(X_train))
        # print('length sequence:', sequence.shape[0])
        # print('number of features:', sequence.shape[1])
        lstm_model.zero_grad()
        input = torch.from_numpy(sequence).float()
        y_pred = lstm_model(input)
        # print('length prediction sequence:', len(y_pred))
        # print(y_pred)

        our_loss = ((y_pred.data.numpy() - y_train[i])**2).mean()
        loss = loss_fn(y_pred, torch.from_numpy(y_train[i]).float())

        train_losses.append(loss)

        # VALID
        valid_loss = 0
        for j in range(len(X_valid)):
            input = torch.from_numpy(X_valid[j]).float()
            y_pred = lstm_model(input)
            our_loss = ((y_pred.data.numpy() - y_valid[j])**2).mean() 
            valid_loss += our_loss 
        valid_losses.append(valid_loss/len(X_valid))

        # print('our loss', our_loss)
        # print('MSEloss:', loss.item())
        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='valid')
plt.title('Loss')
plt.legend()
plt.show()

#%%
total_loss = 0
for i in range(len(X_test)):
    input = torch.from_numpy(X_test[i]).float()
    # print(input.shape)
    pred = lstm_model(input)
    # print(y_test.shape
    # print(len(y_test[i]))
    total_loss += ((pred.data.numpy() - y_test[i])**2).mean()
print('total test loss', total_loss.item()/len(X_test))

# total_loss = 0
# for i in range(len(X_valid)):
#     input = torch.from_numpy(X_valid[i]).float()
#     pred = lstm_model(input)
#     total_loss += loss_fn(pred, torch.from_numpy(y_valid[i]).float())
# print('total validation loss', total_loss.item()/len(X_valid))

#%%
