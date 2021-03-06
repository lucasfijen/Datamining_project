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
import seaborn as sns; sns.set()

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

# database = pd.read_pickle('database_basic_old_normalisation.pkl')
# database = pd.read_csv('database_new_standardisation.csv', index_col=0)
# database = database.reset_index()
# database = database.set_index(['id', 'date'])
# database.head()

database_standard = pd.read_csv('database_basic_stand.csv', index_col=0)
database_standard = database_standard.reset_index()
# database_standard = database_standard.set_index(['id', 'date'])
database_norm = pd.read_csv('database_basic_norm.csv', index_col=0)
database_norm = database_norm.reset_index()
# database_norm = database_norm.set_index(['id', 'date'])

# database = database_norm

print(database_norm.max())
print(database_standard.max())

database = database_norm

#%%

X_train = []
X_valid = []
X_test = []
y_train = []
y_valid = []
y_test = []

y_train_bool = []
y_valid_bool = []
y_test_bool = []

train_lengths = []
# length train + validation
valid_lengths = []
for _, group in database.groupby(['id']):
    length = group.shape[0]
    train_valid_split = int(length * TRAIN_P)
    valid_test_split = int(length * (1 - TEST_P))

    train_lengths.append(train_valid_split)
    valid_lengths.append(valid_test_split)

    y_train_bool.append(1 - group.iloc[:train_valid_split]['shifted_target_mood_bool'].values.reshape(-1, 1))
    y_valid_bool.append(1 - group.iloc[train_valid_split:valid_test_split]['shifted_target_mood_bool'].values.reshape(-1, 1))
    y_test_bool.append(1 - group.iloc[valid_test_split:]['shifted_target_mood_bool'].values.reshape(-1, 1))

    y_train.append(group.iloc[:train_valid_split]['target_mood'].values)
    y_valid.append(group.iloc[train_valid_split:valid_test_split]['target_mood'].values)
    y_test.append(group.iloc[valid_test_split:]['target_mood'].values)

    cols = [c for c in group.columns if ('bool' not in c) and ('target' not in c) and ('date' not in c) and ('id' not in c)]
    group = group[cols]
    
    X_train.append(group.iloc[:train_valid_split].values)
    X_valid.append(group.iloc[:valid_test_split].values)
    X_test.append(group.values)

print(train_lengths)

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

def MSE_4(input, target):
    return torch.sum((input-target) ** 4) / input.shape[0]

# loss_fn = MSE_4

train_losses = []
valid_losses = []
for e in range(50):
    for i, sequence in enumerate(X_train):
        # if (i == 3) or (i == 5):
        #     continue
        
        # print(e, i, '/', len(X_train))
        # print('length sequence:', sequence.shape[0])
        # print('number of features:', sequence.shape[1])
        lstm_model.zero_grad()
        input = torch.from_numpy(sequence).float()
        y_pred = lstm_model(input)
        # print('length prediction sequence:', len(y_pred))
        # print(y_pred)

        # print(y_pred.shape)
        # print(y_train_bool[i].shape)
        # exit()

        y_train_predict_corrected = np.squeeze(y_pred.data.numpy()) * np.squeeze(y_train_bool[i])
        y_train_corrected = np.squeeze(y_train[i]) * np.squeeze(y_train_bool[i])

        train_nr_not_interpolated = np.count_nonzero(y_train_bool[i])

        
        train_loss = ((y_train_predict_corrected - y_train_corrected)**2).sum() / train_nr_not_interpolated 
        # print(train_loss)
        train_losses.append(train_loss)
        
        # our_loss = ((y_pred.data.numpy() - y_train[i])**2).mean()
        train_loss = MSE_4(y_pred, torch.from_numpy(y_train[i]).float())

        # VALID
        valid_loss = 0
        for j in range(len(X_valid)):

            input = torch.from_numpy(X_valid[j]).float()
            y_pred = lstm_model(input)
            
            # CUT OFF PREDICTION PART THAT WAS ALREADY IN TRAIN
            y_pred = y_pred[train_lengths[j]:]

            y_valid_predict_corrected = np.squeeze(y_pred.data.numpy()) * np.squeeze(y_valid_bool[j])
            y_valid_corrected = np.squeeze(y_valid[j]) * np.squeeze(y_valid_bool[j])
        
            valid_nr_not_interpolated = np.count_nonzero(y_valid_bool[j])

            our_loss = ((y_valid_predict_corrected - y_valid_corrected)**2).sum() / valid_nr_not_interpolated 
            # print(our_loss)
            valid_loss += our_loss 
        valid_losses.append(valid_loss/len(X_valid))

        # print('our loss', our_loss)
        
        # print('MSEloss:', train_loss.item())
        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        train_loss.backward()

        # Update parameters
        optimiser.step()
    
    if (e % 5) == 0 or (e == 24):
        print('Epoch:', e)
        print('train loss:', train_loss.item())
        print('valid loss:', valid_loss/len(X_valid))
#%%

# plt.plot(train_losses, label='train')
# plt.plot(valid_losses, label='valid')
# plt.title('Loss')
# plt.savefig('LSTM_standardised.png', dpi=200, facecolor='w')
# plt.legend()
# plt.show()

#%%
total_loss = 0

all_predictions = []

for i in range(len(X_test)):
    input = torch.from_numpy(X_test[i]).float()
    pred = lstm_model(input)
    
    # CUT OFF PREDICTION PART THAT WAS ALREADY IN TRAIN
    y_pred = pred.data.numpy()
    y_pred = y_pred[valid_lengths[i]:]
    
    all_predictions.append(np.squeeze(y_pred))

    # CORRECT FOR INTERPOLATION
    y_test_predict_corrected = np.squeeze(y_pred) * np.squeeze(y_test_bool[i])
    y_test_corrected = np.squeeze(y_test[i]) * np.squeeze(y_test_bool[i])

    test_nr_not_interpolated = np.count_nonzero(y_test_bool[i])
    total_loss += ((y_test_predict_corrected - y_test_corrected)**2).sum() / test_nr_not_interpolated 


print('NORMALIZED')
all_predictions = np.concatenate(all_predictions)
print(list(all_predictions))
print(len(all_predictions))

print('total test loss', total_loss.item()/len(X_test))

#%%
