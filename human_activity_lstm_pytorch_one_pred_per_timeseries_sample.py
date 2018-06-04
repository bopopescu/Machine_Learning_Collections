# coding: utf-8

import shutil, os, csv, itertools, glob

import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle as pk
import re
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
RANDOM_SEED = 42

# CUDA trick: for checking if the GPU (CUDA) is available
cuda = torch.cuda.is_available()


##########################################################################################
# Common Utillities
def batch_startend_list(total_length, batch_size):
    """
    Create Batch (Start, End) tuple zip for indexing
    for example, we want to create a list of indices
    For example
    > batch_startend_list(100, 20)
    The return is [(0, 20), (20, 40), (40, 60),
                    (60, 80), (80, 100)]
    """
    return zip(range(0, total_length, batch_size),
        range(batch_size, total_length + 1,batch_size))


def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print("Pickle file can not be openned")
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print("load_pickle failed once, try again")
        p.close()
        p = open(filename, "r")
        picklelicious = pk.load(p)
    p.close()
    return picklelicious


def save_pickle(data_object, filename):
    pickle_file = open(filename, 'w')
    pk.dump(data_object, pickle_file)
    pickle_file.close()

def read_data(filename):
    print("Loading Data......")
    df = pd.read_csv(filename, header=None) # this is to read the data file without header
    data = df.values
    return data

def read_line(csvfile, line):
    with open(csvfile, 'r') as f:
        data = next(itertools.islice(csv.reader(f), line, None))
    return data

def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x_axis', 'y_axis', 'z_axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 8),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

def replace2(x):
    # replace the string character and convert to number
    if pd.isna(x):
        res = np.nan
    else:
        res = np.float(re.sub('[;]', '', x))
    return res

def encode_label(x):
    # x being a numpy object vector
    res = np.zeros((len(x), 6))
    res[x == "Walking"] = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res[x == "Jogging"] = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    res[x == "Upstairs"] = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    res[x == "Downstairs"] = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    res[x == "Sitting"] = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    res[x == "Standing"] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return res
def encode_label2(x):
    # x being a numpy object vector
    res = np.zeros(len(x))
    res[x == "Walking"] = 0
    res[x == "Jogging"] = 1
    res[x == "Upstairs"] = 2
    res[x == "Downstairs"] = 3
    res[x == "Sitting"] = 4
    res[x == "Standing"] = 5
    return res


# ### Classification by PyTorch (LSTM Deep Learning model) using sensor data (accelerometer) to detect the bus ride or not
#
# Details check out the link https://pytorch.org/docs/master/nn.html#torch-nn-init

####################################################################################################
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout,
                        bidirectional, num_classes, batch_size,
                        batch_first = True):
        super(LSTMClassifier, self).__init__()
        self.arch = "lstm"  # define the archtecture type, LSTM, VGG, MLP or CNN, etc.
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_first = batch_first
        # Define the LSTM basic config, not to the architecture yet
        # Without batch_first, input dimensions are [seq_length, batch_size, num_features].
            # batch_first switches the seq_length and batch_size for input
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers = num_layers,
                            dropout = dropout,
                            batch_first = batch_first,
                            bidirectional = bidirectional  # it's a boolean variable
                            )
        self.hidden2label = nn.Sequential(nn.Linear(hidden_dim * self.num_dir, hidden_dim),
                                            nn.ReLU(True),
                                            nn.Dropout(dropout),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(True),
                                            nn.Dropout(dropout),
                                            nn.Linear(hidden_dim, self.num_classes))
        self.hidden = self.init_hidden()
        # This version de-precated the batch size

    def init_hidden(self):
        if cuda:
            # define the the hidden state at time t=0, ct is the cell state at time t
            # both ht, ct are the hidden states in the LSTM architecture
            h0 = Variable(torch.zeros(self.num_layers * self.num_dir,
                        self.batch_size,
                        self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers * self.num_dir,
                        self.batch_size,
                        self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers * self.num_dir,
                        self.batch_size,
                        self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers * self.num_dir,
                        self.batch_size,
                        self.hidden_dim))
            return (h0, c0)

    def forward(self, x):
        # here, x is the input (sentence or time series of multi-dimensional)
        # each x input is, say, batch_sze X 200 X 3, the 200 seconds of x,y,z-axis accelerometer readings
        # change it to 200 X batch_size X 3, here 1 is the batch size, for now dsable the batch size to
        # simplify the process
        seq_length = x[0].shape[0]
        if self.batch_first:
            x = x.view(self.batch_size, seq_length, -1)
        else:
            x = x.view(seq_length, self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)  # take last prediction of the sequence
        y = self.hidden2label(lstm_out.view(self.batch_size, seq_length, -1))
        #return y
        y_scores = F.log_softmax(y, dim=2) # return probability using log_softmax, dim=1 means sum is 1 for column (like axis=1)
        y_scores = torch.exp(y_scores)
        # Cross entropy funciton does it already
        # return just one record based on average all rows, so convert it to
        # batch_size X num_classes
        y_scores = torch.log(y_scores.mean(dim=1))
        # Need to return log scale for loss function below
        return y_scores


# Print the Network
LSTMClassifier(input_dim=3, hidden_dim=28, num_layers=5, dropout=0.5,
                      bidirectional=True, num_classes=6, batch_size = 32)


####################################################################################################
# ### Define Data Loader
# Data is downloaded from http://www.cis.fordham.edu/wisdm/dataset.php

columns = ['user','activity','timestamp', 'x_axis', 'y_axis', 'z_axis']
df = pd.read_csv('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
df['z_axis'] = df['z_axis'].apply(replace2)
df = df.dropna()

df.head()

df.info()

# Data Exploration
df['activity'].value_counts().plot(kind='bar', title='Training examples by activity type');
df['user'].value_counts().plot(kind='bar', title='Training examples by user')
plot_activity("Jogging", df)

# Data Transformation
df.sort_values(by=['user', 'timestamp'], ascending=[True, True], inplace=True)
df_ = df[['user', 'activity']]
df_["cnt1"] = 0
df_.loc[np.array(df_["user"].shift(periods=1,axis=0) != df_["user"]) | \
        np.array(df_["activity"].shift(periods=1,axis=0) != df_["activity"]), "cnt1"] = 1
df['cnt1'] = df_['cnt1'].cumsum()

allids = df.cnt1.unique()
segments = []
labels = []
N_TIME_STEPS = 200  # short than 1000 data points will be filtered out
step = 10  # smaller the better, the data will be large
for i in allids:
    df_ = df.query("cnt1 == " + str(i))
    for j in range(0, len(df_) - N_TIME_STEPS, step):
        xs = df_['x_axis'].values[j: j + N_TIME_STEPS]
        ys = df_['y_axis'].values[j: j + N_TIME_STEPS]
        zs = df_['z_axis'].values[j: j + N_TIME_STEPS]
        # label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]  # use one label for all time points
        label = df_['activity'].values[j: j + N_TIME_STEPS] # use each data point's label
        segments.append([xs, ys, zs])
        labels.append(stats.mode(encode_label2(label))[0][0])  # get the single label value based on mode
    del df_

N_FEATURES = 3  # x,y,z-axis
# get label types
#reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
segments = np.asarray(segments, dtype= np.float32)
segments = segments.reshape(-1, N_TIME_STEPS, N_FEATURES)
x_ = np.asarray(labels, dtype= object).reshape(-1)
label_unique_values = np.unique(np.random.choice(x_, size=100000))
#labels = np.asarray(pd.get_dummies(labels), dtype = np.int)
labels = np.asarray(labels, dtype = np.int)
print(np.array(segments).shape)
print(np.array(labels).shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
        segments, labels, test_size=0.2, random_state=RANDOM_SEED)

print((X_train.shape, X_test.shape))


####################################################################################################
# ### Model Building

# Try to clear everything
for obj_ in ["model", "optimizer", "loss_function", "loss_history", "loss"]:
    try:
        exec("del "+obj_)
    except:
        pass
# 3 dimension data as input dimension
BATCHSIZE = 32
model = LSTMClassifier(input_dim=3, hidden_dim=64, num_layers=2, dropout=0.05,
                      bidirectional=True, num_classes=6, batch_size = BATCHSIZE)
# Make sure the input length is multiple of batch_size
#loss_function = nn.NLLLoss()
loss_function = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.99)
#optimizer = optim.Adam(model.parameters(), weight_decay = 1e-6)
optimizer = optim.Adamax(model.parameters(), weight_decay = 1e-4)

# See what the scores are before training
# Here we don't need to train, so the code is wrapped in torch.no_grad()
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test, dtype=torch.long)
loss_history = []
zips = batch_startend_list(X_train.shape[0], BATCHSIZE)
zips_size = len(list(zips))
# Train the model
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    batch_counter = 1
    for start, end in batch_startend_list(X_train.shape[0], BATCHSIZE):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        optimizer.zero_grad()
        input_, actual_label_ = X_train[start:end], y_train[start:end]
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        # Step 3. Run our forward pass.
        target_ = model(input_)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        #actual_label_ = actual_label_.view(BATCHSIZE * y_train.shape[1])
        #target_ = target_.view(BATCHSIZE * y_train.shape[1], -1)
        loss = loss_function(target_, actual_label_)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        lookback_ = (-1) * zips_size
        if batch_counter % max(1,int(zips_size/20)) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_counter, zips_size,
                100. * batch_counter / zips_size,
                np.mean(np.asarray(loss_history[lookback_:]))))
        batch_counter += 1
    # Validation on Test after each epoch
    test_loss = 0
    correct = 0
    model.zero_grad()
    optimizer.zero_grad()
    BATCHSIZE_TEST = X_test.shape[0]
    model.batch_size = BATCHSIZE_TEST  # this is to fit test in the batch mode
    with torch.no_grad():
        data, target = X_test, y_test  # just some legacy convention
        model.hidden = model.init_hidden() # VERY CRITICAL to initialize hidden layers again
        output = model(data)
        #target = target.view(BATCHSIZE_TEST * y_test.shape[1])
        #output = output.view(BATCHSIZE_TEST * y_test.shape[1], -1)
        test_loss += loss_function(output, target).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()/len(target) #whole batch is just one label value
    print('\nTest set: Epoch {}, Average loss: {:.4f}, Accuracy: {:.4f} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, 100. * correct))
    model.batch_size = BATCHSIZE
    ###### Save Or Load Model ######
    best_metric = np.inf
    if False: #test_loss < best_metric:
        best_metric = test_loss + 0.0
        torch.save(model.state_dict(), "state_human_activity_torch_lstm.tar")
        torch.save(model, "model_human_activity_torch_lstm.tar")
    if False:
        # two methods of loading model, but the first one by loading model state is more
        # reliable in case if the model class is not easily serizlied
        # second method is more straightforward method for saving and loading model
        # but for some model structure, it might run into risk of being broken
        model_loaded1 = torch.load('./model_human_activity_torch_lstm.tar')
        model_loaded2 = LSTMClassifier(input_dim=3, hidden_dim=200, num_layers=2, dropout=0.5,
                              bidirectional=True, num_classes=6, batch_size = BATCHSIZE)
        model_loaded2.load_state_dict(torch.load("state_human_activity_torch_lstm.tar"))

