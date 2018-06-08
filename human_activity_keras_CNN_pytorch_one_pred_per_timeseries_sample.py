# coding: utf-8

import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1_l2
from keras.models import load_model

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle as pk
import re
from scipy import stats

RANDOM_SEED = 10038


##########################################################################################
# Common Utillities
def batch_startend_list(total_length, batch_size):
    return zip(range(0, total_length, batch_size),
        range(batch_size, total_length + 1,batch_size))

def plot_activity(activity, df):
    data = df[df['activity'] == activity][['x_axis', 'y_axis', 'z_axis']][:200]
    axis = data.plot(subplots=True, figsize=(16, 8),
                     title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

def replace2(x):
    if pd.isna(x):
        res = np.nan
    else:
        res = np.float(re.sub('[;]', '', x))
    return res

def encode_label(x):
    res = np.zeros((len(x), 6))
    res[x == "Walking"] = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res[x == "Jogging"] = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    res[x == "Upstairs"] = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    res[x == "Downstairs"] = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    res[x == "Sitting"] = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    res[x == "Standing"] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return res
def encode_label2(x):
    res = np.zeros(len(x))
    res[x == "Walking"] = 0
    res[x == "Jogging"] = 1
    res[x == "Upstairs"] = 2
    res[x == "Downstairs"] = 3
    res[x == "Sitting"] = 4
    res[x == "Standing"] = 5
    return res

####################################################################################################
# ### Define Data Loader
# Data is downloaded from http://www.cis.fordham.edu/wisdm/dataset.php and it uses raw data

# The data is x,y,z-axis accelerometer sensor data, and the 6 activity classes are
# Class Distribution
#    Walking: 424,400 (38.6%)
#    Jogging: 342,177 (31.2%)
#    Upstairs: 122,869 (11.2%)
#    Downstairs: 100,427 (9.1%)
#    Sitting: 59,939 (5.5%)
#    Standing: 48,395 (4.4%)

columns = ['user','activity','timestamp', 'x_axis', 'y_axis', 'z_axis']
df = pd.read_csv('./WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
df['z_axis'] = df['z_axis'].apply(replace2)
df = df.dropna()
# Data Transformation
df.sort_values(by=['user', 'timestamp'], ascending=[True, True], inplace=True)
df_ = df[['user', 'activity']]
df_["cnt1"] = 0
df_.loc[np.array(df_["user"].shift(periods=1,axis=0) != df_["user"]) | \
        np.array(df_["activity"].shift(periods=1,axis=0) != df_["activity"]), "cnt1"] = 1
df['cnt1'] = df_['cnt1'].cumsum()
# add magnitude
df = df.eval("magnitude = (x_axis**2 + y_axis**2 + z_axis**2)**0.5")
# Scale data
x_scaler = preprocessing.StandardScaler()
x_ = x_scaler.fit_transform(df[["x_axis", "y_axis", "z_axis", "magnitude"]].values)
df["x_axis"] = x_[:,0]
df["y_axis"] = x_[:,1]
df["z_axis"] = x_[:,2]
df["magnitude"] = x_[:,3]

df["x1"] = np.abs(df["x_axis"])
df["y1"] = np.abs(df["y_axis"])
df["z1"] = np.abs(df["z_axis"])
x_ = df[["x1", "y1", "z1"]].max(axis=1)
df['max1'] = x_
x_ = df[["x1", "y1", "z1"]].median(axis=1)
df['max2'] = x_
x_ = df[["x1", "y1", "z1"]].min(axis=1)
df['max3'] = x_

allids = df.cnt1.unique()
X_train=[]
y_train=[]
N_TIME_STEPS = 400
step = 20
N_FEATURES = 7
NUM_CLASSES = 6 # there are 6 activities to classify

def data_generator(df, allids):
    X_train=[]
    y_train=[]
    N_TIME_STEPS = 400
    step = 20
    N_FEATURES = 7
    NUM_CLASSES = 6 # there are 6 activities to classify
    for i in allids:
        df_ = df.query("cnt1 == " + str(i))
        for j in range(0, len(df_) - N_TIME_STEPS, step):
            xs = df_['x_axis'].values[j: j + N_TIME_STEPS]
            ys = df_['y_axis'].values[j: j + N_TIME_STEPS]
            zs = df_['z_axis'].values[j: j + N_TIME_STEPS]
            z2s = df_['magnitude'].values[j: j + N_TIME_STEPS]
            z3s = df_['max1'].values[j: j + N_TIME_STEPS]
            z4s = df_['max2'].values[j: j + N_TIME_STEPS]
            z5s = df_['max3'].values[j: j + N_TIME_STEPS]
            label = df_['activity'].values[j: j + N_TIME_STEPS]
            X_train.append([xs, ys, zs, z2s, z3s, z4s, z5s])
            y_ =np.zeros(NUM_CLASSES)
            y_[int(stats.mode(encode_label2(label))[0][0])] = 1.
            y_train.append(y_)
        del df_
    X_train = np.asarray(X_train, dtype= np.float32)
    X_train = X_train.reshape(-1, N_TIME_STEPS, N_FEATURES)
    y_train = np.asarray(y_train, dtype = np.int)
    return (X_train, y_train)

X_train, y_train = data_generator(df, allids)
print(X_train.shape, y_train.shape)

####################################################################################################
# CNN Model Training
# Instantiate the cross validator
NFOLD = 5
skf = KFold(n_splits=NFOLD, shuffle=True)
# Loop through the indices the split() method returns
history = []
for index, (train_indices, val_indices) in enumerate(skf.split(range(len(allids)))):
    print("Training on fold " + str(index+1) + "/"+str(NFOLD)+"...")
    # Generate batches from indices
    xtrain, ytrain = data_generator(df, np.array(allids)[train_indices])
    xval, yval = data_generator(df, np.array(allids)[val_indices])
    # Clear model, and create it
    model = None
    reg = 1e-2
    model = Sequential()
    model.add(Conv1D(256, 2, activation='relu', input_shape=(N_TIME_STEPS,N_FEATURES)))
    model.add(Conv1D(256, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    #model.add(Conv1D(32, 2, activation='relu', kernel_regularizer=l1_l2(l1=reg, l2=0.), activity_regularizer=l1_l2(l1=reg, l2=0.)))
    #model.add(MaxPooling1D(2))
    #model.add(Dropout(0.5))
    model.add(Conv1D(512, 2, activation='relu'))
    model.add(Conv1D(512, 2, activation='relu'))
    model.add(GlobalAveragePooling1D())
    #model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    #model.add(Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=reg, l2=0.), activity_regularizer=l1_l2(l1=reg, l2=0.)))
    #model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax',
                    kernel_regularizer=l1_l2(l1=reg, l2=0.),
                    activity_regularizer=l1_l2(l1=reg, l2=0.)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='best_CNN_keras_model.hdf5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    allcallbacks = [checkpointer, EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max')]
    modelhist = model.fit(xtrain, ytrain, batch_size=32, epochs=10, verbose=1, validation_data=(xval, yval), callbacks=[checkpointer])
    # (test_crossentropy, test_accuracy) = model.evaluate(xval, yval, batch_size=32)
    history.append(modelhist.history['val_acc'])

history = np.array(history)
best_mean = 0
best_epoch = 0
for i in range(history.shape[1]):
    a = np.mean(history[:,i])
    if a > best_mean:
        best_mean = a + 0
        best_epoch = i + 1
print("Best Cross Validated Accruacy is {:.4}% at epoch {}".format(best_mean*100., best_epoch))
# "Log: Best Cross Validated Accruacy is 93.34% at epoch 4 with reg = 1e-2"
# "Log: Best Cross Validated Accruacy is 94.47% at epoch 10 with reg = 4e-3"

# Load in the pre-buitl model
model2 = load_model('best_CNN_keras_model.hdf5')
train_preds = model2.predict_proba(X_train, verbose=0)
train_classes = model2.predict_classes(X_train)
train_actual_label = np.array(list(map(lambda x: np.where(x==1)[0][0], y_train)))
print("Accuracy of test data on moving window is {:.4}%".format(np.sum(train_classes==train_actual_label)/len(train_actual_label) * 100.))
