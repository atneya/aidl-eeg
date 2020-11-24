import numpy as np
import tensorflow as tf
import argparse
import scipy
import os
from os import listdir
from scipy.io import loadmat
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def crossval(generate_model, n_epochs, X_train, y_train, X_test, y_test, filename = None):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    bestmodel = None
    bestAcc = 0
    cvscores = []
    fold = 1
    for train, test in kfold.split(X_train, y_train):
        model = generate_model()
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold} ...')
        model.fit(X_train[train], y_train[train],epochs=n_epochs, verbose=0, validation_split=0.2)
        scores = model.evaluate(X_train[test], y_train[test], verbose=1)
        print("Score for fold %d - %s: %.2f%%" % (fold, model.metrics_names[1], scores[1]*100))
        if(scores[1] > bestAcc):
            bestAcc = scores[1]
            bestmodel = model
        cvscores.append(scores[1] * 100)
        fold += 1
    print('------------------------------------------------------------------------')
    print("Avg accuracies: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    test_loss, test_acc = bestmodel.evaluate(X_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    if filename:
        pickle.dump( bestmodel, open( filename, "wb" ) )

def threedConv():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(32, kernel_size=10, padding="same", activation='relu', kernel_initializer='he_uniform', input_shape=X.shape[1:]))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Conv3D(64, kernel_size=3, activation='relu', padding="same", kernel_initializer='he_uniform'))
    #model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(tf.keras.layers.BatchNormalization(center=True, scale=True))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    return model

def fully_conv():
    return tf.keras.Sequential([
    tf.keras.layers.SeparableConv1D(
128, 3, data_format='channels_last', padding="same", activation='relu',input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(
128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(
128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(
128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


def single_conv():
    return tf.keras.Sequential([
    tf.keras.layers.Conv1D(
32, 3, activation='relu',input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(4)
])

def LSTM():
    return tf.keras.Sequential([
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

def twodConv():
    return tf.keras.Sequential([
    tf.keras.layers.Conv2D(
32, 3, padding="same", activation="relu",input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling2D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.BatchNormalization(center=True, scale=True),
    tf.keras.layers.Conv2D(
64, 3, padding="same", activation="relu",input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling2D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.BatchNormalization(center=True, scale=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

def parse_args(parser):
    parser.add_argument('-m', '--model', type = str, choices=['d', 'sc', 'fc', 'LSTM', '3dConv'], help = "specify a model type", required = True)
    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = parse_args(parser)
model_tag = args.model

directory = "./data/eeg_feature_smooth/1/"
directories = ["./data/eeg_feature_smooth/{}/".format(i+1) for i in range(3)]
 
channel_coords = [['0', '0', 'AF3', 'FP1', 'FPZ', 'FP2', 'AF4', '0', '0'], ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'], ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'], ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'], ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'], ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'], ['0', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '0'], ['0', '0', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '0', '0']]

channel_list = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

coord_dict = {}
for n in range(len(channel_list)):
    for i, l in enumerate(channel_coords):
        for j, x in enumerate(l):
            if (channel_list[n] == x):
                coord_dict[n] = (i,j)

n = 24
perSample = ['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS']
array = np.zeros(shape=(len(directories),len(os.listdir(directories[0])), n, 4, 8, 9, 5, 64)) # features = 4 datatypes*(8 x 9 eeg channel locs)*5 frequency bands*64 timestamps(zero padded) // trials = (3 sessions) x 15 people x 24 labels 
li = []
for h, dire in enumerate(directories):
    data = [loadmat(dire + file) for file in os.listdir(dire)]
    for i, bigsample in enumerate(data):
        for j in range(n):
            for k, key in enumerate(perSample):
                sample = np.transpose(np.array(bigsample[key + str(j+1)]), (0,2,1))
                sample = np.pad(sample, [(0,0), (0,0), (0, 64-sample.shape[2])], mode = 'constant')
                for l, channel in enumerate(sample):
                    array[h][i][j][k][coord_dict[l][0]][coord_dict[l][1]] = channel

_X = array.reshape(np.prod(array.shape[0:3]), *array.shape[3:])

session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}

y = np.array(session1_label * 15 + session2_label * 15 + session3_label * 15)
y.shape

X = _X.transpose(0, 5, 1,2,3,4)
print(X.shape)
X = X.reshape(X.shape[0], X.shape[1], np.prod(X.shape[2:]))
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Starting Training")
if model_tag == 'd':
    crossval(dense, 100, X_train, y_train, X_test, y_test)
elif model_tag == 'fc':
    crossval(fully_conv, 100, X_train, y_train, X_test, y_test)
elif model_tag == 'sc':
    crossval(single_conv, 100, X_train, y_train, X_test, y_test)
elif model_tag == 'LSTM':
    print(4)
    crossval(LSTM, 100, X_train, y_train, X_test, y_test)
else: 
    X = _X.transpose(0, 5, 2,3, 4, 1)[:,:,:,:,:,[0,2]]
    
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], np.prod(X.shape[4:]))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    
    print("Starting Training")
    crossval(threedConv, 100, X_train, y_train, X_test, y_test)
