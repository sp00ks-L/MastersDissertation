# Title      :	third model.py
# Objective  :	Experiment with other types of conv layers as well as attention
# Created by :	Luke
# Created on :	Sun 11/07/21 14:18

from load_img_data import load_data
import sys
from datetime import datetime

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Input, SpatialDropout2D
from keras.layers.convolutional import (Conv2D, MaxPooling2D)
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.optimizers import *
from sklearn.metrics import (auc, average_precision_score, f1_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine.input_layer import InputLayer

sys.path.insert(
    0, r'C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Modelling\Models\load_img_data.py')

plt.style.use('ggplot')

img_height = img_width = 128
batch_size = 32

# Data Directories
# train_dir = 'E:/CNN Data Small Sample/Train'
# test_dir = 'E:/CNN Data Small Sample/Test'
# validate_dir = 'E:/CNN Data Small Sample/Validate'

train_dir = 'E:/CNN Combined Sample/Train'
test_dir = 'E:/CNN Combined Sample/Test'
validate_dir = 'E:/CNN Combined Sample/Validate'


# Load the data
train_x, train_y = load_data(train_dir, img_height, img_width)
test_x, test_y = load_data(test_dir, img_height, img_width)
val_x, val_y = load_data(validate_dir, img_height, img_width)

# Concat Data and split
data_x = np.concatenate((train_x, val_x, test_x))
data_y = np.concatenate((train_y, val_y, test_y))

train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
assert train_ratio + validation_ratio + test_ratio == 1

train_x, test_x, train_y, test_y = train_test_split(
    data_x, data_y, test_size=1 - train_ratio)

val_x, test_x, val_y, test_y = train_test_split(
    test_x, test_y, test_size=test_ratio/(test_ratio + validation_ratio))


# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)
save_model_dir = r'E:\CNN Models\weights.best.hdf5'
checkpoint = ModelCheckpoint(save_model_dir, monitor='val_precision',
                             save_best_only=True, mode='max', save_weights_only=False)

# Metrics
rec = tf.keras.metrics.Recall(name='recall')
prec = tf.keras.metrics.Precision(name='precision')
auc_score = tf.keras.metrics.AUC(curve='PR')

# Hyperparam imports
he = tf.keras.initializers.HeUniform()
glorot = tf.keras.initializers.GlorotUniform()
loss = BinaryCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=0.001)


# Define Model
def create_model():

    model = Sequential([
        Conv2D(32, 5, activation='relu', kernel_initializer=he),
        Conv2D(32, 8, activation='relu', kernel_initializer=he),
        Conv2D(32, 5, activation='relu', kernel_initializer=he),
        MaxPooling2D(pool_size=(2, 2), strides=None),
        BatchNormalization(momentum=0.7, epsilon=0.001),
        Conv2D(32, 7, activation='relu', kernel_initializer=he),
        Conv2D(32, 4, activation='relu', kernel_initializer=he),
        Conv2D(32, 4, activation='relu', kernel_initializer=he),
        Conv2D(16, 9, activation='relu', kernel_initializer=he),
        MaxPooling2D(pool_size=(3, 3), strides=None),
        BatchNormalization(momentum=0.7, epsilon=0.001),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=he),
        Dense(1, activation='sigmoid', kernel_initializer=glorot)
    ])

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[rec, prec, auc_score])

    return model


model = create_model()
history = model.fit(
    train_x,
    train_y,
    validation_data=(val_x, val_y),
    epochs=100,
    batch_size=batch_size,
    callbacks=[early_stop, checkpoint]
)

########################################
# Below is just for plotting the model
# Code is identical to final_model.py, just includes performance plotting

# Assessing Model Performance
y_pred_keras = model.predict(test_x).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

precision, recall, thresholds = precision_recall_curve(test_y, y_pred_keras)
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)
best_threshold = thresholds[ix]

# Use lambdas


def optim_threshold(x): return 0 if (x <= best_threshold) else 1
def default_threshold(x): return 0 if (x <= 0.5) else 1


# Just exploring the difference between the 'optimal' threshold and a default threshold of 0.5
final_preds = [optim_threshold(x) for x in y_pred_keras]
alt_preds = [default_threshold(x) for x in y_pred_keras]

# Plot AUC-ROC
plt.figure(1, figsize=(10, 7))
plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
plt.plot(fpr_keras, tpr_keras, label=f'CNN (area = {auc_keras:.4f})')
plt.xlabel('False positive rate', fontsize=14)
plt.ylabel('True positive rate', fontsize=14)
plt.title('Combined Chromosome ROC curve', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='best', fontsize=14)
plt.show()

# Plot Precision-Recall Curve
no_skill = len(test_y[test_y == 1]) / len(test_y)
pr_score = average_precision_score(test_y, final_preds)

plt.figure(2, figsize=(10, 7))
plt.plot([0, 1], [no_skill, no_skill], 'k--', label='Baseline')
plt.plot(recall, precision, label=f'CNN (area = {pr_score:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Combined Chromosome Precision-Recall curve', fontsize=18)
plt.tick_params(labelsize=14)
plt.legend(loc='best')
plt.show()
