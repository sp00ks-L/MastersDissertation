# Title      :	final_model.py
# Objective  :	A script for my final model - this script does not include any plotting of performance,
#               only creation of the predictions and labels
# Created by :	Luke
# Created on :	Tue 13/07/21 09:47

from load_img_data import load_data
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import visualkeras
from PIL import ImageFont
from sklearn.metrics import (auc, average_precision_score, f1_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split

sys.path.insert(
    0, r'/Sussex Python/Dissertation/Modelling/Models/load_img_data.py')

plt.style.use('ggplot')

img_height = img_width = 128
batch_size = 32

# Data Directories
# train_dir = 'E:/CNN Data Small Sample/Train'
# test_dir = 'E:/CNN Data Small Sample/Test'
# validate_dir = 'E:/CNN Data Small Sample/Validate'

train_dir = 'E:/CNN Combined Sample 2/Train'
test_dir = 'E:/CNN Combined Sample 2/Test'
validate_dir = 'E:/CNN Combined Sample 2/Validate'

# Load the data
train_x, train_y = load_data(train_dir, img_height, img_width)
test_x, test_y = load_data(test_dir, img_height, img_width)
val_x, val_y = load_data(validate_dir, img_height, img_width)

# If you do not want to combine chromosomal data, then comment out from here
# Concat Data and split
data_x = np.concatenate((train_x, val_x, test_x))
data_y = np.concatenate((train_y, val_y, test_y))

train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
# just incase you got the ratios wrong will throw an error
assert train_ratio + validation_ratio + test_ratio == 1

train_x, test_x, train_y, test_y = train_test_split(
    data_x, data_y, test_size=1 - train_ratio)

val_x, test_x, val_y, test_y = train_test_split(
    test_x, test_y, test_size=test_ratio/(test_ratio + validation_ratio))
# to here ---------------------------------------------------------------


# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)


save_model_dir = r'E:\CNN Models\weights.best.hdf5'
# I saved the whole model, not just weights so that the architecture did not have to be re-specified in the prediction scipts
checkpoint = ModelCheckpoint(save_model_dir, monitor='val_precision',
                             save_best_only=True, mode='max', save_weights_only=False)

# Metrics
rec = tf.keras.metrics.Recall(name='recall')
prec = tf.keras.metrics.Precision(name='precision')
auc_score = tf.keras.metrics.AUC(curve='PR')

# Hyperparam imports
he = tf.keras.initializers.HeUniform()  # used for ReLU
glorot = tf.keras.initializers.GlorotUniform()  # used for sigmoid
loss = BinaryCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=0.001)


# Define Model
def create_model():
    # Filter sizes for each layer derived using Hyperband
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

model.fit(
    train_x,
    train_y,
    validation_data=(val_x, val_y),
    # Over estimate epochs thanks to earlystopping
    epochs=100,
    batch_size=batch_size,
    callbacks=[early_stop, checkpoint]
)

# Visual keras only needed to plot the model architecture
# from collections import defaultdict

# color_map = defaultdict(dict)
# color_map[Conv2D]['fill'] = '#ffd166'
# color_map[MaxPooling2D]['fill'] = '#ef476f'
# color_map[Dense]['fill'] = '#118ab2'
# color_map[Flatten]['fill'] = '#06d6a0'
# color_map[BatchNormalization]['fill'] = '#9b70e2'

# font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model, legend=True, font=font, to_file=r'C:\Users\Luke\Desktop\Thesis Figures\final_model.png', scale_xy=1, scale_z=1, color_map=color_map) # font is optional

# Finding best threshold that maximises f1 score
y_pred_keras = model.predict(test_x).ravel()
precision, recall, thresholds = precision_recall_curve(test_y, y_pred_keras)
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)
best_threshold = thresholds[ix]
# Write the best threshold to file for reading in at prediction time
with open(r"E:\CNN Models\best_threshold.txt", "w+") as f:
    f.write(str(best_threshold))
