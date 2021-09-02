# Title      :	final_model.py
# Objective  :	A script for my final model - this script does not include any plotting of performance,
#               only creation of the predictions and labels
# Created by :	Luke
# Created on :	Tue 13/07/21 09:47

from pathlib import Path

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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from models.load_img_data import load_data

plt.style.use('ggplot')

img_height = img_width = 128
batch_size = 32


def load_and_split(train_dir, test_dir, validate_dir):
    train_x, train_y = load_data(train_dir, img_height, img_width)
    test_x, test_y = load_data(test_dir, img_height, img_width)
    val_x, val_y = load_data(validate_dir, img_height, img_width)

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

    return train_x, train_y, test_x, test_y, val_x, val_y


# Metrics
rec = tf.keras.metrics.Recall(name='recall')
prec = tf.keras.metrics.Precision(name='precision')
auc_score = tf.keras.metrics.AUC(curve='PR')

# Hyperparam imports
he = tf.keras.initializers.HeUniform()  # used for ReLU
glorot = tf.keras.initializers.GlorotUniform()  # used for sigmoid
loss = BinaryCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=0.001)


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


def train_and_save(model, train_x, train_y, val_x, val_y):
    model_path = Path(__file__).parent.absolute()
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    # I saved the whole model, not just weights so that the architecture did not have to be re-specified in the prediction scipts
    save_model_dir = model_path.joinpath("weights.best.hdf5")
    checkpoint = ModelCheckpoint(save_model_dir, monitor='val_precision',
                                 save_best_only=True, mode='max', save_weights_only=False)

    model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        # Over estimate epochs thanks to earlystopping
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint]
    )

    # Finding best threshold that maximises f1 score
    y_pred_keras = model.predict(val_x).ravel()
    precision, recall, thresholds = precision_recall_curve(val_y, y_pred_keras)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(fscore)
    best_threshold = thresholds[ix]
    # Write the best threshold to file for reading in at prediction time
    threshold_path = model_path.joinpath("best_threshold.txt")
    with open(threshold_path, "w+") as f:
        f.write(str(best_threshold))
