# Title      :	third_model_hypertuning.py
# Objective  :	Perform hypertuning on model 3
# Created by :	Luke
# Created on :	Mon 12/07/21 18:37

import sys
from operator import mod

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.optimizers import *
from sklearn.metrics import auc, roc_curve
from tensorflow.python.keras.engine.input_layer import InputLayer
from datetime import datetime
import keras_tuner as kt

sys.path.insert(0, r'C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Modelling\Models\load_img_data.py')
from load_img_data import load_data

plt.style.use('ggplot')

img_height = img_width = 128
batch_size = 32

# Data Directories
train_dir = 'E:/CNN Data Small Sample/Train' 
test_dir = 'E:/CNN Data Small Sample/Test'
validate_dir = 'E:/CNN Data Small Sample/Validate'

# Load the data
# train_x, train_y = load_data(train_dir, img_height, img_width)
# test_x, test_y = load_data(test_dir, img_height, img_width)
# val_x, val_y = load_data(validate_dir, img_height, img_width)

train_x, train_y = np.load(r'E:\CNN Numpy Arrays\train_x.npy'), np.load(r'E:\CNN Numpy Arrays\train_y.npy')
test_x, test_y = np.load(r'E:\CNN Numpy Arrays\test_x.npy'), np.load(r'E:\CNN Numpy Arrays\test_y.npy')
val_x, val_y = np.load(r'E:\CNN Numpy Arrays\val_x.npy'), np.load(r'E:\CNN Numpy Arrays\val_y.npy')

data_x = np.concatenate((train_x, val_x, test_x))
data_y = np.concatenate((train_y, val_y, test_y))

train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15
assert train_ratio + validation_ratio + test_ratio == 1



train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=1 - train_ratio)

val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=test_ratio/(test_ratio + validation_ratio)) 

# train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, 
#     train_size=0.7)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss',                           patience=5,                                                 restore_best_weights=True)
now = datetime.now()
log_dir = r'C:\Users\Luke\Desktop\CNN Log Files\Final Model Logs'
dt_string = now.strftime("\Model Log_%d-%m_%H-%M-%S.csv")
csv_log = CSVLogger(log_dir + '\csv logs' + dt_string, separator=",")

# Metrics
rec = tf.keras.metrics.Recall(name='recall')
prec = tf.keras.metrics.Precision(name='precision')
auc_score = tf.keras.metrics.AUC(curve='PR')

# Hyperparam imports
he = tf.keras.initializers.HeUniform()
glorot = tf.keras.initializers.GlorotUniform()
loss = BinaryCrossentropy(from_logits=True)
optimizer = Adam(learning_rate=0.001)
regulariser = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)

# Define Model
def create_model():

    model = Sequential([
        Conv2D (32, 5, activation='relu', kernel_initializer=he),
        Conv2D (32, 8, activation='relu', kernel_initializer=he),
        Conv2D (32, 5, activation='relu', kernel_initializer=he),
        MaxPooling2D(pool_size=(2, 2), strides=None),
        BatchNormalization(momentum=0.7, epsilon=0.001),
        Conv2D (32, 7, activation='relu', kernel_initializer=he),
        Conv2D (32, 4, activation='relu', kernel_initializer=he),
        Conv2D (32, 4, activation='relu', kernel_initializer=he),
        Conv2D (16, 9, activation='relu', kernel_initializer=he),
        MaxPooling2D(pool_size=(3, 3), strides=None),
        BatchNormalization(momentum=0.7, epsilon=0.001),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=he),
        Dense(1, activation='sigmoid', kernel_initializer=glorot)
    ])

    model.compile(loss=loss, optimizer=optimizer, metrics=[rec, prec, auc_score])

    return model

model = create_model()

def hypertune_model(hp):

    
    lr = hp.Choice('learning_rate', values=[0.004, 0.005, 0.006, 0.007, 0.008, 0.009])


    model = Sequential([
        Conv2D (32, 5, activation='relu', kernel_initializer=he),
        Conv2D (32, 8, activation='relu', kernel_initializer=he),
        Conv2D (32, 5, activation='relu', kernel_initializer=he),
        MaxPooling2D(pool_size=(2, 2), strides=None),
        BatchNormalization(momentum=0.7, epsilon=0.001),
        Conv2D (32, 7, activation='relu', kernel_initializer=he),
        Conv2D (32, 4, activation='relu', kernel_initializer=he),
        Conv2D (32, 4, activation='relu', kernel_initializer=he),
        Conv2D (16, 9, activation='relu', kernel_initializer=he),
        MaxPooling2D(pool_size=(3, 3), strides=None),
        BatchNormalization(momentum=0.7, epsilon=0.001),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=he),
        Dense(1, activation='sigmoid', kernel_initializer=glorot)
    ])

    model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=[rec, prec, auc_score])

    return model

hyper_path = r'C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Modelling\Models\Hypertuning'

now = datetime.now()
dt_string = now.strftime("Log_%d-%m_%H-%M-%S")

tuner = kt.Hyperband(hypertune_model,
                     objective=kt.Objective('val_precision', 'max'),
                     max_epochs=100,
                     factor=3,
                     hyperband_iterations=2,
                     directory=hyper_path,
                     project_name=dt_string)

tuner.search(train_x, train_y, epochs=100, validation_data=(val_x, val_y), callbacks=[early_stop])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n")
# for i in range(1, 8):
#     print(f"Filter {i} - {best_hps[f'filters{i}']}")

print(f"Learning Rate - {best_hps[f'learning_rate']}")
# for i in range(1, 3):
#     print(f"Momentum {i} - {best_hps[f'momentum{i}']}")



