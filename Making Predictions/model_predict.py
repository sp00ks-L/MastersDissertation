# Title      :	model_predict.py
# Objective  :	Create a script that will make predictions on new data
# Created by :	Luke
# Created on :	Sun 11/07/21 17:55

import keras
from load_img_data import load_data

"""
This script makes a single set of predictions
"""

model_loc = r'E:/CNN Models/weights.best.hdf5'
predict_data_loc = r'E:/CNN Extra Predict Data/5kb windows'
pred_save_loc = r'E:/CNN Models/Predictions'
threshold_loc = r'E:/CNN Models/best_threshold.txt'
model = keras.models.load_model(model_loc)

img_height = img_width = 128
batch_size = 32
data = load_data(predict_data_loc, img_height, img_width)

print(data)
# Retrieve threshold from file
f = open(threshold_loc, "r")
threshold = float(f.read())
f.close()


def f(x):
    return 0 if (x <= threshold) else 1


predictions = model.predict(data)
label_predictions = [f(y) for y in predictions]

# convert to str for file writing
predictions = [str(p[0]) + '\n' for p in predictions]
label_predictions = [str(p) + '\n' for p in label_predictions]


# print(pred_array)
with open(pred_save_loc + '\labels.txt', 'w') as preds, open(pred_save_loc + '\confidence.txt', 'w') as probs:
    preds.writelines(label_predictions)
    probs.writelines(predictions)

print("Predictions made")
