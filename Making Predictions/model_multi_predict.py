# Title      :	model_multi_predict.py
# Objective  :	Create a script that invokes model_predict2 to create an 'average' prediction
# Created by :	Luke
# Created on :	Mon 02/08/21 12:25


from os import remove
from pathlib import Path
import keras
import pandas as pd
import model_predict2
from load_img_data import load_data

"""
When set to 1, will perform the same as the original model_predict.py
Currently setting this value above one does not yield any advantage
It will just produce multiple predictions that are equivalent
i.e for image 1, the predictions will be 0.33, 0.33, 0.33... eyc
"""
n_predictions = 5

predict_data_loc = r'E:/CNN Extra Predict Data/5kb windows'
conf_file = r'E:/CNN Models/Predictions/confidence.csv'
pred_file = r'E:/CNN Models/Predictions/labels.csv'
threshold_file = r'E:/CNN Models/best_threshold.txt'


def load_model():
    model_loc = r'E:/CNN Models/weights.best.hdf5'
    model = keras.models.load_model(model_loc)
    return model


def clear_files(conf_file, pred_file):
    conf_path = Path(conf_file)
    pred_path = Path(pred_file)
    if conf_path.exists() and conf_path.stat().st_size >= 0:
        try:
            remove(conf_file)
        except PermissionError:
            print(
                "\nThe confidence.txt file is still open. Please close the file and re-run the script")
            exit()
    if pred_path.exists() and pred_path.stat().st_size >= 0:
        try:
            remove(pred_path)
        except PermissionError:
            print(
                "\nThe labels.txt file is still open. Please close the file and re-run the script")
            exit()


img_height = img_width = 128
clear_files(conf_file, pred_file)
model = load_model()
data = load_data(predict_data_loc, img_height, img_width)
f = open(threshold_file, "r")
threshold = float(f.read())
f.close()


for i in range(n_predictions):
    # this calls the function in model_predict2
    model_predict2.main(model, data)


conf_data = pd.read_csv(conf_file)
pred_data = pd.read_csv(pred_file)

conf_data['mean'] = conf_data.mean(axis=1)
conf_data.to_csv(conf_file, index=False)


def f(x): return 0 if x < 1 else 1


# A way to filter sites that have been called repeatedly. e.g set threshold to 8/10 times peak must be called
pred_data['binding_site'] = pred_data.sum(axis=1) >= (n_predictions * 0.8)
# This is a primitive mean of multiple predictions
pred_data['binding_site2'] = pred_data.mean(axis=1) >= 0.8

final_preds = [f(x) for x in pred_data['binding_site'].astype(
    int) + pred_data['binding_site2'].astype(int) - 1]
pred_data['final_binding_site'] = final_preds
# pred_data['Binding Site'] = pred_data['Binding Site'].astype(int)
pred_data.to_csv(pred_file, index=False)

print("Predictions and Confidence levels produced")
