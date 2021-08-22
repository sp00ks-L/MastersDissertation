# Title      :	model_predict2.py
# Objective  :	Create a script that aggregates several predictions from the model
# Created by :	Luke
# Created on :	Thu 29/07/21 12:49

import csv
import sys
from os import remove
from pathlib import Path
import keras
import pandas as pd


def load_model():
    model_loc = r'E:/CNN Models/weights.best.hdf5'
    model = keras.models.load_model(model_loc)
    return model


def clear_file(filename):
    # If prediction file exists, clear it
    my_path = Path(filename)
    if my_path.exists():
        remove(filename)


def main(model, data):
    pred_file = r'E:/CNN Models/Predictions/labels.csv'
    threshold_file = r'E:/CNN Models/best_threshold.txt'
    conf_file = r'E:/CNN Models/Predictions/confidence.csv'

    # Retrieve threshold from file
    f = open(threshold_file, "r")
    threshold = float(f.read())
    f.close()
    # Threshold function
    def f(x): return 0 if (x <= threshold) else 1

    predictions = [p for p in model.predict(data)]
    label_predictions = [[f(y)] for y in predictions]

    # Check if confidence file already present
    conf_path = Path(conf_file)
    pred_path = Path(pred_file)
    if conf_path.exists() and conf_path.stat().st_size > 0:
        my_data = pd.read_csv(conf_file)
        tmp_df = pd.DataFrame(predictions)
        my_data = pd.concat([my_data, tmp_df], axis=1)
        my_data.columns = [x for x in range(len(my_data.columns))]
        my_data.to_csv(conf_file, index=False)
    if not conf_path.exists():
        with open(conf_file, 'w') as out_file:
            writer = csv.writer(out_file, lineterminator='\n')
            # Write inital header
            writer.writerow([0])
            writer.writerows(predictions)

    if pred_path.exists() and pred_path.stat().st_size > 0:
        my_data = pd.read_csv(pred_file)
        tmp_df = pd.DataFrame(label_predictions)
        my_data = pd.concat([my_data, tmp_df], axis=1)
        my_data.columns = [x for x in range(len(my_data.columns))]
        my_data.to_csv(pred_file, index=False)
    if not pred_path.exists():
        with open(pred_file, 'w') as out_file:
            writer = csv.writer(out_file, lineterminator='\n')
            # Write inital header
            writer.writerow([0])
            writer.writerows(label_predictions)


if __name__ == '__main__':
    sys.exit(main())
