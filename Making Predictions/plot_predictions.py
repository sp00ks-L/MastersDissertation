# Title      :	plot_predictions.py
# Objective  :	A script to plot the predictions made by the model
# Created by :	Luke
# Created on :	Sun 11/07/21 20:57

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')


chip_data = pd.read_csv(
    rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_rif1_chip_chrI.csv")
peak_data = pd.read_csv(
    r'C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_rif1_peakcaller_chrI.csv')
dir = r'E:\CNN Models\Predictions'
label_preds = pd.read_csv(dir + r'\labels.csv')
label_probs = pd.read_csv(dir + r'\confidence.csv')
threshold_loc = r'E:\CNN Models\best_threshold.txt'
f = open(threshold_loc, "r")
threshold = float(f.read())
f.close()

max_dist = max([float(_) for _ in label_probs['mean']]) - threshold

xs = [x for x in range(len(chip_data))]
label_preds_list = list(label_preds['binding_site'].astype(int))
label_probs_list = [float(prob)
                    for prob in label_probs['mean'] if prob > threshold]

slice_size = 110000
window_size = 5000
jump_size = window_size // 2
# Finding start and end index for each binding prediction area

binding_regions = []
binding_img_indices = np.where(np.array(label_preds_list) == 1)[0]
binding_img_indices *= jump_size

# Expand binding regions to encompass the 5kb window
for start_ind in binding_img_indices:
    if start_ind >= slice_size:
        break
    end_ind = start_ind + window_size
    if end_ind <= slice_size:  # if binding region fits within data slice
        binding_regions.append([start_ind, end_ind])
    else:
        # if binding region doesnt fit in slice, we need to clip the region
        binding_regions.append([start_ind, slice_size])


# Find the peak regions from the peak-caller
peaks = []
for i in range(len(peak_data)):
    peak_start = peak_data['start'][i]
    peak_end = peak_data['end'][i]
    if peak_data['start'][i] >= slice_size:
        break
    if peak_start < slice_size and peak_end <= slice_size:
        peaks.append([peak_start, peak_end])
    else:
        peaks.append([peak_start, slice_size])


plt.figure(figsize=(10, 7))
sns.lineplot(x=xs[:slice_size], y=chip_data['norm'][:slice_size])
label_added = False
for peak in peaks:
    if not label_added:
        plt.fill_between(peak, y1=1, y2=5, color='blue',
                         alpha=0.2, label='Original Peak')
        label_added = True
    else:
        plt.fill_between(peak, y1=1, y2=5, color='blue', alpha=0.2)

label_added = False
for i in range(len(binding_regions)):
    conf_level = label_probs_list[i] - threshold
    conf_level_pct = conf_level / max_dist
    if not label_added:
        plt.fill_between(binding_regions[i], y1=2, y2=4, color='green',
                         alpha=conf_level_pct, label='Candidate Binding Site')
        label_added = True
    else:
        plt.fill_between(binding_regions[i], y1=2,
                         y2=4, color='green', alpha=conf_level_pct)
plt.xlabel("Chromosome Position", fontsize=20)
plt.ylabel("Normalised ChIP Data", fontsize=20)
plt.title('5kb Chromosome Predictions w/ Confidence Levels', fontsize=30)
plt.tick_params(labelsize=20)
plt.legend(loc='best', fontsize=20)
plt.show()
