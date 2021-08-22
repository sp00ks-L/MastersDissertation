# Title      :	Images2Predict.py
# Objective  :	Create a script that will generate images sequentially along a chromosome (chromosome I) for my model to predict
# Created by :	Luke
# Created on :	Sun 11/07/21 16:29


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

plt.style.use('ggplot')

# Importing chromosome and origin data for Chromosome
# I typically used Chromosome 1 for my prediction data
chromosome_number = 1
if chromosome_number == 1:
    num = 'I'
elif chromosome_number == 2:
    num = 'II'
elif chromosome_number == 3:
    num = 'III'

chip_data = pd.read_csv(
    rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_rif1_chip_chr{num}.csv")
origin_rif = pd.read_csv(
    rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_origins_rif1d_chr{num}.csv")
origin_wt = pd.read_csv(
    rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_origins_wt_chr{num}.csv")


chromo_length = len(chip_data)

"""
Plot the norm chip data in 5kb slices
Start with taking a picture every 2.5kb :: 0 - 5kb -> 2.5kb - 7.5kb
"""


window_size = 5000
jump_size = window_size // 2

save_dir = r'E:\CNN Predict Data\5kb windows'
# The number of imgs to produce will depend on the selected window_size
max_img_count = len(
    [x for x in range(0, chromo_length-window_size+1, jump_size)])
print(f"{max_img_count} total images to produce\n")
img_cnt = 0
with tqdm(total=max_img_count, ncols=100, desc="Producing Images") as pbar:
    # adjust loop so window size fits into chromosome: not out of index error
    for i in range(0, (chromo_length-window_size+1), jump_size):
        data_slice = list(chip_data['norm'][i:i + window_size])
        xs = [_ for _ in range(i, i+window_size)]
        # Would be interesting to vary this size, technically my model was trained on imgs that were a lot smaller
        plt.figure(figsize=(5, 4))
        sns.lineplot(x=xs, y=data_slice, lw=1, color='#257ab5')
        plt.fill_between(xs, data_slice, color='#81b0d4')
        plt.axis('off')
        plt.savefig(save_dir + rf'\{img_cnt}.png')
        img_cnt += 1
        plt.clf()
        plt.close()
        pbar.update(1)
