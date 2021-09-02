# Title      :	predict_img_gen.py
# Objective  :	Refactor the prediction image script for use within the CNN chip-seq tool
# Created by :	Luke
# Created on :	Thu 02/09/21 14:59


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

plt.style.use("ggplot")


def make_prediction_imgs(chromosome_number, data_path):
    if chromosome_number == 1:
        num = 'I'
    elif chromosome_number == 2:
        num = 'II'
    elif chromosome_number == 3:
        num = 'III'

    raw_data_path = data_path.joinpath("raw_data")
    chip_data = pd.read_csv(raw_data_path.joinpath(
        f"pombe_rif1_chip_chr{num}.csv"))
    origin_rif = pd.read_csv(raw_data_path.joinpath(
        f"pombe_origins_rif1d_chr{num}.csv"))
    origin_wt = pd.read_csv(raw_data_path.joinpath(
        f"pombe_origins_wt_chr{num}.csv"))

    chromo_length = len(chip_data)
    window_size = 5000
    jump_size = window_size // 2
    save_path = data_path.joinpath("predict_data")
    try:
        Path.mkdir(save_path, parents=True)
    except FileExistsError:
        pass

    max_img_count = len(
        [x for x in range(0, chromo_length-window_size+1, jump_size)])
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
            plt.savefig(save_path.joinpath(f"{img_cnt}.png"))
            img_cnt += 1
            plt.clf()
            plt.close()
            pbar.update(1)
