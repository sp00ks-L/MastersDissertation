# Title      :	chromo_sampler.py
# Objective  :	Create a script to efficiently sample images around origin regions of the chromosome provided
# Created by :	Luke
# Created on :	Thu 01/07/21 09:37

import glob
from os import remove, walk
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .helpers import count_origins, get_origin_locs

plt.style.use('ggplot')

# Importing chromosome and origin data for Chromosome
"""
Train - Chr 2
Test - Chr 1
Validation - Chr 3

My general folder structure was as follows:
Train
  |
   ------ Binding
   ------ Non-binding
Test
  |
   ------ Binding
   ------ Non-binding
Validation
  |
   ------ Binding
   ------ Non-binding
"""


def clear_subfolder(folder_path, binding=True):
    # Function to clear binding and non-binding subfolders
    if binding:
        local_path = folder_path.joinpath("Binding", "*")
    else:
        local_path = folder_path.joinpath("Non-binding", "*")
    files = glob.glob(str(local_path))
    for f in files:
        remove(f)


def clear_folders(sample_dir):
    # Quicker way to clear current image data to produce new data
    train_dir = sample_dir.joinpath("Train")
    test_dir = sample_dir.joinpath("Test")
    val_dir = sample_dir.joinpath("Validate")

    dirs = [train_dir, test_dir, val_dir]

    for dir in dirs:
        # Iterates through train, test, val and clears subfolders
        for bool in [True, False]:
            clear_subfolder(dir, binding=bool)

    print("Folder(s) cleared")


def get_chromo_data(data_path, chromosome_number):
    if chromosome_number == 1:
        num = 'I'
    elif chromosome_number == 2:
        num = 'II'
    elif chromosome_number == 3:
        num = 'III'

    chip_path = data_path.joinpath("raw_data", f"pombe_rif1_chip_chr{num}.csv")
    label_path = data_path.joinpath("model_data", f"chr{chromosome_number}_labels.csv")

    chip_data = pd.read_csv(chip_path)
    chr_labels = pd.read_csv(label_path)

    origin_labels = list(chr_labels['# Labels'])

    return chip_data, origin_labels



def chromo_sampler(data_path, sample_size, window_size=5000, chromosome_number=2):
    assert int(chromosome_number)
    chip_data, origin_labels = get_chromo_data(data_path, chromosome_number)
    if chromosome_number == 1:
        out_dir = data_path.joinpath("model_data", "Test")
    elif chromosome_number == 2:
        out_dir = data_path.joinpath("model_data", "Train")
    elif chromosome_number == 3:
        out_dir = data_path.joinpath("model_data", "Validate")
    else:
        raise ValueError(f"There is no Chromosome {chromosome_number}")

    binding_cnt = 0
    non_binding_cnt = 0
    extend_range = False
    # This function returns a list of starting and ending indices for the origin labels
    origin_start, origin_end = get_origin_locs(origin_labels)
    walk_path = walk(out_dir)

    img_collection = []

    # Creates a list of imgs that are already produced
    for _, _, files in walk_path:
        for file in files:
            img_collection.append(int(file[3:-4]))

    # tqdm is a way to easily implement a progress bar
    with tqdm(total=sample_size*2, ncols=100, desc=f"Chromosome {chromosome_number}") as pbar:
        while binding_cnt != sample_size or non_binding_cnt != sample_size:
            if binding_cnt == sample_size and not extend_range:
                """
                My sampling system is 'directed' in that it is given a window around an origin of interest to sample
                This significantly reduces the required time to produce binding images - my method is explained stepwise below
                Once we have all the necessary binding images, this sub-routine effectively extends the search area to encompass a wider section of the genome (-+ 5kb). Because the search area is wider, it now favours sampling regions that do not contain origins of interest (non-binding imgs)
                """
                # Could have used numpy broadcasting
                origin_start = [x - 5000 for x in origin_start]
                origin_end = [x + 5000 for x in origin_end]
                extend_range = True

            """
            1. rnd_origin_index: picks random int between 0 and the number of origins - this in essence randomly chooses an origin out of the population of origins
                - by chosing random slices around an origin of interest then the sampling does not require as long to produce the required number of binding images if this was truly random
            2. img_sample_start: selects a random int between the starting and ending indices of the selected origin
            3. Create 5kb window by adding 5000 to img_sample_start 
            4. Slice this 5kb window from the label data
            5. If this 5kb slice has >= 1 origin --> label as binding: else --> label as non-binding
            """
            rnd_origin_index = np.random.randint(0, len(origin_start))
            img_sample_start = np.random.randint(
                origin_start[rnd_origin_index], origin_end[rnd_origin_index])
            if img_sample_start in img_collection:  # this prevents duplicate imgs being produced
                continue
            img_collection.append(img_sample_start)
            img_sample_end = img_sample_start + window_size

            label_region = origin_labels[img_sample_start:img_sample_end]

            if not count_origins(label_region) and non_binding_cnt != sample_size:
                binding_site = False
            elif count_origins(label_region) >= 1 and binding_cnt != sample_size:
                binding_site = True
            else:
                # If any edge cases are caught here, they will not be plotted to save uncessary computation
                continue

            # Extract same 5kb slice from the chip data
            data_slice = list(
                chip_data['norm'][img_sample_start:img_sample_end])
            xs = [_ for _ in range(img_sample_start, img_sample_end)]

            # Plotting
            # size somewhat arbitrary. imgs scaled down within model architecture
            plt.figure(figsize=(5, 4))
            sns.lineplot(x=xs, y=data_slice, lw=1,
                            color='#257ab5')  # darker blue
            # lighter blue
            plt.fill_between(xs, data_slice, color='#81b0d4')
            # Used the below line to test how scaling the y-axis influenced the model
            # plt.ylim(chip_data.norm.min(), chip_data.norm.max())
            plt.axis('off')
            if binding_site:
                plt.savefig(out_dir.joinpath("Binding", f"img{img_sample_end}.png"),
                            bbox_inches='tight',
                            pad_inches=0)
                binding_cnt += 1
                pbar.update(1)
            else:
                plt.savefig(out_dir.joinpath("Non-binding", f"img{img_sample_end}.png"),
                            bbox_inches='tight',
                            pad_inches=0)
                non_binding_cnt += 1
                pbar.update(1)
            # Due to repeated creation of plots, this clears memory each time to avoid python throwing warnings
            plt.clf()
            plt.close()
