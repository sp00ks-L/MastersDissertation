# Title     : data_labelling.py
# Objective : Derive a way to label regions of the genome as Rif1 binding or non-binding
# Created by: Luke
# Created on: 24/08/2021 14:31

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_labels(data_dir, chromosome_number=2):
    # Sort out directories
    if chromosome_number == 1:
        num = 'I'
    elif chromosome_number == 2:
        num = 'II'
    elif chromosome_number == 3:
        num = 'III'
    else:
         raise ValueError ("There are only 3 chromosomes. Enter 1, 2, or 3")

    chip_path = data_dir.joinpath("raw_data", f"pombe_rif1_chip_chr{num}.csv")
    rif_origins_path = data_dir.joinpath("raw_data", f"pombe_origins_rif1d_chr{num}.csv")
    wt_origins_path = data_dir.joinpath("raw_data", f"pombe_origins_wt_chr{num}.csv")
    chip_data = pd.read_csv(chip_path)
    rif_origins = pd.read_csv(rif_origins_path)
    wt_origins = pd.read_csv(wt_origins_path)

    # Allows for selecting part of chromosome to label or entire chromosome
    start = 0
    end = len(chip_data)
    # for testing purposes
    # end = 300000 

    # Extracts origin data
    wt_origin_starts = list(wt_origins['xmin'])
    rif_origin_starts = list(rif_origins['xmin'])
    wt_origin_ends = list(wt_origins['xmax'])
    rif_origin_ends = list(rif_origins['xmax'])
    # Origin Efficiency
    wt_efficiency = list(wt_origins['efficiency'])
    rif_efficiency = list(rif_origins['efficiency'])

    # Pre-allocate arrays
    origin_wt_locs = [0] * (end - start)
    origin_rif_locs = [0] * (end - start)

    # Produce data
    datarange = [x for x in range(start, end)]
    with tqdm(total=(len(wt_origin_starts) + len(rif_origin_starts)), ncols=100, desc=f"Chromosome {num} labels") as pbar:
        for wt_start, wt_end, wt_eff in zip(wt_origin_starts, wt_origin_ends, wt_efficiency):
            if wt_start in datarange and wt_end in datarange:
                for coord in range(wt_start - start, wt_end - start):
                    origin_wt_locs[coord] = wt_eff
            pbar.update(1)    
        for rif_start, rif_end, rif_eff in zip(rif_origin_starts, rif_origin_ends, rif_efficiency):
            if rif_start in datarange and rif_end in datarange:
                for coord in range(rif_start - start, rif_end - start):
                    origin_rif_locs[coord] = rif_eff
            pbar.update(1)

    overlap_data = [0] * len(origin_wt_locs)
    for i in range(len(origin_wt_locs)):
        # I experimented with the below lower threshold
        if origin_wt_locs[i] <= 0.005 and origin_rif_locs[i] >= 0.2:
            overlap_data[i] = 1

    overlap_data = np.asarray(overlap_data).reshape(-1, 1)

    output_path = data_dir.joinpath("model_data", f"chr{chromosome_number}_labels.csv")
    np.savetxt(output_path, overlap_data, fmt='%10.5f', delimiter=',', header="Labels")



