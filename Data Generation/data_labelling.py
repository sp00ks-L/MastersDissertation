# Title     : data_labelling.py
# Objective : Derive a way to label regions of the genome as Rif1 binding or non-binding
# Created by: Luke
# Created on: 20/06/2021 17:00

import numpy as np
import pandas as pd


def generate_labels(chromosome_number=2):
    # Sort out directories
    if chromosome_number == 1:
        num = 'I'
    elif chromosome_number == 2:
        num = 'II'
    elif chromosome_number == 3:
        num = 'III'
    else:
        # probably not required
        assert ("There are only 3 chromosomes. Enter 1, 2, or 3")

    chip_data = pd.read_csv(
        rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_rif1_chip_chr{num}.csv")
    rif_origins = pd.read_csv(
        rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_origins_rif1d_chr{num}.csv")
    wt_origins = pd.read_csv(
        rf"C:\Users\Luke\Sussex Code\Sussex Python\Dissertation\Data\Complete Data\pombe_origins_wt_chr{num}.csv")

    # Allows for selecting part of chromosome to label or entire chromosome
    start = 0
    end = len(chip_data)

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
    for wt_start, wt_end, wt_eff in zip(wt_origin_starts, wt_origin_ends, wt_efficiency):
        if wt_start in datarange and wt_end in datarange:
            for coord in range(wt_start - start, wt_end - start):
                origin_wt_locs[coord] = wt_eff
    for rif_start, rif_end, rif_eff in zip(rif_origin_starts, rif_origin_ends, rif_efficiency):
        if rif_start in datarange and rif_end in datarange:
            for coord in range(rif_start - start, rif_end - start):
                origin_rif_locs[coord] = rif_eff

    overlap_data = [0] * len(origin_wt_locs)
    for i in range(len(origin_wt_locs)):
        # I experimented with the below lower threshold
        if origin_wt_locs[i] <= 0.005 and origin_rif_locs[i] >= 0.2:
            overlap_data[i] = 1

    overlap_data = np.asarray(overlap_data).reshape(-1, 1)

    np.savetxt(f"chr{chromosome_number}_labels.csv", overlap_data,
               fmt='%10.5f', delimiter=',', header="Labels")

    return overlap_data


"""
I included the count_origins function just to see roughly how many labels my method was producing
I was aiming for hundreds per chromosome rather than thousands of candidate binding regions
it is a copy to the one found in the helpers.py file - just had some issues importing it so I included it locally
"""


def count_origins(data):
    # Accepts the overlap data. Simple count of how many distinct origins are present
    cnt = 0
    for i in range(len(data)):
        if data[i] and not data[i + 1]:
            # If current is 1 and next is 0, end of of origin region
            cnt += 1
    return cnt


# Produce data for all 3 chromosomes
overlap_data1 = generate_labels(1)
print(count_origins(overlap_data1))

overlap_data2 = generate_labels(2)
print(count_origins(overlap_data2))

overlap_data3 = generate_labels(3)
print(count_origins(overlap_data3))
