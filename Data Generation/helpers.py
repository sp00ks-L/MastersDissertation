# Title     : helpers.py
# Objective : Collate helper functions to avoid repetition
# Created by: Luke
# Created on: 27/06/2021 12:25


def count_origins(data):
    # Accepts the overlap data. Simple count of how many distinct origins are present
    # Currently only counts and origin if ALL of the origin is present in the window
    # Adapt it to accept a percentage of the origin being present i.e. 80% origin present =
    cnt = 0
    for i in range(len(data) - 1):
        if data[i] and not data[i + 1]:
            # If current is 1 and next is 0, end of of origin region
            cnt += 1
    return cnt


def get_origin_locs(origin_labels):
    """
    This function retrieves the index of the start and end of an origin
    it then 'extends' this window by performing start - 5000 and end + 2000
    this widens the window around the origin so it can then be passed to my sampler
    by providing genomic regions like this, we have a relatively wide region with a good chance
    of it including an origin of interest
    """
    curr = 0
    index_start = []
    index_end = []

    for i in range(len(origin_labels)):
        if origin_labels[i] and not curr:
            curr = 1
            index_start.append(i - 5000)
        elif not origin_labels[i] and curr:
            curr = 0
            index_end.append(i + 2000)

    return index_start, index_end
