# Title      :	load_img_data.py
# Objective  :	Create a set of functions for loading and standardising the image data that needs to be loaded for training a CNN
# Created by :	Luke
# Created on :	Tue 06/07/21 14:19


from os import walk

import numpy as np
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img


def load_data(dir, img_height, img_width):
    num_observations = len(next(walk(dir.joinpath("Binding")))[2])
    classes = ["Binding", "Non-binding"]
    # Pre-allocate
    x = []
    y = np.zeros((num_observations*2, 1))
    # Pre-shuffle, we know the first part of the array will only be binding imgs
    # Therefore we can set the first 'num_observations' = 1
    y[:num_observations] = 1

    for c in (classes):
        tmp_path = dir.joinpath(c)
        img_list = next(walk(tmp_path))[2]

        # Re-scale img values from 0-255 to 0.0-1.0
        for i in range(num_observations):
            img = load_img(tmp_path.joinpath(img_list[i]))
            img = img.resize((img_height, img_width))
            img_arr = img_to_array(img)
            img_arr /= 255.0
            x.append(img_arr)

    x = np.array(x)
    # Shuffle each array
    rnd_indices = np.arange(num_observations * 2)
    np.random.shuffle(rnd_indices)
    x = x[rnd_indices]
    y = y[rnd_indices]

    print(f"\nLoaded {2*num_observations} images from {dir}")
    return x, y
