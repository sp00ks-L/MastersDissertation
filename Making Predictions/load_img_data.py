# Title      :	load_img_data.py
# Objective  :	An updated img loading script for use specifically with the prediction images
# Created by :	Luke
# Created on :	Sun 11/07/21 18:37


from os import walk

import numpy as np
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img


def load_data(dir, img_height, img_width):
    print("Loading Images")
    num_observations = len(next(walk(dir))[2])
    # Pre-allocate
    x = []

    arr_ind = 0
    img_list = next(walk(dir))[2]

    for i in range(num_observations):
        img = load_img(dir + r'\\' + img_list[i])
        img = img.resize((img_height, img_width))
        img_arr = img_to_array(img)
        img_arr /= 255.0
        x.append(img_arr)

    x = np.array(x)
    # Shuffle each array
    # rnd_indices = np.arange(num_observations)
    # np.random.shuffle(rnd_indices)
    # x = x[rnd_indices]

    print(f"Loaded {num_observations} images")
    return x
