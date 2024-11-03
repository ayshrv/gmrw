import base64
import csv

# import torch
import functools
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")

from scripts.kubric_point_tracking import create_point_tracking_dataset, plot_tracks


def read_tsv(infile, TSV_FIELDNAMES):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for item in reader:
            item["id"] = int(item["id"])
            item["video"] = np.frombuffer(
                base64.b64decode(item["video"]), dtype=np.float32
            ).reshape((1, 24, 256, 256, 3))
            item["query_points"] = np.frombuffer(
                base64.b64decode(item["query_points"]), dtype=np.float32
            ).reshape((1, 256, 3))
            item["target_points"] = np.frombuffer(
                base64.b64decode(item["target_points"]), dtype=np.float32
            ).reshape((1, 256, 24, 2))
            item["occluded"] = np.frombuffer(
                base64.b64decode(item["occluded"]), dtype=bool
            ).reshape((1, 256, 24))
            in_data.append(item)
    return in_data


ds = create_point_tracking_dataset(
    split="validation",
    batch_dims=[1],
    shuffle_buffer_size=None,
    repeat=False,
    vflip=False,
    random_crop=False,
)
ds = tfds.as_numpy(ds)

data = []
for i, item in tqdm(enumerate(ds), desc="adding to data"):
    item["id"] = i
    data.append(item)

out_folder = "datasets/movi/movi_e/images_data_256x256/"

for i, item in tqdm(enumerate(data), desc="writing images"):
    images = item["video"][0]  # (T, H, W, 3)
    images = (images + 1) / 2.0
    for j in tqdm(range(images.shape[0])):
        file_path = out_folder + str(i) + "_" + str(j) + ".png"
        plt.imsave(
            file_path,
            images[j],
            format="png",
        )
