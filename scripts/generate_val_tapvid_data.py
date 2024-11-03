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

from scripts.kubric_point_tracking import (create_point_tracking_dataset,
                                           plot_tracks)


def read_tsv(infile, TSV_FIELDNAMES):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = TSV_FIELDNAMES)
        for item in reader:
            item['id'] = int(item['id'])
            item['video'] = np.frombuffer(base64.b64decode(item['video']), dtype=np.float32).reshape((1, 24, 256, 256, 3))
            item['query_points'] = np.frombuffer(base64.b64decode(item['query_points']), dtype=np.float32).reshape((1, 256, 3))
            item['target_points'] = np.frombuffer(base64.b64decode(item['target_points']), dtype=np.float32).reshape((1, 256, 24, 2))
            item['occluded'] = np.frombuffer(base64.b64decode(item['occluded']), dtype=bool).reshape((1, 256, 24))
            in_data.append(item)
    return in_data

ds = create_point_tracking_dataset(
      split='validation',
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

TSV_FIELDNAMES = ['id', 'video', 'query_points', 'target_points','occluded']
tsvfilename = "datasets/movi/movi_e/validation_data_temp.tsv"
with open(tsvfilename, 'wt') as tsvfile:
    writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)
    for i, item in tqdm(enumerate(data), desc="writing to tsv"):
        writer.writerow({
            "id": str(item["id"]),
            "video": str(base64.b64encode(item["video"]), "utf-8"),
            "query_points": str(base64.b64encode(item["query_points"]), "utf-8"),
            "target_points": str(base64.b64encode(item["target_points"]), "utf-8"),
            "occluded": str(base64.b64encode(item["occluded"]), "utf-8"),
        })


saved_data = read_tsv(tsvfilename, TSV_FIELDNAMES)

for i in range(len(data)):
    for key in ["id", "video", "query_points", "target_points", "occluded"]:
        flag = (saved_data[i][key] == data[i][key])
        if isinstance(flag, bool) is not True:
            flag = flag.all()
        print(i, flag)