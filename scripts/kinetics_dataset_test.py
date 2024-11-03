import os
import argparse
import os.path as osp
from glob import glob
import mediapy as media
from tqdm import tqdm

data_root = "datasets/kinetics700-2020"
split = "train"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--local_id", default=0, type=int, help="")

args = parser.parse_args()

video_list = sorted(glob(osp.join(data_root, split, "*/*.mp4")))

print(f"Total Length: {len(video_list)}") # 536499

# 536499 / 20000 = 26.82495

video_list = video_list[20000*args.local_id:20000*(args.local_id+1)]

print(f"Trimmed Length: {len(video_list)}")

ignored_ids = []

for video_id in tqdm(video_list):
    try:
        frames = media.read_video(video_id)
    except:
        print(f"Error in video id {video_id}")
        ignored_ids.append(video_id)

print(f"\nIgnored ids:")
print(ignored_ids)

with open(f"scripts/kinetics_dataset_log_id{args.local_id}.log", 'w+') as f:
    for items in ignored_ids:
        f.write('%s\n' %items)

    print(f"scripts/kinetics_dataset_log_id{args.local_id}.log")