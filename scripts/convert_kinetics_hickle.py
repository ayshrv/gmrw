import hickle as hkl
import h5py
from glob import glob
import mediapy as media
import numpy as np
import cv2
import time
import os 
import os.path as osp
from tqdm import tqdm
import multiprocessing
import functools
import argparse


def apply_png_across(imgs, dtype=np.uint8, axis=0):
    """
    applies png compression across given axis (only supports 0 for now)
    Args:
        imgs: batch of images [B, H, W, C]
        dtype: numpy dtype to convert to (np.int16 or np.int8)
    """
    assert axis==0, "Currently only supports axis=0"
    assert imgs.shape[3]==3, f"imgs shape is incorrect. Current shape: {imgs.shape}, expected shape[3] to be 3."
    assert isinstance(imgs, np.ndarray)
    imgs=imgs.astype(dtype)
    pnglist=[]
    for i in range(imgs.shape[axis]):
        _, buffer = cv2.imencode(".png",imgs[i],[cv2.IMWRITE_PNG_COMPRESSION , 3]) # compression str = 3 for png
        pnglist.append(buffer)
    
    return pnglist


def apply_jpg_across(imgs, dtype=np.uint8, axis=0):
    """
    applies jpg compression across given axis (only supports 0 for now)
    Args:
        imgs: batch of images [B, H, W, C]
        dtype: numpy dtype to convert to (np.int16 or np.int8)
    """
    assert axis==0, "Currently only supports axis=0"
    assert imgs.shape[3] == 3, f"imgs shape is incorrect. Current shape: {imgs.shape}, expected shape[3] to be 3."
    assert isinstance(imgs, np.ndarray)
    imgs=imgs.astype(dtype)
    jpglist=[]
    for i in range(imgs.shape[axis]):
        _, buffer = cv2.imencode(".jpg", imgs[i], [cv2.IMWRITE_JPEG_QUALITY, 95]) # compression str DEFAULT = 95 for jpg
        jpglist.append(buffer)
    
    return jpglist


def retrieve_tensor(buffers, dtype=np.uint8):
    """
    Retrieve image from buffer.
    """
    img = cv2.imdecode(buffers[0], -1)
    H, W, C = img.shape
    imgs = np.empty((len(buffers), H, W, C), dtype=img.dtype)
    imgs[0] = img
    for i in range(1, len(buffers)):
        imgs[i] = cv2.imdecode(buffers[i], -1)
    
    imgs.astype(dtype)    
    return imgs


def verify_load_hkl():
    video_path = "/home/ayshrv/Downloads/_JM4Bj-JjJs_000008_000018.mp4"
    hkl_path = "/home/ayshrv/Downloads/_JM4Bj-JjJs_000008_000018.hkl"


    frames = media.read_video(video_path)
    frames = np.array(frames)
    frames = frames[::10]
    frames_jpg = apply_jpg_across(frames)
    # hkl.dump(frames, hkl_path, mode='w', compression='lzf')

    array_hkl = hkl.load(hkl_path)
    hkl_frames = retrieve_tensor(array_hkl)
    print(frames.shape)
    print(hkl_frames.shape)

    flag = np.all([np.all(frames_jpg[i] == array_hkl[i]) for i in range(len(array_hkl))])
    print(flag)


def preprocess_mp4_to_hkl(input_data_root, output_data_root, folder):
    # data_root = "/home/ayshrv/turbo_nobackup_datasets/kinetics700-2020/train/"
    # folder = "abseiling"
    # new_data_root = f"/home/ayshrv/turbo_nobackup_datasets/kinetics700-2020-processed/train/{folder}"

    folder_path = osp.join(output_data_root, folder)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    video_list = sorted(glob(osp.join(input_data_root, folder, "*.mp4")))

    for i, video_id in tqdm(enumerate(video_list), total=len(video_list)):
        video_name = video_id.split("/")[-1]
        video_name = video_name.replace(".mp4", ".hkl")

        hkl_path = osp.join(output_data_root, folder, video_name)        

        if os.path.isfile(hkl_path):
            continue
        
        try:
            frames = media.read_video(video_id)
        except:
            print(f"Error in video id {video_id}")
            continue

        frames = np.array(frames)
        frames = frames[::10]
        frames = apply_jpg_across(frames)

        hkl.dump(frames, hkl_path, mode='w', compression='lzf')
    
    print(f"{input_data_root}/{folder} done!")


def compare_load_times():
    folder = "abseiling"

    mp4_data_root = f"/home/ayshrv/turbo_nobackup_datasets/kinetics700-2020/train/{folder}"
    hkl_data_root = f"/home/ayshrv/turbo_nobackup_datasets/kinetics700-2020-processed/train/{folder}"

    mp4_video_list = sorted(glob(osp.join(mp4_data_root, "*.mp4")))

    print(f"MP4 Files: {len(mp4_video_list)}")
    t1 = time.time()
    for i, video_id in tqdm(enumerate(mp4_video_list)):
        frames = media.read_video(video_id)
        frames = np.array(frames)

    t2 = time.time()

    print(f"{t2-t1} secs")


    hkl_video_list = sorted(glob(osp.join(hkl_data_root, "*.hkl")))
    print(f"HKL Files: {len(hkl_video_list)}")

    t1 = time.time()
    for i, video_id in tqdm(enumerate(hkl_video_list)):
        array_hkl = hkl.load(video_id)
        hkl_frames = retrieve_tensor(array_hkl)

    t2 = time.time()

    print(f"{t2-t1} secs")


def preprocess_kinetics_folder(folder_i):
    mp4_data_root = f"/home/ayshrv/turbo_nobackup_datasets/kinetics700-2020/train/"
    hkl_data_root = f"/home/ayshrv/turbo_nobackup_datasets/kinetics700-2020-processed/train/"

    folders = sorted(glob(osp.join(mp4_data_root, "*")))
    folder_names = [osp.basename(folder) for folder in folders]

    folder_name = folder_names[folder_i]
    print(f"Processing {folder_name}...")

    preprocess_mp4_to_hkl(
        input_data_root=mp4_data_root, 
        output_data_root=hkl_data_root,
        folder=folder_name,
    )

    print(f"{folder_name} done")

    # partial_func = functools.partial(
    #     preprocess_mp4_to_hkl,
    #     input_data_root=mp4_data_root, 
    #     output_data_root=hkl_data_root,
    # )

    # pool = multiprocessing.Pool(processes=n_pool)

    # list(
    #     tqdm(
    #         pool.imap_unordered(
    #             partial_hickle_video, 
    #             folder_names,
    #         ), 
    #         total=len(folder_names)
    #     )
    # )

parser = argparse.ArgumentParser(description="")
parser.add_argument("--folder_id", default=0, type=int, help="")

args = parser.parse_args()

preprocess_kinetics_folder(args.folder_id)