import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds

data_save_root = "/home/ayshrv/long-range-crw/raft_crw/datasets/movi/movi_e/images_flow_data_256x256"

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip flow values to [-clip_flow, clip_flow]. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def create_dst_dir(dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        

def clean_dst_file(dst_file):
    """Create the output folder, if necessary; empty the output folder of previous predictions, if any
    Args:
        dst_file: Destination path
    """
    # Create the output folder, if necessary
    dst_file_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_file_dir):
        os.makedirs(dst_file_dir)

    # Empty the output folder of previous predictions, if any
    if os.path.exists(dst_file):
        os.remove(dst_file)

def flow_write(flow, dst_file):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(202021.25, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)
        
        
def readFlow(fn):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

whole_ds = tfds.load(
    'movi_e/256x256',
    data_dir='/home/ayshrv/turbo_datasets/movi/',
    shuffle_files=False,
)

for split in ["train", "validation"]:

    ds = tfds.as_numpy(whole_ds[split])

    data_len = len(ds)

    img_out_dir = os.path.join(data_save_root, split, "images")
    forward_flows_out_dir = os.path.join(data_save_root, split, "forward_flows")
    backward_flows_out_dir = os.path.join(data_save_root, split, "backward_flows")
    forward_flows_viz_out_dir = os.path.join(data_save_root, split, "forward_flows_viz")
    backward_flows_viz_out_dir = os.path.join(data_save_root, split, "backward_flows_viz")


    create_dst_dir(img_out_dir)
    create_dst_dir(forward_flows_out_dir)
    create_dst_dir(backward_flows_out_dir)
    create_dst_dir(forward_flows_viz_out_dir)
    create_dst_dir(backward_flows_viz_out_dir)

    for i, item in tqdm(enumerate(ds), desc=f"Running for {split}...", total=data_len):
        frames = item["video"]  # (T, H, W, 3)

        forward_flow = item["forward_flow"]  # (T, H, W, 2)
        forward_flow_range = item["metadata"]["forward_flow_range"]  # (2)
        minv = forward_flow_range[0][None, None, None, None]  # (1, 1, 1, 1)
        maxv = forward_flow_range[1][None, None, None, None]  # (1, 1, 1, 1)
        forward_flow = forward_flow / 65535 * (maxv - minv) + minv
        forward_flow = np.flip(forward_flow, 3)
        forward_flow = forward_flow[:-1]  # (T-1, H, W, 2)

        backward_flow = item["backward_flow"]  # (T, H, W, 2)
        backward_flow_range = item["metadata"]["backward_flow_range"]  # (2)
        minv = backward_flow_range[0][None, None, None, None]  # (1, 1, 1, 1)
        maxv = backward_flow_range[1][None, None, None, None]  # (1, 1, 1, 1)
        backward_flow = backward_flow / 65535 * (maxv - minv) + minv
        backward_flow = np.flip(backward_flow, 3)
        backward_flow = backward_flow[1:]  # (T-1, H, W, 2)
            
        for t in range(frames.shape[0]):
            Image.fromarray(frames[t]).save(f"{img_out_dir}/{i:04}_{t:02}.png")
        for t in range(forward_flow.shape[0]):
            flow_write(forward_flow[t], f"{forward_flows_out_dir}/{i:04}_{t:02}.flo")
        for t in range(backward_flow.shape[0]):
            flow_write(backward_flow[t], f"{backward_flows_out_dir}/{i:04}_{t:02}.flo")
        for t in range(forward_flow.shape[0]):
            Image.fromarray(flow_to_image(forward_flow[t])).save(f"{forward_flows_viz_out_dir}/{i:04}_{t:02}.png")
        for t in range(backward_flow.shape[0]):
            Image.fromarray(flow_to_image(backward_flow[t])).save(f"{backward_flows_viz_out_dir}/{i:04}_{t:02}.png")
        