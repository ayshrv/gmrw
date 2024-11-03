import io
import os
from os import path
import pickle
import random
import logging
from glob import glob
from typing import Iterable, Mapping, Optional, Tuple, Union, Sequence

from PIL import Image
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_datasets as tfds
import mediapy as media
import torch

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]

TRAIN_SIZE = (24, 256, 256, 3)

def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
  """Resize a video to output_size."""
  # If you have a GPU, consider replacing this with a GPU-enabled resize op,
  # such as a jitted jax.image.resize.  It will make things faster.
  return media.resize_video(video, TRAIN_SIZE[1:3])

def convert_grid_coordinates(
    coords: torch.Tensor,
    input_grid_size: Sequence[int],
    output_grid_size: Sequence[int],
    coordinate_format: str = 'xy',
) -> torch.Tensor:
    """Convert grid coordinates to correct format."""

    if isinstance(input_grid_size, tuple):
        input_grid_size = torch.tensor(input_grid_size, device=coords.device)
    if isinstance(output_grid_size, tuple):
        output_grid_size = torch.tensor(output_grid_size, device=coords.device)

    if coordinate_format == 'xy':
        if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
            raise ValueError(
                'If coordinate_format is xy, the shapes must be length 2.'
            )
    elif coordinate_format == 'tyx':
        if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
            raise ValueError(
                'If coordinate_format is tyx, the shapes must be length 3.'
            )
        if input_grid_size[0] != output_grid_size[0]:
            raise ValueError('converting frame count is not supported.')
    else:
        raise ValueError('Recognized coordinate formats are xy and tyx.')

    position_in_grid = coords
    position_in_grid = position_in_grid * output_grid_size / input_grid_size

    return position_in_grid

def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.

    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
         query_points: The query points, an in the format [t, y, x].  Its size is
             [b, n, 3], where b is the batch size and n is the number of queries
         gt_occluded: A boolean array of shape [b, n, t], where t is the number of
             frames.  True indicates that the point is occluded.
         gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
             format [x, y]
         pred_occluded: A boolean array of predicted occlusions, in the same format
             as gt_occluded.
         pred_tracks: An array of track predictions from your algorithm, in the same
             format as gt_tracks.
         query_mode: Either 'first' or 'strided', depending on how queries are
             sampled.  If 'first', we assume the prior knowledge that all points
             before the query point are occluded, and these are removed from the
             evaluation.

    Returns:
            A dict with the following keys:

            occlusion_accuracy: Accuracy at predicting occlusion.
            pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
                predicted to be within the given pixel threshold, ignoring occlusion
                prediction.
            jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
                threshold
            average_pts_within_thresh: average across pts_within_{x}
            average_jaccard: average across jaccard_{x}
    """

    metrics = {}

    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == 'first':
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == 'strided':
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError('Unknown query mode ' + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics['occlusion_accuracy'] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics['average_jaccard'] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics['average_pts_within_thresh'] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
        target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
            where True indicates occluded.
        target_points: Position, of shape [n_tracks, n_frames, 2], where each point
            is [x,y] scaled between 0 and 1.
        frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
            -1 and 1.
        query_stride: When sampling query points, search for un-occluded points
            every query_stride frames and convert each one into a query.

    Returns:
        A dict with the keys:
            video: Video tensor of shape [1, n_frames, height, width, 3].  The video
                has floats scaled to the range [-1, 1].
            query_points: Query points of shape [1, n_queries, 3] where
                each point is [t, y, x] scaled to the range [-1, 1].
            target_points: Target points of shape [1, n_queries, n_frames, 2] where
                each point is [x, y] scaled to the range [-1, 1].
            trackgroup: Index of the original track that each query point was
                sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        'video': frames[np.newaxis, ...],
        'query_points': np.concatenate(queries, axis=0)[np.newaxis, ...],
        'target_points': np.concatenate(tracks, axis=0)[np.newaxis, ...],
        'occluded': np.concatenate(occs, axis=0)[np.newaxis, ...],
        'trackgroup': np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.

    Args:
        target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
            where True indicates occluded.
        target_points: Position, of shape [n_tracks, n_frames, 2], where each point
            is [x,y] scaled between 0 and 1.
        frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
            -1 and 1.

    Returns:
        A dict with the keys:
            video: Video tensor of shape [1, n_frames, height, width, 3]
            query_points: Query points of shape [1, n_queries, 3] where
                each point is [t, y, x] scaled to the range [-1, 1]
            target_points: Target points of shape [1, n_queries, n_frames, 2] where
                each point is [x, y] scaled to the range [-1, 1]
    """

    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        'video': frames[np.newaxis, ...],
        'query_points': query_points[np.newaxis, ...],
        'target_points': target_points[np.newaxis, ...],
        'occluded': target_occluded[np.newaxis, ...],
    }


def create_davis_dataset(
    davis_points_path: str,
    query_mode: str = 'strided',
    full_resolution=False,
    resolution: Optional[Tuple[int, int]] = (256, 256),
    local_rank=0,
    world_rank=-1,
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on DAVIS data."""
    pickle_path = davis_points_path

    # with tf.io.gfile.GFile(pickle_path, 'rb') as f:
    with open(pickle_path, 'rb') as f:
        davis_points_dataset = pickle.load(f)

    if full_resolution:
        ds, _ = tfds.load(
            'davis/full_resolution', split='validation', with_info=True
        )
        to_iterate = tfds.as_numpy(ds)
    else:
        to_iterate = davis_points_dataset.keys()

    if world_rank ==-1:
        for tmp in to_iterate:
            if full_resolution:
                frames = tmp['video']['frames']
                video_name = tmp['metadata']['video_name'].decode()
            else:
                video_name = tmp
                frames = davis_points_dataset[video_name]['video']
                if resolution is not None and resolution != frames.shape[1:3]:
                    frames = resize_video(frames, resolution)

            frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            target_points = davis_points_dataset[video_name]['points']
            target_occ = davis_points_dataset[video_name]['occluded']
            target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

            if query_mode == 'strided':
                converted = sample_queries_strided(target_occ, target_points, frames)
            elif query_mode == 'first':
                converted = sample_queries_first(target_occ, target_points, frames)
            else:
                raise ValueError(f'Unknown query mode {query_mode}.')

            yield {'davis': converted}
    else:  
        to_iterate = list(to_iterate)
        to_iterate = to_iterate[local_rank::world_rank]
        print(f"Davis Dataset size: {len(to_iterate)}")

        for tmp in to_iterate:
            if full_resolution:
                frames = tmp['video']['frames']
                video_name = tmp['metadata']['video_name'].decode()
            else:
                video_name = tmp
                frames = davis_points_dataset[video_name]['video']
                if resolution is not None and resolution != frames.shape[1:3]:
                    frames = resize_video(frames, resolution)

            frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            target_points = davis_points_dataset[video_name]['points']
            target_occ = davis_points_dataset[video_name]['occluded']
            target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

            if query_mode == 'strided':
                converted = sample_queries_strided(target_occ, target_points, frames)
            elif query_mode == 'first':
                converted = sample_queries_first(target_occ, target_points, frames)
            else:
                raise ValueError(f'Unknown query mode {query_mode}.')

            yield {'davis': converted}


def create_rgb_stacking_dataset(
    robotics_points_path: str,
    query_mode: str = 'strided',
    resolution: Optional[Tuple[int, int]] = (256, 256),
    local_rank=0,
    world_rank=-1,
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on robotics data."""
    pickle_path = robotics_points_path

    # with tf.io.gfile.GFile(pickle_path, 'rb') as f:
    with open(pickle_path, 'rb') as f:
        robotics_points_dataset = pickle.load(f)

    if world_rank ==-1:
        robotics_points_dataset = robotics_points_dataset
    else:
        robotics_points_dataset = robotics_points_dataset[local_rank::world_rank]

    print(f"RGB Stacking Dataset size: {len(robotics_points_dataset)}")

    for example in robotics_points_dataset:
        frames = example['video']
        if resolution is not None and resolution != frames.shape[1:3]:
            frames = resize_video(frames, resolution)
        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        target_points = example['points']
        target_occ = example['occluded']
        target_points = target_points * np.array([frames.shape[2], frames.shape[1]])

        if query_mode == 'strided':
            converted = sample_queries_strided(target_occ, target_points, frames)
        elif query_mode == 'first':
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f'Unknown query mode {query_mode}.')

        yield {'robotics': converted}


def create_kinetics_dataset(
    kinetics_path: str, query_mode: str = 'strided',
    resolution: Optional[Tuple[int, int]] = (256, 256),
    local_rank=0,
    world_rank=-1,
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on Kinetics point tracking."""

    # all_paths = tf.io.gfile.glob(path.join(kinetics_path, '*_of_0010.pkl'))
    all_paths = glob(path.join("datasets/tapvid_kinetics", "*_of_0010.pkl"))
    for pickle_path in all_paths:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                data = list(data.values())

        if world_rank ==-1:
            data = data
        else:
            data = data[local_rank::world_rank]

        print("Kinetics Dataset size: ", len(data))

        # idx = random.randint(0, len(data) - 1)
        for idx in range(len(data)):
            example = data[idx]

            frames = example['video']

            if isinstance(frames[0], bytes):
                # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
                def decode(frame):
                    byteio = io.BytesIO(frame)
                    img = Image.open(byteio)
                    return np.array(img)

                frames = np.array([decode(frame) for frame in frames])

            if resolution is not None and resolution != frames.shape[1:3]:
                frames = resize_video(frames, resolution)

            frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
            target_points = example['points']
            target_occ = example['occluded']
            target_points *= np.array([frames.shape[2], frames.shape[1]])

            if query_mode == 'strided':
                converted = sample_queries_strided(target_occ, target_points, frames)
            elif query_mode == 'first':
                converted = sample_queries_first(target_occ, target_points, frames)
            else:
                raise ValueError(f'Unknown query mode {query_mode}.')

            yield {'kinetics': converted}


def create_kinetics_dataset_split_index(
    kinetics_path: str, query_mode: str = 'strided',
    resolution: Optional[Tuple[int, int]] = (256, 256),
    local_rank=0,
    data_split_idx=-1,
) -> Iterable[DatasetElement]:
    """Dataset for evaluating performance on Kinetics point tracking."""

    pickle_path = path.join(kinetics_path, f'{local_rank:>04}_of_0010.pkl')
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            data = list(data.values())

    print("Kinetics Dataset size: ", len(data))

    # idx = random.randint(0, len(data) - 1)
    for idx in range(len(data)):
        example = data[idx]

        if data_split_idx >=0:
            if idx > data_split_idx:
                break
            if idx != data_split_idx:
                print(f"Skipping data element {idx}")
                continue


        frames = example['video']

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        if resolution is not None and resolution != frames.shape[1:3]:
            frames = resize_video(frames, resolution)

        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        target_points = example['points']
        target_occ = example['occluded']
        target_points *= np.array([frames.shape[2], frames.shape[1]])

        if query_mode == 'strided':
            converted = sample_queries_strided(target_occ, target_points, frames)
        elif query_mode == 'first':
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            raise ValueError(f'Unknown query mode {query_mode}.')

        yield {'kinetics': converted, 'video_id': f"{local_rank}_{idx}"}