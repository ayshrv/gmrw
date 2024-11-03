import logging
from typing import Iterable, Mapping, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def end_point_error(pred_flow, gt_flow, mask=None, order="HWC"):
    """
    flow, gt_flow (B, H, W, 2) / (B, 2, H, W)
    mask: (B, H, W)
    error: float
    """

    import torch

    if order == "CHW":
        pred_flow = pred_flow.permute(0, 2, 3, 1)
        gt_flow = gt_flow.permute(0, 2, 3, 1)
    elif order == "HWC":
        pass
    else:
        raise NotImplementedError

    if mask is None:
        mask = torch.ones(pred_flow.shape[:3], device=pred_flow.device)

    a = (gt_flow[:, :, :, 0] - pred_flow[:, :, :, 0]) ** 2
    b = (gt_flow[:, :, :, 1] - pred_flow[:, :, :, 1]) ** 2
    a = mask * a
    b = mask * b
    error = (a + b) ** 0.5
    error = error.sum() / (mask.sum() + 1e-6)
    return error


def traj_error(pred_trajs, gt_trajs, mask=None):
    """
    trajs, gt_trajs: list, B*[(T, N, 2)]
    mask: list, B*[(T, N)]
    """
    from torch.linalg import vector_norm
    final_values = []
    assert len(pred_trajs) == len(gt_trajs)
    if mask is not None:
        assert len(mask) == len(pred_trajs)
    for i in range(len(pred_trajs)):
        t_pred = pred_trajs[i]  # (T, N, 2)
        t_gt = gt_trajs[i]  # (T, N, 2)

        if len(t_pred[1]) == 0 and len(t_gt[1]) == 0:
            logger.info("No points in this sample, skip in traj_error...")
            continue
        diff = t_pred - t_gt  # (T, N, 2)
        te = vector_norm(diff, ord=2, dim=-1)  # (T, N)
        if mask is not None:
            m = mask[i]  # (T, N)
            te = te * m
            te = te.sum() / (m.sum() + 1e-6)
        else:
            te = te.mean()
        final_values.append(te)
    
    value = sum(final_values) / len(final_values)

    return value


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    points_eval_window: int=-1,
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
        gt_occluded: A boolean array of shape [b, n, t], where t is the number
        of frames.  True indicates that the point is occluded.
        gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
        in the format [x, y]
        pred_occluded: A boolean array of predicted occlusions, in the same
        format as gt_occluded.
        pred_tracks: An array of track predictions from your algorithm, in the
        same format as gt_tracks.
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

    # Don't evaluate the query point.  Numpy doesn't have one_hot, so we
    # replicate it by indexing into an identity matrix.
    one_hot_eye = np.eye(gt_tracks.shape[2])
    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = one_hot_eye[query_frame] == 0

    if points_eval_window != -1:
        B, _, T, _ = gt_tracks.shape
        eval_window = np.zeros_like(evaluation_points)
        for b in range(B):
            for i in range(evaluation_points.shape[1]):
                min_i = max(-1, query_frame[b, i] - points_eval_window)
                max_i = min(T, query_frame[b, i] + points_eval_window) # inclusive in range
                if min_i > -1:
                    eval_window[b, i, min_i] = 1.0
                if max_i < T:
                    eval_window[b, i, max_i] = 1.0
                # eval_window[b, i, min_i:max_i+1] = 1.0
        evaluation_points = evaluation_points * eval_window

    # If we're using the first point on the track as a query, don't evaluate the
    # other points.
    if query_mode == 'first':
        for i in range(gt_occluded.shape[0]):
            index = np.where(gt_occluded[i] == 0)[0][0]
            evaluation_points[i, :index] = False
    elif query_mode != 'strided':
        raise ValueError('Unknown query mode ' + query_mode)

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
        count_visible_points = np.sum(
            visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2))

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
