import argparse
import base64
import csv
import logging
import os
import random
import sys
import time
import json
import timeit
from types import SimpleNamespace
from pathlib import Path

from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fire import Fire
from tqdm import tqdm

import arguments

from data.tapvid import create_davis_dataset, create_rgb_stacking_dataset, create_kinetics_dataset
from utils.metrics import compute_tapvid_metrics
from utils import utils, tracking_utils
from models import GMRW

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


def read_data_tsv(infile, batch_size=1, debug=False):
    # Verify we can read a tsv
    TSV_FIELDNAMES = ["id", "video", "query_points", "target_points", "occluded"]
    in_data = []

    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for i, item in tqdm(enumerate(reader)):
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
            if debug and len(in_data) >= 8:
                break

    if batch_size != 1:
        batched_in_data = []
        for i in range(0, len(in_data), batch_size):
            new_item = {}
            new_item["id"] = np.concatenate([np.array(item["id"])[None] for item in in_data[i:i+batch_size]], axis=0)
            new_item["video"] = np.concatenate([item["video"] for item in in_data[i:i+batch_size]], axis=0)
            new_item["query_points"] = np.concatenate([item["query_points"] for item in in_data[i:i+batch_size]], axis=0)
            new_item["target_points"] = np.concatenate([item["target_points"] for item in in_data[i:i+batch_size]], axis=0)
            new_item["occluded"] = np.concatenate([item["occluded"] for item in in_data[i:i+batch_size]], axis=0)
            batched_in_data.append(new_item)
        in_data = batched_in_data

    return in_data



def get_gmrw_model(model_path=None):

    args = arguments.train_args("")
    args.eval_only = True
    args.no_of_frames = 2

    args.resume = model_path

    try:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    except:
        args.local_rank = -1

    args = utils.setup_device_training(args)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    utils.print_device_info(args)
    logger.info("Eval parameters %s", args)
    logger.info(f"torch version: {str(torch.__version__)}")
    logger.info("Creating model")

    num_scales = 2 
    gmrw_kwargs = {
        "args": args,
        "feature_channels": 128,
        "num_scales": num_scales,
        "upsample_factor": 8,
        "num_head": 1,
        "attention_type": "swin",
        "ffn_dim_expansion": 4,
        "num_transformer_layers": 6,
        "norm": False,
    }

    model = GMRW(**gmrw_kwargs)
    model.to(args.device)

    non_parallel_model = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        non_parallel_model = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model_dict = checkpoint["model"]
        model_dict = utils.trim_statedict_keys(model_dict)
        non_parallel_model.load_state_dict(model_dict)
        logger.info(f"Loading model weights from {args.resume}")
        print(f"Loading model weights from {args.resume}")

    model.eval()

    return args, model



def run_crw(
    model,
    video,  # (B, T, H, W, C)
    target_points,  # (B, N, T, 2)
    query_points,  # (B, N, 3)
    gt_occluded,  # (B, N, T)
    setting,
    global_step=0,
    upsample_factor=1,
    device="cuda",
    compute_flow_method=None, # weighted_argmax, weighted_sum
    window_threshold=None,
    eval_dataset=None,
):

    original_rgbs = video.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
    trajs_g = target_points.permute(0, 2, 1, 3)  # (B, T, N, 2)

    B, S, C, H1, W1 = original_rgbs.shape
    if upsample_factor > 1:
        rgbs = F.interpolate(original_rgbs.reshape(B*S, C, H1, W1), size=(int(upsample_factor*H1), int(upsample_factor*W1)), mode="bilinear", align_corners=True).reshape(B, S, C, int(upsample_factor*H1), int(upsample_factor*W1))
    else:
        rgbs = original_rgbs


    B, S, C, H2, W2 = rgbs.shape

    img2feature_downsample = 4

    viz_outputs = {"images": original_rgbs, "trajs_g": trajs_g}
    viz_outputs["gt_visibility"] = (gt_occluded == False).permute(0, 2, 1)

    if setting == "forward_backward_query_point":
        forward_rgbs = rgbs
        backward_rgbs = rgbs.flip(1)

        forward_flows = []
        backward_flows = []
        for i in tqdm(range(S - 1), leave=False, desc="Computing forward flows"):
            model_input_i = forward_rgbs[:, i : i + 2]
            forward_all_pairs_i = model.compute_correspondence(
                model_input_i
            ).detach()
            forward_flows_i = utils.compute_flow_from_affinity(
                forward_all_pairs_i,
                H=H1,
                W=W1,
                weight_method=compute_flow_method, # weighted_sum, weighted_argmax
                window_threshold=window_threshold,
                device=device,
            )
            forward_flows.append(forward_flows_i)
            del model_input_i, forward_all_pairs_i,

        torch.cuda.empty_cache()

        for i in tqdm(range(S - 1), leave=False, desc="Computing backward flows"):
            model_input_i = backward_rgbs[:, i : i + 2]
            backward_all_pairs_i = model.compute_correspondence(
                model_input_i
            ).detach()
            backward_flows_i = utils.compute_flow_from_affinity(
                backward_all_pairs_i,
                H=H1,
                W=W1,
                weight_method=compute_flow_method,
                window_threshold=window_threshold,
                device=device,
            )
            backward_flows.append(backward_flows_i)
            del model_input_i, backward_all_pairs_i

        torch.cuda.empty_cache()

        forward_flows = torch.cat(forward_flows, dim=1)
        backward_flows = torch.cat(backward_flows, dim=1)

        backward_flows = backward_flows.flip(1)

        viz_outputs['forward_flows'] = forward_flows
        viz_outputs['backward_flows'] = backward_flows

        # (B, T, N, 2), (B, T, N)
        trajectories, visibilities = tracking_utils.get_tracks_from_flows_query_points(
            query_points=query_points,
            forward_flows=forward_flows,
            backward_flows=backward_flows,
            device=device,
        )
    
    elif setting == "forward_backward_time_independent_query_point":
        trajectories = []
        visibilities = []

        for b in range(B):
            num_query_pts = query_points.shape[1]

            trajectories_b = []
            visibilities_b = []

            query_stride = 10

            for q in tqdm(
                range(0, num_query_pts, query_stride), desc="Running for query points"
            ):
                end_index = min(q + query_stride, num_query_pts)
                if q + query_stride <= num_query_pts:
                    end_index = q + query_stride
                    current_query_stride = query_stride
                else:
                    end_index = num_query_pts
                    current_query_stride = num_query_pts - q

                query_point = query_points[b, q:end_index]  # (query_stride, 3)
                qp_t = query_point[:, 0].to(torch.int)

                # (query_stride, C, H, W)
                query_frame = rgbs[b, qp_t]
                # (query_stride, 1, C, H, W)
                query_frame = query_frame.unsqueeze(1)
                target_frames = rgbs[b]  # (S, C, H, W)
                target_frames = target_frames[None].repeat(current_query_stride, 1, 1, 1, 1) # (query_stride, S, C, H, W)

                flows = []
                for i in tqdm(range(S), leave=False, desc="Computing flows"):
                    model_input_i1 = query_frame
                    model_input_i2 = target_frames[:, i:i+1]
                    all_pairs_i = model.compute_correspondence_for_pairs(
                        model_input_i1, model_input_i2
                    ).detach() # (query_stride, 1, h, w, h, w)
                    flows_i = utils.compute_flow_from_affinity(
                        all_pairs_i,
                        H=H1,
                        W=W1,
                        weight_method=compute_flow_method, # weighted_sum, weighted_argmax
                        window_threshold=window_threshold,
                        device=device,
                    ) # (query_stride, 1, h, w, 2)
                    flows.append(flows_i)
                    del model_input_i1, model_input_i2, all_pairs_i,
                flows = torch.cat(flows, dim=1) # (query_stride, S, H, W, 2)

                # (query_stride, S, N=1, 2), (query_stride, S, N=1)
                trajectories_q, visibilities_q = tracking_utils.get_tracks_from_direct_flows_query_points(
                    query_points=query_point[:, None], # (query_stride, 1, 3)
                    flows=flows,
                    device=device,
                )
                trajectories_q = trajectories_q[:, :, 0] # (query_stride, S, 2)
                visibilities_q = visibilities_q[:, :, 0] # (query_stride, S)
                trajectories_q = trajectories_q[None].permute(0, 2, 1, 3) # (B=1, S, query_stride, 2)
                visibilities_q = visibilities_q[None].permute(0, 2, 1) # (B=1, S, query_stride)

                trajectories_b.append(trajectories_q)
                visibilities_b.append(visibilities_q)

                torch.cuda.empty_cache()

            trajectories_b = torch.cat(trajectories_b, dim=2) # (B=1, S, N, 2)
            visibilities_b = torch.cat(visibilities_b, dim=2) # (B=1, S, N, 2)

            trajectories.append(trajectories_b) 
            visibilities.append(visibilities_b) 
        
        trajectories = torch.cat(trajectories, dim=0) # (B, S, N, 2)
        visibilities = torch.cat(visibilities, dim=0) # (B, S, N)

    elif setting == "forward_backward_time_independent_query_point_efficient":
        trajectories = []
        visibilities = []

        for b in range(B):
            num_query_pts = query_points.shape[1]

            trajectories_b = []
            visibilities_b = []

            unique_query_timesteps = torch.unique(query_points[b, :, 0]).to(torch.int).cpu().numpy()

            query_stride = 1

            unique_num_query_pts = len(unique_query_timesteps)

            unique_pts_index = {unique_query_timesteps[i]:i for i in range(unique_num_query_pts)}

            flows = []

            for q in tqdm(
                range(0, unique_num_query_pts, query_stride), desc="Running for query points"
            ):
                end_index = min(q + query_stride, unique_num_query_pts)
                if q + query_stride <= unique_num_query_pts:
                    end_index = q + query_stride
                    current_query_stride = query_stride
                else:
                    end_index = unique_num_query_pts
                    current_query_stride = unique_num_query_pts - q

                qp_t = unique_query_timesteps[q:end_index]

                # (query_stride, C, H, W)
                query_frame = rgbs[b, qp_t]
                # (query_stride, 1, C, H, W)
                query_frame = query_frame.unsqueeze(1)
                target_frames = rgbs[b]  # (S, C, H, W)
                target_frames = target_frames[None].repeat(current_query_stride, 1, 1, 1, 1) # (query_stride, S, C, H, W)

                flows_q = []
                for i in tqdm(range(S), leave=False, desc="Computing flows"):
                    model_input_i1 = query_frame
                    model_input_i2 = target_frames[:, i:i+1]
                    all_pairs_i = model.compute_correspondence_for_pairs(
                        model_input_i1, model_input_i2
                    ).detach() # (query_stride, 1, h, w, h, w)
                   
                    flows_i = utils.compute_flow_from_affinity(
                        all_pairs_i,
                        H=H1,
                        W=W1,
                        weight_method=compute_flow_method, # weighted_sum, weighted_argmax
                        window_threshold=window_threshold,
                        device=device,
                    ) # (query_stride, 1, h, w, 2)
                    flows_q.append(flows_i)
                    del model_input_i1, model_input_i2, all_pairs_i,
                flows_q = torch.cat(flows_q, dim=1) # (query_stride, S, H, W, 2)
                flows.append(flows_q)
            # (unique_q, S, H, W, 2)
            flows = torch.cat(flows, dim=0)

            for q_i in range(num_query_pts):
                query_point = query_points[b, q_i:q_i+1][None] # (1, 1, 3)
                query_point_int = query_point.to(torch.int).cpu().numpy()[0, 0, 0]
                unique_q_index = unique_pts_index[query_point_int]
                flows_q = flows[unique_q_index][None] # (1, S, H, W, 2)

                # (1, S, N=1, 2), (1, S, N=1)
                trajectories_q, visibilities_q = tracking_utils.get_tracks_from_direct_flows_query_points(
                    query_points=query_point, # (1, 1, 3)
                    flows=flows_q, # (1, S, H, W, 2)
                    device=device,
                )
                trajectories_q = trajectories_q[:, :, 0] # (query_stride=1, S, 2)
                visibilities_q = visibilities_q[:, :, 0] # (query_stride=1, S)
                trajectories_q = trajectories_q[None].permute(0, 2, 1, 3) # (B=1, S, query_stride, 2)
                visibilities_q = visibilities_q[None].permute(0, 2, 1) # (B=1, S, query_stride)

                trajectories_b.append(trajectories_q)
                visibilities_b.append(visibilities_q)

                torch.cuda.empty_cache()

            trajectories_b = torch.cat(trajectories_b, dim=2) # (B=1, S, N, 2)
            visibilities_b = torch.cat(visibilities_b, dim=2) # (B=1, S, N, 2)

            trajectories.append(trajectories_b) 
            visibilities.append(visibilities_b) 
        
        trajectories = torch.cat(trajectories, dim=0) # (B, S, N, 2)
        visibilities = torch.cat(visibilities, dim=0) # (B, S, N)
    
    viz_outputs['trajs_e'] = trajectories
    viz_outputs['visibility'] = visibilities

    pred_tracks = trajectories.permute(0, 2, 1, 3) # (B, N, T, 2)
    occluded = (visibilities == False).permute(0, 2, 1) # (B, N, T)

    query_points = query_points.cpu().numpy()  # (B, N, mer3)
    gt_occluded = gt_occluded.cpu().numpy()  # (B, N, T)
    gt_tracks = target_points.cpu().numpy()  # (B, N, T, 2)
    pred_occluded = occluded.cpu().numpy()  # (B, N, T)
    pred_tracks = pred_tracks.cpu().numpy()  # (B, N, T, 2)

    metrics = compute_tapvid_metrics(
        query_points=query_points,
        gt_occluded=gt_occluded,
        gt_tracks=gt_tracks,
        pred_occluded=pred_occluded,
        pred_tracks=pred_tracks,
        query_mode="strided",
    )

    metrics["total_count"] = np.array([1])
    return metrics


def add_to_metrics(overall_metrics, new_metrics):
    for k, v in new_metrics.items():
        new_v = torch.tensor(v)
        if k not in overall_metrics:
            assert new_v.shape == torch.Size([1])
            overall_metrics[k] = [new_v[0].item()]
        else:
            assert new_v.shape == torch.Size([1])
            overall_metrics[k].append(new_v[0].item())
    return overall_metrics


def average_metrics(overall_metrics):
    for k, v in overall_metrics.items():
        new_v = torch.tensor(v)
        overall_metrics[k] = torch.mean(new_v)
    return overall_metrics


def sum_metrics(overall_metrics):
    for k, v in overall_metrics.items():
        new_v = torch.tensor(v)
        overall_metrics[k] = torch.sum(new_v)
    return overall_metrics


def reduce_dict(input_dict, device, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0).to(device)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v}")


def save_tsv(filename, tsv_fields, data):
    with open(filename, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=tsv_fields)
        for i, item in enumerate(data):
            encoded_item = {
                k: str(v) if isinstance(v, int) else str(base64.b64encode(np.ascontiguousarray(v)), "utf-8")
                for k, v in item.items()
            }
            writer.writerow(encoded_item)
    print(f"Saved {filename}!")


def get_eval_input_from_kubric():

    total_val_data = read_data_tsv(
        "datasets/movi/movi_e/validation_data.tsv", batch_size=1, debug=False
    )

    return total_val_data

def get_eval_input_from_davis(query_mode, local_rank=None, world_rank=None):
    if local_rank is None:
        local_rank = -1
        world_rank = -1
    davis_points_path = "datasets/tapvid_davis/tapvid_davis.pkl"
    yield from create_davis_dataset(
        davis_points_path, query_mode=query_mode, local_rank=local_rank, world_rank=world_rank
    )

def get_eval_input_from_rgbstacking(query_mode, local_rank=None, world_rank=None):
    if local_rank is None:
        local_rank = -1
        world_rank = -1
    robotics_points_path = "datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl"
    yield from create_rgb_stacking_dataset(
        robotics_points_path, query_mode=query_mode, local_rank=local_rank, world_rank=world_rank
    )

def get_eval_input_from_kinetics(query_mode, local_rank=None, world_rank=None):
    if local_rank is None:
        local_rank = -1
        world_rank = -1
    kinetics_points_path = "datasets/tapvid_kinetics"
    yield from create_kinetics_dataset(
        kinetics_points_path, query_mode=query_mode, local_rank=local_rank, world_rank=world_rank
    )

def get_eval_input_from_jhmdb(debug=False, local_rank=None, world_rank=None):
    if local_rank is None:
        local_rank = -1
        world_rank = -1
    jhmdb_points_path = "datasets/jhmdb"
    yield from create_jhmdb_dataset(
        jhmdb_points_path, debug=debug
    )


def main(
    eval_dataset="kubric",
    setting="", # forward_backward_query_point, forward_backward_time_independent_query_point, forward_backward_time_independent_query_point_efficient
    upsample_factor=1,
    device="cuda",
    compute_flow_method="weighted_sum",
    model_path=None,
    window_threshold=None,
):  

    args, model = get_gmrw_model(model_path)

    if eval_dataset == "davis":
        if args.local_rank != -1:
            eval_data_loader = get_eval_input_from_davis(
                query_mode="strided", 
                local_rank=args.local_rank, 
                world_rank=dist.get_world_size()
            )
        else:
            eval_data_loader = get_eval_input_from_davis(query_mode="strided")
    elif eval_dataset == "rgb_stacking":
        if args.local_rank != -1:
            eval_data_loader = get_eval_input_from_rgbstacking(
                query_mode="strided", 
                local_rank=args.local_rank, 
                world_rank=dist.get_world_size()
            )
        else:
            eval_data_loader = get_eval_input_from_rgbstacking(query_mode="strided")
    elif eval_dataset == "kinetics":
        if args.local_rank != -1:
            eval_data_loader = get_eval_input_from_kinetics(
                query_mode="strided", 
                local_rank=args.local_rank, 
                world_rank=dist.get_world_size()
            )
        else:
            eval_data_loader = get_eval_input_from_kinetics(query_mode="strided")
    elif eval_dataset == "kubric":
        total_eval_data_loader = get_eval_input_from_kubric()
    else:
        raise ValueError(f"Unknown eval dataset {eval_dataset}")

    if eval_dataset == "kubric":
        if args.local_rank != -1:
            eval_data_loader = total_eval_data_loader[args.local_rank :: dist.get_world_size()]
        else:
            eval_data_loader = total_eval_data_loader

        print(
            f"Evaluating Val data with {len(eval_data_loader)} out of {len(total_eval_data_loader)} items"
        )

    overall_metrics = {}

    for i, item in enumerate(tqdm(eval_data_loader, desc="Evaluting an iteration")):
            
        if eval_dataset == "davis":
            item = item["davis"]
        elif eval_dataset == "rgb_stacking":
            item = item["robotics"]
        elif eval_dataset == "kinetics":
            item = item["kinetics"]
        elif eval_dataset == "jhmdb":
            item = item["jhmdb"]

        video = item["video"]  # (B, T, H, W, C)
        video = torch.from_numpy(video.copy()).float()

        target_points = item["target_points"]  # (B, N, T, 2)
        target_points = torch.from_numpy(target_points.copy()).float()

        query_points = item["query_points"]  # (B, N, 3)
        query_points = torch.from_numpy(query_points.copy()).float()

        occluded = item["occluded"]  # (B, N, T)
        occluded = torch.from_numpy(occluded.copy()).bool()

        video = video.to(device)
        target_points = target_points.to(device)
        query_points = query_points.to(device)
        occluded = occluded.to(device)

        video = 255.0 * (video + 1) / 2.0

        with torch.no_grad():
            metrics = run_crw(
                model=model,
                video=video,
                target_points=target_points,
                query_points=query_points,
                gt_occluded=occluded,
                global_step=i,
                device=device,
                setting=setting,
                upsample_factor=upsample_factor,
                compute_flow_method=compute_flow_method,
                window_threshold=window_threshold,
                eval_dataset=eval_dataset,
            )

        # print(f"Video: {i}")
        # print(metrics)

        overall_metrics = add_to_metrics(overall_metrics, metrics)



    overall_metrics = sum_metrics(overall_metrics)

    if args.local_rank != -1:
        torch.distributed.barrier()
        overall_metrics = reduce_dict(
            overall_metrics, device=args.device, average=False
        )

    overall_metrics = {
        k: v / overall_metrics["total_count"] for k, v in overall_metrics.items()
    }

    if args.local_rank in [-1, 0]:
        print_metrics(overall_metrics)



if __name__ == "__main__":
    Fire(main)
