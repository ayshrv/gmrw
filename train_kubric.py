import ctypes

libgcc_s = ctypes.CDLL("libgcc_s.so.1")

import datetime
import logging
import time
from collections import OrderedDict

import torch
import torch.distributed as dist

import torchvision
import wandb
from torch import autocast, nn
from torch.distributed.elastic.multiprocessing.errors import record

import arguments
from data import datasets
from models import GMRW
from utils import tracking_utils, utils, viz_utils

torch.autograd.set_detect_anomaly(True)
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

logger = logging.getLogger(__name__)

GLOBAL_STEP = 0

def plot_visualizations(
    GLOBAL_STEP,
    frames,
    outputs,
    gt_flows,
    gt_trajs,
    args,
    html_page=None,
):
    # frames: (B, 2*T, 3, H, W)
    # flows: (B, 2*T-2, H_, W_, H_, W_)
    # gt_flow: (B, T-1, H, W, 2)
    # trajs: (B, T-1, H, W, 2)
    # gt_trajs: (B, T-1, H, W, 2)

    flows = outputs["flows"]
    trajs = outputs["trajs"]

    overall_flows = outputs["overall_pred_flows"]

    visible_flow_mask = outputs["visible_flow_mask"]
    occluded_flow_mask = outputs["occluded_flow_mask"]

    gt_visible_points = outputs["gt_visible_points"]

    html_dict = viz_utils.visualize_kubric(
        step=GLOBAL_STEP,
        images=frames,
        flows=flows,
        overall_flows=overall_flows,
        gt_flows=gt_flows,
        visible_flow_mask=visible_flow_mask,
        occluded_flow_mask=occluded_flow_mask,
        trajs=trajs,
        gt_trajs=gt_trajs,
        gt_visible_points=gt_visible_points,
        args=args,
        use_html=(html_page is not None),
    )

    html_page.add_header(f"Step {GLOBAL_STEP}:")
    html_page.add_images_in_rows(
        ims=html_dict["images"],
        txts=html_dict["texts"],
        links=html_dict["links"],
        width=200 * (args.no_of_frames + 1),
        video_height=1200,
        is_video=False,
        add_new_table=True,
    )  # 2858 × 446
    html_page.save()

    single_step_html_filename = (
        f"index_train_{args.html_filename}_{GLOBAL_STEP:06d}.html"
    )
    utils.save_single_step_html(
        args,
        single_step_html_filename,
        GLOBAL_STEP,    
        html_dict,
        width=200 * (args.no_of_frames + 1),
        video_height=1200,
        is_video=False,
    )


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    epoch,
    args,
    grad_scaler=None,
    html_page=None,
):
    global GLOBAL_STEP
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "GLOBAL_STEP", utils.SmoothedValue(window_size=1, fmt="{value}")
    )
    metric_logger.meters.update(GLOBAL_STEP=GLOBAL_STEP)

    header = "Epoch: [{}]".format(epoch)

    for step, item in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        item = [i.to(args.device) for i in item]

        if not args.smoothness_curriculum:
            smoothness_loss_weight = args.smoothness_loss_weight
        else:
            start_iter = 5000
            end_iter = 15000
            start_value = 0.0
            end_value = 200.0

            if GLOBAL_STEP <= start_iter:
                smoothness_loss_weight = start_value
            elif GLOBAL_STEP >= end_iter:
                smoothness_loss_weight = end_value
            else:
                smoothness_loss_weight = start_value + (GLOBAL_STEP - start_iter) * (
                    end_value - start_value
                ) / (end_iter - start_iter)

        (
            frames,
            affine_mat_f2b,
            affine_mat_b2f,
            gt_flows_f,
            gt_flows_b,
            forward_flows_b,
        ) = item


        gt_flows = gt_flows_f
        # (B, 2*T-2, 2, H, W)
        full_gt_flows = torch.cat((gt_flows_f, gt_flows_b), dim=1)

        with autocast(enabled=args.use_amp, device_type="cuda", dtype=torch.bfloat16):
            all_pairs, loss, diagnostics, variables_log = model(
                frames,
                affine_mat_b2f,
                smoothness_loss_weight,
            )

        loss = loss.mean()

        diagnostics["loss"] = loss.mean().item()
        diagnostics["smoothness_loss_weight"] = smoothness_loss_weight

        if args.use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()

        diagnostics["total gradient norm"] = utils.get_gradient_norm(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        diagnostics["total gradient norm after clipping"] = utils.get_gradient_norm(
            model
        )

        if args.use_amp:
            grad_scaler.step(optimizer)
        else:
            optimizer.step()

        if args.use_amp:
            grad_scaler.update()

        torch.cuda.empty_cache()

        if args.use_data_aug_curriculum and GLOBAL_STEP % 1000 == 0:
            data_augs_diags = data_loader.update_data_aug_params(iter=GLOBAL_STEP)
        else:
            data_augs_diags = {}

        with torch.no_grad():
            outputs = {}
            outputs.update(variables_log)

            # frames: (B, 2T-2, 3, H, W)
            # all_pairs: (B, 2T-3, H_, W_, H_, W_)
            # gt_flows: (B, T-1, 2, H, W)\
            # full_gt_flows: (B, 2T-2, 2, H, W) / (B, T-1, 2, H, W)

            B = gt_flows.shape[0]
            T = gt_flows.shape[1] + 1
            H = gt_flows.shape[3]
            W = gt_flows.shape[4]

            # gt_trajs: (B, T, N, 2)
            gt_trajs = tracking_utils.get_tracks_from_flows(
                gt_flows, device=args.device
            )

            # gt_trajs: B*[(T, N_, 2)]
            # gt_visible_points, gt_occluded_points: (B*[(T, N_)]

            (
                gt_trajs,
                gt_visible_points,
                gt_occluded_points,
            ) = tracking_utils.filter_tracks(
                trajs=gt_trajs,
                flows_f=gt_flows_f,
                flows_b=forward_flows_b,
                boundary_padding=0,
                device=args.device,
            )
            # (B, T, 1, H, W)
            gt_occlusions = (
                tracking_utils.get_occlusions_from_forward_backward_flows(
                    forward_flow=gt_flows_f,
                    backward_flow=forward_flows_b,
                    device=args.device,
                )
            )

            # (B, 2T-3, 2, H, W)
            pred_flows = utils.compute_flow_from_affinity(
                all_pairs,
                H,
                W,
                weight_method="weighted_sum",
                device=args.device,
            )

            forward_pred_flows = pred_flows[:, : T - 1]

            outputs["flows"] = pred_flows

            valid_flow_mask = torch.ones_like(gt_occlusions)
            valid_flow_mask = valid_flow_mask.bool()
            gt_occlusions = gt_occlusions.bool()

            visible_flow_mask = valid_flow_mask & ~gt_occlusions  # (B, T, 1, H, W)
            occluded_flow_mask = valid_flow_mask & gt_occlusions  # (B, T, 1, H, W)

            outputs["visible_flow_mask"] = visible_flow_mask
            outputs["occluded_flow_mask"] = occluded_flow_mask

            # (B, 2T-3, H_, W_, H_, W_)
            probabilities_t = utils.At_from_all_pairs(all_pairs)
            overall_pred_flows = utils.compute_flow_from_affinity(
                probabilities_t,
                H,
                W,
                weight_method="weighted_sum",
                device=args.device,
            )

            outputs["overall_pred_flows"] = overall_pred_flows

            pred_trajs = []
            forward_pred_trajs = []

            for b_idx in range(B):
                flow = overall_pred_flows[b_idx]
                coord = gt_trajs[b_idx][0]  # (N_, 2)
                pred_coords = tracking_utils.get_tracks_from_overall_flows(
                    flow[None], coord[None], device=args.device
                )  # (1, T, N_, 2)
                pred_trajs.append(pred_coords[0])
                forward_pred_trajs.append(pred_coords[0, :T])

            # pred_trajs: B*[(T, N_, 2)]
            outputs["trajs"] = pred_trajs
            outputs["gt_visible_points"] = gt_visible_points

        if GLOBAL_STEP % args.model_save_freq == 0 and args.local_rank in [-2, -1, 0]:
            utils.save_model_checkpoint(
                args,
                model,
                optimizer,
                epoch,
                GLOBAL_STEP,
            )

        if (
            args.visualize
            and args.local_rank in [-2, -1, 0]
            and GLOBAL_STEP % args.train_viz_log_freq == 0
        ):
            plot_visualizations(
                GLOBAL_STEP,
                frames,
                outputs,
                full_gt_flows,
                gt_trajs,
                args,
                html_page,
            )

        diagnostics["lr"] = optimizer.param_groups[0]["lr"]

        diagnostics.update(data_augs_diags)

        if args.local_rank in [-2, -1, 0]:
            wandb.log(
                {
                    k: v.mean().item() if type(v) == torch.Tensor else v
                    for k, v in diagnostics.items()
                },
                step=GLOBAL_STEP,
            )

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["clips/s"].update(
            frames.shape[0] / (time.time() - start_time)
        )
            
        metric_logger.meters.update(GLOBAL_STEP=GLOBAL_STEP)
        GLOBAL_STEP += 1

        torch.cuda.empty_cache()


@record
def main(args):
    utils.create_output_dir(args)
    utils.setup_logging(args)

    if args.local_rank in [-2, -1, 0]:
        html_page = utils.setup_html_page(
            args, html_filename=f"index_train_{args.html_filename}.html"
        )
    else:
        html_page = None

    args = utils.setup_device_training(args)

    utils.set_seed(args.seed, args.n_gpu)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if args.local_rank in [-2, -1, 0]:
        utils.wandb_logging(args)

    utils.print_device_info(args)

    logger.info("Training parameters %s", args)
    logger.info(f"torch version: {str(torch.__version__)}")
    logger.info(f"torchvision version: {str(torchvision.__version__)}")

    logger.info("Preparing training dataloader...")

    dataloader, args = datasets.get_data_loader(args, training=True)

    logger.info("Creating model")
    
    # GMFlow Args
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
        "norm": args.disable_transforms,
        "flash_attention": args.use_flash_attention,
        "gradient_checkpointing": args.use_gradient_checkpointing,
    }
    model = GMRW(**gmrw_kwargs)
    model.to(args.device)

    logger.info(model)

    if args.local_rank in [-2, -1, 0]:
        wandb.watch(model)

    if args.pretrained_gmflow:
        gmflow_model_dict = torch.load(
            args.pretrained_gmflow_path, map_location=args.device
        )["model"]
        new_gmflow_model_dict = OrderedDict()
        for key, item in gmflow_model_dict.items():
            if "backbone" in key or "transformer" in key:
                new_key = key
                if "module" in key:
                    new_key = new_key[7:]
                new_gmflow_model_dict[new_key] = item
        model.load_state_dict(new_gmflow_model_dict, strict=False)
        logger.info(f"Loaded GMFlow model from {args.pretrained_gmflow_path}")
    else:
        logger.info(f"Initialized GMFlow model from scratch")

    non_parallel_model = model
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        non_parallel_model = model.module

    if args.local_rank not in [-2, -1]:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        non_parallel_model = model.module

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of params: {num_params}")

    logger.info("Creating optimizer")
    if args.local_rank not in [-2, -1]:
        args.lr *= dist.get_world_size()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    logger.info(optimizer)

    global GLOBAL_STEP
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        strict = True
        model_dict = checkpoint["model"]
        model_dict = utils.trim_statedict_keys(model_dict)
        non_parallel_model.load_state_dict(model_dict, strict=strict)
        args.start_epoch = checkpoint["epoch"] + 1
        if "global_step" in checkpoint.keys():
            GLOBAL_STEP = checkpoint["global_step"]
        else:
            GLOBAL_STEP = 200 * 348 + 50
        logger.info(f"Loaded GMFlow model from {args.resume}")

    if args.use_amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            data_loader=dataloader,
            epoch=epoch,
            args=args,
            html_page=html_page,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = arguments.train_args()
    main(args)
