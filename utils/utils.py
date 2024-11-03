import base64
import csv
import datetime
import errno
import logging
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

from utils import html

logger = logging.getLogger(__name__)


def log(x, eps=1e-7):
    return torch.log(x + eps)


class LogNLLMaskNeighbourLoss(nn.Module):
    def __init__(self, aggregation=None, autofill=None):
        super().__init__()
        self.aggregation = aggregation
        self.autofill = autofill

    def forward(self, output, label, mask=None):
        """
        output: B, N1, N2
        label: B, N1, N2, binary
        mask: B, N1
        """
        if self.autofill and output.size(2) != label.size(2):
            label = F.pad(label, (0, output.size(2) - label.size(2)), "constant", 0)
        # aggregate neighbor
        if self.aggregation == "sum":
            pos_prob = (output * label).sum(-1)
            loss = log(pos_prob)
        elif self.aggregation is None:
            # no aggregation, mean of log prob
            loss = (log(output) * label).sum(-1)
            loss /= label.sum(-1)
        if mask is not None:
            # mask unmatched items
            loss *= mask
            loss = -loss.sum() / (mask.sum() + 1e-5)  ## Basically masked mean!
        return loss


def align_feat(feat, affine_mat):
    BT, feats_dim, feats_h, feats_w = feat.shape
    affine_grid = torch.nn.functional.affine_grid(
        affine_mat, (BT, feats_dim, feats_h, feats_w), True
    )
    aligned_feat = torch.nn.functional.grid_sample(
        feat, affine_grid, mode="bilinear", align_corners=True
    )
    return aligned_feat


def compute_acc(prob, label, mask=None):
    """
    prob: B, P, P, [0, 1]
    label: B, P, P, binary
    mask: B, P
    """
    # B, P
    with torch.no_grad():
        hit = prob.argmax(-1) == label.argmax(-1)
        hit = hit.float()
        if mask is None:
            acc = hit.mean()
        else:
            hit *= mask
            acc = hit.sum() / (mask.sum() + 1e-5)
        return acc


def compute_flow_from_affinity(
    affinities,
    H=None,
    W=None,
    weight_method="argmax",
    weights_masking=False,
    img2feature_downsample=1,
    device=None,
    save_gpu_memory=False,
    evaluation=False,
    return_xy=False,
    window_threshold=None,
):
    """Affinites must be normalized"""

    # affinites: (B, T, H_, W_, H_, W_) / (B, H_, W_, H_, W_)
    # flows: (B, T, 2, H, W) / (B, 2, H, W)

    if device is None:
        device = affinities.device

    squeeze_affinities = False
    if affinities.ndim == 5:
        affinities = affinities.unsqueeze(1)
        squeeze_affinities = True

    B, T, H_, W_ = affinities.shape[:4]

    do_resize = True
    if H is None or W is None:
        do_resize = False
        H, W = H_, W_

    flat_affinities = affinities.reshape(B * T, H_, W_, H_, W_)

    # (B*T_, H, W, 2)
    flows = compute_flow(flat_affinities, method=weight_method, device=device, window_threshold=window_threshold,)
    if flows.dtype == torch.int64:
        flows = flows.to(torch.float32)
    if flows.dtype == torch.bfloat16:
        flows = flows.to(torch.float32)

    if do_resize:
        # (B*T, H, W, 2)
        flows = resize_flows(flows, H, W)

    # if flat_affinities.dtype == torch.bfloat16:
    #     flows = flows.to(torch.bfloat16)

    # (B, T, 2, H, W)
    flows = flows.reshape(B, T, H, W, 2).permute(0, 1, 4, 2, 3)

    if squeeze_affinities:
        # (B, 2, H, W)
        flows = flows.squeeze(1)

    return flows


def image_gradients(images, stride=1):
    """
    images (torch.Tensor): (B, C, H, W)

    return:
    images_gw (toorch.Tensor): (B, C, H, W-1)
    images_gh (toorch.Tensor): (B, C, H-1, W)
    """
    images_gh = images[:, :, stride:, :] - images[:, :, :-stride, :]
    images_gw = images[:, :, :, stride:] - images[:, :, :, :-stride]
    return images_gw, images_gh


def robust_l1(x):
    """Robust L1 metric."""
    return (x**2 + 0.001**2) ** 0.5


def get_first_and_second_order_grads(image):
    # (B, C, H, W)
    image_gx, image_gy = image_gradients(image.permute(0, 3, 1, 2))
    image_gxx, unused_image_gxy = image_gradients(image_gx)
    unused_image_gyx, image_gyy = image_gradients(image_gy)

    return image_gx, image_gy, image_gxx, image_gyy


def smoothness_loss(flows, images, edge_constant=150.0, order=2, edge_aware=True):
    """
    flows (list): (B, H, W, 2)
    images (list[torch.Tensor]): (B, C, H, W) Image1 from which img_grads are computed
    """
    assert order == 2, "Only second order derivatives implemented"
    assert len(flows) == len(images)

    abs_fn = lambda x: x**2

    loss = 0.0

    all_weights_xx = []
    all_weights_yy = []
    all_img_gx = []
    all_img_gy = []
    all_flow_gx = []
    all_flow_gy = []
    all_flow_gxx = []
    all_flow_gyy = []

    for idx, (flow, image) in enumerate(zip(flows, images)):
        flow = flow.permute(0, 3, 1, 2)

        _, _, flow_h, flow_w = flow.shape
        _, _, image_h, image_w = image.shape
        assert flow_h == image_h and flow_w == image_w

        resized_img = image
        resized_flow = flow

        # (B, C, H, W)
        img_gx, img_gy = image_gradients(resized_img, stride=2)

        if edge_aware:
            # (B, 1, H, W)
            weights_xx = torch.exp(
                -torch.mean(abs_fn(edge_constant * img_gx), dim=1, keepdim=True)
            )
            weights_yy = torch.exp(
                -torch.mean(abs_fn(edge_constant * img_gy), dim=1, keepdim=True)
            )
        else:
            # (B, 1, H, W)
            weights_xx = torch.ones_like(img_gx)[:, :1]
            weights_yy = torch.ones_like(img_gy)[:, :1]

        # Compute second derivatives of the predicted smoothness.
        # (B, C, H, W)
        flow_gx, flow_gy = image_gradients(resized_flow)
        flow_gxx, unused_flow_gxy = image_gradients(flow_gx)
        unused_flow_gyx, flow_gyy = image_gradients(flow_gy)

        # Compute weighted smoothness
        loss += (
            1.0
            * (
                torch.mean(weights_xx * robust_l1(flow_gxx))
                + torch.mean(weights_yy * robust_l1(flow_gyy))
            )
            / 2.0
            / len(flows)
        )

        all_weights_xx.append(weights_xx)
        all_weights_yy.append(weights_yy)
        all_img_gx.append(img_gx)
        all_img_gy.append(img_gy)
        all_flow_gx.append(flow_gx)
        all_flow_gy.append(flow_gy)
        all_flow_gxx.append(flow_gxx)
        all_flow_gyy.append(flow_gyy)

    return (
        loss,
        all_weights_xx,
        all_weights_yy,
        all_img_gx,
        all_img_gy,
        all_flow_gx,
        all_flow_gy,
        all_flow_gxx,
        all_flow_gyy,
    )



def save_model_checkpoint(
    args,
    model,
    optimizer,
    epoch,
    global_step,
    lr_scheduler=None,
    cfg=None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "global_step": global_step,
    }
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    if cfg is not None:
        checkpoint["cfg"] = cfg
    torch.save(
        checkpoint,
        os.path.join(
            args.output_dir, "checkpoints", "model_{}.pth".format(global_step)
        ),
    )
    torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))



def create_output_dir(args):
    if (
        args.local_rank in [-1, 0]
        and os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.eval_only
    ):
        raise IOError(
            "%s \nOutput Directory not empty. Exiting to prevent overwriting..."
            % (args.output_dir)
        )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"))
        os.makedirs(os.path.join(args.output_dir, "predictions"))
        os.makedirs(os.path.join(args.output_dir, "predictions", "train"))
        os.makedirs(os.path.join(args.output_dir, "predictions", "train", "gif"))
        os.makedirs(os.path.join(args.output_dir, "predictions", "val"))
        os.makedirs(os.path.join(args.output_dir, "predictions", "val", "gif"))

    logger.info(f"Created output_dir {args.output_dir}")


def setup_logging(args):
    handlers = [logging.StreamHandler()]
    if args.local_rank in [-1, 0]:
        handlers += [
            logging.FileHandler(
                filename=os.path.join(
                    args.output_dir, "eval_log" if args.eval_only else "log"
                ),
                mode="a",
            )
        ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        handlers=handlers,
    )


def setup_html_page(
    args, html_filename="index_train.html", img_dir="train", title="train_predictions"
):
    html_page = html.HTML(
        web_dir=os.path.join(args.output_dir, "predictions"),
        title=title,
        html_filename=html_filename,
        img_dir=img_dir,
        refresh=0,
    )
    html_page.add_header(args.exp_name)
    html_page.add_table()
    return html_page


def save_single_step_html(
    args,
    html_filename,
    GLOBAL_STEP,
    html_dict,
    width,
    video_height,
    is_video,
    lazy_load=True,
    img_dir="train",
    title="train_predictions",
):
    html_page = html.HTML(
        web_dir=os.path.join(args.output_dir, "predictions"),
        title=title,
        html_filename=html_filename,
        img_dir=img_dir,
        refresh=0,
    )
    html_page.add_header(args.exp_name)
    html_page.add_header(f"Step {GLOBAL_STEP}:")
    html_page.add_images_in_rows(
        ims=html_dict["images"],
        txts=html_dict["texts"],
        links=html_dict["links"],
        width=width,
        video_height=video_height,
        is_video=is_video,
        add_new_table=True,
        lazy_load=lazy_load,
    )  # 2858 × 446
    html_page.save()


def setup_device_training(args):
    # Setup CPU, CUDA, GPU & distributed training
    device = None
    if args.local_rank == -1:
        device = torch.device("cpu")
        args.local_rank = -2
        args.n_gpu = -1
        if torch.cuda.is_available():
            device = torch.device("cuda")
            args.local_rank = -1
            args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    return args


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def wandb_logging(args):
    wandb.init(
        project="long-range-crw",
        entity="ayush-owens-lab",
        name=args.exp_name,
        dir=args.output_dir,
        config=args,
    )
    wandb.run.log_code(".")


def print_device_info(args):
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s eval_only: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank not in [-1, -2]),
        args.eval_only,
    )





class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    # def synchronize_between_processes(self):
    #     import torch.distributed as dist

    #     """
    #     Warning: does not synchronize the deque!
    #     """
    #     if not is_dist_avail_and_initialized():
    #         return
    #     t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
    #     dist.barrier()
    #     dist.all_reduce(t)
    #     t = t.tolist()
    #     self.count = int(t[0])
    #     self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    # def synchronize_between_processes(self):
    #     for meter in self.meters.values():
    #         meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""

        try:
            iterable_len = len(iterable)
        except TypeError:
            iterable_len = 14581
            
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        # space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        space_fmt = ":" + str(len(str(iterable_len))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                # eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_seconds = iter_time.global_avg * (iterable_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            # len(iterable),
                            iterable_len,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            # len(iterable),
                            iterable_len,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {}".format(header, total_time_str))


def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid
    

def compute_flow(corr, device, method="argmax", window_threshold=None,):
    """
    corr: Affinity matrix (torch.Tensor), (B, H, W, H, W), make sure it is normalized
    flow: (B, H, W, 2), xy
    """
    B, H, W, _, _ = corr.shape

    # (B, H, W, H*W)
    corr = corr.flatten(-2, -1)
    if method == "argmax":
        # (B, H, W)
        max_indices = corr.argmax(dim=-1)
        # (B, H, W, H*W)
        probability_matrix = torch.nn.functional.one_hot(max_indices, num_classes=H * W)
    elif method == "weighted_sum":
        # (B, H, W, H*W)
        probability_matrix = corr
    elif method == "weighted_argmax":
        # (2, H, W), xy
        init_grid = coords_grid(1, H, W, device=device)[0]
        # (H*W, 2), xy
        init_grid = init_grid.permute(1, 2, 0).reshape(H*W, 2)

        # (B, H*W)
        max_indices = corr.argmax(dim=-1).reshape(B, H*W)
        # (B, H*W, 2) xy
        max_indices_pos = init_grid[max_indices]
        # (B, H*W, 1, 2) xy
        max_indices_pos = max_indices_pos[:, :, None]
        # (1, 1, H*W, 2) xy
        init_grid = init_grid[None, None]

        # (B, H*W, H*W)
        valid_pos = torch.sum(torch.square(max_indices_pos - init_grid), dim=-1) < window_threshold**2
        valid_pos = valid_pos.reshape(B, H, W, H*W)

        probability_matrix = valid_pos * corr
        sum_of_weights = torch.max(torch.tensor(1e-12), torch.sum(probability_matrix, dim=-1)) # (B, H, W)
        # (B, H, W, H*W)
        probability_matrix = torch.div(probability_matrix, sum_of_weights[..., None])
    else:
        raise NotImplementedError(f"{method} method not implemented in compute_flow")

    _dtype = torch.int16
    # [(H, W)] * 2
    xy = torch.meshgrid(torch.arange(W, dtype=_dtype), torch.arange(H, dtype=_dtype), indexing="xy")
    rr = torch.meshgrid(torch.arange(W, dtype=_dtype), torch.arange(H, dtype=_dtype), indexing="xy")

    xy = [i.to(device) for i in xy]
    rr = [r.to(device) for r in rr]

    # [(B, H, W, H*W)] * 2
    # rr = [i[None, None, None].repeat(B, H, W, 1, 1).flatten(-2, -1) for i in rr]
    rr = [i[None, None, None].flatten(-2, -1) for i in rr]
    # [(B, H, W)] * 2
    rr = [torch.sum(torch.mul(i, probability_matrix), dim=-1) for i in rr]

    for idx in range(B):
        rr[0][idx] -= xy[0]
    for idx in range(B):
        rr[1][idx] -= xy[1]

    # (B, H, W, 2)
    flow = torch.stack(rr, dim=-1)
    return flow


def resize_flows(flows, H, W, order="HWC"):
    """flows: (B, H_, W_, 2) / (B, 2, H_, W_)"""
    if order == "HWC":
        flows = flows.permute(0, 3, 1, 2)
    elif order == "CHW":
        pass
    else:
        raise ValueError("Unknown order: {}".format(order))

    B, _, H_, W_ = flows.shape
    flows = torch.nn.functional.interpolate(
        flows, size=(H, W), mode="bilinear", align_corners=True
    )
    flows[:, 0] = flows[:, 0] * (W / W_)
    flows[:, 1] = flows[:, 1] * (H / H_)

    if order == "HWC":
        # (B, H, W, 2)
        flows = flows.permute(0, 2, 3, 1)
    return flows


def trim_statedict_keys(model):
    new_model = OrderedDict()
    for key, item in model.items():
        new_key = key
        if "module" in key:
            new_key = new_key[7:]  # remove "module"
        new_model[new_key] = item
    return new_model


def normalize_tensor(tensor):
    """
    tensor (torch.Tensor: float): (B, C, H, W)
    """

    assert tensor.ndim == 4

    B, C, H, W = tensor.shape

    # (B, 1)
    t_min = torch.min(tensor.view(B, -1), dim=1, keepdim=True).values
    t_max = torch.max(tensor.view(B, -1), dim=1, keepdim=True).values

    # (B, 1, 1, 1)
    t_min = t_min.view(B, 1, 1, 1)
    t_max = t_max.view(B, 1, 1, 1)

    rescaled_tensor = torch.div(tensor - t_min, t_max - t_min)
    return rescaled_tensor



def At_from_all_pairs(all_pairs):
    # (B, T, H, W, H, W)
    B, T, H, W, _, _ = all_pairs.shape
    As = all_pairs.reshape(B, T, H * W, H * W)
    # (B, H*W, H*W)
    At = As[:, 0]
    probabilities = [At.clone()]
    for i in range(1, T):
        At = At @ As[:, i]
        probabilities.append(At.clone())
    probabilities = torch.stack(probabilities, dim=1)
    probabilities = probabilities.reshape(B, T, H, W, H, W)
    return probabilities


def read_results_tsv(infile):
    # Verify we can read a tsv
    TSV_FIELDNAMES = [
        "id",
        "query_points",
        "pred_tracks",
        "pred_occluded",
        "gt_tracks",
        "gt_occluded",
    ]
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for item in reader:
            item["id"] = int(item["id"])
            item["query_points"] = np.frombuffer(
                base64.b64decode(item["query_points"]), dtype=np.float32
            ).reshape((1, 256, 3))
            item["pred_tracks"] = np.frombuffer(
                base64.b64decode(item["pred_tracks"]), dtype=np.float32
            ).reshape((1, 256, 24, 2))
            item["pred_occluded"] = np.frombuffer(
                base64.b64decode(item["pred_occluded"]), dtype=bool
            ).reshape((1, 256, 24))
            item["gt_tracks"] = np.frombuffer(
                base64.b64decode(item["gt_tracks"]), dtype=np.float32
            ).reshape((1, 256, 24, 2))
            item["gt_occluded"] = np.frombuffer(
                base64.b64decode(item["gt_occluded"]), dtype=bool
            ).reshape((1, 256, 24))
            in_data.append(item)
    return in_data


def read_data_tsv(infile, debug=False):
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
    return in_data


