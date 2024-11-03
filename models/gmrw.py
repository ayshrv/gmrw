import logging
import random
from time import gmtime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from models.gmflow_model.backbone import CNNEncoder
from models.gmflow_model.geometry import coords_grid
from models.gmflow_model.matching import global_correlation_softmax
from models.gmflow_model.transformer import FeatureTransformer
from utils import utils
from models.gmflow_model.utils import (
    feature_add_position,
    normalize_imgs,
    unnormalize_imgs
)

logger = logging.getLogger(__name__)

EPS = 1e-20
NEGATIVE_INF = float("-inf")


class GMRW(nn.Module):
    def __init__(
        self, 
        args,
        num_scales=1,
        upsample_factor=8,
        feature_channels=128,
        attention_type="swin",
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        num_head=1,
        norm=True,
        flash_attention=False,
        gradient_checkpointing=False,
        inherited=False,
    ):  
        super(GMRW, self).__init__()

        self.args = args

        self.norm = norm
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers
        self.num_head = num_head
        self.attention_type = attention_type
        self.ffn_dim_expansion = ffn_dim_expansion
        self.flash_attention = flash_attention

        # CNN backbone
        self.backbone = CNNEncoder(
            output_dim=feature_channels, num_output_scales=num_scales
        )

        # Transformer
        self.transformer = FeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            attention_type=attention_type,
            ffn_dim_expansion=ffn_dim_expansion,
            flash_attention=flash_attention,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.loss_fn = utils.LogNLLMaskNeighbourLoss(aggregation="sum")

        self.attn_splits = 2

        assert self.args.no_of_frames == 2

        # hold temporary tensor shapes
        self.reset_tensor_shapes()

    def reset_tensor_shapes(self):
        self.B = None
        self.T = None
        self.C = None
        self.H = None
        self.W = None
        self.H_ = None
        self.W_ = None
        self.C_ = None


    def extract_feature(self, img0, img1):
        b1 = img0.shape[0]
        b2 = img1.shape[0]

        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(
            concat
        )  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            feature0.append(feature[:b1])
            feature1.append(feature[b1:])

        return feature0, feature1

    def prepare_images(self, images1, images2):
        # images1, images2: (B, 2*T-2, C, H, W)

        # B*(2T-2), C, H, W
        images1 = images1.reshape(-1, self.C, self.H, self.W)
        images2 = images2.reshape(-1, self.C, self.H, self.W)

        return images1, images2

    def get_affine_transformed_labels(self, affine_mat, H_, W_, img2feature_downsample):
        init_grid = coords_grid(self.B, H_, W_, device=self.args.device)
        label_init_grid = utils.align_feat(
            init_grid,
            affine_mat,
        )
        label_init_grid = label_init_grid.permute(0, 2, 3, 1)
        label_init_grid = torch.round(label_init_grid).to(int)
        label_init_grid = label_init_grid[:, :, :, 1] * W_ + label_init_grid[:, :, :, 0]
        label = torch.nn.functional.one_hot(label_init_grid, num_classes=H_ * W_)
        label = label.reshape(self.B, -1, H_ * W_)

        mask = torch.ones(self.B, 1, H_, W_, device=self.args.device)
        mask = utils.align_feat(
            mask,
            affine_mat,
        )

        mask = mask > 0.5
        mask = mask.view(self.B, -1).float()

        return label, mask

    def run_gmflow(self, images1, images2):
        # resolution low to high
        feature1_list, feature2_list = self.extract_feature(
            images1, images2
        )  # list of features
        
        # hi_res_features:
        features_idx = 1

        feature1, feature2 = feature1_list[features_idx], feature2_list[features_idx]

        _, self.C_, self.H_, self.W_ = feature1.shape

        # add position to features
        feature1, feature2 = feature_add_position(
            feature1, feature2, self.attn_splits, self.feature_channels
        ) # (B*(2T-1), C_, H_, W_)

        # Transformer
        feature1, feature2 = self.transformer(
            feature1, feature2, attn_num_splits=self.attn_splits
        )

        temp_B, _, H_, W_ = feature1.shape
        self.H_ = H_
        self.W_ = W_

        if self.args.temp == 0:
            temperature = None
        else:
            temperature = self.args.temp

        probabilities = global_correlation_softmax(
            feature1, 
            feature2, 
            temperature=temperature, 
            flash_attention=False, 
            weighted_argmax=(self.args.compute_flow_method=="weighted_argmax"),
            window_threshold=self.args.window_threshold,
        )

        return probabilities

    def compute_cycle_consistency_loss(self, images1, images2, affine_mat_b2f):
        # images1, images2: (B, T, C, H, W)
        
        # flat_images: (B*T, C, H, W)
        flat_images1, flat_images2 = self.prepare_images(images1, images2)

        probabilities = self.run_gmflow(
            flat_images1, flat_images2
        )

        # _, H_, W_ = probabilities.shape[:3]
        all_pairs = probabilities.reshape(
            self.B, 2 * self.T - 2, self.H_, self.W_, self.H_, self.W_
        )

        As = all_pairs.reshape(
            self.B, 2 * self.T - 2, self.H_ * self.W_, self.H_ * self.W_
        )

        At = As[:, 0]
        for i in range(1, As.shape[1]):
            At = At @ As[:, i]

        label, mask = self.get_affine_transformed_labels(
            affine_mat=affine_mat_b2f,
            H_=self.H_,
            W_=self.W_,
            img2feature_downsample=self.args.img2feature_downsample,
        )

        diags = dict()
        variables_log = dict()

        cycle_loss = self.loss_fn(At, label, mask)  # loss masked mean
        acc = 100 * utils.compute_acc(At, label, mask)

        diags["cycle_loss"] = cycle_loss
        diags["accuracy"] = acc

        smoothness_loss = 0.0

        if self.args.smoothness_loss:
            # flat_images1 # (B*(2T-2), 3, H, W)
            image_h, image_w = flat_images1.shape[2:]
            flat_all_pairs = all_pairs.reshape(
                self.B * (2 * self.T - 2), self.H_, self.W_, self.H_, self.W_
            )
            images_for_sl = 2 * unnormalize_imgs(flat_images1) - 1  # (B*(2T-2), C, H, W)

            flows = utils.compute_flow_from_affinity(
                flat_all_pairs,
                H=image_h,
                W=image_w,
                weight_method="weighted_sum",
                device=self.args.device,
            )
            flows = flows.permute(0, 2, 3, 1)

            (
                smoothness_loss,
                weights_xx,
                weights_yy,
                img_gx,
                img_gy,
                flow_gx,
                flow_gy,
                flow_gxx,
                flow_gyy,
            ) = utils.smoothness_loss(
                [flows],
                [images_for_sl],
                self.args.smoothness_edge_constant,
            )

            variables_log["weights_xx"] = (
                weights_xx[0].view(self.B, 2 * self.T - 2, self.H, self.W - 2).detach()
            )
            variables_log["weights_yy"] = (
                weights_yy[0].view(self.B, 2 * self.T - 2, self.H - 2, self.W).detach()
            )
            variables_log["flow_gxx"] = (
                flow_gxx[0].view(self.B, 2 * self.T - 2, 2, self.H, self.W - 2).detach()
            )
            variables_log["flow_gyy"] = (
                flow_gyy[0].view(self.B, 2 * self.T - 2, 2, self.H - 2, self.W).detach()
            )

        return (
            all_pairs,
            cycle_loss,
            acc,
            smoothness_loss,
            diags,
            variables_log,
        )


    def forward(
        self,
        x,
        affine_mat_b2f,
        smoothness_loss_weight,
    ):
        # Pixels to Features
        B, T, C, H, W = x.shape
        self.B = B
        self.T = T
        self.C = C
        self.H = H
        self.W = W

        self.T = int(self.T / 2)

        assert self.T == self.args.no_of_frames
        assert self.C == 3

        variables_log = {}
        diags = {}

        # [B, 2*T, 3, H, W]
        images = normalize_imgs(x, divideby255=self.norm)

        # [B, T, 3, H, W]
        # forward_images: 1, 2, ..., n-1, n
        # backward_images: n', (n-1)', ..., 2', 1'
        forward_images = images[:, : self.T]
        backward_images = images[:, self.T :]

        # images1: 1, 2, ..., n-1,   n   , (n-1)', ..., 3', 2'
        # images2: 2, 3, ...,  n , (n-1)', (n-2)', ..., 2', 1'
        # images1, images2: (B, 2*T-2, C, H, W)
        images1 = torch.cat([forward_images, backward_images[:, 1:-1]], dim=1)
        images2 = torch.cat([forward_images[:, 1:], backward_images[:, 1:]], dim=1)

        (
            all_pairs,
            cycle_loss,
            acc,
            smoothness_loss,
            diags,
            variables_log,
        ) = self.compute_cycle_consistency_loss(images1, images2, affine_mat_b2f,)

        if self.args.local_rank not in [-2, -1]:
            # Take mean across all GPUs
            cycle_loss /= dist.get_world_size()
            acc /= dist.get_world_size()
            dist.all_reduce(cycle_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
            if self.args.smoothness_loss:
                smoothness_loss /= dist.get_world_size()
                dist.all_reduce(smoothness_loss, op=dist.ReduceOp.SUM)

        diags["cycle loss"] = cycle_loss
        diags["accuracy (out of 100%)"] = acc
        if self.args.smoothness_loss:
            diags["smoothness loss"] = smoothness_loss

        loss = 0.0
        loss += cycle_loss
        if self.args.smoothness_loss:
            loss += smoothness_loss_weight * smoothness_loss
        self.reset_tensor_shapes()

        return all_pairs, loss, diags, variables_log

    def compute_correspondence(self, images):
        B, T, C, H, W = images.shape

        self.B = B
        self.T = T
        self.C = C
        self.H = H
        self.W = W

        # [B, T, 3, H, W]
        images = normalize_imgs(images, divideby255=self.norm)

        # (B, T/2, 3, H, W)
        images1 = images[:, : self.T // 2]
        images2 = images[:, self.T // 2 :]

        # (B*(T/2), 3, H, W)
        flat_images1, flat_images2 = self.prepare_images(images1, images2)

        feature1_list, feature2_list = self.extract_feature(flat_images1, flat_images2)

        features_idx = 1
        # (B*(T/2), C_, H_, W_)
        feature1, feature2 = feature1_list[features_idx], feature2_list[features_idx]

        _, C_, H_, W_ = feature1.shape
        self.C_ = C_
        self.H_ = H_
        self.W_ = W_

        # feature1, feature2: (B*(T/2), C_, H_, W_), (B*(T/2), C, H, W)
        feature1, feature2 = feature_add_position(
            feature1, feature2, self.attn_splits, self.feature_channels
        )

        # (B, T/2, C_, H_, W_)
        feature1 = feature1.reshape(self.B, self.T // 2, self.C_, self.H_, self.W_)
        feature2 = feature2.reshape(self.B, self.T // 2, self.C_, self.H_, self.W_)

        # (B, T, C_, H_, W_)
        features = torch.cat([feature1, feature2], dim=1)

        # (B, T-1, C_, H_, W_,)
        feature1 = features[:, :-1]
        feature2 = features[:, 1:]

        feature1 = feature1.reshape(self.B * (self.T - 1), self.C_, self.H_, self.W_)
        feature2 = feature2.reshape(self.B * (self.T - 1), self.C_, self.H_, self.W_)

        # Transformer
        feature1, feature2 = self.transformer(
            feature1, feature2, attn_num_splits=self.attn_splits
        )

        temp_B, _, H_, W_ = feature1.shape
        self.H_ = H_
        self.W_ = W_

        if self.args.temp == 0:
            temperature = None
        else:
            temperature = self.args.temp

        probabilities = global_correlation_softmax(
            feature1, feature2, temperature=temperature, flash_attention=self.flash_attention
        )

        all_pairs = probabilities.reshape(
            self.B,
            self.T - 1,
            self.H_,
            self.W_,
            self.H_,
            self.W_,
        )

        return all_pairs


    def compute_correspondence_for_pairs(self, images1, images2, return_features=False):

        B, T, C, H, W = images1.shape

        self.B = B
        self.T = T
        self.C = C
        self.H = H
        self.W = W

        # [B, T, 3, H, W]
        images1 = normalize_imgs(images1, divideby255=self.norm)
        images2 = normalize_imgs(images2, divideby255=self.norm)

        # (B*T, 3, H, W)
        flat_images1, flat_images2 = self.prepare_images(images1, images2)

        feature1_list, feature2_list = self.extract_feature(flat_images1, flat_images2)

        # (B*T, C_, H_, W_)
        features_idx = 1
        feature1, feature2 = feature1_list[features_idx], feature2_list[features_idx]

        _, C_, H_, W_ = feature1.shape
        self.C_ = C_
        self.H_ = H_
        self.W_ = W_

        # feature1, feature2: (B*T, C_, H_, W_), (B*T, C, H, W)
        feature1, feature2 = feature_add_position(
            feature1, feature2, self.attn_splits, self.feature_channels
        )

        # Transformer
        feature1, feature2 = self.transformer(
            feature1, feature2, attn_num_splits=self.attn_splits
        )

        if return_features:
            feature1 = feature1.reshape(self.B, self.T, self.C_, self.H_, self.W_)
            feature2 = feature2.reshape(self.B, self.T, self.C_, self.H_, self.W_)
            return feature1, feature2

        if self.args.temp == 0:
            temperature = None
        else:
            temperature = self.args.temp

        probabilities = global_correlation_softmax(
            feature1, feature2, temperature=temperature, flash_attention=self.flash_attention
        )

        all_pairs = probabilities.reshape(
            self.B,
            self.T,
            self.H_,
            self.W_,
            self.H_,
            self.W_,
        )

        return all_pairs