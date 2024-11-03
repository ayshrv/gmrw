import torch
import torch.nn.functional as F

from models.gmflow_model.geometry import (coords_grid, generate_5d_window_grid,
                                          generate_window_grid,
                                          normalize_coords)

# from torch.nn.functional import scaled_dot_product_attention

NEGATIVE_INF = float("-inf")

torch.backends.cuda.enable_flash_sdp(enabled=True)

def global_correlation_softmax(
    feature1,
    feature2,
    # boundary_nodes_mask=None,
    # boundary_padding=0,
    softmax=True,
    temperature=None,
    flash_attention=False,
    weighted_argmax=False,
    window_threshold=5,
):
    # global correlation
    b1, c1, h1, w1 = feature1.shape
    b2, c2, h2, w2 = feature2.shape
    assert b1 == b2
    assert c1 == c2
    b = b1
    c = c1
    feature1 = feature1.view(b, c, -1).permute(0, 2, 1)  # [B, H1*W1, C]
    feature2 = feature2.view(b, c, -1)  # [B, C, H2*W2]

    temperature = c**0.5 if temperature is None else temperature

    if h1 >= 256 and w1 >= 256:
        # print("FLOAT16 forced!!!!")
        feature1 = feature1.to(torch.float16)
        feature2 = feature2.to(torch.float16)

    if flash_attention is True:
        feature2 = feature2.permute(0, 2, 1)
        v = torch.eye(h1*w1, dtype=feature1.dtype, device=feature1.device).unsqueeze(0).repeat(b, 1, 1)
        
        prob = scaled_dot_product_attention(query=feature1, key=feature2, value=v)
    else:
        correlation = torch.matmul(feature1, feature2).view(b, h1, w1, h2, w2) / (
            temperature
        )  # [B, H1, W1, H2, W2]

        correlation = correlation.view(b, h1 * w1, h2 * w2)  # [B, H1*W1, H2*W2]
        
        if softmax:
            if correlation.dtype == torch.bfloat16:
                prob = F.softmax(correlation, dim=-1, dtype=torch.bfloat16)  # [B, H1*W1, H2*W2]
            else:
                prob = F.softmax(correlation, dim=-1)
        else:
            prob = correlation

        if weighted_argmax:

            # (2, H, W), xy
            init_grid = coords_grid(1, h1, w1, device=feature1.device)[0]
            # (H*W, 2), xy
            init_grid = init_grid.permute(1, 2, 0).reshape(h1*w1, 2)

            # (B, H*W)
            max_indices = prob.argmax(dim=-1).reshape(b, h1*w1)
            # (B, H*W, 2) xy
            max_indices_pos = init_grid[max_indices]
            # (B, H*W, 1, 2) xy
            max_indices_pos = max_indices_pos[:, :, None]
            # (1, 1, H*W, 2) xy
            init_grid = init_grid[None, None]

            # (B, H*W, H*W)
            valid_pos = torch.sum(torch.square(max_indices_pos - init_grid), dim=-1) < window_threshold**2
            valid_pos = valid_pos.reshape(b, h1*w1, h2*w2)

            prob = valid_pos * prob
            sum_of_weights = torch.max(torch.tensor(1e-12), torch.sum(prob, dim=-1)) # (B, H*W)
            # (B, H, W, H*W)
            prob = torch.div(prob, sum_of_weights[..., None])

    prob = prob.reshape(b, h1, w1, h2, w2)

    return prob


def local_correlation_softmax_for_fwarp(
    feature1,
    feature2,
    local_radius,
    padding_mode="zeros",
):
    
    NEGATIVE_INF = float("-inf")

    b1, c1, h1, w1 = feature1.shape
    b2, c2, h2, w2 = feature2.shape
    assert b1 == b2
    assert c1 == c2
    b = b1
    c = c1
    h = h1
    w = w1

    coords_init = coords_grid(b, h, w).to(feature1.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(
        -local_radius,
        local_radius,
        -local_radius,
        local_radius,
        local_h,
        local_w,
        device=feature1.device,
    )  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (
        sample_coords[:, :, :, 0] < w
    )  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (
        sample_coords[:, :, :, 1] < h
    )  # [B, H*W, (2R+1)^2]

    valid = (
        valid_x & valid_y
    )  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h1, w1)  # [-1, 1]
    window_feature = F.grid_sample(
        feature2, sample_coords_norm, padding_mode=padding_mode, align_corners=True
    ).permute(
        0, 2, 1, 3
    )  # [B, H*W, C, (2R+1)^2]
    feature1_view = feature1.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature1_view, window_feature).view(b, h * w, -1) / (
        c**0.5
    )  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = NEGATIVE_INF

    corr = corr.view(b, h, w, local_h, local_w)

    full_correlation = torch.full(size=(b, h, w, h, w), fill_value=NEGATIVE_INF, device=feature1.device)

    # (B, H, W, 1, 1, 2)
    new_coords = coords_init.view(b, 2, h, w).permute(0, 2, 3, 1)
    new_coords = new_coords.unsqueeze(-2).unsqueeze(-2)

    # (B, H, W, 2R+1, 2R+1, 5)
    indices5d = generate_5d_window_grid(
        -local_radius, 
        local_radius, 
        -local_radius, 
        local_radius, 
        local_h, 
        local_w, 
        h, w, b,
        device=feature1.device,
    )

    indices5d = indices5d.long()

    # (B, H, W, 2R+1, 2R+1, 5)
    indices5d[..., 3:] = indices5d[..., 3:] + new_coords

    indices5d[..., 3] = (indices5d[..., 3] + w)%w
    indices5d[..., 4] = (indices5d[..., 4] + h)%h

    # (B, H, W, H, W)
    full_correlation = full_correlation.index_put(
        indices=(
            indices5d[..., 0], 
            indices5d[..., 2], 
            indices5d[..., 1], 
            indices5d[..., 4], 
            indices5d[..., 3]
        ), 
        values=corr,
    )
    
    full_correlation = full_correlation.view(b, h*w, h*w)

    prob = F.softmax(full_correlation, -1)  # [B, H*W, H*W]

    prob = prob.reshape(b, h, w, h, w)

    return prob


def correlation_mask_for_local_window(
    correspondence_indices,
    local_radius,
    device,
):
    # correspondence_indices: (B, 2, H, W)
    b, _, h, w = correspondence_indices.shape
    correspondence_coords = correspondence_indices.permute(0, 2, 3, 1)
    # (b, h, w, h, w, 2)
    correspondence_coords = correspondence_coords.view(b, h, w, 1, 1, 2).repeat(1, 1, 1, h, w, 1)



    coord_init = coords_grid(b*h*w, h, w).to(device) # (b*h*w, 2, h, w)
    coord_init = coord_init.permute(0, 2, 3, 1)
    coord_init = coord_init.view(b, h, w, h, w, 2)
    correspondence_coords = torch.cat([correspondence_coords, coord_init], dim=-1) # (B, H, W, H, W, 4)

    correspondence_coords[..., 2] = (correspondence_coords[..., 2] - correspondence_coords[..., 0]).abs()
    correspondence_coords[..., 3] = (correspondence_coords[..., 3] - correspondence_coords[..., 1]).abs()

    # (B, H, W, H, W)
    mask_x = correspondence_coords[..., 2] <= local_radius
    mask_y = correspondence_coords[..., 3] <= local_radius

    mask = mask_x & mask_y
    mask = mask.bool()
    return mask
