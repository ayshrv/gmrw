import torch

def bilinear_sample2d(im, x, y, return_inbounds=False, device=None):
    """Uses flow to compute coordinates, i.e. use relative coordinates"""

    if device is None:
        device = torch.device("cpu")

    # x and y are each B, N
    # output is B, C, N
    B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64).to(device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
    i_y0_x0 = im_flat[idx_y0_x0.long()]
    i_y0_x1 = im_flat[idx_y0_x1.long()]
    i_y1_x0 = im_flat[idx_y1_x0.long()]
    i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = (
        w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    )
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(
            B, N
        )  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output  # B, C, N

def meshgrid2d(B, Y, X, stack=False, norm=False, device=None):
    # returns a meshgrid sized B x Y x X
    if device is None:
        device = torch.device("cpu")

    grid_y = torch.linspace(0.0, Y - 1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x



def get_tracks_from_overall_flows(flows, coord_t0=None, device=None):
    B, T, C, H, W = flows.shape
    assert C == 2

    T = T + 1

    if coord_t0 is None:
        ys, xs = meshgrid2d(B, H, W, device=device)
        xs = xs.view(B, -1)
        ys = ys.view(B, -1)

        coord_t0 = torch.stack([xs, ys], dim=2)  # B, N, 2

    coords = [coord_t0]

    for s in range(T - 1):
        delta = bilinear_sample2d(
            flows[:, s], coord_t0[:, :, 0], coord_t0[:, :, 1], device=device
        ).permute(
            0, 2, 1
        )  # B, N, 2: forward flow at the discrete points

        coord = coord_t0 + delta
        coords.append(coord)
    trajs = torch.stack(coords, dim=1)  # B, T, N, 2

    return trajs


def get_tracks_from_flows(flows, coord=None, device=None):
    B, T, C, H, W = flows.shape
    assert C == 2

    T = T + 1  # flows: (B, T-1, 2, H, W)

    if coord is None:
        ys, xs = meshgrid2d(B, H, W, device=device)
        xs = xs.view(B, -1)
        ys = ys.view(B, -1)

        coord = torch.stack([xs, ys], dim=2)  # B, N, 2

    coords = [coord]

    for s in range(T - 1):
        delta = bilinear_sample2d(
            flows[:, s], coord[:, :, 0], coord[:, :, 1], device=device
        ).permute(
            0, 2, 1
        )  # B, N, 2: forward flow at the discrete points
        coord = coord + delta
        coords.append(coord)
    trajs = torch.stack(coords, dim=1)  # B, T, N, 2

    return trajs

def get_tracks_from_direct_flows_query_points(query_points, flows, device):
    # query_points: (B, N, 3) # (t, y, x)
    B, T, C, H, W = flows.shape
    assert C == 2

    _, N, _ = query_points.shape

    temp = query_points[..., 1].clone()  # y
    query_points[..., 1] = query_points[..., 2].clone()  # x
    query_points[..., 2] = temp  # y

    coords = []
    for t in range(T):
        prev_coord = query_points[:, :, 1:]
        delta = bilinear_sample2d(
            im=flows[:, t],
            x=prev_coord[:, :, 0],
            y=prev_coord[:, :, 1],
            device=device,
        ).permute(0, 2, 1)
        assert delta.shape == (B, N, 2), "Forward flow at the discrete points"
        coord = prev_coord + delta

        # Set the ground truth query point location if the timestep is correct
        query_point_mask = query_points[:, :, 0] == t
        coord = coord * ~query_point_mask.unsqueeze(-1) + query_points[:, :, 1:] * query_point_mask.unsqueeze(-1)

        coords.append(coord)

    # (B, T, N, 2)
    trajectories = torch.stack(coords, dim=1)
    # (B, T, N)
    visibilities = (trajectories[:, :, :, 0] >= 0) & \
                    (trajectories[:, :, :, 1] >= 0) & \
                    (trajectories[:, :, :, 0] < W) & \
                    (trajectories[:, :, :, 1] < H)

    return trajectories, visibilities

def get_tracks_from_flows_query_points(query_points, forward_flows, backward_flows, device):
    # query_points: (B, N, 3) # (t, y, x)
    B, T, C, H, W = forward_flows.shape
    assert C == 2
    B, T, C, H, W = backward_flows.shape
    assert C == 2

    T = T + 1

    _, N, _ = query_points.shape

    temp = query_points[..., 1].clone()  # y
    query_points[..., 1] = query_points[..., 2].clone()  # x
    query_points[..., 2] = temp  # y

    coords = []
    for t in range(T):
        if t == 0:
            coord = torch.zeros_like(query_points[:, :, 1:])
        else:
            prev_coord = coords[t - 1]
            delta = bilinear_sample2d(
                im=forward_flows[:, t - 1],
                x=prev_coord[:, :, 0],
                y=prev_coord[:, :, 1],
                device=device,
            ).permute(0, 2, 1)
            assert delta.shape == (B, N, 2), "Forward flow at the discrete points"
            coord = prev_coord + delta

        # Set the ground truth query point location if the timestep is correct
        query_point_mask = query_points[:, :, 0] == t
        coord = coord * ~query_point_mask.unsqueeze(-1) + query_points[:, :, 1:] * query_point_mask.unsqueeze(-1)

        coords.append(coord)

    for t in range(T - 2, -1, -1):
        coord = coords[t]
        successor_coord = coords[t + 1]

        delta = bilinear_sample2d(
            im=backward_flows[:, t],
            x=successor_coord[:, :, 0],
            y=successor_coord[:, :, 1],
            device=device,
        ).permute(0, 2, 1)
        assert delta.shape == (B, N, 2), "Backward flow at the discrete points"

        # Update only the points that are located prior to the query point
        prior_to_query_point_mask = t < query_points[:, :, 0]
        coord = (coord * ~prior_to_query_point_mask.unsqueeze(-1) +
                    (successor_coord + delta) * prior_to_query_point_mask.unsqueeze(-1))
        coords[t] = coord

    # (B, T, N, 2)
    trajectories = torch.stack(coords, dim=1)
    # (B, T, N)
    boundary_visibilities = (trajectories[:, :, :, 0] >= 0) & \
                    (trajectories[:, :, :, 1] >= 0) & \
                    (trajectories[:, :, :, 0] < W) & \
                    (trajectories[:, :, :, 1] < H)

    forward_visibilities = []
    for t in range(T-1):
        ff = bilinear_sample2d(
            im=forward_flows[:, t],
            x=trajectories[:, t, :, 0],
            y=trajectories[:, t, :, 1],
            device=device, 
        ).permute(0, 2, 1) # (B, N, 2)
        bb = bilinear_sample2d(
            im=backward_flows[:, t],
            x=trajectories[:, t+1, :, 0],
            y=trajectories[:, t+1, :, 1],
            device=device,
        ).permute(0, 2, 1) # (B, N, 2)
        dist = torch.norm(ff+bb, dim=2)
        forward_visible_t = dist < 3 # (B, N)
        forward_visibilities.append(forward_visible_t)
    forward_visibilities.append(torch.ones_like(forward_visibilities[-1]).to(torch.bool))
    forward_visibilities = torch.stack(forward_visibilities, dim=1) # (B, T, N)

    backward_visibilities = []
    for t in range(T-1, 0, -1):
        ff = bilinear_sample2d(
            im=backward_flows[:, t-1],
            x=trajectories[:, t, :, 0],
            y=trajectories[:, t, :, 1],
            device=device, 
        ).permute(0, 2, 1) # (B, N, 2)
        bb = bilinear_sample2d(
            im=forward_flows[:, t-1],
            x=trajectories[:, t-1, :, 0],
            y=trajectories[:, t-1, :, 1],
            device=device,
        ).permute(0, 2, 1) # (B, N, 2)
        dist = torch.norm(ff+bb, dim=2)
        backward_visible_t = dist < 3
        backward_visibilities.append(backward_visible_t)
    backward_visibilities.append(torch.ones_like(backward_visibilities[-1]).to(torch.bool))
    backward_visibilities = backward_visibilities[::-1]
    backward_visibilities = torch.stack(backward_visibilities, dim=1) # (B, T, N)

    visibilities = boundary_visibilities & forward_visibilities & backward_visibilities
    
    for t in range(T):
        query_point_mask = query_points[:, :, 0] == t  # (B, N)
        visibilities_t = visibilities[:, t]
        visibilities_t[query_point_mask] = True
        visibilities[:, t] = visibilities_t

    return trajectories, visibilities



def filter_tracks(
    trajs,
    flows_f,
    flows_b,
    segmentations=None,
    boundary_padding=0,
    device=None,
):
    B, T, N, D = trajs.shape

    _, _, _, H, W = flows_f.shape
    assert flows_f.shape[1] == T - 1  # (B, T-1, H, W)
    assert flows_b.shape[1] == T - 1  # (B, T-1, H, W)

    if segmentations is not None:
        _, _, _, H, W = segmentations.shape  # (B, T, H, W)
        assert segmentations.shape[1] == T

    assert D == 2

    trajs = trajs.round()  # B, T, N, 2

    ## Inframe condition
    max_x = torch.max(trajs[:, :, :, 0], dim=1)[0]  # (B, N)
    min_x = torch.min(trajs[:, :, :, 0], dim=1)[0]
    max_y = torch.max(trajs[:, :, :, 1], dim=1)[0]
    min_y = torch.min(trajs[:, :, :, 1], dim=1)[0]

    # (B, N)
    inbound_pixels_single = (
        (max_x <= W - 1) & (min_x >= 0) & (max_y <= H - 1) & (min_y >= 0)
    )

    # (B, T, N)
    inbound_pixels = inbound_pixels_single[:, None, :].repeat(1, T, 1)

    if segmentations is not None:
        # N
        id0 = bilinear_sample2d(
            segmentations[:, 0],
            trajs[:, 0, :, 0].round(),
            trajs[:, 0, :, 1].round(),
            device=device,
        ).view(B, -1)

        segmentation_t = []
        for t in range(T):
            idt = inbound_pixels_single.clone()
            # so let's require the 3x3 neighborhood to match
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    idi = bilinear_sample2d(
                        segmentations[:, t],
                        trajs[:, t, :, 0].round() + dx,
                        trajs[:, t, :, 1].round() + dy,
                        device=device,
                    ).view(
                        B, -1
                    )  # N

                    idt = idt & (idi == id0)  # N
            segmentation_t.append(idt)
        segmentation_t = torch.stack(segmentation_t, dim=1)  # B, T, N
    else:
        segmentation_t = inbound_pixels.clone()

    visible_t = [inbound_pixels_single.clone()]

    for t in range(T - 1):
        idi = inbound_pixels_single.clone()
        ff = (
            bilinear_sample2d(
                flows_f[:, t],
                trajs[:, t, :, 0].round(),
                trajs[:, t, :, 1].round(),
                device=device,
            )
            .permute(0, 2, 1)
            .view(B, -1, 2)
        )  # B, N, 2

        bf = (
            bilinear_sample2d(
                flows_b[:, t],
                trajs[:, t + 1, :, 0].round(),
                trajs[:, t + 1, :, 1].round(),
                device=device,
            )
            .permute(0, 2, 1)
            .view(B, -1, 2)
        )  # B, N, 2

        dist = torch.norm(ff + bf, dim=2)
        idt = idi & (dist < 0.5)
        visible_t.append(idt)

    visible_t = torch.stack(visible_t, dim=1)  # B, T, N

    # (B, T, N)
    in_frame_pixels = (
        (trajs[:, :, :, 0] <= W - 1 - boundary_padding)
        & (trajs[:, :, :, 0] >= boundary_padding)
        & (trajs[:, :, :, 1] <= H - 1 - boundary_padding)
        & (trajs[:, :, :, 1] >= boundary_padding)
    )

    # (B, T, N) Only points visible in the first frame
    valid_pixels = (
        inbound_pixels
        & in_frame_pixels[:, 0:1]
        & visible_t[:, 0:1]
        & segmentation_t[:, 0:1]
    )

    # (B, T, N)
    visible_pixels = valid_pixels & in_frame_pixels & visible_t & segmentation_t
    occluded_pixels = valid_pixels & (~visible_pixels)

    filtered_trajs = []
    visible_points = []
    occluded_points = []

    for b_idx in range(trajs.shape[0]):
        b_id_ok = valid_pixels[b_idx, 0]  # (N_)

        traj = trajs[b_idx].permute(1, 0, 2)  # (N, T, 2)
        visible_pts = visible_pixels[b_idx].permute(1, 0)  # (N, T)
        occluded_pts = occluded_pixels[b_idx].permute(1, 0)  # (N, T)

        traj = traj[b_id_ok].permute(1, 0, 2)  # (T, N_, 2)
        visible_pts = visible_pts[b_id_ok].permute(1, 0)  # (T, N_)
        occluded_pts = occluded_pts[b_id_ok].permute(1, 0)  # (T, N_)

        filtered_trajs.append(traj)
        visible_points.append(visible_pts)
        occluded_points.append(occluded_pts)

    # filtered_trajs: B*[(T, N_, 2)]
    # visible_points, occluded_points: B*[(T, N_)]
    # visible_points = ! occluded_points

    return filtered_trajs, visible_points, occluded_points


def get_occlusions_from_forward_backward_flows(
    forward_flow, backward_flow, device=None
):
    # forward_flow, backward_flow: (B, T, 2, H, W)
    B, T1, _, H, W = forward_flow.shape
    B, T2, _, H, W = backward_flow.shape
    assert T1 == T2
    T = T1

    if device is None:
        device = forward_flow.device

    ys, xs = meshgrid2d(B * T, H, W, device=device)
    xs = xs.view(B * T, -1)
    ys = ys.view(B * T, -1)

    coords = torch.stack([xs, ys], dim=2)  # B*T, N, 2

    forward_flow_p = forward_flow.reshape(B * T, 2, H, W)  # (B*T, 2, H, W)
    backward_flow_p = backward_flow.reshape(B * T, 2, H, W)  # (B*T, 2, H, W)

    forward_delta = bilinear_sample2d(
        forward_flow_p, coords[:, :, 0], coords[:, :, 1], device=device
    ).permute(
        0, 2, 1
    )  # B*T, N, 2: forward flow at the discrete points

    forward_coords = coords + forward_delta

    backward_delta = bilinear_sample2d(
        backward_flow_p, forward_coords[:, :, 0], forward_coords[:, :, 1], device=device
    ).permute(
        0, 2, 1
    )  # B*T, N, 2: forward flow at the discrete points

    zero_delta = forward_delta + backward_delta  # (B*T, N, 2)
    zero_dist = (zero_delta[..., 0] ** 2 + zero_delta[..., 1] ** 2) ** 0.5  # (B*T, N)
    occlusion = zero_dist > 0.5  # (B*T, N)
    occlusion = occlusion.reshape(B, T, 1, H, W)
    return occlusion.float()


