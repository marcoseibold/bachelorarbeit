import os
import argparse
from pathlib import Path
import time
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import imageio
from PIL import Image, ImageDraw, ImageFont
import nvdiffrast.torch as dr
import trimesh
from scipy.ndimage import gaussian_filter, label    # occupancy filtering
from skimage.metrics import structural_similarity as ssim
import lpips
import torchvision.transforms.functional as TF
import gc

import dataset_loader
from tensorf_appearance import ColorFieldVM

# ---------------- Arguments ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True, help="Path to mesh (OBJ/PLY).")
parser.add_argument("--data_root", type=str, required=True, help="Path to dataset scene folder (transforms_*.json + images).")
parser.add_argument("--model", type=str, required=True, help="Appearance model: voxel_grid | tensorf | merf")
parser.add_argument("--outdir", type=str, default="train_out", help="Output directory")
parser.add_argument("--voxel_res", type=int, default=512, help="Voxel grid resolution (D,H,W)")
parser.add_argument("--img_res", type=int, default=800, help="Image resolution (H and W)")
parser.add_argument("--steps", type=int, default=5000, help="Training iterations")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--save_interval", type=int, default=100, help="Save image/checkpoint every N steps")
parser.add_argument("--early_stop_patience", type=int, default=500, help="Stop training if no improvement in N steps")
parser.add_argument("--early_stop_delta", type=float, default=1e-6, help="Minimum change to consider an improvement")
parser.add_argument("--bbox_size", type=float, default=2.0, help="Scene bounding box (cube side length) used when no mesh provided")
parser.add_argument("--n_samples", type=int, default=64, help="Samples per ray for volumetric rendering")
parser.add_argument("--near", type=float, default=0.1, help="Near plane for ray sampling")  # 2.5 for m360 (counter, stump 1.0)
parser.add_argument("--far", type=float, default=6.0, help="Far plane for ray sampling")    # 100.0 for m360
parser.add_argument("--img_count", type=int, default=5, help="Training image count")
parser.add_argument("--test_img_count", type=int, default=200, help="Test image count")
args = parser.parse_args()

MESH_PATH = args.mesh
DATA_ROOT = args.data_root
OUTDIR = args.outdir
VOXEL_RES = args.voxel_res
IMG_RES = args.img_res
STEPS = args.steps
LR = args.lr
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = args.save_interval
PATIENCE = args.early_stop_patience
DELTA = args.early_stop_delta
BBOX_SIZE = args.bbox_size
N_SAMPLES = args.n_samples
NEAR = args.near
FAR = args.far
IMG_COUNT = args.img_count
TEST_IMG_COUNT = args.test_img_count
MODEL = "tensorf" if args.model.lower() == "tensorf" else "merf" if args.model.lower() == "merf" else "voxel_grid"

os.makedirs(OUTDIR, exist_ok=True)
tb_writer = SummaryWriter(log_dir=os.path.join(OUTDIR, "tensorboard"))

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Appearance model: {MODEL}")

# ---------------- helpers ----------------

def perspective(fovy, aspect, near, far):
    #print(f"fovy (rad): {fovy:.4f} ({np.degrees(fovy):.2f}°)")
    #print(f"aspect: {aspect:.4f}, near: {near}, far: {far}")
    f = 1.0 / np.tan(fovy / 2.0)    # Scaling factor for FoV
    #print(f"scaling factor f: {f:.6f}")
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect    # scale x and y
    if dataset_loader.nerf_synthetic:
        m[1, 1] = -f    # y-flip
    else:
        m[1, 1] = f    # deactivate for m360
    m[2, 2] = (far + near) / (near - far)   # map to clip coords
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    #print(f"Projection matrix: {m}")
    return m    # projection matrix (3D to 2D)

def mvp_from_cam(cam2world, H, W, focal, near=0.1, far=10.0):
    #print(f"Image size: H={H}, W={W}, focal={focal}")
    #print(f"near={near}, far={far}")
    view = np.linalg.inv(cam2world).astype(np.float32)  # view-matrix as inverse of cam2world
    #print(f"View matrix {view}")
    fovy = 2.0 * np.arctan(float(H) / (2.0 * float(focal))) # FoV
    #print(f"Computed fovy = {fovy:.4f} rad = {np.degrees(fovy):.2f}°")
    proj = perspective(fovy, float(W) / float(H), near, far)    # projection matrix
    #print("Projection matrix:")
    #print(proj)
    mvp = proj @ view   # multiply to get transformation matrix (world -> clip space)
    #print("MVP = proj @ view:")
    #print(mvp)
    return mvp

# world->grid mapping will be filled later depending on mesh or bbox
bbox_min_t = None   # Bounding box borders
bbox_max_t = None   # Bounding box borders

def world_to_grid_coords(pts):
    # pts: [...,3] in world coords (torch)
    global bbox_min_t, bbox_max_t   # get borders
    #print("bbox_min_t:", bbox_min_t.detach().cpu().numpy())
    #print("bbox_max_t:", bbox_max_t.detach().cpu().numpy())
    #print("pts shape:", tuple(pts.shape))
    norm = (pts - bbox_min_t) / (bbox_max_t - bbox_min_t + 1e-8)    # min-max norm to [0, 1]
    #print("norm range: (%.4f, %.4f)" % (norm.min().item(), norm.max().item()))
    grid = norm * 2.0 - 1.0 # scale to [-1, 1]
    #print("grid range: (%.4f, %.4f)" % (grid.min().item(), grid.max().item()))
    return grid

def render_via_mesh_rasterization_old(voxel_grid_param, cam2world_np, focal, H, W,
                                  vertices, faces_unbatched, ctx, voxel_sample_mode='bilinear',
                                  near=0.1, far=10.0, device='cuda'):
    # Create MVP (view-projection matrix) for rasterizing
    mvp_np = mvp_from_cam(cam2world_np, H, W, focal, near=near, far=far)  # numpy 4x4
    #print("mvp_np shape:", mvp_np.shape, "\n", mvp_np)
    mvp = torch.tensor(mvp_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,4,4]
    #print("mvp tensor:", mvp.shape)

    # Convert vertices (batched [1,V,3]) to [1,V,4] for homogeneous multiply
    ones = torch.ones_like(vertices[:, :, :1])
    verts_h = torch.cat([vertices, ones], dim=-1)  # [1,V,4]
    #print("verts_h:", verts_h.shape, "example:", verts_h[0, 0])

    # pos_clip: multiply vertices with mvp (transpose to match shapes, vertices in clip-space)
    pos_clip = torch.matmul(verts_h, mvp.transpose(1, 2))  # [1,V,4]
    #print("pos_clip:", pos_clip.shape)

    # rasterize -> rast_out
    rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
    #print("rast_out:", rast_out.shape)

    # interpolate world-space positions (vertices are world coords)
    pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)  # [H,W,3]
    pos_map = pos_map[0]  # [H,W,3] on device
    #print("pos_map:", pos_map.shape, "range:", pos_map.min().item(), pos_map.max().item())

    # convert world pos to voxel grid coords in [-1,1]
    grid_coords = world_to_grid_coords(pos_map)  # expects torch, returns [-1,1]
    #print("grid_coords range:", grid_coords.min().item(), grid_coords.max().item())
    # make sampling grid shape [1,1,H,W,3] for grid_sample
    grid_for_sample = grid_coords.unsqueeze(0).unsqueeze(0)  # [1,1,H,W,3]

    # sample/query voxel grid: voxel_grid_param is [1,C,D,H,W]
    sampled = F.grid_sample(voxel_grid_param, grid_for_sample, mode=voxel_sample_mode,
                            padding_mode='border', align_corners=True)  # [1,C,1,H,W]
    #print("sampled:", sampled.shape)
    # Reorder and squeeze dimensions
    sampled = sampled.permute(0, 2, 3, 4, 1)  # [1,1,H,W,C]
    sampled = sampled.squeeze(1)  # [1,H,W,C]
    sampled = sampled[0]  # [H,W,C]

    # Pick first 3 channels of voxel grid as RGB
    pred_rgb = sampled[..., :3]  # [H,W,3]
    #print("pred_rgb:", pred_rgb.shape)

    # Set background to white
    mask = rast_out[0, ..., 3] > 0  # Pixels hit by rasterization
    if dataset_loader.nerf_synthetic:
        pred_rgb[~mask] = 1.0
    if dataset_loader.m360:
        pred_rgb = torch.flip(pred_rgb, dims=[1])   # for m360
    

    # Antialiasing
    pred_rgb_batched = pred_rgb.unsqueeze(0).contiguous().float() # [1,H,W,3]
    rast_aa = rast_out.contiguous().float()
    pos_clip_aa = pos_clip.contiguous().float()
    pred_rgb_aa = dr.antialias(pred_rgb_batched, rast_aa, pos_clip_aa, faces_unbatched)[0]

    return pred_rgb_aa

def render_via_mesh_rasterization(voxel_grid_param, cam2world_np, focal, H, W,
                                  vertices, faces_unbatched, ctx, voxel_sample_mode='bilinear',
                                  near=0.1, far=10.0, device='cuda'):
    vertex_chunk=10_000_000
    chunk_faces = 16_000_000
    # Create MVP (view-projection matrix) for rasterizing
    mvp_np = mvp_from_cam(cam2world_np, H, W, focal, near=near, far=far)  # numpy 4x4
    #print("mvp_np shape:", mvp_np.shape, "\n", mvp_np)
    mvp = torch.tensor(mvp_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,4,4]
    #print("mvp tensor:", mvp.shape)

    # Convert vertices (batched [1,V,3]) to [1,V,4] for homogeneous multiply
    ones = torch.ones_like(vertices[:, :, :1])
    verts_h = torch.cat([vertices, ones], dim=-1)  # [1,V,4]
    #print("verts_h:", verts_h.shape, "example:", verts_h[0, 0])

    # pos_clip: multiply vertices with mvp (transpose to match shapes, vertices in clip-space)
    B, V, _ = verts_h.shape

    # Chunk verts * mvp -> pos_clip
    mvp_t = mvp[0].transpose(0, 1).contiguous().to(device=device, dtype=torch.float32)  # [4,4]
    vertex_chunk = int(min(vertex_chunk, V))
    pos_clip = torch.empty((1, V, 4), device=device, dtype=torch.float32)

    # pos_clip in chunks
    with torch.no_grad():
        for start in range(0, V, vertex_chunk):
            end = min(start + vertex_chunk, V)
            v_slice2d = verts_h[:, start:end, :].contiguous().view(-1, 4)  # [chunk,4]
            out2d = v_slice2d.mm(mvp_t)                                     # [chunk,4]
            pos_clip[:, start:end, :] = out2d.view(1, end - start, 4)
            del v_slice2d, out2d

    pos_clip = pos_clip.contiguous().float()   # final pos_clip [1,V,4]
    #print("pos_clip:", pos_clip.shape)

    # rasterize -> rast_out
    #print("rast_out:", rast_out.shape)

    # interpolate world-space positions (vertices are world coords)
    #print("pos_map:", pos_map.shape, "range:", pos_map.min().item(), pos_map.max().item())

    # convert world pos to voxel grid coords in [-1,1]
    #print("grid_coords range:", grid_coords.min().item(), grid_coords.max().item())
    # make sampling grid shape [1,1,H,W,3] for grid_sample

    # sample/query voxel grid: voxel_grid_param is [1,C,D,H,W]
    #print("sampled:", sampled.shape)
    # Reorder and squeeze dimensions

    F_total = faces_unbatched.shape[0]
    faces = faces_unbatched.contiguous().to(device=device, dtype=torch.int32)

    device = pos_clip.device
    dtype  = pos_clip.dtype
    P = H * W

    # Output buffers
    if dataset_loader.nerf_synthetic:
        color_buffer = torch.ones((1, H, W, 3), device=device, dtype=dtype)
    else:
        color_buffer = torch.zeros((1, H, W, 3), device=device, dtype=dtype)

    depth_buffer = torch.full((1, H, W, 1), float('inf'),
                            device=device, dtype=dtype)

    vertex_positions = vertices[0].float().unsqueeze(0)

    # face-chunk loop
    for s in range(0, F_total, chunk_faces):
        e = min(s + chunk_faces, F_total)
        f_chunk = faces[s:e]

        gc.collect()
        torch.cuda.synchronize()

        # Rasterize
        rast_chunk, _ = dr.rasterize(ctx, pos_clip, f_chunk,
                                    resolution=[H, W])

        # Interpolate clip coords for depth
        interp_clip, _ = dr.interpolate(pos_clip, rast_chunk, f_chunk)
        w = interp_clip[..., 3:4]
        depth_ndc = interp_clip[..., 2:3] / (w + 1e-9)

        # Interpolate world positions
        interp_pos, _ = dr.interpolate(vertex_positions,
                                    rast_chunk, f_chunk)

        valid = (rast_chunk[..., 3] > 0).unsqueeze(-1)
        closer = valid & (depth_ndc < depth_buffer)

        if closer.any():
            idx = closer.view(-1).nonzero(as_tuple=False).squeeze(-1)

            # gather world positions
            interp_pos_flat = interp_pos.view(1, P, 3)[0]
            chosen_xyz = interp_pos_flat[idx]  # [M,3]

            # world -> grid coords [-1,1]
            grid_coords = world_to_grid_coords(chosen_xyz)
            grid = grid_coords.view(1, 1, -1, 1, 3)

            # sample voxel grid
            sampled = F.grid_sample(
                voxel_grid_param,
                grid,
                mode=voxel_sample_mode,
                padding_mode='border',
                align_corners=True
            )  # [1,C,1,M,1]

            rgb = sampled[0, :3, 0, :, 0].permute(1, 0)
            rgb = torch.clamp(rgb, 0.0, 1.0)

            # write color
            color_flat = color_buffer.view(1, P, 3)[0]
            color_flat[idx] = rgb
            color_buffer = color_flat.view(1, H, W, 3)

            # update depth
            depth_flat = depth_buffer.view(1, P, 1)[0]
            depth_flat[idx, 0] = depth_ndc.view(-1)[idx]
            depth_buffer = depth_flat.view(1, H, W, 1)

        del rast_chunk, interp_clip, interp_pos

    # Pick first 3 channels of voxel grid as RGB
    #print("pred_rgb:", pred_rgb.shape)

    # Set background to white
    
    if not dataset_loader.nerf_synthetic:
        # hit mask = pixels with finite depth
        hit_mask = (depth_buffer[0, ..., 0] < float('inf'))   # [H,W], bool
        hit_mask_flat = hit_mask.view(-1)
        color_flat = color_buffer.view(1, P, 3)[0]  # [P,3]

        if hit_mask.any():
            # compute mean color over hit pixels
            hit_colors = color_flat[hit_mask_flat]
            mean_color = hit_colors.mean(dim=0)  # [3]
        else:
            # fallback: mean over entire buffer
            mean_color = color_flat.mean(dim=0)

        # assign mean_color to all non-hit pixels
        nonhit_idx = (~hit_mask_flat).nonzero(as_tuple=False).squeeze(-1)
        if nonhit_idx.numel() > 0:
            color_flat[nonhit_idx] = mean_color.unsqueeze(0).expand(nonhit_idx.numel(), 3)
            color_buffer = color_flat.view(1, H, W, 3)

    # final image
    pred_rgb = color_buffer[0].permute(1, 2, 0)  # [H,W,3]
    pred_rgb = pred_rgb.permute(2, 0, 1).contiguous()

    if dataset_loader.m360:
        pred_rgb = torch.flip(pred_rgb, dims=[1])   # for m360
        
    # Antialiasing
    # pred_rgb_batched = pred_rgb.unsqueeze(0).contiguous().float() # [1,H,W,3]
    # rast_aa = rast_out.contiguous().float()
    # pos_clip_aa = pos_clip.contiguous().float()
    # pred_rgb_aa = dr.antialias(pred_rgb_batched, rast_aa, pos_clip_aa, faces_unbatched)[0]

    return pred_rgb

def render_via_mesh_rasterization_colorfield_old(
    color_field, cam2world_np, focal, H, W,
    vertices, faces_unbatched, ctx,
    near=0.1, far=10.0, device='cuda'):
    # Create MVP, convert to tensor
    mvp_np = mvp_from_cam(cam2world_np, H, W, focal, near=near, far=far)
    mvp = torch.tensor(mvp_np, dtype=torch.float32, device=device).unsqueeze(0)
    # Rasterize homogene vertices converted to clip space
    ones = torch.ones_like(vertices[:, :, :1])
    verts_h = torch.cat([vertices, ones], dim=-1)   # [1,V,4]
    vertex_chunk = 100000
    V = verts_h.shape[1]
    pos_clip_chunks = []

    with torch.no_grad():
        for start in range(0, V, vertex_chunk):
            end = min(start + vertex_chunk, V)
            v_chunk = verts_h[:, start:end, :]
            pc_chunk = torch.matmul(v_chunk, mvp.transpose(1, 2))
            pos_clip_chunks.append(pc_chunk)

    pos_clip = torch.cat(pos_clip_chunks, dim=1).to(device=device, dtype=torch.float32).contiguous()
    del pos_clip_chunks

    #pos_clip = torch.matmul(verts_h, mvp.transpose(1, 2))   # [1,V,4]
    rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
    # World positions per pixel
    pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)  # [1,H,W,3]
    pos_map = pos_map[0]    # [H,W,3]
    # Get view-direction (point -> camera)
    cam_pos = torch.tensor(cam2world_np[:3, 3], dtype=torch.float32, device=device)  # [3]
    viewdirs = cam_pos.view(1,1,3) - pos_map    # camera - world coords [H,W,3]
    viewdirs = viewdirs / (viewdirs.norm(dim=-1, keepdim=True) + 1e-8)  # normalize

    P = H * W   # Convert to lists:
    xyz = pos_map.reshape(P, 3) # World points
    vdir = viewdirs.reshape(P, 3)   # Normalized view directions

    if (dataset_loader.nerf_synthetic):
        # Mask
        mask = (rast_out[0, ..., 3] > 0)    # [H,W]
        # Only use visible pixels
        valid_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)    # [M]
        rgb_out = torch.ones((P, 3), device=device, dtype=torch.float32)    # white background
        if valid_idx.numel() > 0:
            rgb_valid = color_field(xyz[valid_idx], vdir[valid_idx])    # RGB in [M,3]
            rgb_out[valid_idx] = torch.clamp(rgb_valid, 0.0, 1.0)   # Clamp
        img = rgb_out.view(H, W, 3) # [H,W,3]
    else:
        rgb_all = color_field(xyz, vdir)    # [P, 3]
        img = rgb_all.view(H, W, 3) # [H, W, 3]

    if dataset_loader.m360:
        img = torch.flip(img, dims=[1]) # only for m360

    # Antialiasing
    img_batched = img.unsqueeze(0).contiguous().float()  # [1,H,W,3]
    #rast_aa    = rast_out.contiguous().float()
    pos_clip_aa = pos_clip.contiguous().float()
    img_aa = dr.antialias(img_batched, rast_out, pos_clip_aa, faces_unbatched)[0]

    return img_aa    # [H,W,3]

def render_via_mesh_rasterization_colorfield(
    color_field, cam2world_np, focal, H, W,
    vertices, faces_unbatched, ctx,
    near=0.1, far=10.0, device='cuda'):
    chunk_faces=16_000_000
    vertex_chunk=10_000_000
    # --- Build MVP and verts_h (homogeneous) ---
    mvp_np = mvp_from_cam(cam2world_np, H, W, focal, near=near, far=far)
    mvp = torch.tensor(mvp_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,4,4]

    ones = torch.ones_like(vertices[:, :, :1])
    verts_h = torch.cat([vertices, ones], dim=-1)   # [1, V, 4]
    B, V, _ = verts_h.shape

    # Chunk verts * mvp -> pos_clip
    mvp_t = mvp[0].transpose(0, 1).contiguous().to(device=device, dtype=torch.float32)  # [4,4]
    vertex_chunk = int(min(vertex_chunk, V))
    pos_clip = torch.empty((1, V, 4), device=device, dtype=torch.float32)

    # pos_clip in chunks
    with torch.no_grad():
        for start in range(0, V, vertex_chunk):
            end = min(start + vertex_chunk, V)
            v_slice2d = verts_h[:, start:end, :].contiguous().view(-1, 4)  # [chunk,4]
            out2d = v_slice2d.mm(mvp_t)                                     # [chunk,4]
            pos_clip[:, start:end, :] = out2d.view(1, end - start, 4)
            del v_slice2d, out2d

    pos_clip = pos_clip.contiguous().float()   # final pos_clip [1,V,4]

    device = pos_clip.device
    dtype = pos_clip.dtype

    F_total = faces_unbatched.shape[0]
    faces = faces_unbatched.contiguous().to(device=device, dtype=torch.int32)
    vertex_positions = vertices[0].float().unsqueeze(0).to(device=device, dtype=dtype)  # [1,V,3]

    # Buffers
    if dataset_loader.nerf_synthetic:
        # white background
        color_buffer = torch.ones((1, H, W, 3), device=device, dtype=dtype)
    else:
        color_buffer = torch.zeros((1, H, W, 3), device=device, dtype=dtype)


    depth_buffer = torch.full((1, H, W, 1), float('inf'), device=device, dtype=dtype)

    # camera position (for viewdir)
    cam_pos = torch.tensor(cam2world_np[:3, 3], dtype=torch.float32, device=device)
    P = H * W

    # --- build per-pixel ray directions in world space ---
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device)
    )

    dirs_cam = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)  # [H,W,3]

    R = torch.tensor(cam2world_np[:3, :3], device=device, dtype=dtype)
    dirs_world = (dirs_cam @ R.T)
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-8)

    dirs_world_flat = dirs_world.view(-1, 3)  # [P,3]

    # iterate face chunks
    for s in range(0, F_total, chunk_faces):
        e = min(s + chunk_faces, F_total)
        f_chunk = faces[s:e].contiguous()  # [chunk,3]

        gc.collect()
        torch.cuda.synchronize()

        # rasterize chunk
        rast_chunk, _ = dr.rasterize(ctx, pos_clip, f_chunk, resolution=[H, W])

        # interpolate clip coords (for depth) and world positions
        interp_clip_chunk, _ = dr.interpolate(pos_clip, rast_chunk, f_chunk)    # [1,H,W,4]
        interp_pos_chunk, _ = dr.interpolate(vertex_positions, rast_chunk, f_chunk)  # [1,H,W,3]

        # depth_ndc = z / w
        w_comp = interp_clip_chunk[..., 3:4]
        depth_ndc = interp_clip_chunk[..., 2:3] / (w_comp + 1e-9)  # [1,H,W,1]

        # valid fragments mask (coverage > 0)
        valid_mask = (rast_chunk[..., 3] > 0.0).unsqueeze(-1)  # [1,H,W,1]

        # closer fragments than current
        closer_mask = (depth_ndc < depth_buffer) & valid_mask  # [1,H,W,1]

        if closer_mask.any():
            # flatten indices of pixels that changed
            mask_flat = closer_mask.view(-1)       # length P
            idxs = mask_flat.nonzero(as_tuple=False).squeeze(-1)  # indices in flattened P

            if idxs.numel() > 0:
                # gather world positions for those pixels
                interp_pos_flat = interp_pos_chunk.view(1, P, 3)[0]   # [P,3]
                chosen_xyz = interp_pos_flat[idxs]                    # [M,3]

                # compute viewdirs
                cam_pos_expand = cam_pos.view(1, 3)
                viewdirs_chosen = cam_pos_expand - chosen_xyz
                viewdirs_chosen = viewdirs_chosen / (viewdirs_chosen.norm(dim=-1, keepdim=True) + 1e-8)

                # sample color field
                rgb_chosen = color_field(chosen_xyz, viewdirs_chosen)  # [M,3]
                rgb_chosen = torch.clamp(rgb_chosen, 0.0, 1.0)

                # write colors back
                color_flat = color_buffer.view(1, P, 3)[0]   # [P,3]
                color_flat[idxs, :] = rgb_chosen
                color_buffer = color_flat.view(1, H, W, 3)

                # update depth buffer at these indices
                depth_flat = depth_buffer.view(1, P, 1)[0]  # [P,1]
                depth_flat[idxs, 0] = depth_ndc.view(-1)[idxs]
                depth_buffer = depth_flat.view(1, H, W, 1)

        # free temps
        del interp_clip_chunk, interp_pos_chunk, w_comp, depth_ndc, valid_mask, closer_mask

    if not dataset_loader.nerf_synthetic:
        hit_mask = (depth_buffer[0, ..., 0] < float('inf'))
        hit_mask_flat = hit_mask.view(-1)

        nonhit_idx = (~hit_mask_flat).nonzero(as_tuple=False).squeeze(-1)

        if nonhit_idx.numel() > 0:
            # View directions
            viewdirs_bg = -dirs_world_flat[nonhit_idx]  # towards camera

            # FIXED xyz -> decouple background from scene geometry
            xyz_bg = cam_pos.view(1, 3).expand(viewdirs_bg.shape[0], 3)
            # alternative:
            # xyz_bg = torch.zeros_like(viewdirs_bg)

            rgb_bg = color_field(xyz_bg, viewdirs_bg)
            rgb_bg = torch.clamp(rgb_bg, 0.0, 1.0)

            color_flat = color_buffer.view(1, P, 3)[0]
            color_flat[nonhit_idx] = rgb_bg
            color_buffer = color_flat.view(1, H, W, 3)

    # final image
    img = color_buffer[0].permute(1, 2, 0)  # [H,W,3]
    img = img.permute(2, 0, 1).contiguous()

    if dataset_loader.m360:
        img = torch.flip(img, dims=[1])

    # Antialiasing
    # img_batched = img.unsqueeze(0).contiguous().float()  # [1,H,W,3]
    # #rast_aa    = rast_out.contiguous().float()
    # pos_clip_aa = pos_clip.contiguous().float()
    # img_aa = dr.antialias(img_batched, rast_chunk, pos_clip_aa, faces_unbatched)[0]

    return img  # [H,W,3]



# ---------------- Prepare scene (mesh or bbox) & load dataset if requested ----------------
if DATA_ROOT is not None:
    print(f"[INFO] Loading dataset from {DATA_ROOT} ...")
    use_dataset = True
    (dataset_cam2worlds,
     dataset_target_imgs,
     dataset_target_masks,
     dataset_focal,
     dataset_intrinsics_per_frame) = dataset_loader.load_dataset(DATA_ROOT, split="train")
    DATA_H, DATA_W = dataset_target_imgs[0].shape[:2]
    print(f"[INFO] Using dataset resolution H={DATA_H}, W={DATA_W}")
    print(f"[INFO] Train frames: {len(dataset_cam2worlds)}, focal≈{dataset_focal:.3f}")

# print("Example cam2world matrix (frame 0):")
# print(dataset_cam2worlds[0])
# cam_rot = dataset_cam2worlds[0][:3, :3]
# cam_trans = dataset_cam2worlds[0][:3, 3]
# print("Rotation:\n", cam_rot)
# print("Translation:\n", cam_trans)
# print("Image shape:", dataset_target_imgs[0].shape)  # z.B. (IMG_RES, IMG_RES, 3)
# print("Mask shape:", dataset_target_masks[0].shape)  # z.B. (IMG_RES, IMG_RES)

# ---------------- If mesh is provided: load and prepare rasterization targets ----------------
mesh = None

if dataset_loader.m360:
    NEAR = 1.0
    FAR = 50.0

# print(f"Near: {NEAR}, Far: {FAR}")
# print("[DBG] DATA_H, DATA_W:", DATA_H, DATA_W)
# print("[DBG] dataset_focal:", dataset_focal)
# print("[DBG] IMG_RES global:", IMG_RES)
# print("[DBG] intr_per_frame present?:", dataset_intrinsics_per_frame is not None)
# if dataset_intrinsics_per_frame is not None:
#     print("[DBG] intr_per_frame[0]:", dataset_intrinsics_per_frame[0])
# print("[DBG] fov_deg:", 2.0 * math.degrees(math.atan(DATA_H / (2.0 * float(dataset_focal)))))


import open3d

if MESH_PATH is not None:
    loaded = trimesh.load(MESH_PATH, process=True)  # Load mesh (correct triang, norms)
    if isinstance(loaded, trimesh.Scene):   # If Scene, try to create single trimesh
        try:
            mesh = loaded.dump(concatenate=True)    # Combine all scenes to one mesh
            if not isinstance(mesh, trimesh.Trimesh):
                raise RuntimeError("scene.dump did not return a Trimesh")
        except Exception:
            geoms = list(loaded.geometry.values())  # Load all geometries 
            if len(geoms) == 0:
                raise ValueError("Scene contains no geometry")
            mesh = trimesh.util.concatenate(tuple(geoms))   # Concatenate manually
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):   # Make sure mesh is a trimesh
        raise TypeError("Expected trimesh.Trimesh")

    mesh_vertices = mesh.vertices.astype(np.float32)    # Convert to float32
    mesh.vertices = mesh_vertices

    # r = np.array([[1, 0, 0],
    #           [0, 0,-1],
    #           [0, 1, 0]])
    # mesh.vertices = mesh.vertices @ r.T # Rotate coordinates with rotation matrix

    # verts = mesh.vertices
    # faces = mesh.faces
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces, color='lightblue', edgecolor='gray', alpha=0.8)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.title("Loaded Mesh")
    # plt.savefig(os.path.join(OUTDIR, "mesh.png"), dpi=150)
    # plt.show()

    # prepare tensors for rasterizer
    vertices_np = mesh.vertices.astype(np.float32)  # Get vertex positions of mesh
    faces_np = mesh.faces.astype(np.int32)  # Get vertex triangle indices
    vertices = torch.tensor(vertices_np, device=DEVICE).unsqueeze(0)  # [1,V,3]
    faces = torch.tensor(faces_np, device=DEVICE).unsqueeze(0)        # [1,F,3]
    faces_unbatched = faces[0].contiguous() # Remove batch dimension [F,3]

    # compute bbox from mesh (world->grid mapping)
    bbox_min = vertices[0].min(dim=0)[0].cpu().numpy() - 1e-4   # Calculate min for all vertices
    bbox_max = vertices[0].max(dim=0)[0].cpu().numpy() + 1e-4   # Calculate max for all vertices
    bbox_min = bbox_min.astype(np.float32); bbox_max = bbox_max.astype(np.float32)
    bbox_min_t = torch.tensor(bbox_min, dtype=torch.float32, device=DEVICE) # Convert to tensors
    bbox_max_t = torch.tensor(bbox_max, dtype=torch.float32, device=DEVICE)

    print(f"[DEBUG] Mesh bbox_min: {bbox_min}, bbox_max: {bbox_max}")

    print("vertices shape:", vertices.shape)        # [1, V, 3]
    print("faces shape:", faces.shape)              # [1, F, 3]
    print("faces_unbatched shape:", faces_unbatched.shape)  # [F, 3]

    # rasterize targets: if dataset provided, use dataset poses, else generate views by angles
    # if use_dataset:
    #     # create mvp per dataset pose (from dataset_cam2worlds + dataset_focal)
    #     for cam2world in dataset_cam2worlds:
    #         mvp_np = mvp_from_cam(cam2world, H, W, dataset_focal, near=0.1, far=10.0)
    #         mvps.append(torch.tensor(mvp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0))

    #     # rasterize mesh using these mvps -> form target_imgs/masks for mesh-based training
    #     for mvp in mvps:
    #         ones = torch.ones_like(vertices[:, :, :1])
    #         pos_clip = torch.matmul(torch.cat([vertices, ones], dim=-1), mvp.transpose(1,2))    # [1,V,4]
    #         #print("pos_clip shape:", pos_clip.shape)
    #         rast_out, _ = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=[H, W])   # [1,H,W,4]
    #         #print("rast_out shape:", rast_out.shape)
    #         #print("vertices shape:", vertices.shape)
    #         if use_vertex_colors:
    #             rgb_t, _ = dr.interpolate(vcol_tensor, rast_out, faces_unbatched)
    #             rgb_t = rgb_t[0].cpu().numpy()  # [H,W,3]
    #         elif use_texture:
    #             uv_map, _ = dr.interpolate(uv_attr, rast_out, faces_unbatched) # interpolate uv coordinates
    #             uv_grid = uv_map * 2.0 - 1.0    # Scale to [-1, 1]
    #             uv_grid = uv_grid.to(dtype=torch.float32, device=texture_tensor.device)
    #             tex_sampled = F.grid_sample(texture_tensor, uv_grid, mode="bilinear", align_corners=True)   # texture sampling
    #             rgb_t = tex_sampled.permute(0,2,3,1)[0].detach().cpu().numpy()  # reordering
    #         else:
    #             rgb_t = np.ones((H, W, 3), dtype=np.float32)    # Fallback: all pixel white
    #         mask = (rast_out[0, ..., 3] > 0).cpu().numpy()  # Extract mask out of rasterization
    #         if mask.ndim == 2:  # ensure background = white where mask==False
    #             rgb_t[~mask] = 1.0
    #         else:
    #             #fallback
    #             rgb_t[mask == False] = 1.0
    #         target_imgs.append(rgb_t.astype(np.float32))    # Save target image [H,W,3]
    #         target_masks.append(mask.astype(np.bool_))  # Save target mask [H,W]
    #     print("[INFO] Mesh rasterized for dataset poses (targets ready).")
    # else:
    #     # no dataset: generate a few angular views and rasterize mesh for targets (fallback)
    #     ANGLES = [0,45,90,135,180]
    #     for angle in ANGLES:
    #         angle_rad = np.radians(angle)
    #         radius = 5.0; height = -2.0
    #         eye = np.array([np.sin(angle_rad)*radius, height, np.cos(angle_rad)*radius], dtype=np.float32)
    #         center = np.array([0.0,0.0,0.0], dtype=np.float32)
    #         up = np.array([0.0,1.0,0.0], dtype=np.float32)
    #         view = look_at(eye, center, up)
    #         proj = perspective(np.radians(45.0), float(W)/float(H), 0.1, 10.0)
    #         mvp = proj @ view
    #         mvps.append(torch.tensor(mvp, dtype=torch.float32, device=DEVICE).unsqueeze(0))
    #     for mvp in mvps:
    #         ones = torch.ones_like(vertices[:, :, :1])
    #         pos_clip = torch.matmul(torch.cat([vertices, ones], dim=-1), mvp.transpose(1,2))
    #         rast_out, _ = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=[H,W])
    #         if use_vertex_colors:
    #             rgb_t, _ = dr.interpolate(vcol_tensor, rast_out, faces_unbatched)
    #             rgb_t = rgb_t[0].cpu().numpy()
    #         elif use_texture:
    #             uv_map, _ = dr.interpolate(uv_attr, rast_out, faces_unbatched)
    #             uv_grid = uv_map * 2.0 - 1.0
    #             uv_grid = uv_grid.to(dtype=torch.float32, device=texture_tensor.device)
    #             tex_sampled = F.grid_sample(texture_tensor, uv_grid, mode="bilinear", align_corners=True)
    #             rgb_t = tex_sampled.permute(0,2,3,1)[0].detach().cpu().numpy()
    #         else:
    #             rgb_t = np.zeros((H,W,3), dtype=np.float32)
    #         mask = (rast_out[0, ..., 3] > 0).cpu().numpy()
    #         target_imgs.append(rgb_t.astype(np.float32))
    #         target_masks.append(mask.astype(np.bool_))
    #     print("[INFO] Mesh rasterized for angular fallback views.")

ctx = dr.RasterizeCudaContext() # Create nvdiffrast CUDA-Rasterizer-Context
# H = IMG_RES; W = IMG_RES    # Set resolution
H = DATA_H; W = DATA_W

if MODEL == "tensorf":  # ColorFieldVM setup
    extent_vec = (bbox_max_t - bbox_min_t).abs()    # Vector with scene size
    scene_extent = float(torch.as_tensor(extent_vec).max().item()) * 0.5  # scale to [-extent, +extent]

    # Build Appearance-Modell
    color_field = ColorFieldVM(
        n_voxels=3000**3,   # res (1024 for NeRF Synthetic)
        device=DEVICE,
        app_n_comp=16,  # rank
        sh_degree=2,    # spherical harmonic degree
        scene_extent=scene_extent
    ).to(DEVICE)

    # Optimizer with seperate learning rates
    opt = torch.optim.Adam(color_field.get_optparam_groups(), betas=(0.9, 0.99))
    # lambda_tv = 1e-2    # total variation weight

if MODEL == "voxel_grid":   # Voxel grid setup
    voxel_grid = torch.nn.Parameter(torch.rand(1, 3, VOXEL_RES, VOXEL_RES, VOXEL_RES, device=DEVICE))
    optimizer = torch.optim.Adam([voxel_grid], lr=LR)
best_loss = float('inf'); no_improve_steps = 0

# vg = voxel_grid.detach()
# print("voxel_grid shape:", vg.shape)   # [1, C, D, H, W]
# print("device:", vg.device, "dtype:", vg.dtype)
# print("min/max/mean/std:", vg.min().item(), vg.max().item(), vg.mean().item(), vg.std().item())

start_time = time.time()

print("[TRAIN] Start training...")
scaler = torch.cuda.amp.GradScaler()    # Initialize GradScaler for AMP
views_per_it = 1
num_views = len(dataset_cam2worlds)

# Training loop:
for it in range(1, STEPS+1):
    t0 = time.time()
    total_mse_sum = torch.tensor(0.0, device=DEVICE)   # sum of squared errors (over all valid scalar entries)
    total_num_entries = 0.0 # counter
    if MODEL == "tensorf": 
        lr_factor = 0.1 ** (1 / STEPS)
        TV_loss_weight = 1.0

    idxs = np.random.choice(num_views, size=min(views_per_it, num_views), replace=False)
    for vi in idxs: # iterate over dataset poses
        cam2world = dataset_cam2worlds[vi]
        cam2world_t = torch.from_numpy(cam2world).to(device=DEVICE, dtype=torch.float32)
        with torch.cuda.amp.autocast(): # AMP for memory saving
            cam2world_np = cam2world_t.detach().cpu().numpy()
            if MODEL == "tensorf":
                pred_map = render_via_mesh_rasterization_colorfield(
                        color_field, cam2world_np, float(dataset_focal), H, W,
                        vertices, faces_unbatched, ctx, near=NEAR, far=FAR, device=DEVICE
                    )
            if MODEL == "voxel_grid":
                pred_map = render_via_mesh_rasterization(voxel_grid, cam2world_np, float(dataset_focal),
                                                        H, W, vertices, faces_unbatched, ctx,
                                                        voxel_sample_mode='bilinear', near=NEAR, far=FAR, device=DEVICE)
            # pred_map: torch tensor [H,W,3] on device
            tgt = torch.tensor(dataset_target_imgs[vi], dtype=torch.float32, device=DEVICE) # target image

            mse_sum = ((pred_map - tgt) ** 2).sum() # calculate MSE
            entries = float(pred_map.numel())   # H * W * 3

            total_mse_sum = total_mse_sum + mse_sum # add mse to total
            total_num_entries += entries    # add entries to total

    if total_num_entries == 0.0:    # Skip iteration if no valid pixels
        print(f"[WARN] Iter {it}: total_num_entries == 0 -> skipping backward step.")
        continue

    if MODEL == "tensorf":
        opt.zero_grad() # reset gradients
    if MODEL == "voxel_grid":
        optimizer.zero_grad()   # Reset gradients before backprop

    mean_loss = total_mse_sum / float(total_num_entries)   # MSE, tensor on DEVICE

    if MODEL == "tensorf":
        TV_loss_weight *= lr_factor
        loss = mean_loss + TV_loss_weight * color_field.TV_loss()    # TV loss
        scaler.scale(loss).backward()   # backpropagation
        scaler.step(opt)  # parameter update
        scaler.update() # update AMP scaling
        total_loss = float(loss.item())
    if MODEL == "voxel_grid":
        scaler.scale(mean_loss).backward()   # backpropagation
        scaler.step(optimizer)  # parameter update
        scaler.update()
        total_loss = mean_loss.item()

    tb_writer.add_scalar("Loss/train", total_loss, it)  # Write loss to tensorboard
    t1 = time.time()
    if it % 10 == 0 or it == 1: # Print Loss for every 10 steps
        print(f"Iter {it}/{STEPS} loss={total_loss:.6f} time={t1-t0:.3f}s")

    # early stopping
    if total_loss < best_loss - DELTA:
        best_loss = total_loss
        no_improve_steps = 0
    else:
        no_improve_steps += 1
    if no_improve_steps >= PATIENCE:
        print(f"Early stopping at iteration {it} (no improvement in {PATIENCE} steps)")
        break

    # Save intermediates
    if it % SAVE_INTERVAL == 0 or it == 1 or it == STEPS:
        with torch.no_grad():   # Deactivate Autograd, no training
            step_out = Path(OUTDIR) / f"step_{it:06d}"  # Directory for steps
            step_out.mkdir(parents=True, exist_ok=True)

            n_preview = min(IMG_COUNT, len(dataset_target_imgs))   # Set number of images
            for vi in range(n_preview):
                cam2world_np = dataset_cam2worlds[vi]      # numpy (4x4)
                focal_to_use = float(dataset_focal)
                # render predicted views
                if MODEL == "tensorf":
                    pred_map = render_via_mesh_rasterization_colorfield(
                        color_field, cam2world_np, float(dataset_focal), H, W,
                        vertices, faces_unbatched, ctx, near=NEAR, far=FAR, device=DEVICE
                    )  # returns torch Tensor [H,W,3] on DEVICE
                if MODEL == "voxel_grid":
                   pred_map = render_via_mesh_rasterization(
                        voxel_grid, cam2world_np, focal_to_use, H, W,
                        vertices, faces_unbatched, ctx,
                        voxel_sample_mode='bilinear', near=NEAR, far=FAR, device=DEVICE
                    )  # returns torch Tensor [H,W,3] on DEVICE 

                pm = pred_map.detach().cpu().numpy()    # copy to cpu
                pred_rgb = np.clip(pm, 0.0, 1.0)    # [0, 1]
                # Overlay step-count onto image
                try:
                    # Convert float [0,1] -> uint8
                    pred_u8 = (pred_rgb * 255.0).astype(np.uint8)
                    pred_pil = Image.fromarray(pred_u8)
                except Exception:
                    plt.imsave(str(step_out / f"pred_view{vi}.png"), pred_rgb)
                else:
                    draw = ImageDraw.Draw(pred_pil)
                    font_size = max(12, IMG_RES // 16)
                    font = ImageFont.load_default()
                    text = f"{it}"
                    pad = 6
                    x, y = pad, pad
                    draw.text((x+1, y+1), text, font=font, fill=(0,0,0))
                    draw.text((x, y), text, font=font, fill=(255,255,255))
                    pred_pil.save(str(step_out / f"pred_view{vi}.png"))
                # Save Target view
                if it == 1:
                    plt.imsave(str(step_out / f"target_view{vi}.png"), np.clip(dataset_target_imgs[vi],0,1))

            # voxel export -> point cloud in world coords
            # vg = voxel_grid.detach().cpu().numpy()[0]  # [3,D,H,W]
            # np.save(str(step_out / "voxel_grid.npy"), vg)   # save voxelgrid as .npy
            # occ_values = np.linalg.norm(vg, axis=0)  # calculate occupancy [D,H,W]

            # occ_smooth = gaussian_filter(occ_values, sigma=1.0) # remove noise
            # thr = np.percentile(occ_smooth, 95) # 95 percentile of voxels
            # occ = occ_smooth >= thr # Mask
            # lab, nlab = label(occ)  # Check for connected regions
            # if nlab > 0:
            #     sizes = np.bincount(lab.ravel()); sizes[0] = 0  # count voxels, ignore background
            #     occ = lab == sizes.argmax() # return biggest component

            # dz, hy, wx = np.nonzero(occ)    # indices of voxels
            # num_pts = len(dz)
            # print(f"Found {num_pts} occupied voxels")

            # if num_pts > 0:
            #     ix = wx.astype(np.float32); iy = hy.astype(np.float32); iz = dz.astype(np.float32)
            #     cx = (ix + 0.5) / float(VOXEL_RES); cy = (iy + 0.5) / float(VOXEL_RES); cz = (iz + 0.5) / float(VOXEL_RES)
            #     # set bbox
            #     if use_mesh:
            #         bbox_min_np = np.array(bbox_min, dtype=np.float32); bbox_max_np = np.array(bbox_max, dtype=np.float32)
            #     else:
            #         bbox_min_np = np.array([-BBOX_SIZE/2.0]*3, dtype=np.float32)
            #         bbox_max_np = np.array([ BBOX_SIZE/2.0]*3, dtype=np.float32)
            #     world_x = bbox_min_np[0] + cx * (bbox_max_np[0] - bbox_min_np[0])   # calculate world coordinates
            #     world_y = bbox_min_np[1] + cy * (bbox_max_np[1] - bbox_min_np[1])
            #     world_z = bbox_min_np[2] + cz * (bbox_max_np[2] - bbox_min_np[2])
            #     points = np.stack([world_x, world_y, world_z], axis=1)  # [num_pts, 3]
            #     cols = vg[:, dz, hy, wx].T  # extract colors
            #     cols = np.clip(cols, 0.0, 1.0)  # [0, 1]
            #     cols_u8 = (cols * 255.0).astype(np.uint8)   # convert to uint8

            #     MAX_POINTS = 500000 # downsample if too many points
            #     if points.shape[0] > MAX_POINTS:
            #         idxs = np.random.choice(points.shape[0], size=MAX_POINTS, replace=False)
            #         points = points[idxs]; cols_u8 = cols_u8[idxs]
            #         print(f"Downsampled voxel cloud to {MAX_POINTS} points for export.")

            #     ply_path = step_out / "voxel_cloud.ply" # write ASCII .ply
            #     with open(ply_path, "w") as f:
            #         f.write("ply\nformat ascii 1.0\n")
            #         f.write(f"element vertex {len(points)}\n")
            #         f.write("property float x\nproperty float y\nproperty float z\n")
            #         f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            #         f.write("end_header\n")
            #         for (x, y, z), (r, g, b) in zip(points, cols_u8):
            #             f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            #     print(f"Saved voxel cloud ({len(points)} points) -> {ply_path}")

            print(f"Saved checkpoints to {step_out}")

tb_writer.close()   # Close tensorboard
print("Training finished. Results in:", OUTDIR)

end_time = time.time()
training_time = end_time - start_time
hours, rem = divmod(training_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Training duration: {int(hours)}:{int(minutes):02d}:{seconds:.2f}")

print(f"GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# Novel View Synthesis
print("[NVS] Rendering novel views...")

novel_out = Path(OUTDIR) / "novel_views"    # Directory for novel views
novel_out.mkdir(parents=True, exist_ok=True)

if DATA_ROOT is not None:
    try:
        (dataset_cam2worlds_test,
         dataset_target_imgs_test,
         dataset_target_masks_test,
         _,
         dataset_intrinsics_per_frame_test) = dataset_loader.load_dataset(DATA_ROOT, split="test")
        print(f"[INFO] Test frames: {len(dataset_cam2worlds_test)}")
    except Exception as e:
        print(f"[WARN] No test split found ({e}); using train split as test.")
        dataset_cam2worlds_test = list(dataset_cam2worlds)
        dataset_target_imgs_test = list(dataset_target_imgs)
        dataset_target_masks_test = list(dataset_target_masks)
        dataset_intrinsics_per_frame_test = dataset_intrinsics_per_frame


# # Voxel-hit analysis (Train vs Test)
# @torch.no_grad()
# def voxel_hits_for_views_inline(cam2worlds, focal_px, H, W, n_samples, near, far, voxel_res, device, tile_hw=128):
#     D = Hvoxel = Wvoxel = voxel_res
#     hits = torch.zeros((D, Hvoxel, Wvoxel), dtype=torch.bool, device='cpu')

#     t_vals = torch.linspace(near, far, steps=n_samples, device=device)
#     clamp_max = torch.tensor([voxel_res-1, voxel_res-1, voxel_res-1], device=device, dtype=torch.long)

#     # Pixelzentren (H,W), indexing='ij' => erste Achse = y (H), zweite = x (W)
#     ys = (torch.arange(0, H, device=device).float() + 0.5)
#     xs = (torch.arange(0, W, device=device).float() + 0.5)
#     try:
#         py, px = torch.meshgrid(ys, xs, indexing='ij')  # [H,W], [H,W]
#     except TypeError:
#         py, px = torch.meshgrid(ys, xs)
#         py = py.t()
#         px = px.t()

#     cx = W * 0.5
#     cy = H * 0.5
#     x_cam_full =  (px - cx) / float(focal_px)
#     y_cam_full = -(py - cy) / float(focal_px)
#     z_cam_full = -torch.ones_like(x_cam_full)   # look along -z

#     for cam2world_np in cam2worlds:
#         c2w = torch.from_numpy(cam2world_np).to(device=device, dtype=torch.float32)
#         R = c2w[:3, :3]
#         t = c2w[:3, 3]

#         for y0 in range(0, H, tile_hw):
#             y1 = min(y0 + tile_hw, H)
#             for x0 in range(0, W, tile_hw):
#                 x1 = min(x0 + tile_hw, W)

#                 x_cam = x_cam_full[y0:y1, x0:x1]
#                 y_cam = y_cam_full[y0:y1, x0:x1]
#                 z_cam = z_cam_full[y0:y1, x0:x1]
#                 dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)                # [h,w,3]
#                 dirs_world = (dirs_cam.reshape(-1,3) @ R.t()).reshape_as(dirs_cam)
#                 dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-12)
#                 origin = t.view(1,1,3).expand_as(dirs_world)

#                 P = dirs_world.numel() // 3
#                 o = origin.reshape(P, 3)
#                 d = dirs_world.reshape(P, 3)
#                 pts = o.unsqueeze(1) + t_vals.view(1, -1, 1) * d.unsqueeze(1)        # [P,N,3]

#                 g = world_to_grid_coords(pts)                                        # [-1,1]
#                 inside = ((g >= -1.0) & (g <= 1.0)).all(dim=-1)                      # [P,N]

#                 if inside.any():
#                     idx = ((g + 1.0) * 0.5 * voxel_res - 1e-6).long()   # (x,y,z)
#                     idx = idx.clamp(min=0, max=voxel_res - 1)   
#                     idx = idx[inside]
#                     if idx.numel() > 0:
#                         idx = torch.unique(idx, dim=0)
#                         x = idx[:,0].cpu(); y = idx[:,1].cpu(); z = idx[:,2].cpu()
#                         hits[z, y, x] = True

#     return hits


# def print_stats(name, hits, total_voxels):
#     n = int(hits.sum().item())
#     frac = 100.0 * n / total_voxels
#     print(f"[HITS] {name:>12s}: {n:,} voxels hit ({frac:.4f}% of grid)")


# def run_offline_hit_analysis_inline():
#     global dataset_cam2worlds_test

#     assert bbox_min_t is not None and bbox_max_t is not None, "BBox not set."
#     assert dataset_focal is not None, "Focal (Dataset) not set."
#     assert len(dataset_cam2worlds) > 0, "Train-Views missing."

#     if 'dataset_cam2worlds_test' not in globals() or len(dataset_cam2worlds_test) == 0:
#         test_json = os.path.join(DATA_ROOT, "transforms_test.json")
#         meta_test = json.load(open(test_json, "r"))
#         frames_test = meta_test.get("frames", [])
#         if len(frames_test) == 0:
#             raise ValueError("transforms_test.json enthält 0 Frames")
#         dataset_cam2worlds_test = [np.array(fr["transform_matrix"], dtype=np.float32) for fr in frames_test]

#     outdir = Path(OUTDIR) / "hit_analysis"
#     outdir.mkdir(parents=True, exist_ok=True)

#     total_vox = VOXEL_RES ** 3
#     # Test different resolutions
#     res_list = [IMG_RES, max(IMG_RES//2,64), max(IMG_RES//4,32)]
#     res_list = sorted(set(res_list), reverse=True)

#     print("\n[HIT ANALYSIS] Start (Train vs. Test; rays inline, fixed z=-1, ij-indexing, focal scaling)")
#     print(f"[HIT ANALYSIS] Grid {VOXEL_RES}^3, Samples/Ray={N_SAMPLES}, Near={NEAR}, Far={FAR}")

#     report = []
#     for im_res in res_list:
#         H_ = W_ = im_res
#         focal_scaled = float(dataset_focal) * (im_res / float(IMG_RES))

#         print(f"\n[HIT ANALYSIS] Image resolution: {im_res}x{im_res} (focal_scaled={focal_scaled:.3f})")

#         hits_train = voxel_hits_for_views_inline(
#             dataset_cam2worlds[:IMG_COUNT],
#             focal_scaled, H_, W_, N_SAMPLES, NEAR, FAR, VOXEL_RES, DEVICE, tile_hw=128
#         )
#         hits_test = voxel_hits_for_views_inline(
#             dataset_cam2worlds_test[:TEST_IMG_COUNT],
#             focal_scaled, H_, W_, N_SAMPLES, NEAR, FAR, VOXEL_RES, DEVICE, tile_hw=128
#         )

#         inter = hits_train & hits_test
#         union = hits_train | hits_test
#         only_train = hits_train & (~hits_test)
#         only_test  = hits_test  & (~hits_train)

#         n_train = int(hits_train.sum()); n_test = int(hits_test.sum())
#         n_inter = int(inter.sum()); n_union = int(union.sum())
#         n_only_train = int(only_train.sum()); n_only_test = int(only_test.sum())
#         jacc = (n_inter / n_union) if n_union > 0 else 0.0
#         unseen_frac = (n_only_test / n_test) if n_test > 0 else 0.0

#         print_stats("train", hits_train, total_vox)
#         print_stats("test",  hits_test,  total_vox)
#         print(f"[HITS] {'intersection':>12s}: {n_inter:,} voxels")
#         print(f"[HITS] {'union':>12s}: {n_union:,} voxels  (Jaccard={jacc:.4f})")
#         print(f"[HITS] {'only_train':>12s}: {n_only_train:,} voxels")
#         print(f"[HITS] {'only_test':>12s}:  {n_only_test:,} voxels  (unseen-by-train among test hits = {100*unseen_frac:.2f}%)")

#         np.save(outdir / f"hits_train_{im_res}.npy", hits_train.numpy())
#         np.save(outdir / f"hits_test_{im_res}.npy",  hits_test.numpy())
#         np.save(outdir / f"hits_only_test_{im_res}.npy", only_test.numpy())

#         report.append(
#             f"resolution={im_res} train={n_train} test={n_test} inter={n_inter} union={n_union} "
#             f"only_train={n_only_train} only_test={n_only_test} jaccard={jacc:.6f} unseen_test_fraction={unseen_frac:.6f}"
#         )

#     with open(outdir / "summary.txt", "w") as f:
#         f.write("\n".join(report))
#     print(f"\n[HIT ANALYSIS] Summary -> {outdir/'summary.txt'}\n[HIT ANALYSIS] Done.\n")


# # Start
# run_offline_hit_analysis_inline()

# ---------------- NVS helpers ----------------
def compute_psnr(img1, img2):   # calculate PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))

def to_nchw01(x_np):
    # [H,W,3] in [0,1]  ->  [1,3,H,W] torch float32
    x = torch.from_numpy(np.clip(x_np, 0.0, 1.0)).permute(2,0,1).unsqueeze(0)
    x = x.to(memory_format=torch.contiguous_format).contiguous()
    return x.to(device=DEVICE, dtype=torch.float32)

lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()

def compute_lpips(pred_rgb, gt_img):    # calculate LPIPS
    x = to_nchw01(pred_rgb)
    y = to_nchw01(gt_img)
    x = x * 2.0 - 1.0   # LPIPS needs [-1,1]
    y = y * 2.0 - 1.0
    with torch.no_grad():
        v = lpips_fn(x, y).item()
    return float(v)

with torch.no_grad():   # Deactivate Autograd
    n_views = min(TEST_IMG_COUNT, len(dataset_cam2worlds_test))   # Use dataset camera poses
    print(f"[NVS] Using {n_views} dataset poses for novel view synthesis.")
    psnr_list = []; ssim_list = []; lpips_list = []
    for idx, cam2world in enumerate(dataset_cam2worlds_test[:n_views]):
        cam2world_np = cam2world
        # render predicted RGB for pose
        if MODEL == "tensorf":
            pred_rgb = render_via_mesh_rasterization_colorfield(
                color_field, cam2world_np, float(dataset_focal), H, W,
                vertices, faces_unbatched, ctx, near=NEAR, far=FAR, device=DEVICE
            ).cpu().numpy()
        if MODEL == "voxel_grid":
            pred_rgb = render_via_mesh_rasterization(
                    voxel_grid, cam2world_np, float(dataset_focal), H, W,
                    vertices, faces_unbatched, ctx, voxel_sample_mode='bilinear',
                    near=NEAR, far=FAR, device=DEVICE
                ).cpu().numpy()

        # save predicted image
        out_path = novel_out / f"novel_dataset_idx{idx:04d}.png"
        plt.imsave(str(out_path), np.clip(pred_rgb, 0.0, 1.0))

        # if GT exists for this pose, compare and save PSNR + side-by-side
        gt_img = None
        if len(dataset_target_imgs_test) > idx:
            gt_img = dataset_target_imgs_test[idx]  # numpy H,W,3
        elif len(dataset_target_imgs_test) > 0:
            # fallback: match modulo if counts differ
            gt_img = dataset_target_imgs_test[idx % len(dataset_target_imgs_test)]

        if gt_img is not None: # Calculate PSNR, SSIM and LPIPS, plot comparison
            psnr_val = compute_psnr(pred_rgb, gt_img)
            ssim_val = ssim((pred_rgb*255).astype(np.uint8), (gt_img*255).astype(np.uint8), multichannel=True)
            lpips_val = compute_lpips(pred_rgb, gt_img)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
            fig, axs = plt.subplots(1, 2, figsize=(16,8), constrained_layout=True)
            axs[0].imshow(np.clip(gt_img,0,1))
            axs[0].set_title("Ground Truth")
            axs[0].axis("off")
            axs[1].imshow(np.clip(pred_rgb,0,1))
            axs[1].set_title(f"Novel View\nPSNR={psnr_val:.2f} dB\nSSIM={ssim_val:.2f}\nLPIPS={lpips_val:.3f}")
            axs[1].axis("off")
            plt.savefig(novel_out / f"compare_dataset_idx{idx:04d}.png")
            plt.close()
    
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_lpips = sum(lpips_list) / len(lpips_list)
    print("Average metrics:")
    print(f"PSNR: {avg_psnr:.2f} dB SSIM: {avg_ssim:.4f} LPIPS: {avg_lpips:.4f}")

print("[NVS] Done.")