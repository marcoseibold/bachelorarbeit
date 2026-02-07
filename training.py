import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import dataset_loader
import imageio
import lpips
import matplotlib.pyplot as plt
import numpy as np
import nvdiffrast.torch as dr
import open3d
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, label  # occupancy filtering
from skimage.metrics import structural_similarity as ssim
from tensorf_appearance import ColorFieldVM
from torch.utils.tensorboard import SummaryWriter

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True, help="Path to mesh (OBJ/PLY).")
parser.add_argument("--data_root", type=str, required=True, help="Path to dataset scene folder (transforms_*.json + images).")
parser.add_argument("--model", type=str, required=True, help="Appearance model: voxel_grid or tensorf")
parser.add_argument("--outdir", type=str, default="train_out", help="Output directory")
parser.add_argument("--voxel_res", type=int, default=512, help="Voxel grid resolution (D,H,W)")
parser.add_argument("--steps", type=int, default=5000, help="Training iterations")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for voxel grid")
parser.add_argument("--save_interval", type=int, default=100, help="Save image/checkpoint every N steps")
parser.add_argument("--near", type=float, default=0.1, help="Near plane for ray sampling")
parser.add_argument("--far", type=float, default=6.0, help="Far plane for ray sampling")
parser.add_argument("--img_count", type=int, default=5, help="Training image count")
parser.add_argument("--test_img_count", type=int, default=200, help="Test image count")
args = parser.parse_args()

MESH_PATH = args.mesh
DATA_ROOT = args.data_root
OUTDIR = args.outdir
VOXEL_RES = args.voxel_res
STEPS = args.steps
LR = args.lr
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = args.save_interval
NEAR = args.near
FAR = args.far
IMG_COUNT = args.img_count
TEST_IMG_COUNT = args.test_img_count
MODEL = "tensorf" if args.model.lower() == "tensorf" else "voxel_grid"

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
    #with torch.no_grad():
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
        color_buffer = torch.full((1, H, W, 3), 0.5, device=device, dtype=dtype)

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

    # final image
    pred_rgb = color_buffer[0].permute(1, 2, 0)  # [H,W,3]
    pred_rgb = pred_rgb.permute(2, 0, 1).contiguous()

    if dataset_loader.m360:
        pred_rgb = torch.flip(pred_rgb, dims=[1])   # for m360

    # Ensure a valid gradient path if nothing hits the grid
    pred_rgb = pred_rgb + 0.0 * voxel_grid_param.sum()

    return pred_rgb


def render_via_mesh_rasterization_colorfield(
    color_field, cam2world_np, focal, H, W,
    vertices, faces_unbatched, ctx,
    near=0.1, far=10.0, device='cuda'):
    chunk_faces=16_000_000
    vertex_chunk=10_000_000
    # Build MVP and verts_h (homogeneous)
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
        color_buffer = torch.full((1, H, W, 3), 0.5, device=device, dtype=dtype)


    depth_buffer = torch.full((1, H, W, 1), float('inf'), device=device, dtype=dtype)

    # camera position (for viewdir)
    cam_pos = torch.tensor(cam2world_np[:3, 3], dtype=torch.float32, device=device)
    P = H * W

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

    # final image
    img = color_buffer[0].permute(1, 2, 0)  # [H,W,3]
    img = img.permute(2, 0, 1).contiguous()

    if dataset_loader.m360:
        img = torch.flip(img, dims=[1])

    return img  # [H,W,3]

def compute_scene_extent_from_cameras(cam2worlds, margin=1.1):
    cam_centers = np.stack([c2w[:3, 3] for c2w in cam2worlds], axis=0)  # Extract cam poses (N,3)
    center = cam_centers.mean(axis=0)   # Calc center of cameras
    dists = np.linalg.norm(cam_centers - center[None, :], axis=1)   # Calc distances
    radius = dists.max() * margin   # Take max distance as radius
    return center, radius

# Prepare scene (mesh or bbox) & load dataset if requested
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

# Load and prepare rasterization targets
mesh = None

if dataset_loader.m360 or dataset_loader.tnt:
    FAR = 100.0

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
    
    #r = np.array([[1, 0, 0],
    #          [0, 0,-1],
    #         [0, 1, 0]])
    #mesh.vertices = mesh.vertices @ r.T # Rotate coordinates for GT mesh with rotation matrix

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

ctx = dr.RasterizeCudaContext() # Create nvdiffrast CUDA-Rasterizer-Context
H = DATA_H; W = DATA_W  # Set res

if MODEL == "tensorf":  # ColorFieldVM setup
    scene_center, scene_extent = compute_scene_extent_from_cameras(dataset_cam2worlds)

    # Build Appearance-model
    color_field = ColorFieldVM(
        n_voxels=3000**3,   # res
        device=DEVICE,
        app_n_comp=16,  # rank
        sh_degree=2,    # spherical harmonic degree
        scene_extent=scene_extent
    ).to(DEVICE)

    # Optimizer with seperate learning rates
    opt = torch.optim.Adam(color_field.get_optparam_groups(), betas=(0.9, 0.99))

if MODEL == "voxel_grid":   # Voxel grid setup
    voxel_grid = torch.nn.Parameter(torch.rand(1, 3, VOXEL_RES, VOXEL_RES, VOXEL_RES, device=DEVICE))
    optimizer = torch.optim.Adam([voxel_grid], lr=LR)
best_loss = float('inf'); no_improve_steps = 0

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
        TV_loss_weight = 1e-3

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
                    font_size = max(12, 800 // 16)
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

            print(f"Saved checkpoints to {step_out}")

tb_writer.close()   # Close tensorboard
print("[TRAIN] Training finished. Results in:", OUTDIR)

end_time = time.time()
training_time = end_time - start_time   # Calc training time
hours, rem = divmod(training_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"[TRAIN] Training duration: {int(hours)}:{int(minutes):02d}:{seconds:.2f}")

print(f"[TRAIN] GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

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

# NVS helpers
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