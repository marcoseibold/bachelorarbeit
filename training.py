import os
import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True, help="Path to mesh")
parser.add_argument("--outdir", type=str, default="train_out", help="Output directory")
parser.add_argument("--voxel_res", type=int, default=512, help="Voxel grid resolution")
parser.add_argument("--img_res", type=int, default=128, help="Image resolution (H and W)")
parser.add_argument("--steps", type=int, default=1000, help="Training iterations")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--angles", type=int, nargs="+", default=[0, 45, 90], help="List of view angles in degrees for training")
parser.add_argument("--save_interval", type=int, default=100, help="Save image/checkpoint every N steps")
parser.add_argument("--early_stop_patience", type=int, default=50, help="Stop training if no improvement in N steps")
parser.add_argument("--early_stop_delta", type=float, default=1e-5, help="Minimum change to consider an improvement")
args = parser.parse_args()

MESH_PATH = args.mesh
OUTDIR = args.outdir
VOXEL_RES = args.voxel_res
IMG_RES = args.img_res
STEPS = args.steps
LR = args.lr
DEVICE = 'cuda'
ANGLES = args.angles
SAVE_INTERVAL = args.save_interval
PATIENCE = args.early_stop_patience
DELTA = args.early_stop_delta

os.makedirs(OUTDIR, exist_ok=True)

# TensorBoard-Writer
tb_writer = SummaryWriter(log_dir=os.path.join(OUTDIR, "tensorboard"))

# Helper functions
def look_at(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m

def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def get_mvp_matrix(angle_deg, H, W, device):
    angle_rad = np.radians(angle_deg)
    radius = 5.0
    height = -2.0
    eye = np.array([np.sin(angle_rad) * radius, height, np.cos(angle_rad) * radius], dtype=np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view = look_at(eye, center, up)
    proj = perspective(np.radians(45.0), float(W) / float(H), 0.1, 10.0)
    mvp = proj @ view
    return torch.tensor(mvp, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,4,4]

# Load and adjust mesh
mesh = trimesh.load(MESH_PATH, process=True)

if mesh.faces.shape[1] != 3:
    mesh = mesh.triangulate()

mesh.vertices -= mesh.vertices.mean(axis=0)
mesh.vertices /= np.max(np.linalg.norm(mesh.vertices, axis=1))
mesh.vertices *= 2.0
mesh.vertices[:, 1] *= -1

# Create vertices and faces
vertices_np = mesh.vertices.astype(np.float32)
faces_np = mesh.faces.astype(np.int32)
V = vertices_np.shape[0]

vertices = torch.tensor(vertices_np, device=DEVICE).unsqueeze(0)  # [1, V, 3]
faces = torch.tensor(faces_np, device=DEVICE).unsqueeze(0)        # [1, F, 3]
faces_unbatched = faces[0].contiguous()

print("Mesh loaded:", MESH_PATH)
print("Vertices:", vertices.shape, "Faces:", faces.shape)
print("Voxel resolution:", VOXEL_RES, "Image resolution:", IMG_RES)

# Compute bounding box for mapping world->grid
bbox_min = vertices[0].min(dim=0)[0].cpu().numpy() - 1e-4
bbox_max = vertices[0].max(dim=0)[0].cpu().numpy() + 1e-4
bbox_min = bbox_min.astype(np.float32)
bbox_max = bbox_max.astype(np.float32)
print("BBox min:", bbox_min, "max:", bbox_max)

ctx = dr.RasterizeCudaContext()
H = IMG_RES
W = IMG_RES

# Create Target images
print("Rendering target images...")

target_imgs = []
target_masks = []
mvps = []

# Versuche Vertex Colors oder Textur
use_vertex_colors = False
use_texture = False
texture_tensor = None
uv_attr = None

if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors.shape[1] >= 3:
    unique_colors = np.unique(mesh.visual.vertex_colors[:, :3], axis=0)
    if len(unique_colors) > 1:  # mehr als eine Farbe → sinnvoll
        use_vertex_colors = True
        vcol = mesh.visual.vertex_colors[:, :3] / 255.0
        vcol = np.clip(vcol, 0.0, 1.0).astype(np.float32)
        vcol_tensor = torch.tensor(vcol, dtype=torch.float32, device=DEVICE)
        print(f"Using vertex colors with {len(unique_colors)} unique values")

if not use_vertex_colors and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
    import imageio
    # Textur-Datei aus dem Mesh laden (trimesh merkt sich Pfad in material.image)
    tex_path = getattr(mesh.visual.material, "image", None)
    if tex_path is None:
        raise ValueError("Mesh has UVs but no texture image!")
    tex_img = imageio.imread(tex_path)[..., :3]  # [H,W,3]
    tex_img = torch.tensor(tex_img, dtype=torch.float32, device=DEVICE) / 255.0
    tex_img = tex_img.permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    texture_tensor = tex_img
    uv_attr = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=DEVICE)  # [V,2]
    use_texture = True
    print(f"Using texture from {tex_path}, size={tex_img.shape[2:]}")

if not use_vertex_colors and not use_texture:
    v_world = vertices[0]
    vcol = (v_world.cpu().numpy() - bbox_min) / (bbox_max - bbox_min + 1e-8)
    vcol = np.clip(vcol, 0.0, 1.0)
    vcol_tensor = torch.tensor(vcol, dtype=torch.float32, device=DEVICE)
    use_vertex_colors = True
    print("No vertex colors or texture found → using position-based fallback colors")

# Erzeuge Target-Bilder für alle Ansichten
for angle in ANGLES:
    mvp = get_mvp_matrix(angle, H, W, DEVICE)
    mvps.append(mvp)

    ones = torch.ones_like(vertices[:, :, :1])
    vertices_h = torch.cat([vertices, ones], dim=-1)  # [1,V,4]
    pos_clip = torch.matmul(vertices_h, mvp.transpose(1, 2))  # [1,V,4]

    rast_out, rast_db = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=[H, W])

    if use_vertex_colors:
        attr = vcol_tensor  # [V,3]
        rgb_t, _ = dr.interpolate(attr, rast_out, faces_unbatched)  # [1,H,W,3]
        rgb_t = rgb_t[0].cpu().numpy()

    elif use_texture:
        uv_map, _ = dr.interpolate(uv_attr, rast_out, faces_unbatched)  # [1,H,W,2]
        uv_grid = uv_map * 2.0 - 1.0  # [0,1] → [-1,1]
        uv_grid = uv_grid.unsqueeze(0)  # [1,H,W,2]
        # grid_sample erwartet [N,H,W,2]
        tex_sampled = F.grid_sample(texture_tensor, uv_grid.permute(0, 3, 1, 2),
                                    mode="bilinear", align_corners=True)
        rgb_t = tex_sampled.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

    mask = (rast_out[0, ..., 3] > 0).cpu().numpy()
    target_imgs.append(rgb_t.astype(np.float32))
    target_masks.append(mask.astype(np.bool_))

print("Target images generated for angles:", ANGLES)


# Create trainable voxel grid
rng = torch.manual_seed(0)
voxel_grid = torch.nn.Parameter(torch.rand(1, 3, VOXEL_RES, VOXEL_RES, VOXEL_RES, device=DEVICE) * 0.1)
optimizer = torch.optim.Adam([voxel_grid], lr=LR)

# Early stopping variables
best_loss = float('inf')
no_improve_steps = 0

# helper: world points -> grid coordinates in [-1,1]
bbox_min_t = torch.tensor(bbox_min, dtype=torch.float32, device=DEVICE)
bbox_max_t = torch.tensor(bbox_max, dtype=torch.float32, device=DEVICE)
def world_to_grid_coords(pts):  # pts: [H,W,3] torch
    # normalize to [0,1]
    norm = (pts - bbox_min_t) / (bbox_max_t - bbox_min_t + 1e-8)
    # to [-1,1]
    grid = norm * 2.0 - 1.0
    return grid  # [H,W,3]

# Training loop
print("Start training...")

scaler = torch.cuda.amp.GradScaler()
ACCUM_STEPS = 1  # Anzahl der Views pro backward-step; 1 bedeutet direkt

for it in range(1, STEPS + 1):
    t0 = time.time()
    optimizer.zero_grad()
    total_loss = 0.0
    total_pixels = 0

    # Gradient Accumulation über Views
    for vi, angle in enumerate(ANGLES):
        mvp = mvps[vi]
        ones = torch.ones_like(vertices[:, :, :1])
        vertices_h = torch.cat([vertices, ones], dim=-1)
        pos_clip = torch.matmul(vertices_h, mvp.transpose(1,2))

        # --- Rasterizer float32 ---
        rast_out, rast_db = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
        v_world_attr = vertices[0].float()
        pos_map, _ = dr.interpolate(v_world_attr, rast_out, faces_unbatched)
        pos_map = pos_map[0]

        # --- Mixed Precision für Voxel-Sampling + Loss ---
        with torch.cuda.amp.autocast():
            grid = world_to_grid_coords(pos_map).unsqueeze(0).unsqueeze(0)
            sampled = F.grid_sample(voxel_grid, grid, mode='bilinear',
                                    padding_mode='border', align_corners=True)
            pred_rgb = sampled.squeeze(2).permute(0,2,3,1)[0]

            tgt = torch.tensor(target_imgs[vi], dtype=torch.float32, device=DEVICE)
            mask = torch.tensor(target_masks[vi], dtype=torch.float32, device=DEVICE)

            diff = (pred_rgb - tgt) * mask.unsqueeze(-1)
            num_valid = mask.sum()
            if num_valid.item() == 0:
                continue
            loss = (diff ** 2).sum() / (num_valid * 3.0)

            # Scale loss für Gradient Accumulation
            loss = loss / ACCUM_STEPS
            scaler.scale(loss).backward()

            total_loss += loss.item()
            total_pixels += num_valid

        # Optional: Schritt ausführen nach N Views
        if (vi + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Falls keine Accumulation nötig, abschließend Schritt
    if len(ANGLES) % ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    tb_writer.add_scalar("Loss/train", total_loss, it)
    t1 = time.time()
    if it % 10 == 0 or it == 1:
        print(f"Iter {it}/{STEPS} loss={total_loss:.6f} time={t1-t0:.3f}s")

    # Early stopping
    if total_loss < best_loss - DELTA:
        best_loss = total_loss
        no_improve_steps = 0
    else:
        no_improve_steps += 1
    if no_improve_steps >= PATIENCE:
        print(f"Early stopping at iteration {it} (no improvement in {PATIENCE} steps)")
        break

    # Save intermediates (unverändert, nur sicher, dass .float() benutzt wird)
    if it % SAVE_INTERVAL == 0 or it == 1 or it == STEPS:
        with torch.no_grad():
            step_out = Path(OUTDIR) / f"step_{it:06d}"
            step_out.mkdir(parents=True, exist_ok=True)
            for vi, angle in enumerate(ANGLES):
                mvp = mvps[vi]
                ones = torch.ones_like(vertices[:, :, :1])
                vertices_h = torch.cat([vertices, ones], dim=-1)
                pos_clip = torch.matmul(vertices_h, mvp.transpose(1,2))
                rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
                pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)
                pos_map = pos_map[0]

                grid = world_to_grid_coords(pos_map).unsqueeze(0).unsqueeze(0)
                sampled = F.grid_sample(voxel_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
                pred_rgb = sampled.squeeze(2).permute(0,2,3,1)[0].cpu().numpy()
                pred_rgb = np.clip(pred_rgb, 0.0, 1.0)
                plt.imsave(str(step_out / f"pred_view{vi}.png"), pred_rgb)
                plt.imsave(str(step_out / f"target_view{vi}.png"), np.clip(target_imgs[vi], 0.0, 1.0))

            vg = voxel_grid.detach().cpu().numpy()[0]  # [3,D,H,W]
            np.save(str(step_out / "voxel_grid.npy"), vg)

            occ = np.linalg.norm(vg, axis=0) > 0.15
            dz, hy, wx = np.nonzero(occ)
            colors = vg[:, dz, hy, wx].T
            colors = np.clip(colors*255, 0, 255).astype(np.uint8)

            ply_path = step_out / "voxel_cloud.ply"
            with open(ply_path, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(dz)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                for x, y, z, c in zip(wx, hy, dz, colors):
                    f.write(f"{x} {y} {z} {c[0]} {c[1]} {c[2]}\n")

            print(f"Saved checkpoints for {len(ANGLES)} views to {step_out}")

tb_writer.close()

print("Training finished. Final voxel grid saved to", OUTDIR)