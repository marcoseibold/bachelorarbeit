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
import imageio
from PIL import Image

# Training with NeRF GT Mesh support

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True, help="Path to mesh")
parser.add_argument("--outdir", type=str, default="train_out", help="Output directory")
parser.add_argument("--voxel_res", type=int, default=512, help="Voxel grid resolution")
parser.add_argument("--img_res", type=int, default=128, help="Image resolution (H and W)")
parser.add_argument("--steps", type=int, default=1000, help="Training iterations")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
# [0, 15, 45, 60, 90, 105, 135, 150, 180, 195, 225, 240, 270, 285, 315, 330, 360]
# [0, 45, 90, 135, 180, 225, 270, 315]
# --angles $(seq 0 3 357)
parser.add_argument("--angles", type=int, nargs="+", default=[0, 15, 45, 60, 90, 105, 135, 150, 180, 195, 225, 240, 270, 285, 315, 330, 360], help="List of view angles in degrees for training")
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

# --- Load and adjust mesh (robust gegen trimesh.Scene) ---
loaded = trimesh.load(MESH_PATH, process=True)

# Wenn Scene zurückkommt: versuche, zu einem einzelnen Trimesh zu machen.
if isinstance(loaded, trimesh.Scene):
    print(f"Info: '{MESH_PATH}' wurde als trimesh.Scene geladen (mehrere Geometrien).")
    # erster Versuch: .dump(concatenate=True) (wendet Node-Transforms an, falls vorhanden)
    try:
        mesh = loaded.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError("scene.dump did not return a Trimesh")
        print("Scene -> einzelnes Trimesh via scene.dump(concatenate=True).")
    except Exception:
        # Fallback: nur geometrien zusammenhängen (kann Node-Transforms ignorieren)
        geoms = list(loaded.geometry.values())
        if len(geoms) == 0:
            raise ValueError(f"Die Scene enthält keine Geometrien: {MESH_PATH}")
        if len(geoms) == 1:
            mesh = geoms[0]
            print("Scene enthält genau ein Geometrie-Objekt -> verwende dieses.")
        else:
            mesh = trimesh.util.concatenate(tuple(geoms))
            print(f"Scene mit {len(geoms)} Geometrien -> mittels trimesh.util.concatenate zusammengeführt.")
else:
    mesh = loaded  # schon ein Trimesh

# Sicherheit: jetzt muss es ein Trimesh sein
if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError(f"Erwartete trimesh.Trimesh, bekam stattdessen: {type(mesh)}")

# Falls die Faces nicht trianguliert sind, triangulieren
if mesh.faces.ndim == 2 and mesh.faces.shape[1] != 3:
    print("Mesh hat keine Dreiecks-Faces -> trianguliere.")
    mesh = mesh.triangulate()

# Normalisierung / Zentrierung wie vorher
mesh_vertices = mesh.vertices.astype(np.float32)
mesh_vertices -= mesh_vertices.mean(axis=0)
mesh_vertices /= np.max(np.linalg.norm(mesh_vertices, axis=1))
mesh_vertices *= 2.0
mesh_vertices[:, 1] *= -1

# setzte die veränderten Koordinaten zurück in mesh (damit später visual/uv stimmt)
mesh.vertices = mesh_vertices

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
    tex_obj = getattr(mesh.visual.material, "image", None)

    if tex_obj is None:
        raise ValueError("Mesh hat UVs, aber kein texture image im material (material.image ist None).")

    # Normalize into a numpy array [H,W,3], dtype float32 in [0,1]
    def load_image_obj(obj):
        # Path string
        if isinstance(obj, str):
            arr = imageio.imread(obj)
            if arr.ndim == 2:  # grayscale -> to RGB
                arr = np.stack([arr]*3, axis=-1)
            return arr[..., :3].astype(np.uint8)

        # PIL Image
        if isinstance(obj, Image.Image):
            # convert to RGB to drop alpha cleanly
            arr = np.array(obj.convert("RGBA"))[..., :3]
            return arr.astype(np.uint8)

        # numpy array already
        if isinstance(obj, np.ndarray):
            arr = obj
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            return arr[..., :3].astype(np.uint8)

        # file-like (has read)
        if hasattr(obj, "read"):
            arr = imageio.imread(obj)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            return arr[..., :3].astype(np.uint8)

        # Last resort: try imageio.imread() and let it raise a helpful error if it fails
        try:
            arr = imageio.imread(obj)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            return arr[..., :3].astype(np.uint8)
        except Exception as e:
            raise ValueError(f"Unbekannter Typ für material.image: {type(obj)}. Fehler beim Laden: {e}")

    tex_img_np = load_image_obj(tex_obj)  # [H,W,3] uint8

    # normalize + to torch tensor [1,3,H,W]
    tex_img = torch.tensor(tex_img_np, dtype=torch.float32, device=DEVICE) / 255.0
    tex_img = tex_img.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    texture_tensor = tex_img
    uv_attr = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=DEVICE)  # [V,2]
    use_texture = True
    print(f"Using texture from material.image (type={type(tex_obj)}), size={tex_img_np.shape[:2]}")

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
        uv_grid = uv_grid.to(dtype=torch.float32, device=texture_tensor.device)
        # grid_sample erwartet [N,H,W,2]
        tex_sampled = F.grid_sample(texture_tensor, uv_grid,
                                    mode="bilinear", align_corners=True)
        rgb_t = tex_sampled.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

    mask = (rast_out[0, ..., 3] > 0).cpu().numpy()
    target_imgs.append(rgb_t.astype(np.float32))
    target_masks.append(mask.astype(np.bool_))

print("Target images generated for angles:", ANGLES)


# Create trainable voxel grid
#rng = torch.manual_seed(0)
voxel_grid = torch.nn.Parameter(torch.rand(1, 3, VOXEL_RES, VOXEL_RES, VOXEL_RES, device=DEVICE))
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

            # ------------------- Ersatz: Voxel-Grid export -> Punktwolke (Welt-Koordinaten) -------------------
            vg = voxel_grid.detach().cpu().numpy()[0]  # [3, D, H, W]
            np.save(str(step_out / "voxel_grid.npy"), vg)

            # Occupancy als Norm der 3 Kanäle
            occ_values = np.linalg.norm(vg, axis=0)  # [D,H,W]
            maxval = float(occ_values.max()) if occ_values.size and occ_values.max() > 0 else 1.0
            occ_norm = occ_values / (maxval + 1e-12)
            occ = occ_norm > 0.2  # threshold (kann angepasst werden)

            dz, hy, wx = np.nonzero(occ)  # indices in z,y,x order (je nach Konvention)
            num_pts = len(dz)
            print(f"Found {num_pts} occupied voxels (threshold=0.2 after normalizing by max={maxval:.6g}).")

            if num_pts == 0:
                print("Keine belegten Voxels gefunden – PLY wird nicht geschrieben.")
            else:
                # Voxel indices -> world coordinates
                # Indices: x = wx, y = hy, z = dz  (0..VOXEL_RES-1)
                ix = wx.astype(np.float32)
                iy = hy.astype(np.float32)
                iz = dz.astype(np.float32)

                # Voxel center in normalized [0,1]
                cx = (ix + 0.5) / float(VOXEL_RES)
                cy = (iy + 0.5) / float(VOXEL_RES)
                cz = (iz + 0.5) / float(VOXEL_RES)

                # bbox_min / bbox_max sind numpy arrays (wie oben berechnet)
                # map normalized coords -> world coords
                bbox_min_np = np.array(bbox_min, dtype=np.float32)
                bbox_max_np = np.array(bbox_max, dtype=np.float32)
                world_x = bbox_min_np[0] + cx * (bbox_max_np[0] - bbox_min_np[0])
                world_y = bbox_min_np[1] + cy * (bbox_max_np[1] - bbox_min_np[1])
                world_z = bbox_min_np[2] + cz * (bbox_max_np[2] - bbox_min_np[2])

                points = np.stack([world_x, world_y, world_z], axis=1)  # [N,3]

                # Farben aus dem Voxelgrid (3 Kanäle)
                cols = vg[:, dz, hy, wx].T  # [N,3]
                # Clamp/normalisiere Farben: falls Werte außerhalb [0,1], clampte sie, sonst skaliere nicht
                cols = np.clip(cols, 0.0, 1.0)
                cols_u8 = (cols * 255.0).astype(np.uint8)

                # Downsample falls zu groß (vermeidet riesige PLYs)
                MAX_POINTS = 500000
                if points.shape[0] > MAX_POINTS:
                    idxs = np.random.choice(points.shape[0], size=MAX_POINTS, replace=False)
                    points = points[idxs]
                    cols_u8 = cols_u8[idxs]
                    print(f"Downsampled voxel cloud to {MAX_POINTS} points for export.")

                # PLY schreiben (ASCII)
                ply_path = step_out / "voxel_cloud.ply"
                with open(ply_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(points)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write("end_header\n")
                    for (x, y, z), (r, g, b) in zip(points, cols_u8):
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

                print(f"Saved voxel cloud ({len(points)} points) → {ply_path}")


# Analyse: Welche Voxels wurden von den Training Views getroffen?
# print("Analysiere belegte Voxels aus den Trainings-Views...")

# affected_mask = torch.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES),
#                             dtype=torch.bool, device=DEVICE)

# for vi, mvp in enumerate(mvps):
#     ones = torch.ones_like(vertices[:, :, :1])
#     vertices_h = torch.cat([vertices, ones], dim=-1)
#     pos_clip = torch.matmul(vertices_h, mvp.transpose(1,2))

#     # Rasterize
#     rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
#     pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)
#     pos_map = pos_map[0]  # [H,W,3]

#     mask = (rast_out[0, ..., 3] > 0)
#     coords = world_to_grid_coords(pos_map[mask])  # [-1,1] coords
#     idx = ((coords + 1) / 2 * (VOXEL_RES - 1)).long()

#     valid = ((idx >= 0) & (idx < VOXEL_RES)).all(dim=-1)
#     idx = idx[valid]

#     affected_mask[idx[:,0], idx[:,1], idx[:,2]] = True

# num_affected = affected_mask.sum().item()
# total_voxels = VOXEL_RES**3
# print(f"Belegte Voxels: {num_affected} / {total_voxels} "
#       f"({100.0*num_affected/total_voxels:.6f} %)")

# # Export als PLY für Visualisierung
# dz, hy, wx = torch.nonzero(affected_mask, as_tuple=True)
# ply_path = Path(OUTDIR) / "affected_voxels.ply"
# with open(ply_path, "w") as f:
#     f.write("ply\nformat ascii 1.0\n")
#     f.write(f"element vertex {len(dz)}\n")
#     f.write("property float x\nproperty float y\nproperty float z\n")
#     f.write("end_header\n")
#     for x, y, z in zip(wx.cpu().numpy(),
#                        hy.cpu().numpy(),
#                        dz.cpu().numpy()):
#         f.write(f"{x} {y} {z}\n")
# print(f"Voxel-Occupancy als PLY gespeichert: {ply_path}")


# Kamera-Positionen + Mesh visualisieren
# camera_positions = []
# camera_directions = []

# for angle in ANGLES:
#     angle_rad = np.radians(angle)
#     radius = 5.0
#     height = -2.0
#     eye = np.array([
#         np.sin(angle_rad) * radius,
#         height,
#         np.cos(angle_rad) * radius
#     ], dtype=np.float32)
#     camera_positions.append(eye)

#     # Richtung: vom Auge zum Ursprung
#     dir_vec = -eye / np.linalg.norm(eye)
#     camera_directions.append(dir_vec)

# camera_positions = np.array(camera_positions)
# camera_directions = np.array(camera_directions)

# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection="3d")

# # Mesh (leicht transparent, größere Punkte)
# ax.scatter(vertices_np[:,0], vertices_np[:,1], vertices_np[:,2],
#            s=2, alpha=0.3, c="blue", label="Mesh")

# # Farbenliste für Kameras
# colors = ["red", "green", "blue"]
# used_labels = set()

# # Kamera-Positionen + Richtungen mit wechselnden Farben
# for i, (pos, dir_vec) in enumerate(zip(camera_positions, camera_directions)):
#     col = colors[i % len(colors)]
#     label = "Cameras" if col not in used_labels else None
#     ax.scatter(pos[0], pos[1], pos[2], c=col, s=50, label=label)
#     ax.quiver(pos[0], pos[1], pos[2],
#               dir_vec[0], dir_vec[1], dir_vec[2],
#               length=1.0, normalize=True, color=col)
#     used_labels.add(col)

# # Einheitliche Achsenskalierung
# ax.set_box_aspect([1,1,1])

# ax.set_title("Target Cameras + Mesh")
# ax.legend()

# plt.savefig(Path(OUTDIR) / "camera_positions.png")
# plt.close()
# print("Kamerapositionen visualisiert und gespeichert.")


tb_writer.close()

print("Training finished. Final voxel grid saved to", OUTDIR)


# Novel View Synthesis
novel_angles = [160]
print(f"Rendering {len(novel_angles)} novel views...")

with torch.no_grad():
    novel_out = Path(OUTDIR) / "novel_views"
    novel_out.mkdir(parents=True, exist_ok=True)

    for angle in novel_angles:
        # Neue MVP-Matrix für diesen Winkel
        mvp = get_mvp_matrix(angle, H, W, DEVICE)
        ones = torch.ones_like(vertices[:, :, :1])
        vertices_h = torch.cat([vertices, ones], dim=-1)
        pos_clip = torch.matmul(vertices_h, mvp.transpose(1, 2))

        # Rasterize + Weltkoordinaten
        rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
        pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)
        pos_map = pos_map[0]

        # Query Voxelgrid
        grid = world_to_grid_coords(pos_map).unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            voxel_grid, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        pred_rgb = sampled.squeeze(2).permute(0, 2, 3, 1)[0].cpu().numpy()
        pred_rgb = np.clip(pred_rgb, 0.0, 1.0)

        # Bild speichern
        out_path = novel_out / f"novel_angle{angle}.png"
        plt.imsave(str(out_path), pred_rgb)
        print(f"Saved novel view at angle {angle}° → {out_path}")

print("Novel view synthesis complete.")


# Unprojected pixels & target views export
# vis_out = Path(OUTDIR) / "unprojection_vis"
# vis_out.mkdir(parents=True, exist_ok=True)

# print("Erzeuge unprojected pixel cloud from training views...")

# points_list = []
# colors_list = []

# # Rasterize again per training view, unproject pixels -> collect 3D points + colors
# for vi, mvp in enumerate(mvps):
#     ones = torch.ones_like(vertices[:, :, :1])
#     vertices_h = torch.cat([vertices, ones], dim=-1)
#     pos_clip = torch.matmul(vertices_h, mvp.transpose(1, 2))

#     rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
#     pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)  # [1,H,W,3]
#     pos_map = pos_map[0].cpu().numpy()

#     mask = (rast_out[0, ..., 3] > 0).cpu().numpy()  # boolean [H,W]
#     if mask.sum() == 0:
#         continue

#     pts = pos_map[mask]               # [N,3] in world coords
#     cols = target_imgs[vi][mask]      # [N,3] in [0,1] (numpy)
#     if pts.shape[0] > 0:
#         points_list.append(pts)
#         colors_list.append(cols)

# if len(points_list) == 0:
#     print("Keine unprojected pixels gefunden (keine gültigen Rasterhits).")
# else:
#     pts_all = np.concatenate(points_list, axis=0)
#     cols_all = np.concatenate(colors_list, axis=0)

#     # --- save as .ply for external visualization (Meshlab / CloudCompare) ---
#     ply_path = vis_out / "unprojected_points.ply"
#     with open(ply_path, "w") as f:
#         f.write("ply\nformat ascii 1.0\n")
#         f.write(f"element vertex {len(pts_all)}\n")
#         f.write("property float x\nproperty float y\nproperty float z\n")
#         f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
#         f.write("end_header\n")
#         # colors come in [0,1] -> convert to 0..255
#         cols_u8 = np.clip((cols_all * 255.0), 0, 255).astype(np.uint8)
#         for (x, y, z), (r, g, b) in zip(pts_all, cols_u8):
#             f.write(f"{x} {y} {z} {r} {g} {b}\n")
#     print(f"Saved unprojected pointcloud .ply -> {ply_path}")

# # --- Analyse: Train vs Target Coverage ---
# print("Vergleiche Trainings- und Target-Views für Voxel-Coverage...")

# def compute_affected_mask(angles, mvps):
#     mask = torch.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES),
#                        dtype=torch.bool, device=DEVICE)
#     for vi, mvp in enumerate(mvps):
#         ones = torch.ones_like(vertices[:, :, :1])
#         vertices_h = torch.cat([vertices, ones], dim=-1)
#         pos_clip = torch.matmul(vertices_h, mvp.transpose(1,2))
#         rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])
#         pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)
#         pos_map = pos_map[0]

#         mask_pix = (rast_out[0, ..., 3] > 0)
#         coords = world_to_grid_coords(pos_map[mask_pix])  # [N,3] in [-1,1]
#         idx = ((coords + 1) / 2 * (VOXEL_RES - 1)).long()
#         valid = ((idx >= 0) & (idx < VOXEL_RES)).all(dim=-1)
#         idx = idx[valid]
#         mask[idx[:,0], idx[:,1], idx[:,2]] = True
#     return mask

# # Train- und Target-Masken berechnen
# train_mask = compute_affected_mask(ANGLES, mvps)

# # Target-Views separat definieren (z. B. Novel Angles!)
# novel_angles = [160]  # oder mehrere Novel-Angles
# novel_mvps = [get_mvp_matrix(a, H, W, DEVICE) for a in novel_angles]
# target_mask = compute_affected_mask(novel_angles, novel_mvps)

# # Unterschiede
# only_train = train_mask & ~target_mask
# only_target = target_mask & ~train_mask
# both = train_mask & target_mask

# print("Nur Train:", only_train.sum().item())
# print("Nur Target:", only_target.sum().item())
# print("Beide:", both.sum().item())

# # --- Export PLY ---
# dz, hy, wx = torch.nonzero(train_mask | target_mask, as_tuple=True)

# ply_path = Path(OUTDIR) / "train_vs_target_voxels.ply"
# with open(ply_path, "w") as f:
#     f.write("ply\nformat ascii 1.0\n")
#     f.write(f"element vertex {len(dz)}\n")
#     f.write("property float x\nproperty float y\nproperty float z\n")
#     f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
#     f.write("end_header\n")

#     for x, y, z in zip(wx.cpu().numpy(),
#                        hy.cpu().numpy(),
#                        dz.cpu().numpy()):
#         if only_train[z, y, x]:
#             color = (0, 255, 0)  # grün
#         elif only_target[z, y, x]:
#             color = (255, 0, 0)  # rot
#         else:  # both
#             color = (0, 0, 255)  # blau
#         f.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")

# print(f"Train-vs-Target PLY gespeichert: {ply_path}")
