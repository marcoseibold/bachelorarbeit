import os
import argparse
from pathlib import Path
import time

import math
import json
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import imageio
from PIL import Image

# Same as training_nerf, with sanity checks to verify camera poses, no training

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, required=True, help="Path to mesh")
parser.add_argument("--data_root", type=str, default=None, help="Optional: Path to NeRF-Synthetic scene folder (contains transforms_train.json and images) for sanity-check")
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
DATA_ROOT = args.data_root
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
loaded = trimesh.load(MESH_PATH, process=True)

# Preserve raw mesh (use for sanity-checks)
if isinstance(loaded, trimesh.Scene):
    try:
        mesh_raw = loaded.dump(concatenate=True)
        if not isinstance(mesh_raw, trimesh.Trimesh):
            raise RuntimeError("scene.dump did not return a Trimesh")
    except Exception:
        geoms = list(loaded.geometry.values())
        if len(geoms) == 0:
            raise ValueError(f"Die Scene enthält keine Geometrien: {MESH_PATH}")
        if len(geoms) == 1:
            mesh_raw = geoms[0]
        else:
            mesh_raw = trimesh.util.concatenate(tuple(geoms))
else:
    mesh_raw = loaded  # original, unmodified mesh

# make a working copy for normalization / training (keeps mesh_raw intact)
mesh = mesh_raw.copy()


# Safety check
if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError(f"Erwartete trimesh.Trimesh, bekam stattdessen: {type(mesh)}")

# triangulate
if mesh.faces.ndim == 2 and mesh.faces.shape[1] != 3:
    print("Mesh hat keine Dreiecks-Faces -> trianguliere.")
    mesh = mesh.triangulate()

# Normalization / Centering
mesh_vertices = mesh.vertices.astype(np.float32)
mesh_vertices -= mesh_vertices.mean(axis=0)
mesh_vertices /= np.max(np.linalg.norm(mesh_vertices, axis=1))
mesh_vertices *= 2.0
#mesh_vertices[:, 1] *= -1

mesh.vertices = mesh_vertices

r = np.array([[1, 0, 0],
              [0, 0,-1],
              [0, 1, 0]])
mesh.vertices = mesh.vertices @ r.T

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

# Try Vertex Colors or Texture
use_vertex_colors = False
use_texture = False
texture_tensor = None
uv_attr = None

if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors.shape[1] >= 3:
    unique_colors = np.unique(mesh.visual.vertex_colors[:, :3], axis=0)
    if len(unique_colors) > 1:
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

# Create target images
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
        # grid_sample needs [N,H,W,2]
        tex_sampled = F.grid_sample(texture_tensor, uv_grid,
                                    mode="bilinear", align_corners=True)
        rgb_t = tex_sampled.permute(0, 2, 3, 1)[0].detach().cpu().numpy()

    mask = (rast_out[0, ..., 3] > 0).cpu().numpy()
    target_imgs.append(rgb_t.astype(np.float32))
    target_masks.append(mask.astype(np.bool_))

print("Target images generated for angles:", ANGLES)

# SANITY-CHECK: Use first camera pose to render GT mesh and compare to first train image 
if DATA_ROOT is not None:
    try:
        print("Running sanity-check with dataset:", DATA_ROOT)
        # load transforms_train.json
        transforms_path = os.path.join(DATA_ROOT, "transforms_train.json")
        if not os.path.exists(transforms_path):
            transforms_path = os.path.join(DATA_ROOT, "transforms_train.json".replace("train","train"))
        with open(transforms_path, "r") as f:
            meta = json.load(f)
        frame0 = meta["frames"][0]
        fname = os.path.join(DATA_ROOT, frame0["file_path"] + ".png")
        if not os.path.exists(fname):
            alt = fname.replace(".png", ".jpg")
            if os.path.exists(alt):
                fname = alt
            else:
                raise FileNotFoundError(f"Could not find image {fname} or {alt} for sanity-check.")
        
        pil = Image.open(fname).convert("RGBA")
        pil = pil.resize((IMG_RES, IMG_RES), resample=Image.LANCZOS)
        arr = np.array(pil).astype(np.float32) / 255.0
        if arr.shape[-1] == 4:
            gt_rgb = arr[..., :3]
            alpha = arr[..., 3]
            gt_mask = (alpha > 0).astype(np.float32)
        else:
            gt_rgb = arr[..., :3]
            gt_mask = np.ones((IMG_RES, IMG_RES), dtype=np.float32)

        cam_angle_x = float(meta["camera_angle_x"])
        focal_pix = 0.5 * IMG_RES / math.tan(0.5 * cam_angle_x)

        pose0 = np.array(frame0["transform_matrix"], dtype=np.float32)

        Hc = IMG_RES
        Wc = IMG_RES

        def build_rays(pose_c2w, focal, W, H, step=8):
            cx, cy = W/2, H/2
            us = np.arange(0, W, step)
            vs = np.arange(0, H, step)
            uu, vv = np.meshgrid(us, vs)
            uu, vv = uu.ravel(), vv.ravel()
            x = (uu - cx) / focal
            y = -(vv - cy) / focal
            z = -np.ones_like(x)
            dirs_cam = np.stack([x, y, z], axis=1)
            dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
            R, t = pose_c2w[:3,:3], pose_c2w[:3,3]
            dirs_world = (R @ dirs_cam.T).T
            origins = np.tile(t[None,:], (dirs_world.shape[0],1))
            return origins, dirs_world

        origins, dirs = build_rays(pose0, focal_pix, Wc, Hc, step=8)

        # Intersector out of Mesh
        mesh_for_rays = trimesh.Trimesh(vertices=vertices[0].cpu().numpy(),
                                        faces=faces_unbatched.cpu().numpy())
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_for_rays)

        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins=origins, ray_directions=dirs, multiple_hits=False
        )

        print(f"[Sanity] Ray–Mesh Intersections: {len(locations)} Punkte")

        # Plot Mesh + Intersections
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(mesh_for_rays.vertices[:,0],
                        mesh_for_rays.vertices[:,1],
                        mesh_for_rays.faces,
                        mesh_for_rays.vertices[:,2],
                        alpha=0.4)
        if len(locations) > 0:
            ax.scatter(locations[:,0], locations[:,1], locations[:,2],
                    c='red', s=6, label='ray hits')
        cam_origin = pose0[:3,3]
        ax.scatter([cam_origin[0]], [cam_origin[1]], [cam_origin[2]], c='blue', s=50, label='camera')
        ax.legend()
        plt.savefig(os.path.join(OUTDIR, "ray_mesh_intersections.png"), dpi=200)
        plt.close(fig)


        def perspective_from_focal(focal, W, H, near=0.1, far=10.0):
            fovy = 2.0 * math.atan(float(H) / (2.0 * float(focal)))
            f = 1.0 / math.tan(fovy / 2.0)
            m = np.zeros((4,4), dtype=np.float32)
            m[0,0] = f / (W / float(H))
            m[1,1] = -f
            m[2,2] = (far + near) / (near - far)
            m[2,3] = (2*far*near) / (near - far)
            m[3,2] = -1.0
            return m

        dbg_dir = Path(OUTDIR) / "debug_check"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        modes = ["normalized", "no_normalize", "pose_yflip"]
        for mode in modes:
            try:
                if mode == "normalized":
                    verts_tb = vertices  # prepared normalized vertices [1,V,3]
                    faces_tb = faces_unbatched
                    pose_use = pose0.copy()
                elif mode == "no_normalize":
                    # reload original mesh without normalization (if possible)
                    mesh_raw = trimesh.load(MESH_PATH, process=True)
                    if isinstance(mesh_raw, trimesh.Scene):
                        try:
                            mesh_raw = mesh_raw.dump(concatenate=True)
                        except Exception:
                            geoms = list(mesh_raw.geometry.values())
                            mesh_raw = trimesh.util.concatenate(tuple(geoms))
                    if mesh_raw.faces.ndim == 2 and mesh_raw.faces.shape[1] != 3:
                        mesh_raw = mesh_raw.triangulate()
                    verts_np_raw = mesh_raw.vertices.astype(np.float32)
                    faces_np_raw = mesh_raw.faces.astype(np.int32)
                    verts_tb = torch.tensor(verts_np_raw, device=DEVICE).unsqueeze(0)
                    faces_tb = torch.tensor(faces_np_raw, device=DEVICE).unsqueeze(0)[0].contiguous()
                    pose_use = pose0.copy()
                elif mode == "pose_yflip":
                    verts_tb = vertices
                    faces_tb = faces_unbatched
                    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                    pose_use = pose0 @ flip
                else:
                    continue

                proj_np = perspective_from_focal(focal_pix, Wc, Hc, near=0.1, far=10.0)
                view_np = np.linalg.inv(pose_use).astype(np.float32)
                mvp_np = proj_np @ view_np
                mvp = torch.tensor(mvp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # random per-vertex colors
                Vcount = verts_tb.shape[1]
                rand_colors = (np.random.RandomState(42).rand(Vcount, 3)).astype(np.float32)
                vcol_tensor = torch.tensor(rand_colors, dtype=torch.float32, device=DEVICE)

                ones = torch.ones_like(verts_tb[:, :, :1])
                verts_h = torch.cat([verts_tb, ones], dim=-1)
                pos_clip = torch.matmul(verts_h, mvp.transpose(1,2))
                rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_tb, resolution=[Hc, Wc])
                rgb_r, _ = dr.interpolate(vcol_tensor, rast_out, faces_tb)
                pred_rgb = rgb_r[0].cpu().numpy()

                # determine mask
                mask_from_rast = (rast_out[0, ..., 3].cpu().numpy() > 0).astype(np.float32)
                mask_use = gt_mask if gt_mask is not None else mask_from_rast

                valid = mask_use > 0
                if valid.sum() == 0:
                    mse = float('nan')
                else:
                    mse = ((np.clip(pred_rgb,0,1) - np.clip(gt_rgb,0,1))**2 * valid[...,None]).sum() / (valid.sum() * 3.0)

                print(f"[Sanity:{mode}] MSE (masked) between mesh-render and GT first train image: {mse:.6e}")

                # save visualization
                plt.imsave(str(dbg_dir / f"gt_first.png"), np.clip(gt_rgb,0,1))
                plt.imsave(str(dbg_dir / f"pred_mesh_first_{mode}.png"), np.clip(pred_rgb,0,1))
                diff_map = np.abs(pred_rgb - gt_rgb).mean(-1)
                if diff_map.max() > 0:
                    diff_vis = diff_map / diff_map.max()
                else:
                    diff_vis = diff_map
                plt.imsave(str(dbg_dir / f"diff_map_first_{mode}.png"), np.clip(diff_vis,0,1))

            except Exception as e:
                print(f"[Sanity:{mode}] skipped due to error: {e}")

        print(f"[Sanity] finished; debug images (if created) are in {dbg_dir}")

    except Exception as e:
        print(f"[Sanity] skipped (error while preparing sanity-check): {e}")
else:
    print("No --data_root provided -> sanity-check skipped. Provide --data_root to run it.")

# EXTRA VISUALIZATION: NeRF dataset viewpoint
if DATA_ROOT is not None:
    try:
        print("Extra visualization with one dataset viewpoint...")

        # Load first camera
        transforms_path = os.path.join(DATA_ROOT, "transforms_train.json")
        with open(transforms_path, "r") as f:
            meta = json.load(f)
        frame0 = meta["frames"][0]
        fname = os.path.join(DATA_ROOT, frame0["file_path"] + ".png")
        if not os.path.exists(fname):
            fname = fname.replace(".png", ".jpg")
        gt_img = np.array(Image.open(fname).convert("RGB").resize((IMG_RES, IMG_RES)))

        pose0 = np.array(frame0["transform_matrix"], dtype=np.float32)  # camera-to-world
        cam_pos = pose0[:3, 3]

        cam_angle_x = float(meta["camera_angle_x"])
        focal_pix = 0.5 * IMG_RES / math.tan(0.5 * cam_angle_x)

        print("focal_pix:", focal_pix)
        print("img_res:", IMG_RES)
        print("approx FOV deg:", 2*np.degrees(np.arctan(IMG_RES/(2*focal_pix))))

        # Ray-Mesh Intersection
        import trimesh
        from trimesh.ray import ray_pyembree

        # Build rays
        xs = np.linspace(-IMG_RES/2, IMG_RES/2, 32)
        ys = np.linspace(-IMG_RES/2, IMG_RES/2, 32)
        dirs = []
        for y in ys:
            for x in xs:
                d = np.array([x, -y, -focal_pix], dtype=np.float32)
                d = d / np.linalg.norm(d)
                d_world = (pose0[:3, :3] @ d)  # transform to world
                dirs.append(d_world)
        dirs = np.array(dirs)

        origins = np.tile(cam_pos[None, :], (dirs.shape[0], 1))

        mesh_for_rmi = trimesh.Trimesh(vertices=vertices[0].cpu().numpy(), faces=faces_unbatched.cpu().numpy())
        rmi = ray_pyembree.RayMeshIntersector(mesh_for_rmi)
        locs, index_ray, index_tri = rmi.intersects_location(origins, dirs, multiple_hits=False)

        # Plot 3D: Mesh + Hits + Camera
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        mesh_show = mesh.copy()
        mesh_show.visual.face_colors = [100, 100, 200, 100]  # halftransparent
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                        triangles=mesh.faces, color='blue', alpha=0.15, linewidth=0.2)
        ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2], c='r', s=5, label="ray hits")
        ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], c='b', s=50, label="camera")
        ax.legend()
        plt.title("Dataset viewpoint: ray hits + mesh")
        plt.savefig(os.path.join(OUTDIR, "dataset_viewpoint_hits.png"), dpi=200)
        plt.close()

        # Render Mesh with Pose
        def perspective_from_focal(focal, W, H, near=0.1, far=10.0):
            fovy = 2.0 * math.atan(float(H) / (2.0 * float(focal)))
            f = 1.0 / math.tan(fovy / 2.0)
            m = np.zeros((4, 4), dtype=np.float32)
            m[0, 0] = f / (W / float(H))
            m[1, 1] = -f
            m[2, 2] = (far + near) / (near - far)
            m[2, 3] = (2*far*near) / (near - far)
            m[3, 2] = -1.0
            return m

        proj_np = perspective_from_focal(focal_pix, IMG_RES, IMG_RES)
        view_np = np.linalg.inv(pose0).astype(np.float32)
        mvp_np = proj_np @ view_np
        mvp = torch.tensor(mvp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        ones = torch.ones_like(vertices[:, :, :1])
        verts_h = torch.cat([vertices, ones], dim=-1)
        pos_clip = torch.matmul(verts_h, mvp.transpose(1, 2))
        rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[IMG_RES, IMG_RES])
        vcol_tensor = torch.rand(vertices.shape[1], 3, device=DEVICE)  # random vertex colors
        rgb_r, _ = dr.interpolate(vcol_tensor, rast_out, faces_unbatched)
        render_img = rgb_r[0].cpu().numpy()

        # Save GT and Render
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(gt_img)
        axs[0].set_title("GT Image")
        axs[0].axis("off")
        axs[1].imshow(np.clip(render_img, 0, 1))
        axs[1].set_title("Rendered Mesh")
        axs[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "dataset_viewpoint_gt_vs_render.png"), dpi=200)
        plt.close()

        print("Extra visualization saved to", OUTDIR)

    except Exception as e:
        print("[Extra visualization] skipped due to error:", e)


# Create trainable voxel grid
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

# rest of training loop

tb_writer.close()
print("Training finished. Final voxel grid saved to", OUTDIR)
