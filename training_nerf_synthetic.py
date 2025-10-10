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

# ---------------- Arguments ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, default=None, help="(optional) Path to GT mesh (OBJ/PLY). If omitted, dataset-only mode.")
parser.add_argument("--data_root", type=str, default=None, help="Path to NeRF-Synthetic scene folder (transforms_*.json + images).")
parser.add_argument("--outdir", type=str, default="train_out", help="Output directory")
parser.add_argument("--voxel_res", type=int, default=256, help="Voxel grid resolution (D,H,W)")
parser.add_argument("--img_res", type=int, default=128, help="Image resolution (H and W)")
parser.add_argument("--steps", type=int, default=1000, help="Training iterations")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--save_interval", type=int, default=100, help="Save image/checkpoint every N steps")
parser.add_argument("--early_stop_patience", type=int, default=200, help="Stop training if no improvement in N steps")
parser.add_argument("--early_stop_delta", type=float, default=1e-6, help="Minimum change to consider an improvement")
parser.add_argument("--bbox_size", type=float, default=2.0, help="Scene bounding box (cube side length) used when no mesh provided")
parser.add_argument("--n_samples", type=int, default=64, help="Samples per ray for volumetric rendering (dataset-only)")
parser.add_argument("--near", type=float, default=0.1, help="Near plane for ray sampling")
parser.add_argument("--far", type=float, default=6.0, help="Far plane for ray sampling")
parser.add_argument("--normalize_mesh", action="store_true", help="If set, normalize/center/scale mesh on load")
parser.add_argument("--apply_pose_flip", action="store_true", help="If set, apply a diag(1,-1,-1,1) flip to dataset poses (Blender→neRF)")
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
NORMALIZE_MESH = args.normalize_mesh
APPLY_POSE_FLIP = args.apply_pose_flip

os.makedirs(OUTDIR, exist_ok=True)
tb_writer = SummaryWriter(log_dir=os.path.join(OUTDIR, "tensorboard"))

print(f"[INFO] Device: {DEVICE}")

# ---------------- helpers ----------------
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
    m[1, 1] = -f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def mvp_from_cam(cam2world, H, W, focal, near=0.1, far=10.0):
    # view = inverse(cam2world)
    view = np.linalg.inv(cam2world).astype(np.float32)
    fovy = 2.0 * np.arctan(float(H) / (2.0 * float(focal)))
    proj = perspective(fovy, float(W) / float(H), near, far)
    return proj @ view

# ---------------- Dataset loader (NeRF-Synthetic) ----------------
class NeRFSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", img_wh=(128,128)):
        self.root = root
        self.split = split
        self.W, self.H = img_wh
        json_path = os.path.join(root, f"transforms_{split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Expected {json_path}")
        with open(json_path, "r") as f:
            self.meta = json.load(f)

        self.image_paths = []
        self.poses = []
        for frame in self.meta["frames"]:
            p = os.path.join(root, frame["file_path"] + ".png")
            if not os.path.exists(p):
                p_jpg = p.replace(".png", ".jpg")
                if os.path.exists(p_jpg):
                    p = p_jpg
                else:
                    raise FileNotFoundError(f"Image not found: {p}")
            self.image_paths.append(p)
            self.poses.append(np.array(frame["transform_matrix"], dtype=np.float32))
        self.poses = np.stack(self.poses, axis=0)
        # focal from camera_angle_x
        self.focal = 0.5 * self.W / np.tan(0.5 * float(self.meta["camera_angle_x"]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pil = Image.open(self.image_paths[idx])
        pil_rgba = pil.convert("RGBA")
        if (pil_rgba.width, pil_rgba.height) != (self.W, self.H):
            pil_rgba = pil_rgba.resize((self.W, self.H), resample=Image.LANCZOS)
        arr = np.array(pil_rgba)  # [H,W,4]
        if arr.ndim == 2:
            arr = np.stack([arr]*4, axis=-1)
        alpha = arr[..., 3] / 255.0
        rgb = arr[..., :3]
        a = alpha[..., None].astype(np.float32)
        img = (rgb.astype(np.float32) / 255.0) * a + (1.0 - a)
        mask = alpha.astype(np.float32)
        pose = self.poses[idx]
        return {
            "img": torch.from_numpy(img).permute(2,0,1),  # [3,H,W]
            "mask": torch.from_numpy(mask),               # [H,W]
            "pose": torch.from_numpy(pose),               # [4,4]
            "focal": torch.tensor(self.focal, dtype=torch.float32),
            "path": self.image_paths[idx]
        }

# ---------------- Rays + volumetric renderer (dataset-only) ----------------
def get_rays_from_pose(cam2world, focal, H, W, device):
    if isinstance(cam2world, np.ndarray):
        cam2world = torch.from_numpy(cam2world).to(device=device, dtype=torch.float32)
    else:
        cam2world = cam2world.to(device=device, dtype=torch.float32)

    i = (torch.arange(0, W, device=device).float() + 0.5)
    j = (torch.arange(0, H, device=device).float() + 0.5)
    try:
        px, py = torch.meshgrid(i, j, indexing='xy')
    except TypeError:
        px, py = torch.meshgrid(i, j)
        px = px.t(); py = py.t()

    cx = W * 0.5
    cy = H * 0.5
    x_cam = (px - cx) / focal
    y_cam = -(py - cy) / focal
    z_cam = torch.ones_like(x_cam)

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [H,W,3]
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]

    dirs_world = dirs_cam.reshape(-1, 3) @ R.t()
    dirs_world = dirs_world.reshape(H, W, 3)
    dirs_world = dirs_world / (torch.norm(dirs_world, dim=-1, keepdim=True) + 1e-12)
    origin = t.view(1,1,3).expand(H, W, 3)
    return origin, dirs_world

# world->grid mapping will be filled later depending on mesh or bbox
bbox_min_t = None
bbox_max_t = None
def world_to_grid_coords(pts):
    # pts: [...,3] in world coords (torch)
    global bbox_min_t, bbox_max_t
    norm = (pts - bbox_min_t) / (bbox_max_t - bbox_min_t + 1e-8)
    grid = norm * 2.0 - 1.0
    return grid

def render_rays_from_voxelgrid(voxel_grid, cam2world, focal, H, W, N_samples=64, near=0.1, far=6.0, device='cuda'):
    origin, dirs = get_rays_from_pose(cam2world, focal, H, W, device)
    t_vals = torch.linspace(near, far, steps=N_samples, device=device)  # [N]
    pts = origin.unsqueeze(0) + t_vals.view(N_samples,1,1,1) * dirs.unsqueeze(0)  # [N,H,W,3]
    grid_coords = world_to_grid_coords(pts).to(dtype=torch.float32, device=device)  # [-1,1]
    grid = grid_coords.unsqueeze(0)  # [1,N,H,W,3]

    # Sample/query voxel grid (5D sampling): voxel_grid [1,C,D,H,W], grid [1,D_out,H_out,W_out,3]
    sampled = F.grid_sample(voxel_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
    # sampled: [1, C, N, H, W] -> permute to [1,N,H,W,C]
    sampled = sampled.permute(0, 2, 3, 4, 1)
    colors = sampled[..., :3]  # [1,N,H,W,3]
    sigma = torch.norm(colors, dim=-1)  # [1,N,H,W]

    delta = (far - near) / float(N_samples)
    alpha = 1.0 - torch.exp(-F.relu(sigma) * delta)  # [1,N,H,W]

    eps = 1e-10
    one_m_alpha = (1.0 - alpha + eps)
    T = torch.cumprod(torch.cat([torch.ones_like(one_m_alpha[:, :1, ...]), one_m_alpha[:, :-1, ...]], dim=1), dim=1)
    weights = alpha * T  # [1,N,H,W]

    rgb_map = (weights.unsqueeze(-1) * colors).sum(dim=1)[0]  # [H,W,3]
    trans = 1.0 - weights.sum(dim=1)[0]                       # [H,W]
    rgb_map = rgb_map + trans.unsqueeze(-1) * 1.0             # white background
    return rgb_map

def render_via_mesh_rasterization(voxel_grid_param, cam2world_np, focal, H, W,
                                  vertices, faces_unbatched, ctx, voxel_sample_mode='bilinear',
                                  near=0.1, far=10.0, device='cuda'):
    """
    Rasterize the mesh from the given cam2world + focal, interpolate world positions,
    convert them to voxel grid coordinates and sample voxel_grid to get RGB per-pixel.
    Returns a torch tensor [H,W,3] (float32, on device).
    """
    # build MVP like in other places
    mvp_np = mvp_from_cam(cam2world_np, H, W, focal, near=near, far=far)  # numpy 4x4
    mvp = torch.tensor(mvp_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,4,4]

    # prepare vertices (batched [1,V,3]) and ones for homogeneous multiply
    ones = torch.ones_like(vertices[:, :, :1])
    verts_h = torch.cat([vertices, ones], dim=-1)  # [1,V,4]

    # pos_clip: multiply with mvp (transpose to match shapes)
    pos_clip = torch.matmul(verts_h, mvp.transpose(1, 2))  # [1,V,4]

    # rasterize -> rast_out
    rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[H, W])

    # interpolate world-space positions (vertices are world coords)
    pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)  # [1,H,W,3] or [H,W,3] depending
    pos_map = pos_map[0]  # [H,W,3] on device

    # convert world pos to voxel grid coords in [-1,1]
    grid_coords = world_to_grid_coords(pos_map)  # expects torch, returns [-1,1]
    # make sampling grid shape [1,1,H,W,3] for grid_sample
    grid_for_sample = grid_coords.unsqueeze(0).unsqueeze(0)  # [1,1,H,W,3]

    # sample/query voxel grid: voxel_grid_param is [1,C,D,H,W]
    sampled = F.grid_sample(voxel_grid_param, grid_for_sample, mode=voxel_sample_mode,
                            padding_mode='border', align_corners=True)  # [1,C,1,H,W]
    # reorder to [1,1,H,W,C]
    sampled = sampled.permute(0, 2, 3, 4, 1)  # [1,1,H,W,C]
    sampled = sampled.squeeze(1)  # [1,H,W,C]
    sampled = sampled[0]  # [H,W,C]

    # If voxel grid stores colors in first 3 channels, pick them. Else adapt accordingly.
    pred_rgb = sampled[..., :3]  # [H,W,3], torch tensor on device

    # Set background to black
    mask = rast_out[0, ..., 3] > 0
    pred_rgb[~mask] = 1.0
    return pred_rgb

# ---------------- Prepare scene (mesh or bbox) & load dataset if requested ----------------
use_dataset = False
dataset_frames = []   # list of dicts with pose, path, focal
dataset_target_imgs = []
dataset_target_masks = []
dataset_mvps = []
dataset_cam2worlds = []
dataset_focal = None

if DATA_ROOT is not None:
    json_path = os.path.join(DATA_ROOT, "transforms_train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset transforms_train.json not found at {json_path}")
    meta = json.load(open(json_path, "r"))
    frames = meta.get("frames", [])
    if len(frames) == 0:
        raise ValueError("transforms_train.json contains 0 frames")
    use_dataset = True
    dataset_focal = 0.5 * IMG_RES / np.tan(0.5 * float(meta["camera_angle_x"]))
    print(f"[INFO] Dataset found: {len(frames)} frames, focal(derived) = {dataset_focal:.3f}")
    # flip matrix for optional blender->nerf conversion (right-multiply)
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    for fr in frames:
        cam2world = np.array(fr["transform_matrix"], dtype=np.float32)
        if APPLY_POSE_FLIP:
            cam2world = cam2world @ flip
        dataset_cam2worlds.append(cam2world)
        # load image (RGBA), resize to IMG_RES
        img_path = os.path.join(DATA_ROOT, fr["file_path"] + ".png")
        if not os.path.exists(img_path):
            alt = img_path.replace(".png", ".jpg")
            if os.path.exists(alt):
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")
        pil = Image.open(img_path).convert("RGBA")
        if pil.size != (IMG_RES, IMG_RES):
            pil = pil.resize((IMG_RES, IMG_RES), resample=Image.LANCZOS)
        arr = np.array(pil).astype(np.float32)

        # Alpha in [0,1]
        alpha = arr[..., 3] / 255.0

        # Premultiply: RGB * alpha
        a = alpha[..., None].astype(np.float32)            # alpha in [0,1]
        rgb = (arr[..., :3] / 255.0).astype(np.float32) * a + (1.0 - a)

        # Mask
        mask = alpha.astype(np.float32)
        dataset_target_imgs.append(rgb.astype(np.float32))
        dataset_target_masks.append(mask.astype(np.float32))

# ---------------- If mesh is provided: load and prepare rasterization targets ----------------
mesh = None
mvps = []
target_imgs = []
target_masks = []
use_mesh = False

if MESH_PATH is not None:
    import trimesh
    loaded = trimesh.load(MESH_PATH, process=True)
    if isinstance(loaded, trimesh.Scene):
        try:
            mesh = loaded.dump(concatenate=True)
            if not isinstance(mesh, trimesh.Trimesh):
                raise RuntimeError("scene.dump did not return a Trimesh")
        except Exception:
            geoms = list(loaded.geometry.values())
            if len(geoms) == 0:
                raise ValueError("Scene contains no geometry")
            mesh = trimesh.util.concatenate(tuple(geoms))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Expected trimesh.Trimesh")

    # optionally normalize mesh
    mesh_vertices = mesh.vertices.astype(np.float32)
    #mesh_vertices[:, 1] *= -1.0
    if NORMALIZE_MESH:
        mesh_vertices -= mesh_vertices.mean(axis=0)
        mesh_vertices /= (np.max(np.linalg.norm(mesh_vertices, axis=1)))
        #mesh_vertices *= 2.0
        #mesh_vertices[:, 1] *= -1.0
        print("[INFO] Mesh normalized (centered/scaled/y-flip).")
    mesh.vertices = mesh_vertices

    r = np.array([[1, 0, 0],
              [0, 0,-1],
              [0, 1, 0]])
    mesh.vertices = mesh.vertices @ r.T

    # prepare visuals: vertex colors or texture or fallback
    use_vertex_colors = False
    use_texture = False
    texture_tensor = None
    uv_attr = None

    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and mesh.visual.vertex_colors.shape[1] >= 3:
        unique_colors = np.unique(mesh.visual.vertex_colors[:, :3], axis=0)
        if len(unique_colors) > 1:
            use_vertex_colors = True
            vcol = mesh.visual.vertex_colors[:, :3] / 255.0
            vcol = np.clip(vcol, 0.0, 1.0).astype(np.float32)
            vcol_tensor = torch.tensor(vcol, dtype=torch.float32, device=DEVICE)
            print("[INFO] Using vertex colors on mesh.")

    tex_obj = getattr(mesh.visual.material, "image", None) if hasattr(mesh.visual, "material") else None
    if (not use_vertex_colors) and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and tex_obj is not None:
        # load texture robustly
        def load_image_obj(obj):
            if isinstance(obj, str):
                arr = imageio.imread(obj)
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                return arr[..., :3].astype(np.uint8)
            if isinstance(obj, Image.Image):
                arr = np.array(obj.convert("RGB"))
                return arr[..., :3].astype(np.uint8)
            if isinstance(obj, np.ndarray):
                arr = obj
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                return arr[..., :3].astype(np.uint8)
            if hasattr(obj, "read"):
                arr = imageio.imread(obj)
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                return arr[..., :3].astype(np.uint8)
            arr = imageio.imread(obj)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            return arr[..., :3].astype(np.uint8)
        try:
            tex_np = load_image_obj(tex_obj)
            texture_tensor = torch.tensor(tex_np, dtype=torch.float32, device=DEVICE) / 255.0
            texture_tensor = texture_tensor.permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
            uv_attr = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=DEVICE)
            use_texture = True
            print("[INFO] Mesh texture loaded from material.image")
        except Exception as e:
            print("[WARN] Could not load mesh texture:", e)

    if (not use_vertex_colors) and (not use_texture):
        v_world = mesh.vertices.astype(np.float32)
        bbox_min_local = v_world.min(axis=0)
        bbox_max_local = v_world.max(axis=0)
        vcol = (v_world - bbox_min_local) / (bbox_max_local - bbox_min_local + 1e-8)
        vcol = np.clip(vcol, 0.0, 1.0)
        vcol_tensor = torch.tensor(vcol, dtype=torch.float32, device=DEVICE)
        use_vertex_colors = True
        print("[INFO] Using fallback position-based vertex colors for mesh.")

    # prepare tensors for rasterizer
    vertices_np = mesh.vertices.astype(np.float32)
    faces_np = mesh.faces.astype(np.int32)
    vertices = torch.tensor(vertices_np, device=DEVICE).unsqueeze(0)  # [1,V,3]
    faces = torch.tensor(faces_np, device=DEVICE).unsqueeze(0)        # [1,F,3]
    faces_unbatched = faces[0].contiguous()

    # compute bbox from mesh (world->grid mapping)
    bbox_min = vertices[0].min(dim=0)[0].cpu().numpy() - 1e-4
    bbox_max = vertices[0].max(dim=0)[0].cpu().numpy() + 1e-4
    bbox_min = bbox_min.astype(np.float32); bbox_max = bbox_max.astype(np.float32)
    bbox_min_t = torch.tensor(bbox_min, dtype=torch.float32, device=DEVICE)
    bbox_max_t = torch.tensor(bbox_max, dtype=torch.float32, device=DEVICE)

    print(f"[DEBUG] Mesh bbox_min: {bbox_min}, bbox_max: {bbox_max}")

    # rasterize targets: if dataset provided, use dataset poses, else generate views by angles
    ctx = dr.RasterizeCudaContext()
    H = IMG_RES; W = IMG_RES

    if use_dataset:
        # create mvp per dataset pose (from dataset_cam2worlds + dataset_focal)
        for cam2world in dataset_cam2worlds:
            mvp_np = mvp_from_cam(cam2world, H, W, dataset_focal, near=0.1, far=10.0)
            mvps.append(torch.tensor(mvp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0))

        # rasterize mesh using these mvps -> form target_imgs/masks for mesh-based training
        for mvp in mvps:
            ones = torch.ones_like(vertices[:, :, :1])
            pos_clip = torch.matmul(torch.cat([vertices, ones], dim=-1), mvp.transpose(1,2))
            rast_out, _ = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=[H, W])
            if use_vertex_colors:
                rgb_t, _ = dr.interpolate(vcol_tensor, rast_out, faces_unbatched)
                rgb_t = rgb_t[0].cpu().numpy()
            elif use_texture:
                uv_map, _ = dr.interpolate(uv_attr, rast_out, faces_unbatched)
                uv_grid = uv_map * 2.0 - 1.0
                uv_grid = uv_grid.to(dtype=torch.float32, device=texture_tensor.device)
                tex_sampled = F.grid_sample(texture_tensor, uv_grid, mode="bilinear", align_corners=True)
                rgb_t = tex_sampled.permute(0,2,3,1)[0].detach().cpu().numpy()
            else:
                rgb_t = np.ones((H, W, 3), dtype=np.float32)
            mask = (rast_out[0, ..., 3] > 0).cpu().numpy()
            # ensure background = white where mask==False
            if mask.ndim == 2:
                rgb_t[~mask] = 1.0
            else:
                #fallback
                rgb_t[mask == False] = 1.0
            target_imgs.append(rgb_t.astype(np.float32))
            target_masks.append(mask.astype(np.bool_))
        print("[INFO] Mesh rasterized for dataset poses (targets ready).")
    else:
        # no dataset: generate a few angular views and rasterize mesh for targets (fallback)
        ANGLES = [0,45,90,135,180]
        for angle in ANGLES:
            angle_rad = np.radians(angle)
            radius = 5.0; height = -2.0
            eye = np.array([np.sin(angle_rad)*radius, height, np.cos(angle_rad)*radius], dtype=np.float32)
            center = np.array([0.0,0.0,0.0], dtype=np.float32)
            up = np.array([0.0,1.0,0.0], dtype=np.float32)
            view = look_at(eye, center, up)
            proj = perspective(np.radians(45.0), float(W)/float(H), 0.1, 10.0)
            mvp = proj @ view
            mvps.append(torch.tensor(mvp, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        for mvp in mvps:
            ones = torch.ones_like(vertices[:, :, :1])
            pos_clip = torch.matmul(torch.cat([vertices, ones], dim=-1), mvp.transpose(1,2))
            rast_out, _ = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=[H,W])
            if use_vertex_colors:
                rgb_t, _ = dr.interpolate(vcol_tensor, rast_out, faces_unbatched)
                rgb_t = rgb_t[0].cpu().numpy()
            elif use_texture:
                uv_map, _ = dr.interpolate(uv_attr, rast_out, faces_unbatched)
                uv_grid = uv_map * 2.0 - 1.0
                uv_grid = uv_grid.to(dtype=torch.float32, device=texture_tensor.device)
                tex_sampled = F.grid_sample(texture_tensor, uv_grid, mode="bilinear", align_corners=True)
                rgb_t = tex_sampled.permute(0,2,3,1)[0].detach().cpu().numpy()
            else:
                rgb_t = np.zeros((H,W,3), dtype=np.float32)
            mask = (rast_out[0, ..., 3] > 0).cpu().numpy()
            target_imgs.append(rgb_t.astype(np.float32))
            target_masks.append(mask.astype(np.bool_))
        print("[INFO] Mesh rasterized for angular fallback views.")

    use_mesh = True

else:
    # No mesh: dataset-only mode -> use centered bbox
    bbox_min = np.array([-BBOX_SIZE/2.0]*3, dtype=np.float32)
    bbox_max = np.array([ BBOX_SIZE/2.0]*3, dtype=np.float32)
    bbox_min_t = torch.tensor(bbox_min, dtype=torch.float32, device=DEVICE)
    bbox_max_t = torch.tensor(bbox_max, dtype=torch.float32, device=DEVICE)
    print(f"[DEBUG] Dataset-only bbox_min: {bbox_min}, bbox_max: {bbox_max}")

    print(f"[INFO] No mesh provided. Using centered bbox size {BBOX_SIZE}. Dataset-only volumetric mode.")

# If dataset is present but we didn't rasterize mesh, prepare dataset lists for training
if use_dataset and not use_mesh:
    # dataset_target_imgs/masks already loaded earlier
    # create dummy mvps list only for bookkeeping (we will use cam2worlds instead in volumetric render)
    pass

# Sanity-check: if both mesh and dataset present render mesh from first dataset pose and compare
# if use_mesh and use_dataset:
#     try:
#         print("[SANITY] Rendering mesh from first dataset pose and comparing to GT first image...")
#         # use first dataset cam2world
#         cam2world0 = dataset_cam2worlds[0]
#         mvp_np = mvp_from_cam(cam2world0, IMG_RES, IMG_RES, dataset_focal, near=0.1, far=10.0)
#         mvp0 = torch.tensor(mvp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
#         ones = torch.ones_like(vertices[:, :, :1])
#         pos_clip = torch.matmul(torch.cat([vertices, ones], dim=-1), mvp0.transpose(1,2))
#         rast_out, _ = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=[IMG_RES, IMG_RES])
#         if 'vcol_tensor' in locals():
#             rgb_r, _ = dr.interpolate(vcol_tensor, rast_out, faces_unbatched)
#             pred_rgb = rgb_r[0].cpu().numpy()
#         elif 'texture_tensor' in locals() and 'uv_attr' in locals():
#             uv_map, _ = dr.interpolate(uv_attr, rast_out, faces_unbatched)
#             uv_grid = uv_map * 2.0 - 1.0
#             uv_grid = uv_grid.to(dtype=torch.float32, device=texture_tensor.device)
#             tex_sampled = F.grid_sample(texture_tensor, uv_grid, mode="bilinear", align_corners=True)
#             pred_rgb = tex_sampled.permute(0,2,3,1)[0].detach().cpu().numpy()
#         else:
#             pred_rgb = np.zeros((IMG_RES, IMG_RES, 3), dtype=np.float32)

#         gt_rgb = dataset_target_imgs[0]
#         mask0 = dataset_target_masks[0]
#         if mask0.sum() > 0:
#             diff = (pred_rgb - gt_rgb) * mask0[..., None].astype(np.float32)
#             mse = (diff**2).sum() / (mask0.sum() * 3.0)
#         else:
#             mse = np.inf
#         Path(OUTDIR).mkdir(parents=True, exist_ok=True)
#         plt.imsave(str(Path(OUTDIR)/"pred_mesh_first_pose.png"), np.clip(pred_rgb, 0.0, 1.0))
#         plt.imsave(str(Path(OUTDIR)/"gt_first_pose.png"), np.clip(gt_rgb, 0.0, 1.0))
#         diff_vis = np.abs(pred_rgb - gt_rgb)
#         diff_vis = diff_vis / (diff_vis.max() + 1e-12)
#         plt.imsave(str(Path(OUTDIR)/"pred_mesh_first_pose_diff.png"), diff_vis)
#         print(f"[SANITY] Saved pred/gt/diff. Masked MSE={mse:.6e}")
#     except Exception as e:
#         print("[SANITY] Exception during mesh sanity render:", e)


# EXTRA VISUALIZATION: NeRF dataset viewpoint
# if DATA_ROOT is not None:
#     try:
#         print("Extra visualization with one dataset viewpoint...")

#         # Load first camera
#         transforms_path = os.path.join(DATA_ROOT, "transforms_train.json")
#         with open(transforms_path, "r") as f:
#             meta = json.load(f)
#         frame0 = meta["frames"][0]
#         fname = os.path.join(DATA_ROOT, frame0["file_path"] + ".png")
#         if not os.path.exists(fname):
#             fname = fname.replace(".png", ".jpg")
#         gt_img = np.array(Image.open(fname).convert("RGB").resize((IMG_RES, IMG_RES)))

#         pose0 = np.array(frame0["transform_matrix"], dtype=np.float32)  # camera-to-world
#         cam_pos = pose0[:3, 3]

#         cam_angle_x = float(meta["camera_angle_x"])
#         focal_pix = 0.5 * IMG_RES / math.tan(0.5 * cam_angle_x)

#         print("focal_pix:", focal_pix)
#         print("img_res:", IMG_RES)
#         print("approx FOV deg:", 2*np.degrees(np.arctan(IMG_RES/(2*focal_pix))))

#         # Ray-Mesh Intersection
#         import trimesh
#         from trimesh.ray import ray_pyembree

#         # baue Strahlen für ein Subset von Pixeln
#         xs = np.linspace(-IMG_RES/2, IMG_RES/2, 32)
#         ys = np.linspace(-IMG_RES/2, IMG_RES/2, 32)
#         dirs = []
#         for y in ys:
#             for x in xs:
#                 d = np.array([x, -y, -focal_pix], dtype=np.float32)
#                 d = d / np.linalg.norm(d)
#                 d_world = (pose0[:3, :3] @ d)  # nach world transformieren
#                 dirs.append(d_world)
#         dirs = np.array(dirs)

#         origins = np.tile(cam_pos[None, :], (dirs.shape[0], 1))

#         mesh_for_rmi = trimesh.Trimesh(vertices=vertices[0].cpu().numpy(), faces=faces_unbatched.cpu().numpy())
#         rmi = ray_pyembree.RayMeshIntersector(mesh_for_rmi)
#         locs, index_ray, index_tri = rmi.intersects_location(origins, dirs, multiple_hits=False)

#         # --- Plot 3D: Mesh + Hits + Camera ---
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')
#         mesh_show = mesh.copy()
#         mesh_show.visual.face_colors = [100, 100, 200, 100]  # halftransparent
#         ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
#                         triangles=mesh.faces, color='blue', alpha=0.15, linewidth=0.2)
#         ax.scatter(locs[:, 0], locs[:, 1], locs[:, 2], c='r', s=5, label="ray hits")
#         ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], c='b', s=50, label="camera")
#         ax.legend()
#         plt.title("Dataset viewpoint: ray hits + mesh")
#         plt.savefig(os.path.join(OUTDIR, "dataset_viewpoint_hits.png"), dpi=200)
#         plt.close()

#         # Render Mesh
#         def perspective_from_focal(focal, W, H, near=0.1, far=10.0):
#             fovy = 2.0 * math.atan(float(H) / (2.0 * float(focal)))
#             f = 1.0 / math.tan(fovy / 2.0)
#             m = np.zeros((4, 4), dtype=np.float32)
#             m[0, 0] = f / (W / float(H))
#             m[1, 1] = -f
#             m[2, 2] = (far + near) / (near - far)
#             m[2, 3] = (2*far*near) / (near - far)
#             m[3, 2] = -1.0
#             return m

#         proj_np = perspective_from_focal(focal_pix, IMG_RES, IMG_RES)
#         view_np = np.linalg.inv(pose0).astype(np.float32)
#         mvp_np = proj_np @ view_np
#         mvp = torch.tensor(mvp_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

#         ones = torch.ones_like(vertices[:, :, :1])
#         verts_h = torch.cat([vertices, ones], dim=-1)
#         pos_clip = torch.matmul(verts_h, mvp.transpose(1, 2))
#         rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[IMG_RES, IMG_RES])
#         vcol_tensor = torch.rand(vertices.shape[1], 3, device=DEVICE)  # random vertex colors
#         rgb_r, _ = dr.interpolate(vcol_tensor, rast_out, faces_unbatched)
#         render_img = rgb_r[0].cpu().numpy()

#         # Save GT and Render
#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         axs[0].imshow(gt_img)
#         axs[0].set_title("GT Image")
#         axs[0].axis("off")
#         axs[1].imshow(np.clip(render_img, 0, 1))
#         axs[1].set_title("Rendered Mesh")
#         axs[1].axis("off")
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTDIR, "dataset_viewpoint_gt_vs_render.png"), dpi=200)
#         plt.close()

#         print("Extra visualization saved to", OUTDIR)

#     except Exception as e:
#         print("[Extra visualization] skipped due to error:", e)

# Prepare voxel grid + optimizer
voxel_grid = torch.nn.Parameter(torch.rand(1, 3, VOXEL_RES, VOXEL_RES, VOXEL_RES, device=DEVICE))
optimizer = torch.optim.Adam([voxel_grid], lr=LR)
best_loss = float('inf'); no_improve_steps = 0

# print("[PREVIEW] Rendering initial voxelgrid (random init) from first dataset view...")
# with torch.no_grad():
#     step_out = Path(OUTDIR) / "init_preview"
#     step_out.mkdir(parents=True, exist_ok=True)

#     if use_dataset and len(dataset_cam2worlds) > 0:
#         cam2world = torch.from_numpy(dataset_cam2worlds[0]).to(device=DEVICE, dtype=torch.float32)
#         pred_map = render_rays_from_voxelgrid(
#             voxel_grid, cam2world, dataset_focal, IMG_RES, IMG_RES,
#             N_samples=N_SAMPLES, near=NEAR, far=FAR, device=DEVICE
#         )
#         pred_rgb = pred_map.cpu().numpy()
#         plt.imsave(str(step_out / "init_pred.png"), np.clip(pred_rgb, 0.0, 1.0))
#         plt.imsave(str(step_out / "init_target.png"), np.clip(dataset_target_imgs[0], 0.0, 1.0))
#         print("[PREVIEW] Saved init_pred.png and init_target.png")

print("[TRAIN] Start training...")
scaler = torch.cuda.amp.GradScaler()
ACCUM_STEPS = 1

# Training loop:
for it in range(1, STEPS+1):
    t0 = time.time()
    optimizer.zero_grad()
    total_mse_sum = torch.tensor(0.0, device=DEVICE)   # sum of squared errors (over all valid scalar entries)
    total_num_entries = 0.0

    if use_mesh:
        for vi, cam2world in enumerate(dataset_cam2worlds):
            cam2world_t = torch.from_numpy(cam2world).to(device=DEVICE, dtype=torch.float32)
            with torch.cuda.amp.autocast():
                cam2world_np = cam2world_t.cpu().numpy()
                pred_map = render_via_mesh_rasterization(voxel_grid, cam2world_np, float(dataset_focal),
                                                        IMG_RES, IMG_RES, vertices, faces_unbatched, ctx,
                                                        voxel_sample_mode='bilinear', near=NEAR, far=FAR, device=DEVICE)
                # pred_map: torch tensor [H,W,3] on device
                tgt = torch.tensor(dataset_target_imgs[vi], dtype=torch.float32, device=DEVICE)

                mse_sum = ((pred_map - tgt) ** 2).sum()
                entries = float(pred_map.numel())

                total_mse_sum = total_mse_sum + mse_sum
                total_num_entries += entries

    else:
        # Dataset-only volumetric training: render rays and compare to dataset images
        for vi, cam2world in enumerate(dataset_cam2worlds):
            cam2world_t = torch.from_numpy(cam2world).to(device=DEVICE, dtype=torch.float32)
            focal = float(dataset_focal)
            with torch.cuda.amp.autocast():
                pred_map = render_rays_from_voxelgrid(voxel_grid, cam2world_t, focal, IMG_RES, IMG_RES,
                                                      N_samples=N_SAMPLES, near=NEAR, far=FAR, device=DEVICE)
                tgt = torch.tensor(dataset_target_imgs[vi], dtype=torch.float32, device=DEVICE)

                mse_sum = ((pred_map - tgt) ** 2).sum()
                entries = float(pred_map.numel())

                total_mse_sum = total_mse_sum + mse_sum
                total_num_entries += entries

    if total_num_entries == 0.0:
        print(f"[WARN] Iter {it}: total_num_entries == 0 -> skipping backward step.")
        continue

    mean_loss = total_mse_sum / float(total_num_entries)   # tensor on DEVICE

    loss = mean_loss / ACCUM_STEPS

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    total_loss = loss.item()

    tb_writer.add_scalar("Loss/train", total_loss, it)
    t1 = time.time()
    if it % 10 == 0 or it == 1:
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
        with torch.no_grad():
            step_out = Path(OUTDIR) / f"step_{it:06d}"
            step_out.mkdir(parents=True, exist_ok=True)

            # save previews: for mesh-mode use target_imgs, for dataset-mode use dataset images
            n_preview = min(50, len(dataset_target_imgs))
            for vi in range(n_preview):
                cam2world_np = dataset_cam2worlds[vi]      # numpy (4x4)
                focal_to_use = float(dataset_focal)

                pred_map = render_via_mesh_rasterization(
                    voxel_grid, cam2world_np, focal_to_use, IMG_RES, IMG_RES,
                    vertices, faces_unbatched, ctx,
                    voxel_sample_mode='bilinear', near=NEAR, far=FAR, device=DEVICE
                )  # returns torch Tensor [H,W,3] on DEVICE

                pm = pred_map.detach().cpu().numpy()
                #print(f"[PREVIEW] step={it} view={vi} pred_map min/max/mean = {pm.min():.6f}/{pm.max():.6f}/{pm.mean():.6f}")
                pred_rgb = np.clip(pm, 0.0, 1.0)
                # Overlay step-count
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
                # Target view
                plt.imsave(str(step_out / f"target_view{vi}.png"), np.clip(dataset_target_imgs[vi],0,1))

            # voxel export -> point cloud in world coords (like before)
            vg = voxel_grid.detach().cpu().numpy()[0]  # [3,D,H,W]
            np.save(str(step_out / "voxel_grid.npy"), vg)
            occ_values = np.linalg.norm(vg, axis=0)  # [D,H,W]
            maxval = float(occ_values.max()) if occ_values.size and occ_values.max() > 0 else 1.0
            occ_norm = occ_values / (maxval + 1e-12)
            occ = occ_norm > 0.2
            dz, hy, wx = np.nonzero(occ)
            num_pts = len(dz)
            print(f"Found {num_pts} occupied voxels (threshold=0.2 after normalizing by max={maxval:.6g}).")

            if num_pts > 0:
                ix = wx.astype(np.float32); iy = hy.astype(np.float32); iz = dz.astype(np.float32)
                cx = (ix + 0.5) / float(VOXEL_RES); cy = (iy + 0.5) / float(VOXEL_RES); cz = (iz + 0.5) / float(VOXEL_RES)
                # world bbox depends on mesh or dataset bbox
                if use_mesh:
                    bbox_min_np = np.array(bbox_min, dtype=np.float32); bbox_max_np = np.array(bbox_max, dtype=np.float32)
                else:
                    bbox_min_np = np.array([-BBOX_SIZE/2.0]*3, dtype=np.float32)
                    bbox_max_np = np.array([ BBOX_SIZE/2.0]*3, dtype=np.float32)
                world_x = bbox_min_np[0] + cx * (bbox_max_np[0] - bbox_min_np[0])
                world_y = bbox_min_np[1] + cy * (bbox_max_np[1] - bbox_min_np[1])
                world_z = bbox_min_np[2] + cz * (bbox_max_np[2] - bbox_min_np[2])
                points = np.stack([world_x, world_y, world_z], axis=1)
                cols = vg[:, dz, hy, wx].T
                cols = np.clip(cols, 0.0, 1.0)
                cols_u8 = (cols * 255.0).astype(np.uint8)

                MAX_POINTS = 500000
                if points.shape[0] > MAX_POINTS:
                    idxs = np.random.choice(points.shape[0], size=MAX_POINTS, replace=False)
                    points = points[idxs]; cols_u8 = cols_u8[idxs]
                    print(f"Downsampled voxel cloud to {MAX_POINTS} points for export.")

                ply_path = step_out / "voxel_cloud.ply"
                with open(ply_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(points)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write("end_header\n")
                    for (x, y, z), (r, g, b) in zip(points, cols_u8):
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
                print(f"Saved voxel cloud ({len(points)} points) -> {ply_path}")

            print(f"Saved checkpoints to {step_out}")

tb_writer.close()
print("Training finished. Results in:", OUTDIR)

# Novel View Synthesis
print("[NVS] Rendering novel views...")

novel_out = Path(OUTDIR) / "novel_views"
novel_out.mkdir(parents=True, exist_ok=True)

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))

with torch.no_grad():
    # If dataset poses are available, use them as novel views
    if use_dataset:
        n_views = len(dataset_cam2worlds)
        print(f"[NVS] Using {n_views} dataset poses for novel view synthesis.")
        for idx, cam2world in enumerate(dataset_cam2worlds):
            cam2world_np = cam2world
            if use_mesh:
                pred_rgb = render_via_mesh_rasterization(
                    voxel_grid, cam2world_np, float(dataset_focal), IMG_RES, IMG_RES,
                    vertices, faces_unbatched, ctx, voxel_sample_mode='bilinear',
                    near=NEAR, far=FAR, device=DEVICE
                ).cpu().numpy()
            else:
                # dataset-only volumetric rendering (use dataset_focal)
                focal = float(dataset_focal)
                cam2world_t = torch.from_numpy(cam2world).to(device=DEVICE, dtype=torch.float32)
                pred_map = render_rays_from_voxelgrid(voxel_grid, cam2world_t, focal, IMG_RES, IMG_RES,
                                                      N_samples=N_SAMPLES, near=NEAR, far=FAR, device=DEVICE)
                pred_rgb = pred_map.cpu().numpy()

            # save predicted image
            out_path = novel_out / f"novel_dataset_idx{idx:04d}.png"
            plt.imsave(str(out_path), np.clip(pred_rgb, 0.0, 1.0))

            # if GT exists for this pose, compare and save PSNR + side-by-side
            gt_img = None
            if len(dataset_target_imgs) > idx:
                gt_img = dataset_target_imgs[idx]  # numpy H,W,3
            elif len(dataset_target_imgs) > 0:
                # fallback: match modulo if counts differ
                gt_img = dataset_target_imgs[idx % len(dataset_target_imgs)]

            if gt_img is not None:
                psnr_val = compute_psnr(pred_rgb, gt_img)
                fig, axs = plt.subplots(1, 2, figsize=(8,4))
                axs[0].imshow(np.clip(gt_img,0,1))
                axs[0].set_title("Ground Truth")
                axs[0].axis("off")
                axs[1].imshow(np.clip(pred_rgb,0,1))
                axs[1].set_title(f"Novel View\nPSNR={psnr_val:.2f} dB")
                axs[1].axis("off")
                plt.tight_layout()
                plt.savefig(novel_out / f"compare_dataset_idx{idx:04d}.png")
                plt.close()

            #print(f"[NVS] Saved dataset-based novel view idx={idx} -> {out_path}")
    else:
        # No dataset poses: fall back to angular trajectory (as before)
        novel_angles = [0, 60, 120, 180, 240, 300]
        print("[NVS] No dataset poses found — falling back to angular novel views.")
        for angle in novel_angles:
            if use_mesh:
                mvp, _ = get_mvp_matrix(angle, IMG_RES, IMG_RES, DEVICE)
                ones = torch.ones_like(vertices[:, :, :1])
                vertices_h = torch.cat([vertices, ones], dim=-1)
                pos_clip = torch.matmul(vertices_h, mvp.transpose(1, 2))
                rast_out, _ = dr.rasterize(ctx, pos_clip.float(), faces_unbatched, resolution=[IMG_RES, IMG_RES])
                pos_map, _ = dr.interpolate(vertices[0].float(), rast_out, faces_unbatched)
                pos_map = pos_map[0]  # [H,W,3]

                grid = world_to_grid_coords(pos_map).unsqueeze(0).unsqueeze(0)
                sampled = F.grid_sample(voxel_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
                pred_rgb = sampled.squeeze(2).permute(0,2,3,1)[0].cpu().numpy()
            else:
                _, eye = get_mvp_matrix(angle, IMG_RES, IMG_RES, DEVICE)
                cam2world = look_at(eye, np.array([0,0,0],dtype=np.float32), np.array([0,1,0],dtype=np.float32))
                cam2world_t = torch.from_numpy(cam2world).to(device=DEVICE, dtype=torch.float32)
                pred_map = render_rays_from_voxelgrid(voxel_grid, cam2world_t, dataset_focal if dataset_focal is not None else (0.5*IMG_RES/np.tan(0.5*math.radians(45.0))), IMG_RES, IMG_RES,
                                                      N_samples=N_SAMPLES, near=NEAR, far=FAR, device=DEVICE)
                pred_rgb = pred_map.cpu().numpy()

            out_path = novel_out / f"novel_{angle:03d}.png"
            plt.imsave(str(out_path), np.clip(pred_rgb, 0.0, 1.0))
            print(f"[NVS] Saved novel view {angle}° -> {out_path}")

print("[NVS] Done.")
