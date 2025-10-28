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
parser.add_argument("--mesh", type=str, default=None, help="(optional) Path to GT mesh (OBJ/PLY).")
parser.add_argument("--data_root", type=str, default=None, help="Path to NeRF-Synthetic scene folder (transforms_*.json + images).")
parser.add_argument("--outdir", type=str, default="train_out", help="Output directory")
parser.add_argument("--voxel_res", type=int, default=256, help="Voxel grid resolution (D,H,W)")
parser.add_argument("--img_res", type=int, default=800, help="Image resolution (H and W)")
parser.add_argument("--steps", type=int, default=1000, help="Training iterations")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--save_interval", type=int, default=100, help="Save image/checkpoint every N steps")
parser.add_argument("--early_stop_patience", type=int, default=50, help="Stop training if no improvement in N steps")
parser.add_argument("--early_stop_delta", type=float, default=1e-6, help="Minimum change to consider an improvement")
parser.add_argument("--bbox_size", type=float, default=2.0, help="Scene bounding box (cube side length) used when no mesh provided")
parser.add_argument("--n_samples", type=int, default=64, help="Samples per ray for volumetric rendering")
parser.add_argument("--near", type=float, default=0.1, help="Near plane for ray sampling")
parser.add_argument("--far", type=float, default=6.0, help="Far plane for ray sampling")
parser.add_argument("--img_count", type=int, default=100, help="Training image count")
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

os.makedirs(OUTDIR, exist_ok=True)
tb_writer = SummaryWriter(log_dir=os.path.join(OUTDIR, "tensorboard"))

print(f"[INFO] Device: {DEVICE}")

# ---------------- helpers ----------------
def look_at(eye, center, up):
    f = center - eye    # View direction
    f = f / np.linalg.norm(f)   # Normalization
    s = np.cross(f, up) # Right-vector (Cross product)
    s = s / np.linalg.norm(s)   # Normalization
    u = np.cross(s, f)  # Recompute up-vector
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s    # x-axis
    m[1, :3] = u    # y-axis
    m[2, :3] = -f   # z-axis
    m[:3, 3] = -m[:3, :3] @ eye # Translation (4x4 view-matrix)
    return m

def perspective(fovy, aspect, near, far):
    #print(f"fovy (rad): {fovy:.4f} ({np.degrees(fovy):.2f}°)")
    #print(f"aspect: {aspect:.4f}, near: {near}, far: {far}")
    f = 1.0 / np.tan(fovy / 2.0)    # Scaling factor for FoV
    #print(f"scaling factor f: {f:.6f}")
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = -f    # y-flip
    m[2, 2] = (far + near) / (near - far)
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

# ---------------- Rays + volumetric renderer (dataset-only) ----------------
def get_rays_from_pose(cam2world, focal, H, W, device):
    # Make sure cam2world is a float tensor on device (Convert if not)
    if isinstance(cam2world, np.ndarray):
        cam2world = torch.from_numpy(cam2world).to(device=device, dtype=torch.float32)
    else:
        cam2world = cam2world.to(device=device, dtype=torch.float32)

    # Create arrays for pixels, sampling pixel center
    i = (torch.arange(0, W, device=device).float() + 0.5)
    j = (torch.arange(0, H, device=device).float() + 0.5)
    # Create (H,W)-grid with pixel arrays
    try:
        px, py = torch.meshgrid(i, j, indexing='xy')
    except TypeError:
        px, py = torch.meshgrid(i, j)
        px = px.t(); py = py.t()

    cx = W * 0.5    # Center coordinate
    cy = H * 0.5    # Center coordinate
    # Cam coordinates of pixel ray
    x_cam = (px - cx) / focal   # Project pixel in image plane
    y_cam = -(py - cy) / focal  # Project pixel in image plane
    z_cam = torch.ones_like(x_cam)  # Set image plane at z=1 in camera coordinates

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # Create 3D direction vector for every pixel [H,W,3]
    R = cam2world[:3, :3]   # Rotation matrix (camera->world rotation)
    t = cam2world[:3, 3]    # Translation vector (camera position -> world coordinate)

    dirs_world = dirs_cam.reshape(-1, 3) @ R.t()    # Flatten and transform to world coordinates
    dirs_world = dirs_world.reshape(H, W, 3)
    dirs_world = dirs_world / (torch.norm(dirs_world, dim=-1, keepdim=True) + 1e-12)    # Normalization of directions
    origin = t.view(1,1,3).expand(H, W, 3)  # Camera position expanded to [H,W,3] tensor
    return origin, dirs_world

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

def render_rays_from_voxelgrid(voxel_grid, cam2world, focal, H, W, N_samples=64, near=0.1, far=6.0, device='cuda'):
    origin, dirs = get_rays_from_pose(cam2world, focal, H, W, device)   # Create ray origins and directions
    t_vals = torch.linspace(near, far, steps=N_samples, device=device)  # [N] samples along ray
    pts = origin.unsqueeze(0) + t_vals.view(N_samples,1,1,1) * dirs.unsqueeze(0)  # [N,H,W,3]
    grid_coords = world_to_grid_coords(pts).to(dtype=torch.float32, device=device)  # convert to grid coords [-1,1]
    grid = grid_coords.unsqueeze(0)  # [1,N,H,W,3]

    # Sample/query voxel grid (5D sampling): voxel_grid [1,C,D,H,W], grid [1,D_out,H_out,W_out,3]
    sampled = F.grid_sample(voxel_grid, grid, mode='bilinear', padding_mode='border', align_corners=True)
    sampled = sampled.permute(0, 2, 3, 4, 1) # [1, C, N, H, W] to [1,N,H,W,C]
    colors = sampled[..., :3]  # RGB along ray [1,N,H,W,3]
    sigma = torch.norm(colors, dim=-1)  # density as norm of color [1,N,H,W]

    delta = (far - near) / float(N_samples) # distance between samples
    alpha = 1.0 - torch.exp(-F.relu(sigma) * delta)  # convert to opacity [1,N,H,W]

    one_m_alpha = (1.0 - alpha + 1e-10) # calculate cumulative transmissionrate of ray (how much light still gets through)
    T = torch.cumprod(torch.cat([torch.ones_like(one_m_alpha[:, :1, ...]), one_m_alpha[:, :-1, ...]], dim=1), dim=1)
    weights = alpha * T  # calculate contribution to color of samples [1,N,H,W]

    rgb_map = (weights.unsqueeze(-1) * colors).sum(dim=1)[0]  # mult colors with weights [H,W,3]
    trans = 1.0 - weights.sum(dim=1)[0]                       # [H,W]
    rgb_map = rgb_map + trans.unsqueeze(-1) * 1.0             # add white background
    return rgb_map

def render_via_mesh_rasterization(voxel_grid_param, cam2world_np, focal, H, W,
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
    pred_rgb[~mask] = 1.0

    # if it % SAVE_INTERVAL == 0 or it == 1:

    #     verts = verts_h[0, :, :3] / verts_h[0, :, 3:4]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(verts[:, 0].cpu(), verts[:, 1].cpu(), verts[:, 2].cpu(), s=0.2)
    #     cam_pos_w = cam2world_np[:3, 3]
    #     R_w = cam2world_np[:3, :3]  # columns as axes in world coordinates
    #     axis_len = 0.2 * float(np.linalg.norm(bbox_max_t.cpu().numpy() - bbox_min_t.cpu().numpy())) if (bbox_min_t is not None and bbox_max_t is not None) else 0.5

    #     # axe endpoints
    #     x_end = cam_pos_w + axis_len * R_w[:, 0]
    #     y_end = cam_pos_w + axis_len * R_w[:, 1]
    #     z_end = cam_pos_w + axis_len * R_w[:, 2]

    #     # camera position
    #     ax.scatter([cam_pos_w[0]], [cam_pos_w[1]], [cam_pos_w[2]], c='k', s=30, label='camera')

    #     # axes as lines
    #     ax.plot([cam_pos_w[0], x_end[0]], [cam_pos_w[1], x_end[1]], [cam_pos_w[2], x_end[2]], c='r', linewidth=2, label='cam X')
    #     ax.plot([cam_pos_w[0], y_end[0]], [cam_pos_w[1], y_end[1]], [cam_pos_w[2], y_end[2]], c='g', linewidth=2, label='cam Y')
    #     ax.plot([cam_pos_w[0], z_end[0]], [cam_pos_w[1], z_end[1]], [cam_pos_w[2], z_end[2]], c='b', linewidth=2, label='cam Z')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_box_aspect([1, 1, 1])
    #     ax.set_proj_type('ortho')
    #     ax.legend(loc='upper right')
    #     plt.title("verts_h + camera")
    #     plt.savefig(os.path.join(OUTDIR, "verts_h.png"), dpi=150)
    #     plt.show()


    #     pos_clip_3d = pos_clip[0, :, :3] / pos_clip[0, :, 3:4]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     pts = pos_clip_3d.detach().cpu()
    #     ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.2, alpha=0.5)
    #     ax.scatter([0], [0], [0], c='k', s=30, label='camera (origin)')
    #     near_z = near
    #     far_z = far
    #     cube_edges = [
    #         # near face (z = near)
    #         [(-1,-1,near_z), (1,-1,near_z)],
    #         [(1,-1,near_z), (1,1,near_z)],
    #         [(1,1,near_z), (-1,1,near_z)],
    #         [(-1,1,near_z), (-1,-1,near_z)],
    #         # far face (z = far)
    #         [(-1,-1,far_z), (1,-1,far_z)],
    #         [(1,-1,far_z), (1,1,far_z)],
    #         [(1,1,far_z), (-1,1,far_z)],
    #         [(-1,1,far_z), (-1,-1,far_z)],
    #         # connecting edges
    #         [(-1,-1,near_z), (-1,-1,far_z)],
    #         [(1,-1,near_z), (1,-1,far_z)],
    #         [(1,1,near_z), (1,1,far_z)],
    #         [(-1,1,near_z), (-1,1,far_z)],
    #     ]

    #     for e in cube_edges:
    #         ax.plot3D(*zip(*e), color='gray', linewidth=0.8, alpha=0.6)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_xlim([-1,1])
    #     ax.set_ylim([-1,1])
    #     ax.set_zlim([-1,1])
    #     ax.set_box_aspect([1, 1, 1])
    #     ax.set_proj_type('ortho')
    #     ax.legend(loc='upper right')
    #     plt.title("pos_clip + camera")
    #     plt.savefig(os.path.join(OUTDIR, "pos_clip.png"), dpi=150)
    #     plt.show()

        # rast = rast_out[0].cpu().numpy()
        # mask = rast[..., 3] > 0  # Pixel hit by a face
        # plt.figure(figsize=(5,5))
        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')
        # plt.title("Rasterization Mask")
        # plt.savefig(os.path.join(OUTDIR, "rast_out_mask.png"), dpi=150)
        # plt.show()

        # face_idx_map = rast[..., 0]  # Face-Index
        # plt.figure(figsize=(5,5))
        # plt.imshow(face_idx_map, cmap='viridis')
        # plt.axis('off')
        # plt.title("Face Index Map")
        # plt.savefig(os.path.join(OUTDIR, "rast_out_face.png"), dpi=150)
        # plt.show()

        # u = rast_out[0, ..., 1].cpu().numpy()
        # v = rast_out[0, ..., 2].cpu().numpy()
        # w = 1 - u - v
        # plt.subplot(1,2,1); plt.imshow(u, cmap='magma'); plt.title('bary u'); plt.axis('off')
        # plt.subplot(1,2,2); plt.imshow(v, cmap='magma'); plt.title('bary v'); plt.axis('off'); plt.savefig(os.path.join(OUTDIR, "bary_v.png"), dpi=150)
        # plt.imshow(w, cmap='magma'); plt.title("bary w"); plt.show(); plt.savefig(os.path.join(OUTDIR, "bary_w.png"), dpi=150)
        # plt.show()

        # pred_np = pos_map.detach().cpu().numpy()
        # plt.figure(figsize=(5,5))
        # plt.imshow(pos_map[..., 1].detach().cpu().numpy())
        # plt.axis('off')
        # plt.title("pos_map")
        # plt.savefig(os.path.join(OUTDIR, "pos_map.png"), dpi=150)
        # plt.show()

        # verts = grid_coords.reshape(-1, 3)  # [H*W, 3]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(verts[:,0].cpu(), verts[:,1].cpu(), verts[:,2].cpu(), s=0.2)
        # cam_pos_w = torch.tensor(cam2world_np[:3, 3], dtype=torch.float32, device=DEVICE)
        # cam_pos_grid = world_to_grid_coords(cam_pos_w)
        # cpg = cam_pos_grid.detach().cpu().numpy()
        # ax.scatter([cpg[0]], [cpg[1]], [cpg[2]], c='k', s=30, label='camera')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_box_aspect([1,1,1])
        # ax.set_proj_type('ortho')
        # ax.legend(loc='upper right')
        # plt.title("grid_coords + camera")
        # plt.savefig(os.path.join(OUTDIR, "grid_coords.png"), dpi=150)
        # plt.show()

        # rgb = sampled[..., :3]
        # plt.figure(figsize=(5,5))
        # plt.imshow(rgb.detach().cpu().numpy(), cmap='gray')
        # plt.axis('off')
        # plt.title("pred_rgb")
        # plt.savefig(os.path.join(OUTDIR, "pred_rgb.png"), dpi=150)
        # plt.show()

    return pred_rgb

# ---------------- Prepare scene (mesh or bbox) & load dataset if requested ----------------
use_dataset = False
dataset_target_imgs = []
dataset_target_masks = []
dataset_cam2worlds = []
dataset_focal = None

if DATA_ROOT is not None:
    # Open and validate transforms_train.json
    json_path = os.path.join(DATA_ROOT, "transforms_train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset transforms_train.json not found at {json_path}")
    meta = json.load(open(json_path, "r"))
    frames = meta.get("frames", []) # Load frames into list
    if len(frames) == 0:
        raise ValueError("transforms_train.json contains 0 frames")
    use_dataset = True
    dataset_focal = 0.5 * IMG_RES / np.tan(0.5 * float(meta["camera_angle_x"])) # Calculate FoV
    print(f"[INFO] Dataset found: {len(frames)} frames, focal(derived) = {dataset_focal:.3f}")

    for fr in frames:
        cam2world = np.array(fr["transform_matrix"], dtype=np.float32)  # Read transform_matrix
        dataset_cam2worlds.append(cam2world)    # Append camera->world matrices to list
        # Load image
        img_path = os.path.join(DATA_ROOT, fr["file_path"] + ".png")
        if not os.path.exists(img_path):
            alt = img_path.replace(".png", ".jpg")
            if os.path.exists(alt):
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")
        pil = Image.open(img_path).convert("RGBA")  # Open image and convert to RGBA
        if pil.size != (IMG_RES, IMG_RES):
            pil = pil.resize((IMG_RES, IMG_RES), resample=Image.LANCZOS)    # Resize image
        arr = np.array(pil).astype(np.float32)  # Save in array [IMG_RES,IMG_RES,4]

        alpha = arr[..., 3] / 255.0 # Alpha normalized to [0,1]

        # Premultiply: RGB * alpha
        a = alpha[..., None].astype(np.float32)            # Alpha in [0,1], [H,W,1]
        rgb = (arr[..., :3] / 255.0).astype(np.float32) * a + (1.0 - a) # alpha-blending (Background set to white if a = 0)

        mask = alpha.astype(np.float32) # Alpha mask in [0,1]
        dataset_target_imgs.append(rgb.astype(np.float32))  # Save images to list
        dataset_target_masks.append(mask.astype(np.float32))    # Save masks to list

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
use_mesh = False

if MESH_PATH is not None:
    import trimesh
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
    
    #print("Watertight:", mesh.is_watertight)
    #print("Consistent winding:", mesh.is_winding_consistent)

    mesh_vertices = mesh.vertices.astype(np.float32)    # Convert to float32
    mesh.vertices = mesh_vertices

    r = np.array([[1, 0, 0],
              [0, 0,-1],
              [0, 1, 0]])
    mesh.vertices = mesh.vertices @ r.T # Rotate coordinates with rotation matrix

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

    # prepare visuals: vertex colors or texture or fallback
    use_vertex_colors = False
    use_texture = False
    texture_tensor = None
    uv_attr = None

    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and mesh.visual.vertex_colors.shape[1] >= 3:
        unique_colors = np.unique(mesh.visual.vertex_colors[:, :3], axis=0) # Check for different colors
        if len(unique_colors) > 1:
            use_vertex_colors = True
            vcol = mesh.visual.vertex_colors[:, :3] / 255.0
            vcol = np.clip(vcol, 0.0, 1.0).astype(np.float32)   # Norm to [0,1]
            vcol_tensor = torch.tensor(vcol, dtype=torch.float32, device=DEVICE)    # Convert to tensor
            print("[INFO] Using vertex colors on mesh.")

    tex_obj = getattr(mesh.visual.material, "image", None) if hasattr(mesh.visual, "material") else None
    if (not use_vertex_colors) and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and tex_obj is not None:
        # load texture robustly
        def load_image_obj(obj):
            if isinstance(obj, str):
                arr = imageio.imread(obj)   # Image path
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                return arr[..., :3].astype(np.uint8)
            if isinstance(obj, Image.Image):
                arr = np.array(obj.convert("RGB"))  # Pillow-Image
                return arr[..., :3].astype(np.uint8)
            if isinstance(obj, np.ndarray):
                arr = obj   # Array
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                return arr[..., :3].astype(np.uint8)
            if hasattr(obj, "read"):
                arr = imageio.imread(obj)   # File-like object
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                return arr[..., :3].astype(np.uint8)
            arr = imageio.imread(obj)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            return arr[..., :3].astype(np.uint8)
        try:
            tex_np = load_image_obj(tex_obj)    # Load texture with function
            texture_tensor = torch.tensor(tex_np, dtype=torch.float32, device=DEVICE) / 255.0   # Norm
            texture_tensor = texture_tensor.permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
            uv_attr = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=DEVICE)  # Save UV coordinates
            use_texture = True  
            print("[INFO] Mesh texture loaded from material.image")
        except Exception as e:
            print("[WARN] Could not load mesh texture:", e)

    if (not use_vertex_colors) and (not use_texture):   # Position based fallback
        v_world = mesh.vertices.astype(np.float32)
        bbox_min_local = v_world.min(axis=0)
        bbox_max_local = v_world.max(axis=0)
        vcol = (v_world - bbox_min_local) / (bbox_max_local - bbox_min_local + 1e-8)
        vcol = np.clip(vcol, 0.0, 1.0)
        vcol_tensor = torch.tensor(vcol, dtype=torch.float32, device=DEVICE)
        use_vertex_colors = True
        print("[INFO] Using fallback position-based vertex colors for mesh.")

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

    #print("vertices shape:", vertices.shape)        # [1, V, 3]
    #print("faces shape:", faces.shape)              # [1, F, 3]
    #print("faces_unbatched shape:", faces_unbatched.shape)  # [F, 3]

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

    use_mesh = True

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

ctx = dr.RasterizeCudaContext() # Create nvdiffrast CUDA-Rasterizer-Context
H = IMG_RES; W = IMG_RES    # Set resolution

# Prepare voxel grid + optimizer + loss
voxel_grid = torch.nn.Parameter(torch.rand(1, 3, VOXEL_RES, VOXEL_RES, VOXEL_RES, device=DEVICE))
optimizer = torch.optim.Adam([voxel_grid], lr=LR)
best_loss = float('inf'); no_improve_steps = 0

# vg = voxel_grid.detach()
# print("voxel_grid shape:", vg.shape)   # [1, C, D, H, W]
# print("device:", vg.device, "dtype:", vg.dtype)
# print("min/max/mean/std:", vg.min().item(), vg.max().item(), vg.mean().item(), vg.std().item())


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
scaler = torch.cuda.amp.GradScaler()    # Initialize GradScaler for AMP

# Training loop:
for it in range(1, STEPS+1):
    t0 = time.time()
    optimizer.zero_grad()   # Reset gradients before backprop
    total_mse_sum = torch.tensor(0.0, device=DEVICE)   # sum of squared errors (over all valid scalar entries)
    total_num_entries = 0.0 # counter 

    if use_dataset:
        for vi, cam2world in enumerate(dataset_cam2worlds): # iterate over all dataset poses
            cam2world_t = torch.from_numpy(cam2world).to(device=DEVICE, dtype=torch.float32)
            with torch.cuda.amp.autocast(): # AMP for memory saving
                cam2world_np = cam2world_t.cpu().numpy()
                pred_map = render_via_mesh_rasterization(voxel_grid, cam2world_np, float(dataset_focal),
                                                        IMG_RES, IMG_RES, vertices, faces_unbatched, ctx,
                                                        voxel_sample_mode='bilinear', near=NEAR, far=FAR, device=DEVICE)
                # pred_map: torch tensor [H,W,3] on device
                tgt = torch.tensor(dataset_target_imgs[vi], dtype=torch.float32, device=DEVICE) # target image

                mse_sum = ((pred_map - tgt) ** 2).sum() # calculate MSE
                entries = float(pred_map.numel())   # H * W * 3

                total_mse_sum = total_mse_sum + mse_sum # add mse to total
                total_num_entries += entries    # add entries to total

    else:
        # Used when only mesh is given
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

    if total_num_entries == 0.0:    # Skip iteration if no valid pixels
        print(f"[WARN] Iter {it}: total_num_entries == 0 -> skipping backward step.")
        continue

    mean_loss = total_mse_sum / float(total_num_entries)   # tensor on DEVICE

    scaler.scale(mean_loss).backward()   # backpropagation
    scaler.step(optimizer)  # parameter update
    scaler.update() # update AMP scaling
    #optimizer.zero_grad()   # Reset gradients

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
                pred_map = render_via_mesh_rasterization(
                    voxel_grid, cam2world_np, focal_to_use, IMG_RES, IMG_RES,
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
                plt.imsave(str(step_out / f"target_view{vi}.png"), np.clip(dataset_target_imgs[vi],0,1))

            # voxel export -> point cloud in world coords
            vg = voxel_grid.detach().cpu().numpy()[0]  # [3,D,H,W]
            np.save(str(step_out / "voxel_grid.npy"), vg)   # save voxelgrid as .npy
            occ_values = np.linalg.norm(vg, axis=0)  # calculate occupancy [D,H,W]

            from scipy.ndimage import gaussian_filter, label    # occupancy filtering
            occ_smooth = gaussian_filter(occ_values, sigma=1.0) # remove noise
            thr = np.percentile(occ_smooth, 95) # 95 percentile of voxels
            occ = occ_smooth >= thr # Mask
            lab, nlab = label(occ)  # Check for connected regions
            if nlab > 0:
                sizes = np.bincount(lab.ravel()); sizes[0] = 0  # count voxels, ignore background
                occ = lab == sizes.argmax() # return biggest component

            dz, hy, wx = np.nonzero(occ)    # indices of voxels
            num_pts = len(dz)
            print(f"Found {num_pts} occupied voxels")

            if num_pts > 0:
                ix = wx.astype(np.float32); iy = hy.astype(np.float32); iz = dz.astype(np.float32)
                cx = (ix + 0.5) / float(VOXEL_RES); cy = (iy + 0.5) / float(VOXEL_RES); cz = (iz + 0.5) / float(VOXEL_RES)
                # set bbox
                if use_mesh:
                    bbox_min_np = np.array(bbox_min, dtype=np.float32); bbox_max_np = np.array(bbox_max, dtype=np.float32)
                else:
                    bbox_min_np = np.array([-BBOX_SIZE/2.0]*3, dtype=np.float32)
                    bbox_max_np = np.array([ BBOX_SIZE/2.0]*3, dtype=np.float32)
                world_x = bbox_min_np[0] + cx * (bbox_max_np[0] - bbox_min_np[0])   # calculate world coordinates
                world_y = bbox_min_np[1] + cy * (bbox_max_np[1] - bbox_min_np[1])
                world_z = bbox_min_np[2] + cz * (bbox_max_np[2] - bbox_min_np[2])
                points = np.stack([world_x, world_y, world_z], axis=1)  # [num_pts, 3]
                cols = vg[:, dz, hy, wx].T  # extract colors
                cols = np.clip(cols, 0.0, 1.0)  # [0, 1]
                cols_u8 = (cols * 255.0).astype(np.uint8)   # convert to uint8

                MAX_POINTS = 500000 # downsample if too many points
                if points.shape[0] > MAX_POINTS:
                    idxs = np.random.choice(points.shape[0], size=MAX_POINTS, replace=False)
                    points = points[idxs]; cols_u8 = cols_u8[idxs]
                    print(f"Downsampled voxel cloud to {MAX_POINTS} points for export.")

                ply_path = step_out / "voxel_cloud.ply" # write ASCII .ply
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

tb_writer.close()   # Close tensorboard
print("Training finished. Results in:", OUTDIR)

# Novel View Synthesis
print("[NVS] Rendering novel views...")

novel_out = Path(OUTDIR) / "novel_views"    # Directory for novel views
novel_out.mkdir(parents=True, exist_ok=True)
dataset_cam2worlds_test = []
dataset_target_imgs_test = []
dataset_target_masks_test = []

if DATA_ROOT is not None:
    # Open and validate transforms_test.json
    json_path = os.path.join(DATA_ROOT, "transforms_test.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset transforms_test.json not found at {json_path}")
    meta = json.load(open(json_path, "r"))
    frames_test = meta.get("frames", []) # Load frames into list
    if len(frames_test) == 0:
        raise ValueError("transforms_test.json contains 0 frames")
    use_dataset = True
    dataset_focal = 0.5 * IMG_RES / np.tan(0.5 * float(meta["camera_angle_x"])) # Calculate FoV
    print(f"[INFO] Test set found: {len(frames_test)} frames, focal(derived) = {dataset_focal:.3f}")

    for fr in frames_test:
        cam2world = np.array(fr["transform_matrix"], dtype=np.float32)  # Read transform_matrix
        dataset_cam2worlds_test.append(cam2world)    # Append camera->world matrices to list
        # Load image
        img_path = os.path.join(DATA_ROOT, fr["file_path"] + ".png")
        if not os.path.exists(img_path):
            alt = img_path.replace(".png", ".jpg")
            if os.path.exists(alt):
                img_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")
        pil = Image.open(img_path).convert("RGBA")  # Open image and convert to RGBA
        if pil.size != (IMG_RES, IMG_RES):
            pil = pil.resize((IMG_RES, IMG_RES), resample=Image.LANCZOS)    # Resize image
        arr = np.array(pil).astype(np.float32)  # Save in array [IMG_RES,IMG_RES,4]

        alpha = arr[..., 3] / 255.0 # Alpha normalized to [0,1]

        # Premultiply: RGB * alpha
        a = alpha[..., None].astype(np.float32)            # Alpha in [0,1], [H,W,1]
        rgb = (arr[..., :3] / 255.0).astype(np.float32) * a + (1.0 - a) # alpha-blending

        mask = alpha.astype(np.float32) # Alpha mask in [0,1]
        dataset_target_imgs_test.append(rgb.astype(np.float32))  # Save images to list
        dataset_target_masks_test.append(mask.astype(np.float32))    # Save masks to list


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


def compute_psnr(img1, img2):   # calculate PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))

with torch.no_grad():   # Deactivate Autograd
    # If dataset poses are available, use them as novel views
    if use_dataset:
        n_views = min(TEST_IMG_COUNT, len(dataset_cam2worlds_test))   # Use dataset camera poses
        print(f"[NVS] Using {n_views} dataset poses for novel view synthesis.")
        for idx, cam2world in enumerate(dataset_cam2worlds_test[:n_views]):
            cam2world_np = cam2world
            if use_mesh:    # render predicted RGB for pose
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
            if len(dataset_target_imgs_test) > idx:
                gt_img = dataset_target_imgs_test[idx]  # numpy H,W,3
            elif len(dataset_target_imgs_test) > 0:
                # fallback: match modulo if counts differ
                gt_img = dataset_target_imgs_test[idx % len(dataset_target_imgs_test)]

            # Calculate SSIM
            from skimage.metrics import structural_similarity as ssim
            ssim_val = ssim((pred_rgb*255).astype(np.uint8), (gt_img*255).astype(np.uint8), multichannel=True)

            if gt_img is not None:
                psnr_val = compute_psnr(pred_rgb, gt_img)
                fig, axs = plt.subplots(1, 2, figsize=(16,8), constrained_layout=True)
                axs[0].imshow(np.clip(gt_img,0,1))
                axs[0].set_title("Ground Truth")
                axs[0].axis("off")
                axs[1].imshow(np.clip(pred_rgb,0,1))
                axs[1].set_title(f"Novel View\nPSNR={psnr_val:.2f} dB\nSSIM={ssim_val:.2f}")
                axs[1].axis("off")
                plt.savefig(novel_out / f"compare_dataset_idx{idx:04d}.png")
                plt.close()
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
