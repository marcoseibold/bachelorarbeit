import os
import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
import argparse

# Parsing
parser = argparse.ArgumentParser(description="Render mesh")
parser.add_argument("--mesh", type=str, required=True, help="Path to mesh")
parser.add_argument("--outdir", type=str, default="renders", help="Output directory for rendered views")
args = parser.parse_args()

RESOLUTION = (512, 512)
DEVICE = 'cuda'
MESH_PATH = args.mesh
OUTDIR = args.outdir

# Load and adjust mesh
mesh = trimesh.load(MESH_PATH, process=True)
if mesh.faces.shape[1] != 3:
    mesh = mesh.triangulate()

mesh.vertices -= mesh.vertices.mean(axis=0)
mesh.vertices /= np.max(np.linalg.norm(mesh.vertices, axis=1))
mesh.vertices *= 2.0  # Scale
mesh.vertices[:, 1] *= -1  # Flip

# Create vertices and faces
vertices_np = mesh.vertices.astype(np.float32)
faces_np = mesh.faces.astype(np.int32)

vertices = torch.tensor(vertices_np, device=DEVICE).unsqueeze(0)  # [1, V, 3]
faces = torch.tensor(faces_np, device=DEVICE).unsqueeze(0)        # [1, F, 3]

print("Vertices:", mesh.vertices.shape)
print("Faces:", mesh.faces.shape)

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

def get_mvp_matrix(angle_deg):
    angle_rad = np.radians(angle_deg)
    radius = 5.0
    height = 1.0
    eye = np.array([np.sin(angle_rad) * radius, height, np.cos(angle_rad) * radius], dtype=np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    view = look_at(eye, center, up)
    proj = perspective(np.radians(45.0), RESOLUTION[0] / RESOLUTION[1], 0.1, 10.0)
    mvp = proj @ view
    return torch.tensor(mvp, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,4,4]

# random vertex colors
colors = torch.rand((1, vertices.shape[1], 3), device=DEVICE)

# Choose rasterizer context depending on device
if DEVICE.startswith('cuda'):
    ctx = dr.RasterizeCudaContext()
else:
    ctx = dr.RasterizeGLContext()

# Render some angles
angles = [0, 45, 90, 135, 180, 225, 270, 315]
os.makedirs(OUTDIR, exist_ok=True)

for angle in angles:
    mvp = get_mvp_matrix(angle)  # [1, 4, 4]

    ones = torch.ones_like(vertices[:, :, :1])
    vertices_h = torch.cat([vertices, ones], dim=-1)  # [1, V, 4]

    # Transform to clip space
    pos_clip = torch.matmul(vertices_h, mvp.transpose(1, 2))  # [1, V, 4]

    # Optional debug: vertex NDC / screen positions
    pos_clip_ub = pos_clip[0]  # [V, 4]
    pos_ndc = (pos_clip_ub[:, :3] / pos_clip_ub[:, 3:4]).cpu().numpy()  # [V, 3]

    H, W = RESOLUTION

    # Rasterization
    faces_unbatched = faces[0].contiguous()  
    rast_out, rast_db = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=RESOLUTION)  # rast_out: [1,H,W,4]

    # Interpolate world positions (per-pixel)
    v_world_attr = vertices[0]  # [V,3]
    pos_map, _ = dr.interpolate(v_world_attr, rast_out, faces_unbatched)  # [1,H,W,3]
    pos_map = pos_map[0].cpu().numpy()  # [H,W,3]

    # Mask
    hit_mask = (rast_out[0, ..., 3] > 0).cpu().numpy()  # [H,W] bool

    # Depth map
    z_over_w = (pos_clip[..., 2:3] / pos_clip[..., 3:4])  # [1,V,1]
    z_attr = z_over_w[0]  # [V,1]

    # sanity checks
    assert z_attr.ndim == 2 and z_attr.shape[0] > 0 and z_attr.shape[1] == 1
    if z_attr.device != rast_out.device:
        z_attr = z_attr.to(rast_out.device)

    depth_map, _ = dr.interpolate(z_attr, rast_out, faces_unbatched)  # [1,H,W,1]
    depth_map = depth_map[0, ..., 0].cpu().numpy()  # [H,W]

    # Save normalized XYZ
    xyz = pos_map.copy()  # [H,W,3]
    valid = hit_mask
    if valid.any():
        v = xyz[valid]  # (N,3)
        mn = v.min(axis=0)
        mx = v.max(axis=0)
        span = mx - mn
        span[span == 0] = 1.0
        norm = (xyz - mn) / span
    else:
        norm = np.zeros_like(xyz)
    norm[~valid] = 0.0
    img_xyz = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    fname_xyz = os.path.join(OUTDIR, f"view_{angle}_xyz.png")
    plt.imsave(fname_xyz, img_xyz)
    print("Saved", fname_xyz)

    # Save depth heatmap
    depth = depth_map.copy()
    depth[~valid] = np.nan
    dm_valid = depth[valid]
    if dm_valid.size:
        dmin, dmax = float(np.nanmin(dm_valid)), float(np.nanmax(dm_valid))
        if dmax - dmin < 1e-8:
            dmax = dmin + 1e-8
        depth_norm = (depth - dmin) / (dmax - dmin)
        depth_norm[np.isnan(depth_norm)] = 0.0
    else:
        depth_norm = np.zeros_like(depth)
    depth_img = (np.clip(depth_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    fname_depth = os.path.join(OUTDIR, f"view_{angle}_depth.png")
    plt.imsave(fname_depth, depth_img, cmap='gray')
    print("Saved", fname_depth)

    # Export visible pixels as PLY
    pts = pos_map.reshape(-1, 3)
    mask_flat = valid.reshape(-1)
    valid_pts = pts[mask_flat]
    if valid_pts.shape[0] > 0:
        z = valid_pts[:, 2]
        zmin, zmax = z.min(), z.max()
        zspan = zmax - zmin if (zmax - zmin) != 0 else 1.0
        zcol = (z - zmin) / zspan
        colors_pc = np.stack([zcol, zcol, 1.0 - zcol], axis=1)
        colors_u8 = (colors_pc * 255).astype(np.uint8)

        def write_ply(points, colors, filename='points.ply'):
            assert points.shape[0] == colors.shape[0]
            N = points.shape[0]
            header = f'''ply
                format ascii 1.0
                element vertex {N}
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                end_header
                '''
            with open(filename, 'w') as f:
                f.write(header)
                for p, c in zip(points, colors):
                    f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

        fname_ply = os.path.join(OUTDIR, f"view_{angle}.ply")
        write_ply(valid_pts, colors_u8, fname_ply)
        print("Saved", fname_ply, "with", valid_pts.shape[0], "points.")
    else:
        print("No visible pixels -> no PLY written for angle", angle)

    # Reproject all visible world points with the same mvp and compare to original pixel coords + depth.
    mvp_np = mvp[0].cpu().numpy()  # (4,4)
    # get pixel coordinates of valid pixels (row = y, col = x)
    ys, xs = np.nonzero(valid)  # arrays length N
    N = ys.shape[0]
    if N == 0:
        print("No visible pixels for reprojection check.")
        continue

    world_pts = pos_map[ys, xs, :]  # (N,3)
    ones_col = np.ones((N, 1), dtype=np.float32)
    homo = np.concatenate([world_pts.astype(np.float32), ones_col], axis=1)  # (N,4)
    clip = (mvp_np @ homo.T).T  # (N,4)
    # avoid division by zero
    w = clip[:, 3:4].copy()
    valid_w_mask = np.abs(w[:,0]) > 1e-9
    if not np.all(valid_w_mask):
        print(f"Warning: {np.count_nonzero(~valid_w_mask)} points had tiny w during reproj; they will be ignored.")
    clip = clip[valid_w_mask]
    ys_ = ys[valid_w_mask]
    xs_ = xs[valid_w_mask]
    N2 = clip.shape[0]
    ndc = clip[:, :3] / clip[:, 3:4]  # (N2,3)

    # projected pixel coords using same formula as raster script
    px = (ndc[:, 0] * 0.5 + 0.5) * W
    py = (ndc[:, 1] * 0.5 + 0.5) * H

    # compute residuals in pixels (L2)
    res_pixels = np.sqrt((px - xs_.astype(np.float32))**2 + (py - ys_.astype(np.float32))**2)
    # compute residuals in ndc-z vs interpolated depth_map
    reproj_ndc_z = ndc[:, 2]
    # get original interpolated depth_map values at those pixels
    original_depth = depth_map[ys_, xs_].astype(np.float32)
    res_depth = reproj_ndc_z - original_depth

    # Stats
    def stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'max': float(np.max(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99))
        }

    pix_stats = stats(res_pixels)
    depth_stats = stats(res_depth)

    print(f"Reprojection check for angle={angle}:")
    print(f"  visible points used: {N2} / {N}")
    print(f"  pixel residuals (px L2)  mean={pix_stats['mean']:.4f}, median={pix_stats['median']:.4f}, p95={pix_stats['p95']:.4f}, max={pix_stats['max']:.4f}")
    print(f"  ndc-z residuals          mean={depth_stats['mean']:.6f}, median={depth_stats['median']:.6f}, p95={depth_stats['p95']:.6f}, max={depth_stats['max']:.6f}")

    # optionally print a few outliers
    outlier_idx = np.where(res_pixels > 2.0)[0]  # >2 pixel mismatch
    if outlier_idx.size > 0:
        print(f"  Note: {outlier_idx.size} points have >2 px reprojection error. Sample indices: {outlier_idx[:5].tolist()}")

    # OVERLAY: plot reproj points over depth image (color by pixel residual)
    bg = depth_norm.copy()
    # choose indices to plot (sample if too many)
    max_points = 20000
    idxs = np.arange(N2)
    if N2 > max_points:
        idxs = np.random.choice(N2, max_points, replace=False)
    plot_xs = xs_[idxs]
    plot_ys = ys_[idxs]
    plot_res = res_pixels[idxs]

    # separate outliers (>2 px) to draw in red
    outlier_mask = plot_res > 2.0
    inlier_mask = ~outlier_mask

    plt.figure(figsize=(6,6))
    plt.imshow(bg, cmap='gray', origin='upper', vmin=0.0, vmax=1.0)
    # inliers: color by residual (clamped)
    if np.any(inlier_mask):
        sc = plt.scatter(plot_xs[inlier_mask], plot_ys[inlier_mask], c=plot_res[inlier_mask],
                         s=1, cmap='viridis', vmin=0.0, vmax=5.0, marker='.', linewidths=0)
        plt.colorbar(sc, label='pixel residual (px)')
    # outliers: big red dots
    if np.any(outlier_mask):
        plt.scatter(plot_xs[outlier_mask], plot_ys[outlier_mask], c='red', s=6, marker='o', label='outliers >2px')

    plt.title(f"Overlay depth and point cloud (angle {angle})")
    plt.xlim(0, W)
    plt.ylim(H, 0)  # invert y-axis so origin at top-left matches image coords
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    overlay_fname = os.path.join(OUTDIR, f"reproj_overlay_{angle}.png")
    plt.savefig(overlay_fname, dpi=150)
    plt.close()
    print("Saved overlay:", overlay_fname)

print("Done.")
