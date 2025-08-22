import os
import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt

RESOLUTION = (512, 512)
DEVICE = 'cuda'
MESH_PATH = 'bunny.obj'

# Load and adjust mesh
mesh = trimesh.load(MESH_PATH, process=True)
#mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
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

# colors = torch.ones((1, vertices.shape[1], 3), dtype=torch.float32, device=DEVICE) * 0.8
colors = torch.rand((1, vertices.shape[1], 3), device=DEVICE)


ctx = dr.RasterizeGLContext()

# Render some angles
angles = [0, 45, 90, 135, 180, 225, 270, 315]
os.makedirs("renders", exist_ok=True)

for angle in angles:
    mvp = get_mvp_matrix(angle)  # [1, 4, 4]

    ones = torch.ones_like(vertices[:, :, :1])
    vertices_h = torch.cat([vertices, ones], dim=-1)  # [1, V, 4]

    # Transform
    pos_clip = torch.matmul(vertices_h, mvp.transpose(1, 2))  # [1, V, 4]

    # Debug: Clip → NDC → Screen
    pos_clip_ub = pos_clip[0]  # [V, 4]
    pos_ndc = (pos_clip_ub[:, :3] / pos_clip_ub[:, 3:4]).cpu().numpy()  # [V, 3]

    H, W = RESOLUTION
    sx = (pos_ndc[:, 0] * 0.5 + 0.5) * W
    sy = (pos_ndc[:, 1] * 0.5 + 0.5) * H

    # print(f"\n📸 Angle {angle}")
    # print("  NDC x range:", pos_ndc[:, 0].min(), pos_ndc[:, 0].max())
    # print("  NDC y range:", pos_ndc[:, 1].min(), pos_ndc[:, 1].max())
    # print("  NDC z range:", pos_ndc[:, 2].min(), pos_ndc[:, 2].max())
    # print("  screen x range:", sx.min(), sx.max())
    # print("  screen y range:", sy.min(), sy.max())

    # print("pos_clip sample:", pos_clip[0, :5])
    # print("pos_ndc sample:", pos_ndc[:5])

    # Rasterization
    faces_unbatched = faces[0].contiguous()  
    rast_out, _ = dr.rasterize(ctx, pos_clip, faces_unbatched, resolution=RESOLUTION)

    # hit_mask = rast_out[0, ..., 3] > 0  # [H, W]
    # print("Hit pixels:", torch.sum(hit_mask).item(), "/", hit_mask.numel())

    # Interpolation
    attr = colors[0]  # [V, 3]
    rgb, _ = dr.interpolate(attr, rast_out, faces_unbatched)  # [1, H, W, 3]
    rgb = rgb[0]  # [H, W, 3]
    rgb = torch.clamp(rgb, 0.0, 1.0).cpu().numpy()


    rgb = np.nan_to_num(rgb, nan=1.0) 

    # Saving
    filename = f"renders/view_{angle}.png"
    plt.imsave(filename, rgb)
    print(f"Image saved: {filename}")

print("All angles rendered!")
