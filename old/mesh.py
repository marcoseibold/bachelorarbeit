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

def parse_mtl_file(mtl_path):
    """Return dict: material_name -> Kd(np.array float32 [3])"""
    materials = {}
    if not os.path.exists(mtl_path):
        return materials
    current = None
    with open(mtl_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0].lower()
            if key == 'newmtl':
                current = ' '.join(parts[1:])
                materials[current] = {'Kd': np.array([0.8, 0.8, 0.8], dtype=np.float32)}
            elif key == 'kd' and current is not None:
                vals = list(map(float, parts[1:4]))
                materials[current]['Kd'] = np.array(vals, dtype=np.float32)
            # Note: map_Kd (texture) is ignored here; can be added later.
    return {k: v['Kd'] for k, v in materials.items()}

V = vertices_np.shape[0]
colors_vertex = None  # will be numpy (V,3) float32

if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == V:
    vc = mesh.visual.vertex_colors[:, :3]  # may be RGBA or RGB in 0..255
    # If values are >1 assume 0-255
    if vc.max() > 1.0:
        vc = vc.astype(np.float32) / 255.0
    colors_vertex = vc.astype(np.float32)
    print("Using vertex colors from mesh.visual.vertex_colors")

if colors_vertex is None:
    # find mtllib relative to obj path
    obj_dir = os.path.dirname(os.path.abspath(MESH_PATH))
    mtllib_name = None
    face_materials = []  # per face material name (in face order encountered)
    # We will parse the obj in order and map each 'f' line to the current material
    cur_mtl = None
    with open(MESH_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0].lower()
            if key == 'mtllib':
                mtllib_name = ' '.join(parts[1:])
            elif key == 'usemtl':
                cur_mtl = ' '.join(parts[1:])
            elif key == 'f':
                # One face encountered: record current material (may be None)
                face_materials.append(cur_mtl)
    # If we found mtllib, try parsing it
    materials_kd = {}
    if mtllib_name is not None:
        mtllib_path = os.path.join(obj_dir, mtllib_name)
        materials_kd = parse_mtl_file(mtllib_path)
        print("Parsed MTL:", list(materials_kd.keys()))
    else:
        print("No mtllib found in OBJ header; falling back to default color")

    # Now map face_materials to face colors
    F = faces_np.shape[0]
    # face_materials length should equal number of faces encountered in file
    if len(face_materials) != F:
        # Sometimes trimesh triangulation / ordering may differ; attempt to use trimesh.visual.face_materials if available
        if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
            print("Warning: face count mismatch. Trying mesh.visual.material or face_materials from trimesh.")
        # fallback: set all faces to default material
        face_colors = np.tile(np.array([0.8, 0.8, 0.8], dtype=np.float32), (F,1))
    else:
        face_colors = np.zeros((F,3), dtype=np.float32)
        for i, mat_name in enumerate(face_materials):
            if mat_name is None:
                face_colors[i] = np.array([0.8,0.8,0.8], dtype=np.float32)
            else:
                kd = materials_kd.get(mat_name, None)
                if kd is None:
                    # fallback default
                    face_colors[i] = np.array([0.8,0.8,0.8], dtype=np.float32)
                else:
                    face_colors[i] = kd

    # Convert face colors to per-vertex color by averaging adjacent faces
    vertex_color_sum = np.zeros((V,3), dtype=np.float32)
    vertex_face_count = np.zeros((V,), dtype=np.int32)
    for fi in range(F):
        f = faces_np[fi]  # indices (0-based) from trimesh
        c = face_colors[fi]
        for vi in f:
            vertex_color_sum[vi] += c
            vertex_face_count[vi] += 1
    # avoid division by zero
    mask = vertex_face_count > 0
    colors_vertex = np.zeros_like(vertex_color_sum)
    colors_vertex[mask] = (vertex_color_sum[mask].T / vertex_face_count[mask]).T
    # for isolated vertices (shouldn't be), set default
    colors_vertex[~mask] = np.array([0.8,0.8,0.8], dtype=np.float32)
    print("Computed per-vertex colors from face materials")

# As a safety, ensure colors_vertex shape and range
if colors_vertex is None:
    colors_vertex = np.tile(np.array([0.8,0.8,0.8], dtype=np.float32), (V,1))
colors_vertex = np.clip(colors_vertex, 0.0, 1.0).astype(np.float32)

# Convert to torch tensor [1, V, 3]
colors = torch.tensor(colors_vertex, device=DEVICE).unsqueeze(0)  # [1, V, 3]

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
    radius = 7.0
    height = -3.0
    eye = np.array([np.sin(angle_rad) * radius, height, np.cos(angle_rad) * radius], dtype=np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    view = look_at(eye, center, up)
    proj = perspective(np.radians(45.0), RESOLUTION[0] / RESOLUTION[1], 0.1, 10.0)
    mvp = proj @ view
    return torch.tensor(mvp, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,4,4]

# colors = torch.ones((1, vertices.shape[1], 3), dtype=torch.float32, device=DEVICE) * 0.8
# colors = torch.rand((1, vertices.shape[1], 3), device=DEVICE)


ctx = dr.RasterizeGLContext()

# Render some angles
angles = [0, 45, 90, 135, 180, 225, 270, 315]
os.makedirs(OUTDIR, exist_ok=True)

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
    filename = os.path.join(OUTDIR, f"view_{angle}.png")
    plt.imsave(filename, rgb)
    print(f"Image saved: {filename}")

print("All angles rendered!")
