import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib
import matplotlib.pyplot as plt

# Geometry evaluation script for Nerf-Synthetic calculating Chamfer Distance, F1-score, Completeness, Accuracy and Normal Alignment of GT and extracted meshes

np.random.seed(42)  # Set seed for same results

# Sample mesh
def sample_geometry(path: str, n_samples: int) -> np.ndarray:
    geom = trimesh.load(path)

    # Merge scene
    if isinstance(geom, trimesh.Scene):
        geom = trimesh.util.concatenate(
            [g for g in geom.geometry.values()]
        )

    # Triangle mesh
    if isinstance(geom, trimesh.Trimesh) and geom.faces is not None:
        rng = np.random.default_rng(42)
        pts, face_idx = trimesh.sample.sample_surface(geom, n_samples, seed=rng)
        normals = geom.face_normals[face_idx]
        return pts.astype(np.float32), normals.astype(np.float32)

    # Point cloud
    if hasattr(geom, "vertices"):
        verts = np.asarray(geom.vertices, dtype=np.float32)

        if len(verts) > n_samples:
            idx = np.random.choice(len(verts), n_samples, replace=False)
            verts = verts[idx]
        normals = np.zeros_like(verts)
        return verts, normals

# Calculate chamfer distance
def chamfer_distance(x: np.ndarray, y: np.ndarray) -> float:
    tree_y = cKDTree(y)
    d_xy, _ = tree_y.query(x, k=1)

    tree_x = cKDTree(x)
    d_yx, _ = tree_x.query(y, k=1)

    return 0.5 * float(d_xy.mean() + d_yx.mean())

# Calculate F1-score
def f1_score_pointclouds(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float
) -> float:
    tree_y = cKDTree(y)
    d_xy, _ = tree_y.query(x, k=1)

    tree_x = cKDTree(x)
    d_yx, _ = tree_x.query(y, k=1)

    precision = (d_xy <= threshold).mean()
    recall = (d_yx <= threshold).mean()

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))

# Calc Normal alignment
def normal_alignment(
    pts_a: np.ndarray,
    nrm_a: np.ndarray,
    pts_b: np.ndarray,
    nrm_b: np.ndarray,
) -> float:
    tree_b = cKDTree(pts_b)
    _, idx = tree_b.query(pts_a, k=1)

    na = nrm_a
    nb = nrm_b[idx]

    # Normalize
    na = na / (np.linalg.norm(na, axis=1, keepdims=True) + 1e-8)
    nb = nb / (np.linalg.norm(nb, axis=1, keepdims=True) + 1e-8)

    dots = np.abs(np.sum(na * nb, axis=1))
    return float(dots.mean())

# Calc Completeness
def completeness(
    gt: np.ndarray,
    pred: np.ndarray,
) -> float:
    tree = cKDTree(pred)
    d, _ = tree.query(gt, k=1)
    return float(d.mean())

# Calc Accuracy
def accuracy(
    pred: np.ndarray,
    gt: np.ndarray
) -> float:
    tree = cKDTree(gt)
    d, _ = tree.query(pred, k=1)
    return float(d.mean())


# Rotation function to align GT meshes
def rotate_x(points: np.ndarray, degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad),  np.cos(rad)],
    ], dtype=np.float32)
    return points @ R.T

# Normalize bboxes for alignment
def normalize_bbox(x: np.ndarray, y: np.ndarray):
    all_pts = np.concatenate([x, y], axis=0)
    minv = all_pts.min(axis=0)
    maxv = all_pts.max(axis=0)

    center = 0.5 * (minv + maxv)
    scale = np.linalg.norm(maxv - minv)

    x = (x - center) / scale
    y = (y - center) / scale
    return x, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mesh_a", type=str)
    p.add_argument("mesh_b", type=str)
    p.add_argument("--samples", type=int, default=100_000, help="Samples per mesh, default 100.000)")
    p.add_argument("--f1_thresh", type=float, default=None, help="F1 threshold, default 1% of bbox diagonal")
    args = p.parse_args()

    print(f"[INFO] Sample {args.samples} points out of {args.mesh_a}")
    A, A_n = sample_geometry(args.mesh_a, args.samples)

    print(f"[INFO] Sample {args.samples} points out of {args.mesh_b}")
    B, B_n = sample_geometry(args.mesh_b, args.samples)

    print("[INFO] Rotate mesh")
    A = rotate_x(A, -90)
    A_n = rotate_x(A_n, -90)

    print("[INFO] Normalize point clouds")
    A, B = normalize_bbox(A, B)

    print("[INFO] Calculate Chamfer Distance:")
    cd = chamfer_distance(A, B)
    print(f"\nChamfer Distance {cd:.8f}")

    if args.f1_thresh is not None:
        threshold = args.f1_thresh
    else:
        # 1% bbox diagonal as threshold
        bbox_diag = np.linalg.norm(np.max(np.concatenate([A, B], axis=0), axis=0) - np.min(np.concatenate([A, B], axis=0), axis=0))
        threshold = 0.01 * bbox_diag
        print(f"[INFO] Automatic F1-Threshold (1% of bbox diag): {threshold:.6f}")

    print(f"[INFO] Calculate F1-Score (threshold={threshold:.6f}):")
    f1 = f1_score_pointclouds(A, B, threshold)
    print(f"F1-Score: {f1:.6f}")

    print("[INFO] Calculate Completeness:")
    comp = completeness(B, A)    # GT=B, Pred=A
    print(f"Completeness: {comp:.6f}")

    print("[INFO] Calculate Normal Alignment:")
    na = normal_alignment(A, A_n, B, B_n)
    print(f"Normal Alignment: {na:.6f}")

    print("[INFO] Calculate Accuracy (Pred → GT):")
    acc = accuracy(A, B)    # GT=B, Pred=A
    print(f"Accuracy: {acc:.8f}")

if __name__ == "__main__":
    main()