import os, json, math
from typing import Optional, List, Tuple, Dict
import numpy as np
from PIL import Image
import re

Array = np.ndarray

# NeRF-Synthetic (JSON)

def _resolve_image_path(root: str, file_path: str) -> str:
    base = os.path.join(root, file_path)
    if os.path.exists(base):
        return base
    for ext in [".png", ".jpg", ".jpeg"]:
        p = base if base.lower().endswith(ext) else base + ext
        if os.path.exists(p): return p
    base2 = os.path.join(root, "images", os.path.basename(base))
    for ext in [".png", ".jpg", ".jpeg"]:
        p = base2 if base2.lower().endswith(ext) else base2 + ext
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Image not found for '{file_path}' under {root}")

def _compute_focal_from_meta(meta: dict, W: int, H: int) -> float:
    # bevorzugt fl_x (Pixel), sonst camera_angle_x (Rad), sonst per-frame fl_x
    if "fl_x" in meta and isinstance(meta["fl_x"], (int, float)):
        W0 = float(meta.get("w", W))
        return float(meta["fl_x"]) * (W / max(W0, 1e-8))
    if "camera_angle_x" in meta:
        return 0.5 * H / math.tan(0.5 * float(meta["camera_angle_x"]))
    if meta.get("frames"):
        fr0 = meta["frames"][0]
        if "fl_x" in fr0:
            W0 = float(meta.get("w", W))
            return float(fr0["fl_x"]) * (W / max(W0, 1e-8))
    raise ValueError("No focal in JSON meta (need fl_x or camera_angle_x).")

def _load_nerf_synth(root: str, img_res: Optional[int], split: str):
    # Find JSON
    p_split = os.path.join(root, f"transforms_{split}.json")
    p_any   = os.path.join(root, "transforms.json")
    json_path = p_split if os.path.exists(p_split) else (p_any if os.path.exists(p_any) else None)
    if not json_path:
        return None

    with open(json_path, "r") as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"{os.path.basename(json_path)} contains 0 frames")

    # Set resolution
    if img_res is None:
        # Use res of first image
        fr0 = frames[0]
        img0_path = _resolve_image_path(root, fr0.get("file_path", fr0.get("img_path", "")))
        with Image.open(img0_path) as pil0:
            W, H = pil0.size
    else:
        H = W = int(img_res)

    focal_px = _compute_focal_from_meta(meta, W, H)

    cam2worlds: List[Array] = []
    imgs: List[Array] = []
    masks: List[Array] = []

    for fr in frames:
        c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        cam2worlds.append(c2w)

        img_path = _resolve_image_path(root, fr.get("file_path", fr.get("img_path", "")))
        pil = Image.open(img_path)
        try:
            if pil.mode in ("RGBA", "LA"): 
                pil = pil.convert("RGBA")
            else:                          
                pil = pil.convert("RGB")

            # Resize
            if img_res is not None and pil.size != (W, H):
                pil = pil.resize((W, H), resample=Image.LANCZOS)

            arr = np.array(pil).astype(np.float32)
        finally:
            pil.close()

        H_cur, W_cur = arr.shape[0], arr.shape[1]

        if arr.ndim == 3 and arr.shape[2] == 4:
            alpha = arr[..., 3] / 255.0
            a = alpha[..., None].astype(np.float32)
            rgb = (arr[..., :3] / 255.0).astype(np.float32) * a + (1.0 - a)  # white BG
            mask = alpha.astype(np.float32)
        else:
            rgb = (arr[..., :3] / 255.0).astype(np.float32)
            mask = np.ones((H_cur, W_cur), dtype=np.float32)

        imgs.append(rgb)
        masks.append(mask)

    return cam2worlds, imgs, masks, float(focal_px), None  # intrinsics_per_frame=None


# Mip-NeRF 360 (COLMAP / LLFF)

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def _list_all_images_under(root: str):
    # pick images (lowest res first)
    import os, re
    def natural_key(s):  # sort 000.png < 10.png
        return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

    candidates = ["images_4", "images_8", "images_2", "images", "rgb", ""]
    exts = (".png", ".jpg", ".jpeg")

    for sub in candidates:
        folder = os.path.join(root, sub)
        if not os.path.isdir(folder):
            continue
        files = [os.path.join(folder, f)
                 for f in sorted(os.listdir(folder), key=natural_key)
                 if os.path.splitext(f)[1].lower() in exts]
        if files:
            print(f"[INFO] Using images from '{sub}/' ({len(files)} files)")
            return files

    raise FileNotFoundError(
        f"No image folders found under {root} (checked images_[8,4,2], images/, rgb/)"
    )


def _read_noncomment_lines(path: str):
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                yield s

def _qvec2rotmat(qw, qx, qy, qz) -> Array:
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = q / n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ], dtype=np.float32)

def _parse_colmap_sparse0(root: str):
    cam_txt = os.path.join(root, "sparse", "0", "cameras.txt")
    img_txt = os.path.join(root, "sparse", "0", "images.txt")
    if not (os.path.exists(cam_txt) and os.path.exists(img_txt)):
        return None

    cams: Dict[int, Dict[str, float]] = {}
    for ln in _read_noncomment_lines(cam_txt):
        parts = ln.split()
        if len(parts) < 5: continue
        cam_id = int(parts[0]); model = parts[1].upper()
        W0 = float(parts[2]); H0 = float(parts[3])
        pars = list(map(float, parts[4:]))

        if model == "PINHOLE":
            fx, fy, cx, cy = pars[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = pars[0]; cx, cy = pars[1:3]
        else:
            fx = fy = pars[0] if pars else 0.5*(W0+H0); cx = W0/2; cy = H0/2

        cams[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, w=W0, h=H0)

    frames = []
    for ln in _read_noncomment_lines(img_txt):
        parts = ln.split()
        if len(parts) < 10: continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = " ".join(parts[9:])
        R_wc = _qvec2rotmat(qw, qx, qy, qz)
        t_wc = np.array([tx, ty, tz], dtype=np.float32)
        R_cw = R_wc.T
        t_cw = (-R_cw @ t_wc).astype(np.float32)
        c2w = np.eye(4, dtype=np.float32)
        cv2gl = np.diag([1, -1, -1]).astype(np.float32)  # OpenCV -> OpenGL
        c2w[:3, :3] = c2w[:3, :3] @ cv2gl
        c2w[:3,:3] = R_cw; c2w[:3,3] = t_cw
        frames.append(dict(c2w=c2w, name=name, **cams.get(cam_id, {})))
    return frames

def _try_load_llff_poses_bounds(root: str):
    path = os.path.join(root, "poses_bounds.npy")
    if not os.path.exists(path): return None
    arr = np.load(path)  # (N,17)
    frames = []
    for row in arr:
        pose = row[:15].reshape(3,5).astype(np.float32)  # [R|t|h,w,f]
        R = pose[:,:3]; t = pose[:,3]; h,w,f = pose[:,4]
        c2w = np.eye(4, dtype=np.float32); c2w[:3,:3] = R; c2w[:3,3] = t
        theta = -np.pi * 0.5  # test +np.pi*0.5
        Rz = np.array([[ np.cos(theta), -np.sin(theta), 0],
                    [ np.sin(theta),  np.cos(theta), 0],
                    [ 0,               0,            1]], dtype=np.float32)
        c2w[:3, :3] = c2w[:3, :3] @ Rz
        frames.append(dict(c2w=c2w, fx=float(f), fy=float(f), cx=float(w/2), cy=float(h/2), w=float(w), h=float(h), name=""))
    return frames

def _read_split_list(root: str, split: str) -> Optional[set]:
    p = os.path.join(root, f"{split}.txt")
    if not os.path.exists(p): return None
    with open(p, "r") as f:
        return {ln.strip() for ln in f if ln.strip()}

def _find_image_for_name(root: str, name: str) -> Optional[str]:
    # images/ or rgb/
    for base in ["images", "rgb", ""]:
        stem = os.path.splitext(name)[0] if name else None
        if stem is None: continue
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = os.path.join(root, base, stem + ext)
            if os.path.exists(cand): return cand
    return None

def _load_m360_raw(root: str, img_res: Optional[int], split: str):
    # Load frames (LLFF oder COLMAP)
    frames = _try_load_llff_poses_bounds(root)
    if frames is None:
        frames = _parse_colmap_sparse0(root)
    if frames is None:
        return None

    # Save index for frames
    for idx, fr in enumerate(frames):
        fr["_idx"] = idx

    # Split per train.txt/test.txt, else 80/20
    lst = _read_split_list(root, split)
    if lst is not None:
        # Split via train.txt/test.txt
        frames = [f for f in frames if f.get("name","") in lst]
    else:
        # Use every 8th image as test
        frames_sorted = sorted(frames, key=lambda f: f.get("name", ""))

        if split == "train":
            frames = [f for i, f in enumerate(frames_sorted) if (i % 8) != 0]
        else:  # split == "test"
            frames = [f for i, f in enumerate(frames_sorted) if (i % 8) == 0]


    cam2worlds: List[Array] = []
    imgs: List[Array] = []
    masks: List[Array] = []
    intr_per_frame: List[Dict[str, float]] = []
    focals: List[float] = []

    any_H = None  # Fallback-Focal

    for f in frames:
        c2w = f["c2w"].astype(np.float32)
        cam2worlds.append(c2w)

        fx, fy = f.get("fx"), f.get("fy")
        cx, cy = f.get("cx"), f.get("cy")
        W0, H0 = f.get("w"), f.get("h")  # Original res

        # Get image path
        name = f.get("name", "")
        img_path = None

        if name:
            img_path = _find_image_for_name(root, name)

        if img_path is None:
            # LLFF/poses_bounds.npy
            if "_LLFF_IMAGE_LIST_CACHE" not in globals():
                # Get all images
                imgs_all = _list_all_images_under(root)
                globals()["_LLFF_IMAGE_LIST_CACHE"] = imgs_all
                if not imgs_all:
                    raise FileNotFoundError(
                        f"No images found under {root} (looked in images/, rgb/, images_[2|4|8]/)"
                    )
            imgs_all = globals()["_LLFF_IMAGE_LIST_CACHE"]

            global_idx = f.get("_idx", len(cam2worlds) - 1)
            if global_idx < len(imgs_all):
                img_path = imgs_all[global_idx]
            else:
                raise FileNotFoundError(
                    f"Not enough images for LLFF frames: need > {global_idx}, "
                    f"found {len(imgs_all)} under {root}"
                )

        pil = Image.open(img_path)
        try:
            has_alpha = (pil.mode in ("RGBA", "LA"))
            pil = pil.convert("RGBA") if has_alpha else pil.convert("RGB")

            # Get res out of image
            if W0 is None or H0 is None:
                W0, H0 = pil.size

            # Set res
            if img_res is None:
                W_loaded, H_loaded = pil.size
                if W0 is not None and H0 is not None and (W0 != 0 and H0 != 0):
                    scale_x = float(W_loaded) / float(W0)
                    scale_y = float(H_loaded) / float(H0)
                else:
                    scale_x = scale_y = 1.0
                W = int(W_loaded)
                H = int(H_loaded)
            else:
                max_side = float(max(W0, H0))
                if max_side <= img_res:
                    W = int(W0)
                    H = int(H0)
                    scale_x = 1.0
                    scale_y = 1.0
                else:
                    s = float(img_res) / max_side  # Scaling
                    W = int(round(W0 * s))
                    H = int(round(H0 * s))
                    scale_x = float(W) / float(W0)
                    scale_y = float(H) / float(H0)

                if pil.size != (W, H):
                    pil = pil.resize((W, H), resample=Image.LANCZOS)

            arr = np.array(pil).astype(np.float32)
        finally:
            pil.close()

        any_H = H if any_H is None else any_H

        # RGBA -> RGB + Mask
        if arr.ndim == 3 and arr.shape[2] == 4:
            alpha = arr[..., 3] / 255.0
            a = alpha[..., None].astype(np.float32)
            rgb = (arr[..., :3] / 255.0).astype(np.float32) * a + (1.0 - a)
            mask = alpha.astype(np.float32)
        else:
            rgb = (arr[..., :3] / 255.0).astype(np.float32)
            mask = np.ones((H, W), dtype=np.float32)

        imgs.append(rgb)
        masks.append(mask)

        # Scale intrinsics
        fx_scaled = fx * scale_x if fx is not None else None
        fy_scaled = fy * scale_y if fy is not None else None
        cx_scaled = cx * scale_x if cx is not None else (W / 2.0)
        cy_scaled = cy * scale_y if cy is not None else (H / 2.0)

        intr_per_frame.append(dict(
            fx=fx_scaled,
            fy=fy_scaled,
            cx=cx_scaled,
            cy=cy_scaled,
            W=W,
            H=H,
        ))

        if fx_scaled is not None and fy_scaled is not None:
            focals.append(float(0.5 * (fx_scaled + fy_scaled)))

    if len(focals) > 0:
        focal_px = float(np.mean(focals))
    else:
        if any_H is None:
            raise ValueError("Cannot infer focal length: no valid image height found.")
        focal_px = 0.5 * any_H / math.tan(0.5 * math.radians(60.0))  # Fallback

    return cam2worlds, imgs, masks, focal_px, intr_per_frame

# Dataset booleans
nerf_synthetic = False
m360 = False

# Load NeRF-Synthetic or Mip-NeRF 360 dataset
def load_dataset(data_root: str, img_res: Optional[int]=None, split: str = "train"):
    global nerf_synthetic, m360

    out = _load_nerf_synth(data_root, img_res, split)
    if out is not None:
        nerf_synthetic = True
        return out
    out = _load_m360_raw(data_root, img_res, split)
    if out is not None:
        m360 = True
        return out
    raise RuntimeError(f"No supported dataset found under {data_root} for split='{split}'.")
