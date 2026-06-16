from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def imread_rgb(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(p: Path, img: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def ensure_bin(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8) * 255)

def local_mean_std(img: np.ndarray, mask: np.ndarray):
    vals = img[mask > 0].astype(np.float32)
    if vals.size == 0:
        c = img.shape[2]
        return np.zeros((c,), np.float32), np.ones((c,), np.float32)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.maximum(std, 1.0)
    return mean, std

def make_ring(mask: np.ndarray, dilate_px: int = 18, erode_px: int = 2):
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
    dil = cv2.dilate(mask, kd, iterations=1)
    ero = cv2.erode(mask, ke, iterations=1)
    ring = ((dil > 0) & (ero == 0) & (mask == 0)).astype(np.uint8) * 255
    return ring

def recolor_one(
    factual: np.ndarray,
    composite: np.ndarray,
    mask: np.ndarray,
    mix: float = 0.68,          # 越大越贴近背景统计
    dark_shift: float = 14.0,   # 往深灰拉
    contrast_scale: float = 0.72, # 降低缺陷内部对比
    feather_sigma: float = 1.2, # 轻微软化边界
    max_alpha: float = 0.92,    # 不要完全吃掉缺陷
):
    factual = factual.astype(np.float32)
    composite = composite.astype(np.float32)
    mask = ensure_bin(mask)
    m = mask > 0

    if np.count_nonzero(m) == 0:
        return composite.astype(np.uint8)

    ring = make_ring(mask, dilate_px=18, erode_px=2)
    bg_mean, bg_std = local_mean_std(factual, ring)

    defect_vals = composite[m]
    # 强制灰度化
    gray = defect_vals.mean(axis=1, keepdims=True)
    gray3 = np.repeat(gray, 3, axis=1)

    fg_mean = gray3.mean(axis=0)
    fg_std = gray3.std(axis=0)
    fg_std = np.maximum(fg_std, 1.0)

    target_mean = np.clip(bg_mean - dark_shift, 0, 255)
    target_std = np.maximum(bg_std * contrast_scale, 1.0)

    matched = (gray3 - fg_mean) / fg_std * target_std + target_mean
    matched = np.clip(matched, 0, 255)

    # 与原缺陷混合，避免完全丢失缺陷感
    toned_vals = defect_vals * (1.0 - mix) + matched * mix

    toned_full = composite.copy()
    toned_full[m] = toned_vals

    alpha = (mask.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    alpha = np.clip(alpha * max_alpha, 0.0, 1.0)[..., None]

    out = factual * (1.0 - alpha) + toned_full * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process_sample(sdir: Path, overwrite: bool = False):
    factual_p = sdir / "factual.png"
    composite_p = sdir / "initial_composite.png"
    mask_p = sdir / "strict_mask.png"

    if not (factual_p.exists() and composite_p.exists() and mask_p.exists()):
        return False

    factual = imread_rgb(factual_p)
    composite = imread_rgb(composite_p)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    toned = recolor_one(
        factual, composite, mask,
        mix=0.68,
        dark_shift=14.0,
        contrast_scale=0.72,
        feather_sigma=1.2,
        max_alpha=0.92,
    )

    out_p = sdir / "toned_composite.png"
    imwrite_rgb(out_p, toned)

    if overwrite:
        imwrite_rgb(composite_p, toned)

    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    count = 0
    for sdir in sorted(root.iterdir()):
        if not (sdir.is_dir() and sdir.name.startswith("sample_")):
            continue
        ok = process_sample(sdir, overwrite=args.overwrite)
        if ok:
            count += 1
            if count % 20 == 0:
                print(f"[OK] processed {count}")
    print(f"[DONE] processed total = {count}")

if __name__ == "__main__":
    main()
