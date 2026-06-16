from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

def imread_rgb(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(p: Path, img: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def ensure_bin(mask):
    return ((mask > 127).astype(np.uint8) * 255)

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def local_mean_std(img, mask):
    vals = img[mask > 0].astype(np.float32)
    if vals.size == 0:
        c = img.shape[2]
        return np.zeros((c,), np.float32), np.ones((c,), np.float32)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.maximum(std, 1.0)
    return mean, std

def tone_match_inside_mask(factual, composite, mask, alpha=0.18):
    factual = factual.astype(np.float32)
    composite = composite.astype(np.float32)
    m = mask > 0

    out = composite.copy()

    defect_region = composite[m]
    if defect_region.size == 0:
        return composite.astype(np.uint8)

    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(mask, kd, iterations=1)
    ero = cv2.erode(mask, ke, iterations=1)
    ring = ((dil > 0) & (ero == 0) & (mask == 0)).astype(np.uint8) * 255

    bg_mean, bg_std = local_mean_std(factual, ring)
    fg_mean, fg_std = local_mean_std(composite, mask)

    orig_vals = composite[m]
    matched_vals = (orig_vals - fg_mean) / fg_std * bg_std + bg_mean
    matched_vals = np.clip(matched_vals, 0, 255)

    vals = orig_vals * (1.0 - alpha) + matched_vals * alpha
    out[m] = vals
    return out.astype(np.uint8)

def make_boundary_band(mask, dilate_px=10, erode_px=1):
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
    dil = cv2.dilate(mask, kd, iterations=1)
    ero = cv2.erode(mask, ke, iterations=1)
    band = ((dil > 0) & (ero == 0)).astype(np.uint8) * 255
    return band

def poisson_blend(factual, src_img, mask):
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return src_img.copy()
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    src_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(factual, cv2.COLOR_RGB2BGR)
    out_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask, center, cv2.MIXED_CLONE)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def soft_blend_poisson(composite, poisson, beta=0.22):
    composite = composite.astype(np.float32)
    poisson = poisson.astype(np.float32)
    out = composite * (1.0 - beta) + poisson * beta
    return np.clip(out, 0, 255).astype(np.uint8)

def preview_boundary(img, band):
    out = img.copy()
    sel = band > 0
    out[sel] = (0.7 * out[sel] + 0.3 * np.array([255, 120, 0])).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factual", required=True)
    ap.add_argument("--composite", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    factual = imread_rgb(Path(args.factual))
    composite = imread_rgb(Path(args.composite))
    mask = cv2.imread(str(Path(args.mask)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.mask)
    mask = ensure_bin(mask)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tone_matched = tone_match_inside_mask(factual, composite, mask, alpha=0.18)
    poisson_raw = poisson_blend(factual, tone_matched, mask)
    poisson = soft_blend_poisson(composite, poisson_raw, beta=0.22)
    band = make_boundary_band(mask, dilate_px=10, erode_px=1)
    preview = preview_boundary(poisson, band)

    imwrite_rgb(out_dir / "tone_matched.png", tone_matched)
    imwrite_rgb(out_dir / "poisson_composite.png", poisson)
    imwrite_rgb(out_dir / "light_refined.png", poisson)
    cv2.imwrite(str(out_dir / "strict_mask.png"), mask)
    cv2.imwrite(str(out_dir / "boundary_band.png"), band)
    imwrite_rgb(out_dir / "boundary_preview.png", preview)

    print("[OK]", out_dir)

if __name__ == "__main__":
    main()
