from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def read_image_any(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read: {path}")
    return img


def derive_mask_and_rgb(defect_img: np.ndarray):
    # RGBA 优先用 alpha
    if defect_img.ndim == 3 and defect_img.shape[2] == 4:
        rgb = defect_img[:, :, :3]
        alpha = defect_img[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
        return rgb, mask

    # 普通 RGB/BGR：用边界颜色估计背景，再做差分
    if defect_img.ndim == 2:
        rgb = cv2.cvtColor(defect_img, cv2.COLOR_GRAY2BGR)
    else:
        rgb = defect_img[:, :, :3]

    h, w = rgb.shape[:2]

    border_pixels = np.concatenate([
        rgb[0, :, :],
        rgb[-1, :, :],
        rgb[:, 0, :],
        rgb[:, -1, :],
    ], axis=0).astype(np.float32)

    bg_color = np.median(border_pixels, axis=0)
    diff = np.linalg.norm(rgb.astype(np.float32) - bg_color[None, None, :], axis=2)

    # 先宽松阈值，再做形态学清理
    mask = (diff > 18).astype(np.uint8) * 255
    if mask.sum() == 0:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 只保留最大连通域
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n > 1:
        best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == best, 255, 0).astype(np.uint8)

    return rgb, mask


def crop_fg(rgb: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("empty mask after extraction")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def resize_with_mask(rgb: np.ndarray, mask: np.ndarray, scale: float):
    h, w = rgb.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    rgb2 = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    mask2 = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return rgb2, mask2


def rotate_with_mask(rgb: np.ndarray, mask: np.ndarray, angle_deg: float):
    h, w = rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy

    rgb2 = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    mask2 = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    return rgb2, mask2


def match_local_brightness(defect_rgb: np.ndarray, defect_mask: np.ndarray, bg_patch: np.ndarray):
    out = defect_rgb.copy().astype(np.float32)
    m = defect_mask > 0
    if m.sum() < 10:
        return defect_rgb

    defect_lab = cv2.cvtColor(defect_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    dL = defect_lab[:, :, 0][m]
    bL = bg_lab[:, :, 0].reshape(-1)

    mean_d = float(dL.mean())
    std_d = float(dL.std() + 1e-6)
    mean_b = float(bL.mean())
    std_b = float(bL.std() + 1e-6)

    defect_lab[:, :, 0][m] = np.clip((defect_lab[:, :, 0][m] - mean_d) / std_d * std_b + mean_b, 0, 255)
    out = cv2.cvtColor(defect_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def alpha_paste(bg: np.ndarray, defect_rgb: np.ndarray, defect_mask: np.ndarray, x0: int, y0: int, feather_sigma: float = 1.8):
    out = bg.copy()
    h, w = defect_rgb.shape[:2]
    H, W = bg.shape[:2]

    x1 = min(W, x0 + w)
    y1 = min(H, y0 + h)
    x0c = max(0, x0)
    y0c = max(0, y0)

    if x1 <= x0c or y1 <= y0c:
        return out, np.zeros((H, W), dtype=np.uint8)

    sx0 = x0c - x0
    sy0 = y0c - y0
    sx1 = sx0 + (x1 - x0c)
    sy1 = sy0 + (y1 - y0c)

    fg = defect_rgb[sy0:sy1, sx0:sx1].astype(np.float32)
    mk = defect_mask[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0

    if feather_sigma > 0:
        mk = cv2.GaussianBlur(mk, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
        mk = np.clip(mk, 0.0, 1.0)

    roi = out[y0c:y1, x0c:x1].astype(np.float32)
    blend = roi * (1.0 - mk[..., None]) + fg * mk[..., None]
    out[y0c:y1, x0c:x1] = np.clip(blend, 0, 255).astype(np.uint8)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0c:y1, x0c:x1] = (mk * 255).astype(np.uint8)
    full_mask = (full_mask > 20).astype(np.uint8) * 255
    return out, full_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_dir", type=str, required=True)
    parser.add_argument("--defect_dir", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    bg_dir = Path(args.bg_dir)
    defect_dir = Path(args.defect_dir)
    out_root = Path(args.out_root)
    out_img_dir = out_root / "images"
    out_mask_dir = out_root / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    bg_paths = list_images(bg_dir)
    defect_paths = list_images(defect_dir)

    if len(bg_paths) == 0:
        raise RuntimeError(f"no backgrounds found in {bg_dir}")
    if len(defect_paths) == 0:
        raise RuntimeError(f"no defects found in {defect_dir}")

    for i in range(args.num_samples):
        bg_path = bg_paths[i % len(bg_paths)]
        defect_path = random.choice(defect_paths)

        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg is None:
            print(f"[SKIP] bg read fail: {bg_path}")
            continue

        dimg = read_image_any(defect_path)
        try:
            defect_rgb, defect_mask = derive_mask_and_rgb(dimg)
            defect_rgb, defect_mask = crop_fg(defect_rgb, defect_mask)
        except Exception as e:
            print(f"[SKIP] defect fail: {defect_path.name} | {e}")
            continue

        H, W = bg.shape[:2]
        dh, dw = defect_rgb.shape[:2]

        # 尺度控制：偏向中小缺陷
        target_long = random.randint(48, 140)
        scale = target_long / max(dh, dw)
        defect_rgb, defect_mask = resize_with_mask(defect_rgb, defect_mask, scale)

        # 轻微旋转
        angle = random.uniform(-25, 25)
        defect_rgb, defect_mask = rotate_with_mask(defect_rgb, defect_mask, angle)

        dh, dw = defect_rgb.shape[:2]
        if dh >= H or dw >= W:
            scale2 = min((H - 10) / max(dh, 1), (W - 10) / max(dw, 1), 1.0)
            defect_rgb, defect_mask = resize_with_mask(defect_rgb, defect_mask, scale2)
            dh, dw = defect_rgb.shape[:2]

        # 随机位置
        x0 = random.randint(0, max(0, W - dw))
        y0 = random.randint(0, max(0, H - dh))

        bg_patch = bg[y0:y0+dh, x0:x0+dw].copy()
        defect_rgb = match_local_brightness(defect_rgb, defect_mask, bg_patch)

        pasted, mask_full = alpha_paste(bg, defect_rgb, defect_mask, x0, y0, feather_sigma=1.8)

        stem = f"paste_{i:03d}"
        cv2.imwrite(str(out_img_dir / f"{stem}.png"), pasted)
        cv2.imwrite(str(out_mask_dir / f"{stem}.png"), mask_full)
        print(f"[OK] {stem} | bg={bg_path.name} | defect={defect_path.name}")

    print(f"\nDone. saved to: {out_root}")


if __name__ == "__main__":
    main()
