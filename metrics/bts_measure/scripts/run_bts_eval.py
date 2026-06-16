from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def find_image_by_stem(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_mask(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"failed to read mask: {mask_path}")
    return (m > 127).astype(np.uint8) * 255


def load_l_channel(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # OpenCV 的 L 通道范围是 0~255，做相对比较够用
    L = lab[:, :, 0].astype(np.float32)
    return L


def get_boundary(mask_u8: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    ero = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = cv2.subtract(dil, ero)
    return (boundary > 0).astype(np.uint8)


def signed_distance(mask_u8: np.ndarray) -> np.ndarray:
    fg = (mask_u8 > 0).astype(np.uint8)
    bg = 1 - fg
    dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
    # 外部为正、内部为负
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)


def bilinear_sample(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape[:2]
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
        return float("nan")

    x0 = int(math.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(math.floor(y))
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]

    val = (
        v00 * (1 - dx) * (1 - dy) +
        v01 * dx * (1 - dy) +
        v10 * (1 - dx) * dy +
        v11 * dx * dy
    )
    return float(val)


def profile_along_normal(L: np.ndarray, px: float, py: float, nx: float, ny: float, r: int):
    vals = []
    for s in range(-r, r + 1):
        x = px + s * nx
        y = py + s * ny
        v = bilinear_sample(L, x, y)
        vals.append(v)
    arr = np.array(vals, dtype=np.float32)
    if np.any(np.isnan(arr)):
        return None
    return arr


def count_sign_changes(g: np.ndarray, tau: float = 0.0) -> int:
    if tau > 0:
        g2 = g.copy()
        g2[np.abs(g2) < tau] = 0.0
    else:
        g2 = g

    s = np.sign(g2)
    n = 0
    for i in range(len(s) - 1):
        if s[i] == 0 or s[i + 1] == 0:
            continue
        if s[i] != s[i + 1]:
            n += 1
    return int(n)


def compute_bts_for_pair(
    image_path: Path,
    mask_path: Path,
    r: int = 5,
    alpha: float = 1.0,
    beta: float = 0.4,
    eps: float = 1e-6,
    tau: float = 1.0,
    boundary_stride: int = 2,
    sdf_blur_sigma: float = 1.0,
):
    L = load_l_channel(image_path)
    mask = load_mask(mask_path)

    boundary = get_boundary(mask)
    ys, xs = np.where(boundary > 0)
    if len(xs) == 0:
        return None

    sdf = signed_distance(mask)
    if sdf_blur_sigma > 0:
        sdf = cv2.GaussianBlur(sdf, (0, 0), sigmaX=sdf_blur_sigma, sigmaY=sdf_blur_sigma)

    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1, ksize=3)

    idxs = np.arange(0, len(xs), max(1, boundary_stride))
    point_scores = []
    valid_points = 0

    h, w = L.shape[:2]

    for idx in idxs:
        x = float(xs[idx])
        y = float(ys[idx])

        nx = float(gx[int(y), int(x)])
        ny = float(gy[int(y), int(x)])
        norm = math.sqrt(nx * nx + ny * ny)
        if norm < 1e-6:
            continue
        nx /= norm
        ny /= norm

        # 保证采样线不会出界
        if (x - r * nx < 0 or x - r * nx > w - 1 or
            x + r * nx < 0 or x + r * nx > w - 1 or
            y - r * ny < 0 or y - r * ny > h - 1 or
            y + r * ny < 0 or y + r * ny > h - 1):
            continue

        ell = profile_along_normal(L, x, y, nx, ny, r)
        if ell is None or len(ell) < 3:
            continue

        g = np.diff(ell)
        if len(g) < 2:
            continue
        h2 = np.diff(g)

        denom = float(np.sum(np.abs(g)) + eps)
        E_p = float(np.sum(np.abs(h2)) / denom)
        N_p = count_sign_changes(g, tau=tau)

        BTS_p = math.exp(-alpha * E_p - beta * N_p)
        point_scores.append(BTS_p)
        valid_points += 1

    if len(point_scores) == 0:
        return None

    point_scores = np.array(point_scores, dtype=np.float32)
    return {
        "image": image_path.name,
        "mask": mask_path.name,
        "bts": float(point_scores.mean()),
        "bts_std_point": float(point_scores.std()),
        "num_boundary_samples": int(valid_points),
    }


def eval_one_method(method_dir: Path, outputs_root: Path, args):
    images_dir = method_dir / "images"
    masks_dir = method_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        print(f"[SKIP] missing images/masks: {method_dir}")
        return None

    method_name = method_dir.name
    out_csv = outputs_root / f"{method_name}_per_image.csv"
    out_json = outputs_root / f"{method_name}_summary.json"

    rows = []
    for mask_path in sorted(masks_dir.iterdir()):
        if mask_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = mask_path.stem
        image_path = find_image_by_stem(images_dir, stem)
        if image_path is None:
            print(f"[WARN] image missing for mask: {mask_path.name}")
            continue

        result = compute_bts_for_pair(
            image_path=image_path,
            mask_path=mask_path,
            r=args.r,
            alpha=args.alpha,
            beta=args.beta,
            eps=args.eps,
            tau=args.tau,
            boundary_stride=args.boundary_stride,
            sdf_blur_sigma=args.sdf_blur_sigma,
        )
        if result is not None:
            rows.append(result)

    if len(rows) == 0:
        print(f"[WARN] no valid pairs for method: {method_name}")
        return None

    bts_vals = np.array([x["bts"] for x in rows], dtype=np.float32)
    summary = {
        "method": method_name,
        "num_images": int(len(rows)),
        "bts_mean": float(bts_vals.mean()),
        "bts_std": float(bts_vals.std()),
        "bts_median": float(np.median(bts_vals)),
        "bts_min": float(bts_vals.min()),
        "bts_max": float(bts_vals.max()),
        "params": {
            "r": args.r,
            "alpha": args.alpha,
            "beta": args.beta,
            "eps": args.eps,
            "tau": args.tau,
            "boundary_stride": args.boundary_stride,
            "sdf_blur_sigma": args.sdf_blur_sigma,
        },
    }

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "mask", "bts", "bts_std_point", "num_boundary_samples"]
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] {method_name:<12} "
        f"n={summary['num_images']:<4d} "
        f"BTS_mean={summary['bts_mean']:.4f} "
        f"BTS_std={summary['bts_std']:.4f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods_root",
        type=str,
        default="/root/autodl-tmp/Gear_Causal_Project_v2/metrics/bts_measure/inputs",
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="/root/autodl-tmp/Gear_Causal_Project_v2/metrics/bts_measure/outputs",
    )
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--boundary_stride", type=int, default=2)
    parser.add_argument("--sdf_blur_sigma", type=float, default=1.0)
    args = parser.parse_args()

    methods_root = Path(args.methods_root)
    outputs_root = Path(args.outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for method_dir in sorted(methods_root.iterdir()):
        if not method_dir.is_dir():
            continue
        s = eval_one_method(method_dir, outputs_root, args)
        if s is not None:
            summaries.append(s)

    if len(summaries) == 0:
        print("[WARN] no valid method results")
        return

    summary_csv = outputs_root / "bts_summary_all_methods.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "num_images", "bts_mean", "bts_std", "bts_median", "bts_min", "bts_max"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nsummary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
