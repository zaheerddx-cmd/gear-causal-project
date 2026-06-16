from __future__ import annotations
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))

def _load_gray(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))

def _save_rgb(p: Path, arr: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(p)

def _save_gray(p: Path, arr: np.ndarray):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(p)

def _bin_mask(mask_u8: np.ndarray) -> np.ndarray:
    return ((mask_u8 > 127).astype(np.uint8) * 255)

def _mask_ratio(mask_u8: np.ndarray) -> float:
    return float(np.count_nonzero(mask_u8 > 127)) / float(mask_u8.shape[0] * mask_u8.shape[1])

def _mask_center(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 127)
    if len(xs) == 0:
        h, w = mask_u8.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())

def _largest_contour(mask_u8: np.ndarray):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    return cnt

def _contour_to_seg_line(mask_u8: np.ndarray, class_id: int = 0, eps_ratio: float = 0.006) -> str:
    mask_u8 = _bin_mask(mask_u8)
    h, w = mask_u8.shape
    cnt = _largest_contour(mask_u8)
    if cnt is None or len(cnt) < 3:
        return ""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, max(1.0, peri * eps_ratio), True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        pts = cnt.reshape(-1, 2)
    nums = []
    for x, y in pts:
        nums.append(f"{x / w:.6f}")
        nums.append(f"{y / h:.6f}")
    return f"{class_id} " + " ".join(nums)

def _bbox_from_mask(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 127)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1

def _bbox_to_yolo_line(bbox, shape_hw, class_id: int = 0) -> str:
    if bbox is None:
        return ""
    h, w = shape_hw
    x0, y0, x1, y1 = bbox
    xc = ((x0 + x1) / 2.0) / w
    yc = ((y0 + y1) / 2.0) / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def _overlay_mask(img_rgb: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    out = img_rgb.copy()
    m = mask_u8 > 127
    fill = np.zeros_like(out, dtype=np.uint8)
    fill[:] = (230, 120, 70)
    out = np.where(m[..., None], (out * (1 - alpha) + fill * alpha).astype(np.uint8), out)
    return out

def _draw_bbox(img_rgb: np.ndarray, bbox) -> np.ndarray:
    out = img_rgb.copy()
    if bbox is None:
        return out
    x0, y0, x1, y1 = bbox
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 90, 90), 2)
    return out

def _warp_with_scale(arr: np.ndarray, cx: float, cy: float, scale: float, interp, border_value):
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), 0.0, scale)
    return cv2.warpAffine(arr, M, (w, h), flags=interp, borderValue=border_value)

def _post_scale_one(factual_rgb, defect_rgb, mask_u8, scale: float, dilate_ks: int, close_ks: int):
    mask = _bin_mask(mask_u8)
    h, w = mask.shape
    cx, cy = _mask_center(mask)

    # 1) 先放大 mask
    scaled_mask = _warp_with_scale(mask, cx, cy, scale, cv2.INTER_NEAREST, 0)

    if dilate_ks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ks, dilate_ks))
        scaled_mask = cv2.dilate(scaled_mask, kernel, iterations=1)

    if close_ks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        scaled_mask = cv2.morphologyEx(scaled_mask, cv2.MORPH_CLOSE, kernel)

    scaled_mask = _bin_mask(scaled_mask)

    # 2) 只放大缺陷残差，不动背景
    residual = defect_rgb.astype(np.float32) - factual_rgb.astype(np.float32)
    residual = residual * (mask[..., None] / 255.0)
    warped_res = np.zeros_like(residual, dtype=np.float32)
    for c in range(3):
        warped_res[..., c] = _warp_with_scale(residual[..., c], cx, cy, scale, cv2.INTER_LINEAR, 0)

    # 3) 合成
    alpha = cv2.GaussianBlur((scaled_mask > 127).astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    candidate = factual_rgb.astype(np.float32) + warped_res
    candidate = np.clip(candidate, 0, 255)

    out = factual_rgb.astype(np.float32) * (1.0 - alpha[..., None]) + candidate * alpha[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, scaled_mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--target-min", type=float, default=0.045)
    ap.add_argument("--target-max", type=float, default=0.085)
    ap.add_argument("--min-scale", type=float, default=1.20)
    ap.add_argument("--max-scale", type=float, default=2.00)
    ap.add_argument("--dilate-ks", type=int, default=11)
    ap.add_argument("--close-ks", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-all", action="store_true", help="所有样本都处理；默认只处理面积较小的")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(src_root)

    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

    defect_root = dst_root / "defect"
    yolo_root = dst_root / "yolo_dual_export"

    done = 0
    skipped = 0

    for sdir in sorted(defect_root.iterdir()):
        if not (sdir.is_dir() and sdir.name.startswith("sample_")):
            continue

        stem = sdir.name
        factual_p = sdir / "factual.png"
        lq_p = sdir / "defect_lq.png"
        hq_p = sdir / "defect_hq.png"
        mask_p = sdir / "mask.png"

        if not (factual_p.exists() and lq_p.exists() and mask_p.exists()):
            skipped += 1
            continue

        factual = _load_rgb(factual_p)
        defect_lq = _load_rgb(lq_p)
        mask = _bin_mask(_load_gray(mask_p))

        cur_ratio = _mask_ratio(mask)

        if (not args.force_all) and cur_ratio >= args.target_min:
            skipped += 1
            continue

        low = max(args.target_min, cur_ratio * 1.35)
        high = max(low + 1e-4, args.target_max)
        target_ratio = random.uniform(low, high)

        if cur_ratio <= 1e-6:
            skipped += 1
            continue

        scale = math.sqrt(target_ratio / cur_ratio)
        scale = max(args.min_scale, min(args.max_scale, scale))

        new_img, new_mask = _post_scale_one(
            factual_rgb=factual,
            defect_rgb=defect_lq,
            mask_u8=mask,
            scale=scale,
            dilate_ks=args.dilate_ks,
            close_ks=args.close_ks,
        )

        # sample 内：以 defect_lq 为标准，hq 同步
        _save_rgb(lq_p, new_img)
        _save_rgb(hq_p, new_img)
        _save_gray(mask_p, new_mask)

        # bbox / seg
        bbox = _bbox_from_mask(new_mask)
        det_line = _bbox_to_yolo_line(bbox, new_mask.shape, class_id=0)
        seg_line = _contour_to_seg_line(new_mask, class_id=0)

        # yolo_dual_export 同步
        img_p = yolo_root / "images" / f"{stem}.png"
        if img_p.parent.exists():
            _save_rgb(img_p, new_img)

        for msub in ["masks_final", "masks", "masks_coarse"]:
            mp = yolo_root / msub / f"{stem}.png"
            if mp.parent.exists():
                _save_gray(mp, new_mask)

        det_p = yolo_root / "labels_det" / f"{stem}.txt"
        if det_p.parent.exists():
            det_p.parent.mkdir(parents=True, exist_ok=True)
            det_p.write_text((det_line + "\n") if det_line else "", encoding="utf-8")

        seg_p = yolo_root / "labels_seg" / f"{stem}.txt"
        if seg_p.parent.exists():
            seg_p.parent.mkdir(parents=True, exist_ok=True)
            seg_p.write_text((seg_line + "\n") if seg_line else "", encoding="utf-8")

        prev_seg = yolo_root / "preview_seg" / f"{stem}.png"
        if prev_seg.parent.exists():
            _save_rgb(prev_seg, _overlay_mask(new_img, new_mask, alpha=0.30))

        prev_det = yolo_root / "preview_det" / f"{stem}.png"
        if prev_det.parent.exists():
            _save_rgb(prev_det, _draw_bbox(new_img, bbox))

        meta_p = yolo_root / "meta" / f"{stem}.json"
        meta = {
            "stem": stem,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "area_ratio": _mask_ratio(new_mask),
            "post_scale": scale,
            "source": "post_scaleup_spalling.py",
        }
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        done += 1
        if done % 50 == 0:
            print(f"[OK] processed {done}")

    print(f"[DONE] processed={done}, skipped={skipped}")
    print(f"[OUT] {dst_root}")

if __name__ == "__main__":
    main()
