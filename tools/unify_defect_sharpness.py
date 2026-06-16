from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim == 2:
        return img_rgb.astype(np.uint8)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def _bin_mask(mask_u8: np.ndarray) -> np.ndarray:
    return ((mask_u8 > 127).astype(np.uint8) * 255)


def _lap_energy(gray_u8: np.ndarray, mask_u8: np.ndarray) -> float:
    m = mask_u8 > 0
    if int(np.count_nonzero(m)) < 20:
        return 0.0
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap[m])))


def _feather_alpha(mask_u8: np.ndarray, feather_sigma: float) -> np.ndarray:
    a = (mask_u8.astype(np.float32) / 255.0)
    if feather_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    return np.clip(a, 0.0, 1.0)


def _blend_inside(base_rgb: np.ndarray, blurred_rgb: np.ndarray, mask_u8: np.ndarray, feather_sigma: float) -> np.ndarray:
    alpha = _feather_alpha(mask_u8, feather_sigma)
    out = base_rgb.astype(np.float32).copy()
    src = blurred_rgb.astype(np.float32)
    out = out * (1.0 - alpha[..., None]) + src * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_local_sharpness_rgb(
    defect_rgb: np.ndarray,
    factual_rgb: np.ndarray,
    mask_u8: np.ndarray,
    target_ratio: float = 1.02,
    ring_dilate_px: int = 14,
    ring_gap_px: int = 3,
    sigma_candidates: tuple[float, ...] = (0.0, 0.35, 0.6, 0.9, 1.2, 1.5, 1.8),
    feather_sigma: float = 2.4,
) -> np.ndarray:
    mask = _bin_mask(mask_u8)
    if int(np.count_nonzero(mask)) < 30:
        return defect_rgb

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, 2 * ring_gap_px + 1), max(3, 2 * ring_gap_px + 1)))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, 2 * ring_dilate_px + 1), max(5, 2 * ring_dilate_px + 1)))

    inner = cv2.dilate(mask, k1, iterations=1)
    outer = cv2.dilate(mask, k2, iterations=1)
    ring = cv2.subtract(outer, inner)

    factual_gray = _to_gray(factual_rgb)
    defect_gray = _to_gray(defect_rgb)

    target_sharp = _lap_energy(factual_gray, ring)
    if target_sharp <= 1e-6:
        target_sharp = _lap_energy(defect_gray, ring)
    if target_sharp <= 1e-6:
        return defect_rgb

    desired = target_ratio * target_sharp

    best_img = defect_rgb
    best_err = float("inf")

    for sigma in sigma_candidates:
        if sigma <= 1e-8:
            cand = defect_rgb.copy()
        else:
            blurred = cv2.GaussianBlur(defect_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
            cand = _blend_inside(defect_rgb, blurred, mask, feather_sigma)

        cand_gray = _to_gray(cand)
        sharp = _lap_energy(cand_gray, mask)
        err = abs(sharp - desired)

        if err < best_err:
            best_err = err
            best_img = cand

    return best_img



def _degrade_from_hq(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    down = cv2.resize(img_rgb, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return up

def _overlay_mask(img_rgb: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    out = img_rgb.copy()
    mask = mask_u8 > 0
    fill = np.zeros_like(out, dtype=np.uint8)
    fill[:] = (230, 120, 70)
    out = np.where(mask[:, :, None], (out * (1 - alpha) + fill * alpha).astype(np.uint8), out)
    return out


def _redraw_preview_det(img_rgb: np.ndarray, bbox_xyxy) -> np.ndarray:
    out = img_rgb.copy()
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        return out
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 90, 90), 2)
    return out


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def process_root(root: Path, alpha: float, target_ratio: float):
    defect_root = root / "defect"
    yolo_root = root / "yolo_dual_export"

    sample_dirs = sorted([p for p in defect_root.iterdir() if p.is_dir() and p.name.startswith("sample_")]) if defect_root.exists() else []
    done = 0

    for sdir in sample_dirs:
        stem = sdir.name
        factual_p = sdir / "factual.png"
        defect_hq_p = sdir / "defect_hq.png"
        defect_lq_p = sdir / "defect_lq.png"
        mask_p = sdir / "mask.png"

        if not (factual_p.exists() and defect_hq_p.exists() and mask_p.exists()):
            continue

        try:
            factual = np.array(Image.open(factual_p).convert("RGB"))
            defect_hq = np.array(Image.open(defect_hq_p).convert("RGB"))
            mask = np.array(Image.open(mask_p).convert("L"))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"[WARN] skip unreadable core files: {sdir} | {e}")
            continue

        new_hq = normalize_local_sharpness_rgb(
            defect_hq, factual, mask,
            target_ratio=target_ratio,
            ring_dilate_px=14,
            ring_gap_px=3,
            sigma_candidates=(0.0, 0.35, 0.6, 0.9, 1.2, 1.5, 1.8),
            feather_sigma=2.4,
        )
        Image.fromarray(new_hq).save(defect_hq_p)

        if defect_lq_p.exists():
            try:
                defect_lq = np.array(Image.open(defect_lq_p).convert("RGB"))
                new_lq = normalize_local_sharpness_rgb(
                    defect_lq, factual, mask,
                    target_ratio=1.00,
                    ring_dilate_px=14,
                    ring_gap_px=3,
                    sigma_candidates=(0.0, 0.25, 0.45, 0.7, 1.0, 1.3),
                    feather_sigma=2.2,
                )
            except (UnidentifiedImageError, OSError, ValueError) as e:
                print(f"[WARN] rebuild broken defect_lq: {defect_lq_p} | {e}")
                new_lq = _degrade_from_hq(new_hq)
            Image.fromarray(new_lq).save(defect_lq_p)
        else:
            new_lq = _degrade_from_hq(new_hq)
            Image.fromarray(new_lq).save(defect_lq_p)

        # 同步 yolo 导出
        img_p = yolo_root / "images" / f"{stem}.png"
        if img_p.exists():
            Image.fromarray(new_hq).save(img_p)

        prev_seg_p = yolo_root / "preview_seg" / f"{stem}.png"
        if prev_seg_p.exists():
            Image.fromarray(_overlay_mask(new_hq, mask, alpha=alpha)).save(prev_seg_p)

        meta_p = yolo_root / "meta" / f"{stem}.json"
        prev_det_p = yolo_root / "preview_det" / f"{stem}.png"
        if prev_det_p.exists():
            meta = _load_json(meta_p) if meta_p.exists() else None
            bbox = meta.get("bbox_xyxy") if isinstance(meta, dict) else None
            Image.fromarray(_redraw_preview_det(new_hq, bbox)).save(prev_det_p)

        done += 1
        if done % 100 == 0:
            print(f"[OK] processed {done}")

    print(f"[DONE] total processed = {done}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="输出根目录，例如 .../defect_datasetpack_xxx")
    ap.add_argument("--alpha", type=float, default=0.30, help="preview_seg 透明度")
    ap.add_argument("--target-ratio", type=float, default=1.02, help="缺陷区目标清晰度 / 背景ring清晰度")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    process_root(root, alpha=args.alpha, target_ratio=args.target_ratio)


if __name__ == "__main__":
    main()
