from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def imread_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"read image failed: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def ensure_gray3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    return img


def bin_mask(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8) * 255)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def polygon_to_mask(h: int, w: int, points):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_mask_from_json(js, h: int, w: int, class_name: str | None = None) -> np.ndarray:
    """
    兼容几种常见格式：
    1) LabelMe: {"shapes":[{"label":"...", "points":[[x,y],...]}]}
    2) 简单 list/dict annotation
    3) bbox: {"x","y","width","height"} / {"bbox":[x,y,w,h]}
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # LabelMe
    if isinstance(js, dict) and "shapes" in js:
        for shp in js["shapes"]:
            label = shp.get("label", "")
            if class_name is not None and label != class_name:
                continue
            pts = shp.get("points", None)
            if pts and len(pts) >= 3:
                mask = np.maximum(mask, polygon_to_mask(h, w, pts))
            elif "points" in shp and len(shp["points"]) == 2:
                (x1, y1), (x2, y2) = shp["points"]
                x1, x2 = sorted([int(round(x1)), int(round(x2))])
                y1, y2 = sorted([int(round(y1)), int(round(y2))])
                mask[y1:y2, x1:x2] = 255
        return mask

    # list annotations
    anns = None
    if isinstance(js, list):
        anns = js
    elif isinstance(js, dict):
        for k in ["annotations", "objects", "labels", "items"]:
            if k in js and isinstance(js[k], list):
                anns = js[k]
                break

    if anns is not None:
        for ann in anns:
            label = ann.get("label") or ann.get("class") or ann.get("name") or ""
            if class_name is not None and label != class_name:
                continue

            if "points" in ann and len(ann["points"]) >= 3:
                mask = np.maximum(mask, polygon_to_mask(h, w, ann["points"]))
                continue

            if "polygon" in ann and len(ann["polygon"]) >= 3:
                mask = np.maximum(mask, polygon_to_mask(h, w, ann["polygon"]))
                continue

            if "bbox" in ann and len(ann["bbox"]) >= 4:
                x, y, bw, bh = ann["bbox"][:4]
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + bw)), int(round(y + bh))
                mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 255
                continue

            keys = ann.keys()
            if {"x", "y", "width", "height"}.issubset(keys):
                x1, y1 = int(round(ann["x"])), int(round(ann["y"]))
                x2 = int(round(ann["x"] + ann["width"]))
                y2 = int(round(ann["y"] + ann["height"]))
                mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 255
                continue

    return mask


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_by_mask(img: np.ndarray, mask: np.ndarray):
    bbox = bbox_from_mask(mask)
    if bbox is None:
        raise RuntimeError("mask is empty")
    x1, y1, x2, y2 = bbox
    patch = img[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2].copy()
    return patch, patch_mask, bbox


def random_bg_paths(bg_input: Path):
    if bg_input.is_file():
        return [bg_input]
    paths = []
    for p in sorted(bg_input.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def paste_patch(bg: np.ndarray, patch: np.ndarray, patch_mask: np.ndarray,
                scale_range=(0.9, 1.3), rotate_deg=8, feather_sigma=1.2):
    H, W = bg.shape[:2]
    ph, pw = patch.shape[:2]

    scale = random.uniform(*scale_range)
    angle = random.uniform(-rotate_deg, rotate_deg)

    nh = max(4, int(round(ph * scale)))
    nw = max(4, int(round(pw * scale)))

    patch_rs = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    mask_rs = cv2.resize(patch_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    patch_rt = cv2.warpAffine(patch_rs, M, (nw, nh), flags=cv2.INTER_LANCZOS4, borderValue=0)
    mask_rt = cv2.warpAffine(mask_rs, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    mask_rt = bin_mask(mask_rt)

    ys, xs = np.where(mask_rt > 0)
    if len(xs) == 0:
        raise RuntimeError("rotated mask empty")

    mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    patch_rt = patch_rt[my1:my2, mx1:mx2]
    mask_rt = mask_rt[my1:my2, mx1:mx2]
    nh, nw = patch_rt.shape[:2]

    if nh >= H or nw >= W:
        scale2 = min((H - 4) / max(nh, 1), (W - 4) / max(nw, 1), 1.0)
        nh2 = max(4, int(round(nh * scale2)))
        nw2 = max(4, int(round(nw * scale2)))
        patch_rt = cv2.resize(patch_rt, (nw2, nh2), interpolation=cv2.INTER_LANCZOS4)
        mask_rt = cv2.resize(mask_rt, (nw2, nh2), interpolation=cv2.INTER_NEAREST)
        mask_rt = bin_mask(mask_rt)
        nh, nw = patch_rt.shape[:2]

    x1 = random.randint(0, max(0, W - nw))
    y1 = random.randint(0, max(0, H - nh))
    x2, y2 = x1 + nw, y1 + nh

    out = bg.copy()
    alpha = mask_rt.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    base_roi = out[y1:y2, x1:x2].astype(np.float32)
    patch_roi = patch_rt.astype(np.float32)
    out[y1:y2, x1:x2] = np.clip(base_roi * (1 - alpha) + patch_roi * alpha, 0, 255).astype(np.uint8)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask_rt

    # 给大模型 refine 用：边界环
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(full_mask, k1, iterations=1)
    ero = cv2.erode(full_mask, k2, iterations=1)
    refine_mask = ((dil > 0) & (ero == 0)).astype(np.uint8) * 255

    return out, full_mask, refine_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-image", required=True)
    ap.add_argument("--src-json", required=True)
    ap.add_argument("--bg", required=True, help="背景图或背景目录")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_image_p = Path(args.src_image)
    src_json_p = Path(args.src_json)
    bg_p = Path(args.bg)
    out_root = Path(args.out_root)

    src = imread_rgb(src_image_p)
    H, W = src.shape[:2]
    js = load_json(src_json_p)

    mask = extract_mask_from_json(js, H, W, args.class_name)
    mask = bin_mask(mask)
    if np.count_nonzero(mask) == 0:
        raise RuntimeError("解析不到有效 mask，请检查 json 格式或 class-name")

    patch, patch_mask, bbox = crop_by_mask(src, mask)

    bg_list = random_bg_paths(bg_p)
    if not bg_list:
        raise RuntimeError(f"背景为空: {bg_p}")

    out_root.mkdir(parents=True, exist_ok=True)
    imwrite_rgb(out_root / "source_patch_preview.png", patch)
    cv2.imwrite(str(out_root / "source_patch_mask.png"), patch_mask)

    for i in range(args.num):
        bg_img = imread_rgb(random.choice(bg_list))
        comp, full_mask, refine_mask = paste_patch(
            bg_img, patch, patch_mask,
            scale_range=(0.9, 1.25),
            rotate_deg=6,
            feather_sigma=1.0,
        )
        sdir = out_root / f"sample_{i:06d}"
        sdir.mkdir(parents=True, exist_ok=True)

        imwrite_rgb(sdir / "factual.png", bg_img)
        imwrite_rgb(sdir / "initial_composite.png", comp)
        imwrite_rgb(sdir / "defect_patch.png", patch)
        cv2.imwrite(str(sdir / "mask.png"), full_mask)
        cv2.imwrite(str(sdir / "refine_mask.png"), refine_mask)

    print(f"[OK] done: {out_root}")


if __name__ == "__main__":
    main()
