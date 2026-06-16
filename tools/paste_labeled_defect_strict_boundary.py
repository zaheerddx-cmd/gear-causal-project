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
        raise FileNotFoundError(f"read failed: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def ensure_bin(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8) * 255)


def polygon_to_mask(h: int, w: int, points):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_mask_from_json(js, h: int, w: int, class_name: str | None = None) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1) COCO 风格：annotations -> segmentation
    if isinstance(js, dict) and "annotations" in js and isinstance(js["annotations"], list):
        found = False
        for ann in js["annotations"]:
            seg = ann.get("segmentation", None)
            if seg is None:
                continue

            if isinstance(seg, list):
                for poly in seg:
                    if not isinstance(poly, list):
                        continue
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(mask, [pts_i], 255)
                    found = True

        if not found:
            raise RuntimeError("annotations 存在，但没有解析到有效 segmentation polygon")
        return ensure_bin(mask)

    # 2) LabelMe 风格兜底
    if isinstance(js, dict) and "shapes" in js:
        found = False
        for shp in js["shapes"]:
            pts = shp.get("points", None)
            if pts and len(pts) >= 3:
                pts = np.array(pts, dtype=np.float32)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(mask, [pts_i], 255)
                found = True
        if not found:
            raise RuntimeError("shapes 存在，但没有解析到有效 points")
        return ensure_bin(mask)

    raise RuntimeError("不支持的 JSON 结构；当前只支持 COCO-style annotations.segmentation 或 LabelMe shapes.points")

    anns = None
    if isinstance(js, list):
        anns = js
    elif isinstance(js, dict):
        for k in ["annotations", "objects", "labels", "items", "instances"]:
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
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
                continue
            keys = ann.keys()
            if {"x", "y", "width", "height"}.issubset(keys):
                x1, y1 = int(round(ann["x"])), int(round(ann["y"]))
                x2 = int(round(ann["x"] + ann["width"]))
                y2 = int(round(ann["y"] + ann["height"]))
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
                continue

    return ensure_bin(mask)


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_by_mask(img: np.ndarray, mask: np.ndarray):
    bbox = bbox_from_mask(mask)
    if bbox is None:
        raise RuntimeError("mask empty")
    x1, y1, x2, y2 = bbox
    patch = img[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2].copy()
    return patch, patch_mask, bbox


def make_boundary(mask: np.ndarray, edge_px: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_px + 1, 2 * edge_px + 1))
    ero = cv2.erode(mask, k, iterations=1)
    boundary = ((mask > 0) & (ero == 0)).astype(np.uint8) * 255
    return boundary


def make_boundary_band(mask: np.ndarray, dilate_px: int = 9, erode_px: int = 1) -> np.ndarray:
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    dil = cv2.dilate(mask, kd, iterations=1)
    ero = cv2.erode(mask, ke, iterations=1)
    band = ((dil > 0) & (ero == 0)).astype(np.uint8) * 255
    return band


def collect_bg_factuals(bg_root: Path):
    paths = []
    if bg_root.is_file():
        return [bg_root]
    for p in bg_root.rglob("*"):
        if p.is_file() and p.name == "factual.png":
            paths.append(p)
    if not paths:
        for p in bg_root.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(p)
    return sorted(paths)


def paste_hard(bg: np.ndarray, patch: np.ndarray, patch_mask: np.ndarray,
               scale_range=(0.95, 1.20), rotate_deg=5):
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
    mask_rt = ensure_bin(mask_rt)

    bbox = bbox_from_mask(mask_rt)
    if bbox is None:
        raise RuntimeError("rotated mask empty")
    x1m, y1m, x2m, y2m = bbox
    patch_rt = patch_rt[y1m:y2m, x1m:x2m]
    mask_rt = mask_rt[y1m:y2m, x1m:x2m]

    nh, nw = patch_rt.shape[:2]
    if nh >= H or nw >= W:
        scale2 = min((H - 4) / max(nh, 1), (W - 4) / max(nw, 1), 1.0)
        nh2 = max(4, int(round(nh * scale2)))
        nw2 = max(4, int(round(nw * scale2)))
        patch_rt = cv2.resize(patch_rt, (nw2, nh2), interpolation=cv2.INTER_LANCZOS4)
        mask_rt = cv2.resize(mask_rt, (nw2, nh2), interpolation=cv2.INTER_NEAREST)
        mask_rt = ensure_bin(mask_rt)
        nh, nw = patch_rt.shape[:2]

    x1 = random.randint(0, max(0, W - nw))
    y1 = random.randint(0, max(0, H - nh))
    x2, y2 = x1 + nw, y1 + nh

    out = bg.copy()
    roi = out[y1:y2, x1:x2]
    m = (mask_rt > 0)
    roi[m] = patch_rt[m]
    out[y1:y2, x1:x2] = roi

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask_rt
    return out, full_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-image", required=True)
    ap.add_argument("--src-json", required=True)
    ap.add_argument("--bg-root", required=True, help="刚才大模型生成背景的根目录，可递归找 factual.png")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_image_p = Path(args.src_image)
    src_json_p = Path(args.src_json)
    bg_root = Path(args.bg_root)
    out_root = Path(args.out_root)

    src = imread_rgb(src_image_p)
    H, W = src.shape[:2]
    js = load_json(src_json_p)

    strict_mask = extract_mask_from_json(js, H, W, args.class_name)
    if np.count_nonzero(strict_mask) == 0:
        raise RuntimeError("解析不到有效 mask，请检查 json 结构或 class-name")

    patch, patch_mask, _ = crop_by_mask(src, strict_mask)

    bg_paths = collect_bg_factuals(bg_root)
    if not bg_paths:
        raise RuntimeError(f"找不到背景: {bg_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    imwrite_rgb(out_root / "source_patch.png", patch)
    cv2.imwrite(str(out_root / "source_patch_mask.png"), patch_mask)

    for i in range(args.num):
        bg_img = imread_rgb(random.choice(bg_paths))
        comp, full_mask = paste_hard(
            bg_img, patch, patch_mask,
            scale_range=(0.95, 1.20),
            rotate_deg=5,
        )

        strict_boundary = make_boundary(full_mask, edge_px=1)
        boundary_band = make_boundary_band(full_mask, dilate_px=9, erode_px=1)

        sdir = out_root / f"sample_{i:06d}"
        sdir.mkdir(parents=True, exist_ok=True)

        imwrite_rgb(sdir / "factual.png", bg_img)
        imwrite_rgb(sdir / "initial_composite.png", comp)
        imwrite_rgb(sdir / "defect_patch.png", patch)
        cv2.imwrite(str(sdir / "strict_mask.png"), full_mask)
        cv2.imwrite(str(sdir / "boundary.png"), strict_boundary)
        cv2.imwrite(str(sdir / "boundary_band.png"), boundary_band)

        # 给后续模型重绘的说明
        with open(sdir / "redraw_prompt.txt", "w", encoding="utf-8") as f:
            f.write(
                "Edit initial_composite.png using factual.png as background reference.\n"
                "Preserve the defect region inside strict_mask.png.\n"
                "Only refine the transition and local realism near boundary_band.png.\n"
                "Do not move the defect, do not shrink or expand beyond strict_mask.png.\n"
                "Keep the exact boundary shape defined by strict_mask.png and boundary.png.\n"
                "Make the pasted defect blend naturally with the background, but do not blur away the boundary.\n"
            )

    print(f"[OK] done: {out_root}")
    print(f"[OK] backgrounds used from: {bg_root}")
    print(f"[OK] bg count: {len(bg_paths)}")


if __name__ == "__main__":
    main()
