from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def keep_largest_component(bin_mask: np.ndarray) -> np.ndarray:
    bin01 = (bin_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if num <= 1:
        return bin01 * 255
    best_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ((labels == best_idx).astype(np.uint8) * 255)

def smooth_inside_coarse(trimmed_u8: np.ndarray, coarse_u8: np.ndarray) -> np.ndarray:
    M = ((coarse_u8 > 127).astype(np.uint8) * 255)
    X = ((trimmed_u8 > 0).astype(np.uint8) * 255)

    X = cv2.bitwise_and(X, M)
    X = keep_largest_component(X)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 第一轮：去毛刺 + 圆滑化
    X = cv2.morphologyEx(X, cv2.MORPH_CLOSE, k_close)
    X = cv2.morphologyEx(X, cv2.MORPH_OPEN,  k_open)

    soft = cv2.GaussianBlur(X.astype(np.float32), (0, 0), sigmaX=2.4, sigmaY=2.4)
    X = ((soft >= 118).astype(np.uint8) * 255)

    # 第二轮：再钝化一点
    X = cv2.bitwise_and(X, M)
    X = cv2.morphologyEx(X, cv2.MORPH_CLOSE, k_close)
    soft2 = cv2.GaussianBlur(X.astype(np.float32), (0, 0), sigmaX=1.6, sigmaY=1.6)
    X = ((soft2 >= 124).astype(np.uint8) * 255)

    X = cv2.bitwise_and(X, M)
    X = keep_largest_component(X)
    return X

def mask_to_polygons(mask_u8: np.ndarray, min_area: float = 20.0, eps_ratio: float = 0.010):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 3:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(1e-6, eps_ratio * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] >= 3:
            polys.append(pts)
    polys.sort(key=lambda x: cv2.contourArea(x.astype(np.float32)), reverse=True)
    return polys[:1]

def write_yoloseg_label(txt_path: Path, polys, class_id: int, w: int, h: int):
    lines = []
    for pts in polys:
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts])
        lines.append(f"{class_id} {coords}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

def overlay_fill(img_bgr: np.ndarray, mask_u8: np.ndarray,
                 fill=(80, 120, 255), alpha=0.68, edge=(255, 255, 255), edge_th=3):
    out = img_bgr.copy()
    m = mask_u8 > 0
    color = np.zeros_like(out, dtype=np.uint8)
    color[:] = fill
    out = np.where(m[..., None], (out * (1 - alpha) + color * alpha).astype(np.uint8), out)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, edge, edge_th, cv2.LINE_AA)
    return out

def add_title(img_bgr: np.ndarray, text: str):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (18, 18, 18), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2, cv2.LINE_AA)
    return out

def make_mask_only(mask_u8: np.ndarray) -> np.ndarray:
    canvas = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    canvas[mask_u8 > 0] = (220, 220, 220)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas

def crop_zoom(img_bgr: np.ndarray, mask_u8: np.ndarray, pad: int = 28, out_side: int = 480) -> np.ndarray:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return cv2.resize(img_bgr, (out_side, out_side), interpolation=cv2.INTER_CUBIC)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img_bgr.shape[1] - 1, x1 + pad)
    y1 = min(img_bgr.shape[0] - 1, y1 + pad)

    crop = img_bgr[y0:y1+1, x0:x1+1]
    h, w = crop.shape[:2]
    scale = min(out_side / max(w, 1), out_side / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((out_side, out_side, 3), 18, dtype=np.uint8)
    xx = (out_side - nw) // 2
    yy = (out_side - nh) // 2
    canvas[yy:yy+nh, xx:xx+nw] = crop
    return canvas

def letterbox(img: np.ndarray, side: int = 320, bg=(18, 18, 18)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(side / max(w, 1), side / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((side, side, 3), bg, dtype=np.uint8)
    x0 = (side - nw) // 2
    y0 = (side - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def build_contact_sheet(image_paths, out_path: Path, cols: int = 4, rows: int = 4, cell_side: int = 320):
    per_board = cols * rows
    chunk = image_paths[:per_board]
    board = np.full((rows * cell_side, cols * cell_side, 3), 18, dtype=np.uint8)
    for i, p in enumerate(chunk):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        tile = letterbox(img, side=cell_side)
        r = i // cols
        c = i % cols
        y0 = r * cell_side
        x0 = c * cell_side
        board[y0:y0+cell_side, x0:x0+cell_side] = tile
    cv2.imwrite(str(out_path), board)

def load_class_id(meta_path: Path, single_class: bool) -> int:
    if single_class:
        return 0
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        defect_type = str(meta.get("defect_type", "")).strip()
    except Exception:
        defect_type = ""
    mapping = {"Spalling": 0, "Crack": 1}
    return mapping.get(defect_type, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", type=str, required=True)
    ap.add_argument("--trimmed-root", type=str, required=True)
    ap.add_argument("--single-class", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.68)
    ap.add_argument("--poly-eps-ratio", type=float, default=0.010)
    args = ap.parse_args()

    export_root = Path(args.export_root)
    trimmed_root = Path(args.trimmed_root)

    polished_dir   = trimmed_root / "polished_masks"
    showcase_dir   = trimmed_root / "showcase_strong"
    maskonly_dir   = trimmed_root / "mask_only"
    zoom_dir       = trimmed_root / "zoom_showcase"
    overview_dir   = trimmed_root / "overview_v2"
    labels_train   = trimmed_root / "labels" / "train"
    labels_val     = trimmed_root / "labels" / "val"
    images_train   = trimmed_root / "images" / "train"
    images_val     = trimmed_root / "images" / "val"

    for p in [polished_dir, showcase_dir, maskonly_dir, zoom_dir, overview_dir]:
        ensure_dir(p)

    preview_list = []
    showcase_list = []
    zoom_list = []

    sample_dirs = sorted([p for p in export_root.iterdir() if p.is_dir() and p.name.startswith("sample_")])

    count = 0
    for sdir in sample_dirs:
        name = sdir.name
        defect_path = sdir / "defect_hq.png"
        coarse_path = sdir / "mask.png"
        meta_path   = sdir / "meta.json"

        old_mask = trimmed_root / "refined_masks" / f"{name}.png"
        if not old_mask.exists():
            continue

        defect = cv2.imread(str(defect_path), cv2.IMREAD_COLOR)
        coarse = cv2.imread(str(coarse_path), cv2.IMREAD_GRAYSCALE)
        trimmed = cv2.imread(str(old_mask), cv2.IMREAD_GRAYSCALE)
        if defect is None or coarse is None or trimmed is None:
            continue

        polished = smooth_inside_coarse(trimmed, coarse)
        cv2.imwrite(str(polished_dir / f"{name}.png"), polished)

        polys = mask_to_polygons(polished, eps_ratio=args.poly_eps_ratio)
        class_id = load_class_id(meta_path, args.single_class)

        train_label = labels_train / f"{name}.txt"
        val_label   = labels_val / f"{name}.txt"
        out_label = train_label if train_label.exists() else val_label
        if polys:
            write_yoloseg_label(out_label, polys, class_id, defect.shape[1], defect.shape[0])

        strong = overlay_fill(defect, polished, alpha=args.alpha, edge_th=3)
        strong = add_title(strong, f"{name}")
        mask_only = add_title(make_mask_only(polished), f"{name}")
        zoom = add_title(crop_zoom(strong, polished, pad=30, out_side=520), f"{name}")

        p1 = showcase_dir / f"{name}.jpg"
        p2 = maskonly_dir / f"{name}.jpg"
        p3 = zoom_dir / f"{name}.jpg"
        cv2.imwrite(str(p1), strong)
        cv2.imwrite(str(p2), mask_only)
        cv2.imwrite(str(p3), zoom)

        showcase_list.append(p1)
        preview_list.append(p2)
        zoom_list.append(p3)
        count += 1

    build_contact_sheet(showcase_list, overview_dir / "showcase_board_001.jpg", cols=4, rows=4, cell_side=320)
    build_contact_sheet(preview_list,  overview_dir / "maskonly_board_001.jpg", cols=4, rows=4, cell_side=320)
    build_contact_sheet(zoom_list,     overview_dir / "zoom_board_001.jpg", cols=4, rows=4, cell_side=320)

    print("=" * 60)
    print("done")
    print("processed    :", count)
    print("polished_dir :", polished_dir)
    print("showcase_dir :", showcase_dir)
    print("maskonly_dir :", maskonly_dir)
    print("zoom_dir     :", zoom_dir)
    print("overview_dir :", overview_dir)
    print("=" * 60)

if __name__ == "__main__":
    main()
