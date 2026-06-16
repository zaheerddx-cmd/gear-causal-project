from __future__ import annotations
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REQUIRED = ["defect_hq.png", "defect_lq.png", "factual.png", "mask.png", "meta.json"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_valid_sample_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("sample_") and all((p / x).exists() for x in REQUIRED)

def get_samples(root: Path):
    return sorted([p for p in root.iterdir() if is_valid_sample_dir(p)])

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

def mask_to_polygons(mask_u8: np.ndarray, min_area: float = 20.0, largest_only: bool = True):
    _, bin_mask = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    items = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area and len(c) >= 3:
            items.append((area, c))

    if not items:
        return []

    items.sort(key=lambda x: x[0], reverse=True)
    if largest_only:
        items = [items[0]]

    polys = []
    for _, c in items:
        peri = cv2.arcLength(c, True)
        eps = max(1e-6, 0.0025 * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            pts = c.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 3:
            polys.append(pts)
    return polys

def write_yoloseg_label(txt_path: Path, polys, class_id: int, w: int, h: int):
    lines = []
    for pts in polys:
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts])
        lines.append(f"{class_id} {coords}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

def draw_preview(img_bgr: np.ndarray, polys, class_id: int):
    out = img_bgr.copy()
    for pts in polys:
        pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts_i], True, (0, 255, 0), 2)
        x, y = pts_i[0, 0]
        cv2.putText(out, f"cls {class_id}", (int(x), max(20, int(y))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--image-name", type=str, default="defect_hq.png",
                    choices=["defect_hq.png", "defect_lq.png", "factual.png"])
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single-class", action="store_true")
    ap.add_argument("--largest-only", action="store_true")
    ap.add_argument("--save-preview", action="store_true")
    args = ap.parse_args()

    export_root = Path(args.export_root)
    out_root = Path(args.out_root)

    samples = get_samples(export_root)
    if not samples:
        raise RuntimeError(f"没有找到有效 sample 目录: {export_root}")

    random.seed(args.seed)
    random.shuffle(samples)

    n_total = len(samples)
    n_val = max(1, int(round(n_total * args.val_ratio))) if n_total > 1 else 0
    val_names = set(p.name for p in samples[:n_val])

    for split in ["train", "val"]:
        ensure_dir(out_root / "images" / split)
        ensure_dir(out_root / "labels" / split)
    if args.save_preview:
        ensure_dir(out_root / "preview")

    converted = 0
    empty = 0

    for sample_dir in sorted(samples):
        split = "val" if sample_dir.name in val_names else "train"

        img_path = sample_dir / args.image_name
        mask_path = sample_dir / "mask.png"
        meta_path = sample_dir / "meta.json"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        h, w = mask.shape[:2]
        class_id = load_class_id(meta_path, args.single_class)
        polys = mask_to_polygons(mask, min_area=20.0, largest_only=args.largest_only)

        out_img = out_root / "images" / split / f"{sample_dir.name}.png"
        out_lab = out_root / "labels" / split / f"{sample_dir.name}.txt"
        shutil.copy2(img_path, out_img)

        if polys:
            write_yoloseg_label(out_lab, polys, class_id, w, h)
            converted += 1
            if args.save_preview:
                preview = draw_preview(img, polys, class_id)
                cv2.imwrite(str(out_root / "preview" / f"{sample_dir.name}.jpg"), preview)
        else:
            out_lab.write_text("", encoding="utf-8")
            empty += 1

    names = ["defect"] if args.single_class else ["Spalling", "Crack"]
    yaml_text = f"""path: {out_root}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(names):
        yaml_text += f"  {i}: {name}\n"
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    print("=" * 60)
    print("done")
    print("total     :", n_total)
    print("converted :", converted)
    print("empty     :", empty)
    print("yaml      :", out_root / "dataset.yaml")
    print("=" * 60)

if __name__ == "__main__":
    main()
