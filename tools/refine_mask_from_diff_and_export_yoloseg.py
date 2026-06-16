from __future__ import annotations
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

REQ = ["defect_hq.png", "factual.png", "mask.png", "meta.json"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_valid_sample_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("sample_") and all((p / x).exists() for x in REQ)

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

def keep_best_component(bin_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num <= 1:
        return bin_mask

    ref = (ref_mask > 0).astype(np.uint8)
    best_idx = -1
    best_score = -1.0

    for i in range(1, num):
        comp = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        overlap = int((comp & ref).sum())
        score = overlap * 10.0 + area
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return np.zeros_like(bin_mask)
    return ((labels == best_idx).astype(np.uint8) * 255)

def refine_mask(defect_bgr: np.ndarray, factual_bgr: np.ndarray, coarse_mask_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coarse = ((coarse_mask_u8 > 127).astype(np.uint8) * 255)

    # 1) 用 coarse mask 扩张成搜索区，避免纯 diff 漫出去
    search = cv2.dilate(coarse, np.ones((21, 21), np.uint8), iterations=1)

    # 2) RGB + Lab-L 双通道差分
    diff_rgb = cv2.absdiff(defect_bgr, factual_bgr).astype(np.float32)
    diff_rgb_mean = diff_rgb.mean(axis=2)

    defect_lab = cv2.cvtColor(defect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    factual_lab = cv2.cvtColor(factual_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff_l = np.abs(defect_lab[..., 0] - factual_lab[..., 0])

    score = 0.55 * diff_l + 0.45 * diff_rgb_mean
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=1.2, sigmaY=1.2)

    # 3) 仅在搜索区里做阈值
    vals = score[search > 0]
    if vals.size < 20:
        return coarse, np.clip(score, 0, 255).astype(np.uint8)

    # 更稳一点：高分位阈值，而不是全图 Otsu
    thr = max(6.0, float(np.percentile(vals, 72)))
    cand = ((score >= thr) & (search > 0)).astype(np.uint8) * 255

    # 4) 保留和 coarse 最相关的主连通域
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = keep_best_component(cand, coarse)

    # 5) 和 coarse 求一个折中：避免太飘，也避免太胖
    coarse_erode = cv2.erode(coarse, np.ones((5, 5), np.uint8), iterations=1)
    fused = np.maximum(cand, coarse_erode)
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fused = keep_best_component(fused, coarse)

    return fused, np.clip(score, 0, 255).astype(np.uint8)

def mask_to_polygons(mask_u8: np.ndarray, min_area: float = 20.0):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or len(c) < 3:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(1e-6, 0.0025 * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            pts = c.reshape(-1, 2).astype(np.float32)
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

def overlay_fill(img_bgr: np.ndarray, mask_u8: np.ndarray, fill=(60, 80, 255), alpha=0.38, edge=(255, 255, 255)):
    out = img_bgr.copy()
    mask_bool = mask_u8 > 0

    color = np.zeros_like(out, dtype=np.uint8)
    color[:] = fill
    out = np.where(mask_bool[..., None], (out * (1 - alpha) + color * alpha).astype(np.uint8), out)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, edge, 2, cv2.LINE_AA)
    return out

def add_title(img_bgr: np.ndarray, text: str):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(out, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
    return out

def to_heat(score_u8: np.ndarray):
    return cv2.applyColorMap(score_u8, cv2.COLORMAP_TURBO)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single-class", action="store_true")
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

    ensure_dir(out_root / "preview")
    ensure_dir(out_root / "refined_masks")

    converted = 0
    empty = 0

    for sample_dir in sorted(samples):
        split = "val" if sample_dir.name in val_names else "train"

        defect_path = sample_dir / "defect_hq.png"
        factual_path = sample_dir / "factual.png"
        coarse_mask_path = sample_dir / "mask.png"
        meta_path = sample_dir / "meta.json"

        defect = cv2.imread(str(defect_path), cv2.IMREAD_COLOR)
        factual = cv2.imread(str(factual_path), cv2.IMREAD_COLOR)
        coarse = cv2.imread(str(coarse_mask_path), cv2.IMREAD_GRAYSCALE)

        if defect is None or factual is None or coarse is None:
            continue

        refined, score_u8 = refine_mask(defect, factual, coarse)
        polys = mask_to_polygons(refined, min_area=20.0)
        class_id = load_class_id(meta_path, args.single_class)

        out_img = out_root / "images" / split / f"{sample_dir.name}.png"
        out_lab = out_root / "labels" / split / f"{sample_dir.name}.txt"
        out_mask = out_root / "refined_masks" / f"{sample_dir.name}.png"

        shutil.copy2(defect_path, out_img)
        cv2.imwrite(str(out_mask), refined)

        if polys:
            write_yoloseg_label(out_lab, polys, class_id, defect.shape[1], defect.shape[0])
            converted += 1
        else:
            out_lab.write_text("", encoding="utf-8")
            empty += 1

        p1 = add_title(factual, "factual")
        p2 = add_title(defect, "defect_hq")
        p3 = add_title(overlay_fill(defect, ((coarse > 127).astype(np.uint8) * 255)), "coarse mask")
        p4 = add_title(overlay_fill(defect, refined), "refined mask")
        p5 = add_title(to_heat(score_u8), "diff heat")
        preview = cv2.hconcat([p1, p2, p3, p4, p5])
        cv2.imwrite(str(out_root / "preview" / f"{sample_dir.name}.jpg"), preview)

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
    print("preview   :", out_root / "preview")
    print("masks     :", out_root / "refined_masks")
    print("=" * 60)

if __name__ == "__main__":
    main()
