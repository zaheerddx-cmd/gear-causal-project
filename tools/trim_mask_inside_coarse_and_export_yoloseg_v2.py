from __future__ import annotations
import argparse
import json
import math
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

def keep_largest_component(bin_mask: np.ndarray) -> np.ndarray:
    bin01 = (bin_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if num <= 1:
        return bin01 * 255
    best_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ((labels == best_idx).astype(np.uint8) * 255)

def compute_diff_score(defect_bgr: np.ndarray, factual_bgr: np.ndarray) -> np.ndarray:
    diff_rgb = cv2.absdiff(defect_bgr, factual_bgr).astype(np.float32)
    diff_rgb_mean = diff_rgb.mean(axis=2)

    defect_lab = cv2.cvtColor(defect_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    factual_lab = cv2.cvtColor(factual_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff_l = np.abs(defect_lab[..., 0] - factual_lab[..., 0])

    score = 0.55 * diff_l + 0.45 * diff_rgb_mean
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return score.astype(np.float32)

def smooth_mask_inside_coarse(mask_u8: np.ndarray, coarse_mask_u8: np.ndarray) -> np.ndarray:
    M = ((coarse_mask_u8 > 127).astype(np.uint8) * 255)
    X = ((mask_u8 > 0).astype(np.uint8) * 255)

    X = cv2.bitwise_and(X, M)
    X = keep_largest_component(X)

    # 先做一次形态学去毛刺
    X = cv2.morphologyEx(X, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    X = cv2.morphologyEx(X, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 再做轻微模糊+回阈值，让边界更顺
    soft = cv2.GaussianBlur(X.astype(np.float32), (0, 0), sigmaX=1.6, sigmaY=1.6)
    X = ((soft >= 110).astype(np.uint8) * 255)

    # 绝不允许超过 coarse
    X = cv2.bitwise_and(X, M)
    X = keep_largest_component(X)

    # 最后再压一遍
    X = cv2.morphologyEx(X, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    X = cv2.bitwise_and(X, M)
    X = keep_largest_component(X)
    return X

def refine_mask_inside_coarse(
    defect_bgr: np.ndarray,
    factual_bgr: np.ndarray,
    coarse_mask_u8: np.ndarray,
    core_ks: int = 5,
    percentile: float = 60.0,
    min_keep_ratio: float = 0.60,
):
    M = ((coarse_mask_u8 > 127).astype(np.uint8) * 255)
    if M.sum() == 0:
        return M, np.zeros_like(M), 1.0

    score = compute_diff_score(defect_bgr, factual_bgr)

    Core = cv2.erode(M, np.ones((core_ks, core_ks), np.uint8), iterations=1)
    Ring = cv2.subtract(M, Core)

    ring_vals = score[Ring > 0]
    if ring_vals.size < 10:
        Final = smooth_mask_inside_coarse(M, M)
        score_u8 = np.clip(score, 0, 255).astype(np.uint8)
        keep_ratio = float((Final > 0).sum()) / max(1, int((M > 0).sum()))
        return Final, score_u8, keep_ratio

    coarse_area = int((M > 0).sum())
    tried = [percentile, max(52.0, percentile - 8.0), max(46.0, percentile - 12.0)]
    best_final = None
    best_ratio = 0.0

    for p in tried:
        thr = max(6.0, float(np.percentile(ring_vals, p)))
        RingKeep = (((score >= thr) & (Ring > 0)).astype(np.uint8) * 255)

        Final = cv2.bitwise_or(Core, RingKeep)
        Final = cv2.bitwise_and(Final, M)
        Final = smooth_mask_inside_coarse(Final, M)

        ratio = float((Final > 0).sum()) / max(1, coarse_area)
        if ratio > best_ratio:
            best_ratio = ratio
            best_final = Final
        if ratio >= min_keep_ratio:
            score_u8 = np.clip(score, 0, 255).astype(np.uint8)
            return Final, score_u8, ratio

    score_u8 = np.clip(score, 0, 255).astype(np.uint8)
    return best_final, score_u8, best_ratio

def mask_to_polygons(mask_u8: np.ndarray, min_area: float = 20.0, eps_ratio: float = 0.006):
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
        if pts.shape[0] < 3:
            continue
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

def overlay_fill(img_bgr: np.ndarray, mask_u8: np.ndarray, fill=(70, 110, 255), alpha=0.38, edge=(255, 255, 255)):
    out = img_bgr.copy()
    m = mask_u8 > 0
    color = np.zeros_like(out, dtype=np.uint8)
    color[:] = fill
    out = np.where(m[..., None], (out * (1 - alpha) + color * alpha).astype(np.uint8), out)

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

def build_contact_sheets(image_paths, out_dir: Path, prefix: str, cols: int = 4, rows: int = 4, cell_side: int = 320):
    ensure_dir(out_dir)
    per_board = cols * rows
    if not image_paths:
        return []

    saved = []
    for bi in range(0, len(image_paths), per_board):
        chunk = image_paths[bi:bi+per_board]
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

        idx = bi // per_board + 1
        out_path = out_dir / f"{prefix}_board_{idx:03d}.jpg"
        cv2.imwrite(str(out_path), board)
        saved.append(out_path)
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single-class", action="store_true")
    ap.add_argument("--core-ks", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=60.0)
    ap.add_argument("--min-keep-ratio", type=float, default=0.60)
    ap.add_argument("--poly-eps-ratio", type=float, default=0.006)
    ap.add_argument("--board-cols", type=int, default=4)
    ap.add_argument("--board-rows", type=int, default=4)
    ap.add_argument("--board-cell", type=int, default=320)
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
    ensure_dir(out_root / "showcase")
    ensure_dir(out_root / "overview")
    ensure_dir(out_root / "refined_masks")

    converted = 0
    empty = 0
    ratios = []
    preview_paths = []
    showcase_paths = []

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

        refined, score_u8, keep_ratio = refine_mask_inside_coarse(
            defect, factual, coarse,
            core_ks=args.core_ks,
            percentile=args.percentile,
            min_keep_ratio=args.min_keep_ratio,
        )
        ratios.append(keep_ratio)

        polys = mask_to_polygons(refined, min_area=20.0, eps_ratio=args.poly_eps_ratio)
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

        coarse_bin = ((coarse > 127).astype(np.uint8) * 255)

        p1 = add_title(factual, "factual")
        p2 = add_title(defect, "defect_hq")
        p3 = add_title(overlay_fill(defect, coarse_bin), "coarse mask")
        p4 = add_title(overlay_fill(defect, refined), f"trimmed mask r={keep_ratio:.2f}")
        p5 = add_title(to_heat(score_u8), "diff heat")
        preview = cv2.hconcat([p1, p2, p3, p4, p5])

        showcase = overlay_fill(defect, refined, fill=(70, 110, 255), alpha=0.42, edge=(255, 255, 255))
        showcase = add_title(showcase, f"{sample_dir.name}")

        preview_path = out_root / "preview" / f"{sample_dir.name}.jpg"
        showcase_path = out_root / "showcase" / f"{sample_dir.name}.jpg"
        cv2.imwrite(str(preview_path), preview)
        cv2.imwrite(str(showcase_path), showcase)

        preview_paths.append(preview_path)
        showcase_paths.append(showcase_path)

    preview_boards = build_contact_sheets(
        preview_paths, out_root / "overview", "preview",
        cols=args.board_cols, rows=args.board_rows, cell_side=args.board_cell
    )
    showcase_boards = build_contact_sheets(
        showcase_paths, out_root / "overview", "showcase",
        cols=args.board_cols, rows=args.board_rows, cell_side=args.board_cell
    )

    names = ["defect"] if args.single_class else ["Spalling", "Crack"]
    yaml_text = f"""path: {out_root}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(names):
        yaml_text += f"  {i}: {name}\n"
    (out_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    min_ratio = float(np.min(ratios)) if ratios else 0.0
    max_ratio = float(np.max(ratios)) if ratios else 0.0

    print("=" * 60)
    print("done")
    print("total          :", n_total)
    print("converted      :", converted)
    print("empty          :", empty)
    print("ratio_mean     :", round(mean_ratio, 4))
    print("ratio_min      :", round(min_ratio, 4))
    print("ratio_max      :", round(max_ratio, 4))
    print("yaml           :", out_root / "dataset.yaml")
    print("preview_dir    :", out_root / "preview")
    print("showcase_dir   :", out_root / "showcase")
    print("overview_dir   :", out_root / "overview")
    print("refined_masks  :", out_root / "refined_masks")
    print("preview_boards :", len(preview_boards))
    print("showcase_boards:", len(showcase_boards))
    print("=" * 60)

if __name__ == "__main__":
    main()
