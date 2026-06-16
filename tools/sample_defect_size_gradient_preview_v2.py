from pathlib import Path
import argparse, random
import cv2
import numpy as np

def overlay(img_bgr, mask_u8, alpha=0.30):
    out = img_bgr.copy()
    m = mask_u8 > 127
    color = np.zeros_like(out, dtype=np.uint8)
    color[:] = (60, 140, 255)
    out = np.where(m[..., None], (out * (1 - alpha) + color * alpha).astype(np.uint8), out)
    return out

def add_title(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 2, cv2.LINE_AA)
    return out

def letterbox(img, side=320, bg=(18,18,18)):
    h, w = img.shape[:2]
    scale = min(side / max(w, 1), side / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((side, side, 3), bg, dtype=np.uint8)
    x0 = (side - nw) // 2
    y0 = (side - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def make_board(items, out_path, cols=2, cell=320):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    board = np.full((rows * cell, cols * cell, 3), 18, dtype=np.uint8)
    for i, item in enumerate(items):
        img = cv2.imread(str(item["img"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        vis = overlay(img, mask, alpha=0.30)
        vis = add_title(vis, f'{item["stem"]} | r={item["ratio"]:.4f}')
        tile = letterbox(vis, side=cell)
        r = i // cols
        c = i % cols
        y0 = r * cell
        x0 = c * cell
        board[y0:y0+cell, x0:x0+cell] = tile
    cv2.imwrite(str(out_path), board)
    print("[OK]", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    random.seed(args.seed)

    # 自动兼容两种结构
    defect_dir = root / "defect"
    yolo_dir = root / "yolo_dual_export"

    items = []

    if defect_dir.exists():
        for sdir in sorted(defect_dir.iterdir()):
            if not (sdir.is_dir() and sdir.name.startswith("sample_")):
                continue
            img = sdir / "defect_lq.png"
            if not img.exists():
                img = sdir / "defect_hq.png"
            mask = sdir / "mask.png"
            if not img.exists() or not mask.exists():
                continue
            m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            ratio = float(np.count_nonzero(m > 127)) / float(m.shape[0] * m.shape[1])
            items.append({"stem": sdir.name, "img": img, "mask": mask, "ratio": ratio})
    else:
        images_dir = yolo_dir / "images"
        masks_dir = None
        for cand in ["masks_final", "masks", "masks_coarse"]:
            d = yolo_dir / cand
            if d.exists():
                masks_dir = d
                break
        if not images_dir.exists() or masks_dir is None:
            raise FileNotFoundError(f"既没找到 {defect_dir}，也没找到 {yolo_dir}/images + masks_*")
        for p in sorted(images_dir.glob("*.png")):
            stem = p.stem
            mp = masks_dir / f"{stem}.png"
            if not mp.exists():
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            ratio = float(np.count_nonzero(m > 127)) / float(m.shape[0] * m.shape[1])
            items.append({"stem": stem, "img": p, "mask": mp, "ratio": ratio})

    if not items:
        raise RuntimeError("没有找到可用样本")

    bins = [
        ("bin_00_lt_0015", 0.0,   0.015),
        ("bin_01_0015_0030", 0.015, 0.030),
        ("bin_02_0030_0050", 0.030, 0.050),
        ("bin_03_0050_0080", 0.050, 0.080),
        ("bin_04_ge_0080",   0.080, 1.000),
    ]

    out_dir = root / "size_gradient_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for name, lo, hi in bins:
        pool = [x for x in items if lo <= x["ratio"] < hi]
        picks = random.sample(pool, min(args.per_bin, len(pool))) if pool else []
        make_board(picks, out_dir / f"{name}.jpg", cols=2, cell=320)
        lines.append(f"{name}: range=[{lo:.4f}, {hi:.4f}) count={len(pool)} picked={len(picks)}")
        for p in picks:
            lines.append(f"  {p['stem']} ratio={p['ratio']:.6f}")

    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("[OK]", out_dir / "summary.txt")

if __name__ == "__main__":
    main()
