from __future__ import annotations
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

def make_board(items, out_path: Path, cols=4, cell=320):
    if not items:
        print(f"[WARN] no items for {out_path}")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), board)
    print("[OK]", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-all", type=int, default=12)
    ap.add_argument("--n-large", type=int, default=12)
    ap.add_argument("--large-thr", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    defect_dir = root / "defect"
    if not defect_dir.exists():
        raise FileNotFoundError(defect_dir)

    random.seed(args.seed)
    items = []

    for sdir in sorted(defect_dir.iterdir()):
        if not (sdir.is_dir() and sdir.name.startswith("sample_")):
            continue
        # 你现在以 defect_lq 为最终标准域
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
        items.append({
            "stem": sdir.name,
            "img": img,
            "mask": mask,
            "ratio": ratio,
        })

    if not items:
        raise RuntimeError("no valid samples found")

    ratios = np.array([x["ratio"] for x in items], dtype=np.float32)
    print(f"total={len(items)}")
    print(f"ratio min/mean/max = {ratios.min():.4f} / {ratios.mean():.4f} / {ratios.max():.4f}")
    print(f"q50/q75/q90/q95 = {np.quantile(ratios,0.5):.4f} / {np.quantile(ratios,0.75):.4f} / {np.quantile(ratios,0.90):.4f} / {np.quantile(ratios,0.95):.4f}")

    all_pick = random.sample(items, min(args.n_all, len(items)))
    large_pool = [x for x in items if x["ratio"] >= args.large_thr]
    large_pick = random.sample(large_pool, min(args.n_large, len(large_pool)))

    out_dir = root / "size_preview"
    make_board(all_pick, out_dir / "random_all.jpg")
    make_board(large_pick, out_dir / f"random_large_ge_{args.large_thr:.3f}.jpg")

    txt = out_dir / "stats.txt"
    txt.write_text(
        "\n".join([
            f"total={len(items)}",
            f"ratio min={ratios.min():.6f}",
            f"ratio mean={ratios.mean():.6f}",
            f"ratio max={ratios.max():.6f}",
            f"q50={np.quantile(ratios,0.5):.6f}",
            f"q75={np.quantile(ratios,0.75):.6f}",
            f"q90={np.quantile(ratios,0.90):.6f}",
            f"q95={np.quantile(ratios,0.95):.6f}",
            f"large_thr={args.large_thr}",
            f"large_count={len(large_pool)}",
        ]),
        encoding="utf-8"
    )
    print("[OK]", txt)

if __name__ == "__main__":
    main()
