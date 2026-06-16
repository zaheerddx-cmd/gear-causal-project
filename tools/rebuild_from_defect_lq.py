from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--replace-hq", action="store_true", help="用 defect_lq 覆盖 defect_hq.png")
    args = ap.parse_args()

    root = Path(args.root)
    defect_root = root / "defect"
    yolo_root = root / "yolo_dual_export"

    if not defect_root.exists():
        raise FileNotFoundError(defect_root)

    done = 0
    skipped = 0

    for sdir in sorted(defect_root.iterdir()):
        if not (sdir.is_dir() and sdir.name.startswith("sample_")):
            continue

        stem = sdir.name
        lq_p = sdir / "defect_lq.png"
        hq_p = sdir / "defect_hq.png"
        mask_p = sdir / "mask.png"

        if not (lq_p.exists() and mask_p.exists()):
            skipped += 1
            continue

        try:
            defect_lq = np.array(Image.open(lq_p).convert("RGB"))
            mask = np.array(Image.open(mask_p).convert("L"))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"[WARN] skip unreadable: {sdir} | {e}")
            skipped += 1
            continue

        # 1) 可选：直接让 hq 也跟 lq 一样
        if args.replace_hq:
            Image.fromarray(defect_lq).save(hq_p)

        # 2) 重建 yolo export images
        img_p = yolo_root / "images" / f"{stem}.png"
        if img_p.exists():
            Image.fromarray(defect_lq).save(img_p)

        # 3) 重建 preview_seg
        prev_seg_p = yolo_root / "preview_seg" / f"{stem}.png"
        if prev_seg_p.exists():
            Image.fromarray(_overlay_mask(defect_lq, mask, alpha=args.alpha)).save(prev_seg_p)

        # 4) 重建 preview_det
        meta_p = yolo_root / "meta" / f"{stem}.json"
        prev_det_p = yolo_root / "preview_det" / f"{stem}.png"
        if prev_det_p.exists():
            meta = _load_json(meta_p) if meta_p.exists() else None
            bbox = meta.get("bbox_xyxy") if isinstance(meta, dict) else None
            Image.fromarray(_redraw_preview_det(defect_lq, bbox)).save(prev_det_p)

        done += 1
        if done % 100 == 0:
            print(f"[OK] processed {done}")

    print(f"[DONE] processed={done}, skipped={skipped}")


if __name__ == "__main__":
    main()
