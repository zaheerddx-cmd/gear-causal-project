from __future__ import annotations
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def ensure_mask_u8(mask):
    return ((mask > 0).astype(np.uint8) * 255)

def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, flood_inv)
    return filled

def refine_mask(mask_u8: np.ndarray, close_ks: int = 9) -> np.ndarray:
    mask_u8 = ensure_mask_u8(mask_u8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    out = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)
    out = fill_holes(out)
    return ensure_mask_u8(out)

def overlay_mask(img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    sel = mask_u8 > 0
    color = np.array([255, 0, 0], dtype=np.uint8)  # BGR 蓝色
    out[sel] = (0.55 * out[sel] + 0.45 * color).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--close-ks", type=int, default=9)
    args = ap.parse_args()

    model = YOLO(args.model)
    src = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = []
    if src.is_dir():
        imgs = [p for p in sorted(src.iterdir()) if p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [src]

    for i, img_path in enumerate(imgs):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            retina_masks=True,
            verbose=False,
            save=False,
            device=0,
        )

        res = results[0]
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            cv2.imwrite(str(out_dir / img_path.name), img_bgr)
            continue

        confs = res.boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(confs))
        mask = res.masks.data[best_idx].detach().cpu().numpy()
        mask_u8 = ensure_mask_u8(mask)

        refined = refine_mask(mask_u8, close_ks=args.close_ks)
        vis = overlay_mask(img_bgr, refined)

        stem = img_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_vis.png"), vis)
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), refined)

        print(f"[OK] {img_path.name} conf={confs[best_idx]:.3f}")

if __name__ == "__main__":
    main()
