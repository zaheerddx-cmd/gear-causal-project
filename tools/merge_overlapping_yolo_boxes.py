from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

def xywhn_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w / 2.0) * W
    y1 = (yc - h / 2.0) * H
    x2 = (xc + w / 2.0) * W
    y2 = (yc + h / 2.0) * H
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def xyxy_to_xywhn(box, W, H):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / W
    yc = ((y1 + y2) / 2.0) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return xc, yc, w, h

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def merge_group(boxes, confs):
    weights = np.array(confs, dtype=np.float32)
    weights = np.clip(weights, 1e-6, None)
    boxes = np.array(boxes, dtype=np.float32)
    merged = np.average(boxes, axis=0, weights=weights)
    merged_conf = float(np.max(confs))
    return merged, merged_conf

def cluster_and_merge(items, iou_thr=0.5):
    # items: [(cls_id, box_xyxy, conf)]
    used = [False] * len(items)
    merged_items = []
    for i in range(len(items)):
        if used[i]:
            continue
        cls_i, box_i, conf_i = items[i]
        group_boxes = [box_i]
        group_confs = [conf_i]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(items)):
                if used[j]:
                    continue
                cls_j, box_j, conf_j = items[j]
                if cls_j != cls_i:
                    continue
                # 和组内任一框重叠就并入
                if any(iou(box_j, gb) >= iou_thr for gb in group_boxes):
                    group_boxes.append(box_j)
                    group_confs.append(conf_j)
                    used[j] = True
                    changed = True

        merged_box, merged_conf = merge_group(group_boxes, group_confs)
        merged_items.append((cls_i, merged_box, merged_conf))
    return merged_items

def load_preds(txt_path, W, H):
    items = []
    if not txt_path.exists():
        return items
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        box = xywhn_to_xyxy(xc, yc, w, h, W, H)
        items.append((cls_id, box, conf))
    return items

def save_preds(txt_path, items, W, H):
    lines = []
    for cls_id, box, conf in items:
        xc, yc, w, h = xyxy_to_xywhn(box, W, H)
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def draw_boxes(img, items):
    out = img.copy()
    for cls_id, box, conf in items:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 64, 64), 2)
        cv2.putText(out, f"{cls_id}:{conf:.2f}", (max(4, x1), max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 64, 64), 2, cv2.LINE_AA)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-images", required=True)
    ap.add_argument("--pred-labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    img_dir = Path(args.source_images)
    lab_dir = Path(args.pred_labels)
    out_dir = Path(args.out_dir)
    out_img = out_dir / "images"
    out_lab = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        txt_path = lab_dir / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        items = load_preds(txt_path, W, H)
        merged = cluster_and_merge(items, iou_thr=args.iou_thr)
        save_preds(out_lab / f"{stem}.txt", merged, W, H)

        vis = draw_boxes(img, merged)
        cv2.imwrite(str(out_img / img_path.name), vis)

    print("done:", out_dir)

if __name__ == "__main__":
    main()
