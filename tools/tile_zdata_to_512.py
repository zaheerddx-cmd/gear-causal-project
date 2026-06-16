#!/usr/bin/env python3
import math
from pathlib import Path
from PIL import Image

IMG_DIR = Path("/root/autodl-tmp/Gear_Causal_Project_v2/ablation_data/common/zdata/images")
LBL_DIR = Path("/root/autodl-tmp/Gear_Causal_Project_v2/ablation_data/common/zdata/labels")

OUT_ROOT = Path("/root/autodl-tmp/Gear_Causal_Project_v2/ablation_data/common/zdata_tiles_512")
OUT_IMG = OUT_ROOT / "images"
OUT_LBL = OUT_ROOT / "labels"

TILE = 512
VALID_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".webp"}

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

def load_yolo_boxes(label_path, w, h):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                # 只支持 YOLO detect: cls cx cy w h
                continue
            cls_id = parts[0]
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes

def save_yolo_boxes(boxes, tile_path):
    with open(tile_path, "w", encoding="utf-8") as f:
        for cls_id, x1, y1, x2, y2 in boxes:
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            f.write(
                f"{cls_id} "
                f"{cx / TILE:.6f} {cy / TILE:.6f} "
                f"{bw / TILE:.6f} {bh / TILE:.6f}\n"
            )

def clip_box_to_tile(box, tx, ty):
    cls_id, x1, y1, x2, y2 = box
    ix1 = max(x1, tx)
    iy1 = max(y1, ty)
    ix2 = min(x2, tx + TILE)
    iy2 = min(y2, ty + TILE)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    # 相交后若太小则丢弃，避免碎框
    inter_w = ix2 - ix1
    inter_h = iy2 - iy1
    orig_w = x2 - x1
    orig_h = y2 - y1
    inter_area = inter_w * inter_h
    orig_area = max(orig_w * orig_h, 1e-6)

    if inter_area / orig_area < 0.20:
        return None
    if inter_w < 4 or inter_h < 4:
        return None

    # 转到 tile 局部坐标
    return (cls_id, ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty)

img_paths = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in VALID_EXTS])

if not img_paths:
    print(f"[ERR] no images found in {IMG_DIR}")
    raise SystemExit(1)

count_imgs = 0
count_tiles = 0

for img_path in img_paths:
    stem = img_path.stem
    label_path = LBL_DIR / f"{stem}.txt"

    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size

    # 若任一边 < 512，则按比例放大到两边都 >= 512
    scale = max(TILE / w0, TILE / h0, 1.0)
    w1 = int(round(w0 * scale))
    h1 = int(round(h0 * scale))

    if scale != 1.0:
        img = img.resize((w1, h1), Image.BILINEAR)
    else:
        w1, h1 = w0, h0

    # 读取并同步缩放框
    boxes = load_yolo_boxes(label_path, w0, h0)
    if scale != 1.0:
        boxes = [
            (cls_id, x1 * scale, y1 * scale, x2 * scale, y2 * scale)
            for cls_id, x1, y1, x2, y2 in boxes
        ]

    # pad 到 512 整数倍
    pw = math.ceil(w1 / TILE) * TILE
    ph = math.ceil(h1 / TILE) * TILE
    if pw != w1 or ph != h1:
        canvas = Image.new("RGB", (pw, ph), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas

    nx = pw // TILE
    ny = ph // TILE

    for j in range(ny):
        for i in range(nx):
            tx = i * TILE
            ty = j * TILE

            tile = img.crop((tx, ty, tx + TILE, ty + TILE))
            out_name = f"{stem}_r{j:02d}_c{i:02d}.png"
            out_img_path = OUT_IMG / out_name
            out_lbl_path = OUT_LBL / f"{Path(out_name).stem}.txt"

            tile.save(out_img_path)

            tile_boxes = []
            for b in boxes:
                cb = clip_box_to_tile(b, tx, ty)
                if cb is not None:
                    tile_boxes.append(cb)

            save_yolo_boxes(tile_boxes, out_lbl_path)
            count_tiles += 1

    count_imgs += 1
    print(f"[OK] {img_path.name} -> {nx*ny} tiles")

print(f"\n[DONE] images={count_imgs}, tiles={count_tiles}")
print(f"[OUT IMG] {OUT_IMG}")
print(f"[OUT LBL] {OUT_LBL}")
