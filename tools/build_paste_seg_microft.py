from pathlib import Path
import random, shutil, os
import cv2
import numpy as np

SRC = Path("/root/autodl-tmp/Gear_Causal_Project_v2/output/paste_realdefect_strict_boundary")
OUT = Path("/root/autodl-tmp/Gear_Causal_Project_v2/datasets/paste_seg_microft")

KEEP = {"sample_000007", "sample_000004", "sample_000012"}
EXTRA_RAND = 12
SEED = 42

random.seed(SEED)

def safe_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def pick_image(sdir: Path) -> Path | None:
    cands = [
        sdir / "toned_composite.png",
        sdir / "refined" / "light_refined.png",
        sdir / "refined" / "poisson_composite.png",
        sdir / "initial_composite.png",
    ]
    for p in cands:
        if p.exists():
            return p
    return None

def mask_to_seg_line(mask_path: Path, class_id: int = 0, eps_ratio: float = 0.008) -> str:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    mask = ((mask > 127).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ""
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return ""
    peri = cv2.arcLength(c, True)
    eps = max(1e-6, eps_ratio * peri)
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 3:
        pts = c.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 3:
        return ""
    h, w = mask.shape[:2]
    coords = []
    for x, y in pts:
        coords.append(f"{np.clip(x / w, 0.0, 1.0):.6f}")
        coords.append(f"{np.clip(y / h, 0.0, 1.0):.6f}")
    return f"{class_id} " + " ".join(coords)

all_dirs = [p for p in sorted(SRC.iterdir()) if p.is_dir() and p.name.startswith("sample_")]
keep_dirs = [p for p in all_dirs if p.name in KEEP]
other_dirs = [p for p in all_dirs if p.name not in KEEP]
random.shuffle(other_dirs)
pick_dirs = keep_dirs + other_dirs[:EXTRA_RAND]

valid = []
for sdir in pick_dirs:
    img = pick_image(sdir)
    mask = sdir / "strict_mask.png"
    if img is None or not mask.exists():
        continue
    if not mask_to_seg_line(mask):
        continue
    valid.append((sdir.name, img, mask))

if len(valid) < 4:
    raise RuntimeError(f"有效样本太少: {len(valid)}")

random.shuffle(valid)
n_val = max(2, int(round(len(valid) * 0.2)))
val_items = valid[:n_val]
train_items = valid[n_val:]

for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

for split, items in [("train", train_items), ("val", val_items)]:
    for stem, img, mask in items:
        ext = img.suffix.lower()
        dst_img = OUT / "images" / split / f"{stem}{ext}"
        safe_link(img, dst_img)
        line = mask_to_seg_line(mask)
        (OUT / "labels" / split / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")

yaml_text = f"""path: {OUT}
train: images/train
val: images/val
names:
  0: Spalling
"""
(OUT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

print("OUT =", OUT)
print("total =", len(valid))
print("train =", len(train_items))
print("val =", len(val_items))
print("picked =", [x[0] for x in valid])
