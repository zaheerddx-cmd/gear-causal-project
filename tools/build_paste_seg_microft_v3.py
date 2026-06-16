from pathlib import Path
import random, shutil, os
import cv2
import numpy as np

SRC = Path("/root/autodl-tmp/Gear_Causal_Project_v2/output/paste_realdefect_strict_boundary")
OUT = Path("/root/autodl-tmp/Gear_Causal_Project_v2/datasets/paste_seg_microft_v3")

KEEP = {"sample_000007", "sample_000004", "sample_000012"}
EXTRA_RAND = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def safe_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def pick_image(sdir: Path) -> Path | None:
    cands = [
        sdir / "refined" / "light_refined.png",
        sdir / "initial_composite.png",
        sdir / "refined" / "poisson_composite.png",
        sdir / "toned_composite.png",
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

def recolor_mask_region(img: np.ndarray, mask: np.ndarray, delta: float) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask > 127
    if np.count_nonzero(m) == 0:
        return img
    vals = out[m]
    gray = vals.mean(axis=1, keepdims=True)
    gray3 = np.repeat(gray, 3, axis=1)
    gray3 = np.clip(gray3 + delta, 0, 255)
    # 保留一点原纹理，不要完全抹平
    vals2 = vals * 0.45 + gray3 * 0.55
    out[m] = vals2
    return np.clip(out, 0, 255).astype(np.uint8)

all_dirs = [p for p in sorted(SRC.iterdir()) if p.is_dir() and p.name.startswith("sample_")]
keep_dirs = [p for p in all_dirs if p.name in KEEP]
other_dirs = [p for p in all_dirs if p.name not in KEEP]
random.shuffle(other_dirs)
picked_dirs = keep_dirs + other_dirs[:EXTRA_RAND]

base_items = []
for sdir in picked_dirs:
    img = pick_image(sdir)
    mask = sdir / "strict_mask.png"
    if img is None or not mask.exists():
        continue
    line = mask_to_seg_line(mask)
    if not line:
        continue
    base_items.append((sdir.name, img, mask, line))

if len(base_items) < 4:
    raise RuntimeError(f"有效基样本太少: {len(base_items)}")

# 总量控制：8个基样本左右 + 4个轻微颜色变体 = 12个左右
variant_items = []
for stem, img, mask, line in base_items:
    variant_items.append((f"{stem}_orig", img, mask, line, "orig"))

# 从基样本里随机挑 4 个，分别加 light / dark 变体
extra_candidates = base_items.copy()
random.shuffle(extra_candidates)
extra_pick = extra_candidates[:4]

extra_modes = ["light", "dark", "light", "dark"]
for (stem, img, mask, line), mode in zip(extra_pick, extra_modes):
    variant_items.append((f"{stem}_{mode}", img, mask, line, mode))

random.shuffle(variant_items)
n_val = max(3, int(round(len(variant_items) * 0.25)))
val_items = variant_items[:n_val]
train_items = variant_items[n_val:]

for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def write_item(split, item):
    stem2, img_p, mask_p, line, mode = item
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return

    if mode == "light":
        out = recolor_mask_region(img, mask, delta=16.0)
    elif mode == "dark":
        out = recolor_mask_region(img, mask, delta=-16.0)
    else:
        out = img

    out_img = OUT / "images" / split / f"{stem2}.png"
    out_lab = OUT / "labels" / split / f"{stem2}.txt"

    cv2.imwrite(str(out_img), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    out_lab.write_text(line + "\n", encoding="utf-8")

for item in train_items:
    write_item("train", item)
for item in val_items:
    write_item("val", item)

yaml_text = f"""path: {OUT}
train: images/train
val: images/val
names:
  0: Spalling
"""
(OUT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

print("OUT =", OUT)
print("base_count =", len(base_items))
print("total_variant_count =", len(variant_items))
print("train =", len(train_items))
print("val =", len(val_items))
print("picked_base =", [x[0] for x in base_items])
print("picked_variants =", [x[0] for x in variant_items])
