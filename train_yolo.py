from pathlib import Path
import random
import shutil
import subprocess
import yaml
import os
import sys

# =========================
# 路径配置
# =========================
project_root = Path("/root/autodl-tmp/Gear_Causal_Project_v2")

src_img = project_root / "datasets/gen_aug_manual_500_testreal/train/images"
src_lbl = project_root / "datasets/gen_aug_manual_500_testreal/train/labels"

out = project_root / "datasets/gear_yolov8_detect_1000_200_300"

train_n = 1000
val_n = 200
test_n = 300
seed = 42

epochs = 100
imgsz = 640
batch = 16
device = 0
workers = 8

run_project = project_root / "runs/detect"
run_name = "gear_yolov8_detect_1000_200_300"

# =========================
# 检查原始路径
# =========================
if not src_img.exists():
    raise FileNotFoundError(f"图像路径不存在: {src_img}")

if not src_lbl.exists():
    raise FileNotFoundError(f"标注路径不存在: {src_lbl}")

# =========================
# 收集图像和 label
# =========================
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = sorted([p for p in src_img.iterdir() if p.suffix.lower() in img_exts])

pairs = []
bad_no_label = []
bad_box_format = []

for img in images:
    lbl = src_lbl / f"{img.stem}.txt"

    if not lbl.exists():
        bad_no_label.append(img.name)
        continue

    text = lbl.read_text(encoding="utf-8").strip()

    ok = True

    # 空 label 允许存在，表示负样本
    if text:
        for line in text.splitlines():
            parts = line.strip().split()

            # YOLO detection box 格式必须是 5 列:
            # class x_center y_center width height
            if len(parts) != 5:
                ok = False
                break

            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except Exception:
                ok = False
                break

            # 简单检查归一化坐标
            if not all(0 <= v <= 1 for v in vals):
                ok = False
                break

    if not ok:
        bad_box_format.append(lbl.name)
        continue

    pairs.append((img, lbl))

need = train_n + val_n + test_n

print("=" * 80)
print("数据检查")
print("=" * 80)
print(f"图像总数: {len(images)}")
print(f"可用图像/box标注对: {len(pairs)}")
print(f"缺少 label 的图像数: {len(bad_no_label)}")
print(f"疑似不是 YOLO box 格式的 label 数: {len(bad_box_format)}")

if bad_box_format[:10]:
    print("前 10 个格式异常 label:")
    for x in bad_box_format[:10]:
        print("  ", x)

if len(pairs) < need:
    raise RuntimeError(f"可用数据不足，需要 {need} 对，但只有 {len(pairs)} 对。")

# =========================
# 切分数据
# =========================
random.seed(seed)
random.shuffle(pairs)

selected = pairs[:need]

splits = {
    "train": selected[:train_n],
    "val": selected[train_n:train_n + val_n],
    "test": selected[train_n + val_n:train_n + val_n + test_n],
}

if out.exists():
    shutil.rmtree(out)

for split, items in splits.items():
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img, lbl in items:
        dst_img = out / "images" / split / img.name
        dst_lbl = out / "labels" / split / lbl.name

        try:
            dst_img.hardlink_to(img)
        except Exception:
            shutil.copy2(img, dst_img)

        try:
            dst_lbl.hardlink_to(lbl)
        except Exception:
            shutil.copy2(lbl, dst_lbl)

# =========================
# 推断类别
# =========================
cls_ids = set()

for _, lbl in selected:
    text = lbl.read_text(encoding="utf-8").strip()
    if not text:
        continue

    for line in text.splitlines():
        parts = line.strip().split()
        if parts:
            cls_ids.add(int(float(parts[0])))

nc = max(cls_ids) + 1 if cls_ids else 1

if nc == 1:
    names = {0: "gear"}
else:
    names = {i: f"class_{i}" for i in range(nc)}

data_yaml = {
    "path": str(out),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": names,
}

data_yaml_path = out / "data.yaml"

with open(data_yaml_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

print("=" * 80)
print("数据集已生成")
print("=" * 80)
print(f"输出目录: {out}")
print(f"data.yaml: {data_yaml_path}")
print(f"类别: {names}")
for k, v in splits.items():
    print(f"{k}: {len(v)}")

# =========================
# 自动搜索 YOLO 权重
# =========================
print("=" * 80)
print("自动搜索本地 YOLO 权重")
print("=" * 80)

all_pts = list(project_root.rglob("*.pt"))

def is_detect_weight(p: Path):
    name = p.name.lower()
    full = str(p).lower()

    # 排除非检测任务权重
    banned = ["seg", "pose", "cls", "obb"]
    if any(x in name for x in banned):
        return False
    if any(f"/{x}/" in full for x in banned):
        return False

    return True

candidates = [p for p in all_pts if is_detect_weight(p)]

def score_weight(p: Path):
    s = 0
    name = p.name.lower()
    full = str(p).lower()

    if name == "best.pt":
        s += 100
    if name == "last.pt":
        s += 80
    if "runs/detect" in full:
        s += 50
    if "yolov8" in name:
        s += 30
    if "yolov8n" in name:
        s += 20
    if "yolov8s" in name:
        s += 15
    if "detect" in full:
        s += 10

    # 越新的权重稍微优先
    try:
        s += p.stat().st_mtime / 1e10
    except Exception:
        pass

    return s

candidates = sorted(candidates, key=score_weight, reverse=True)

if candidates:
    model = str(candidates[0])
    print("找到本地检测权重，自动使用:")
    print(model)
    print()
    print("候选权重前 10 个:")
    for p in candidates[:10]:
        print("  ", p)
else:
    model = "yolov8n.pt"
    print("没有找到本地检测权重，自动使用官方预训练权重:")
    print(model)

# =========================
# 训练
# =========================
print("=" * 80)
print("开始训练 YOLOv8 detection")
print("=" * 80)

train_cmd = [
    "yolo", "detect", "train",
    f"model={model}",
    f"data={data_yaml_path}",
    f"epochs={epochs}",
    f"imgsz={imgsz}",
    f"batch={batch}",
    f"device={device}",
    f"workers={workers}",
    "patience=30",
    f"project={run_project}",
    f"name={run_name}",
]

print("训练命令:")
print(" ".join(train_cmd))
subprocess.run(train_cmd, check=True)

best_pt = run_project / run_name / "weights/best.pt"

if not best_pt.exists():
    raise FileNotFoundError(f"训练完成但没有找到 best.pt: {best_pt}")

print("=" * 80)
print("训练完成")
print("=" * 80)
print(f"best.pt: {best_pt}")

# =========================
# 验证集评估
# =========================
print("=" * 80)
print("验证集评估")
print("=" * 80)

val_cmd = [
    "yolo", "detect", "val",
    f"model={best_pt}",
    f"data={data_yaml_path}",
    "split=val",
    f"imgsz={imgsz}",
    f"device={device}",
    "plots=True",
]

print(" ".join(val_cmd))
subprocess.run(val_cmd, check=True)

# =========================
# 测试集评估
# =========================
print("=" * 80)
print("测试集评估")
print("=" * 80)

test_cmd = [
    "yolo", "detect", "val",
    f"model={best_pt}",
    f"data={data_yaml_path}",
    "split=test",
    f"imgsz={imgsz}",
    f"device={device}",
    "plots=True",
    "save_json=True",
]

print(" ".join(test_cmd))
subprocess.run(test_cmd, check=True)

# =========================
# 测试集预测可视化
# =========================
print("=" * 80)
print("测试集预测可视化")
print("=" * 80)

pred_cmd = [
    "yolo", "detect", "predict",
    f"model={best_pt}",
    f"source={out / 'images/test'}",
    f"imgsz={imgsz}",
    "conf=0.25",
    f"device={device}",
    "save=True",
    "save_txt=True",
    "save_conf=True",
    f"project={run_project}",
    "name=gear_test_predictions",
]

print(" ".join(pred_cmd))
subprocess.run(pred_cmd, check=True)

print("=" * 80)
print("全部完成")
print("=" * 80)
print(f"训练目录: {run_project / run_name}")
print(f"最优权重: {best_pt}")
print(f"测试预测结果: {run_project / 'gear_test_predictions'}")
