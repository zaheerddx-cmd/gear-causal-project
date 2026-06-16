#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?need root}"
DEFECT_DIR="$ROOT/defect"
YOLO_DIR="$ROOT/yolo_dual_export"

# 收集所有 stem
TMP_STEMS="$(mktemp)"
{
  if [ -d "$DEFECT_DIR" ]; then
    find "$DEFECT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'sample_*' -printf '%f\n'
  fi
  for d in \
    "$YOLO_DIR/images" \
    "$YOLO_DIR/labels_det" \
    "$YOLO_DIR/labels_seg" \
    "$YOLO_DIR/preview_det" \
    "$YOLO_DIR/preview_seg" \
    "$YOLO_DIR/meta" \
    "$YOLO_DIR/masks_final" \
    "$YOLO_DIR/masks_coarse" \
    "$YOLO_DIR/masks"
  do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 1 -type f -printf '%f\n' | sed 's/\.[^.]*$//'
  done
} | sort -u > "$TMP_STEMS"

while IFS= read -r stem; do
  [ -n "$stem" ] || continue
  bad=0

  # 1) 检查 sample 目录
  sdir="$DEFECT_DIR/$stem"
  if [ ! -d "$sdir" ]; then
    bad=1
  else
    for f in factual.png defect_hq.png defect_lq.png mask.png meta.json; do
      [ -f "$sdir/$f" ] || bad=1
    done
  fi

  # 2) 检查 flat 导出
  [ -f "$YOLO_DIR/images/$stem.png" ]      || bad=1
  [ -f "$YOLO_DIR/labels_det/$stem.txt" ]  || bad=1
  [ -f "$YOLO_DIR/labels_seg/$stem.txt" ]  || bad=1
  [ -f "$YOLO_DIR/preview_det/$stem.png" ] || bad=1
  [ -f "$YOLO_DIR/preview_seg/$stem.png" ] || bad=1
  [ -f "$YOLO_DIR/meta/$stem.json" ]       || bad=1

  # masks_final / masks_coarse / masks 三选一至少有一个
  has_mask=0
  [ -f "$YOLO_DIR/masks_final/$stem.png" ] && has_mask=1
  [ -f "$YOLO_DIR/masks_coarse/$stem.png" ] && has_mask=1
  [ -f "$YOLO_DIR/masks/$stem.png" ] && has_mask=1
  [ "$has_mask" -eq 1 ] || bad=1

  if [ "$bad" -eq 1 ]; then
    echo "[DEL] $stem"

    rm -rf "$DEFECT_DIR/$stem" 2>/dev/null || true

    rm -f \
      "$YOLO_DIR/images/$stem.png" \
      "$YOLO_DIR/labels_det/$stem.txt" \
      "$YOLO_DIR/labels_seg/$stem.txt" \
      "$YOLO_DIR/preview_det/$stem.png" \
      "$YOLO_DIR/preview_seg/$stem.png" \
      "$YOLO_DIR/meta/$stem.json" \
      "$YOLO_DIR/masks_final/$stem.png" \
      "$YOLO_DIR/masks_coarse/$stem.png" \
      "$YOLO_DIR/masks/$stem.png" \
      2>/dev/null || true
  fi
done < "$TMP_STEMS"

rm -f "$TMP_STEMS"
echo "done"
