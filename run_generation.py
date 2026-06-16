from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path("/root/autodl-tmp/Gear_Causal_Project_v2")
BASE_SCRIPT = PROJECT_ROOT / "27_inference_pipeline_v15_datasetpack_grouped_export_v500_separate_output.py"
OUTPUT_DIR = PROJECT_ROOT / "output" / "defect_datasetpack_spalling_singlepass_v13_final"
V13_WEIGHT = PROJECT_ROOT / "weights" / "gear_and_defect_from_scratch_explainable_v13_defectonly_cropped512.safetensors"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(f"找不到基础推理脚本: {BASE_SCRIPT}")
if not V13_WEIGHT.exists():
    raise FileNotFoundError(f"找不到 v13 权重: {V13_WEIGHT}")

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# 兼容 dataset 文件名
intrinsic_file = PROJECT_ROOT / "causal_dataset_v20_intrinsic_surface_debug_refactored.py"
sygrid_file = PROJECT_ROOT / "causal_dataset_v20_sygrid_refactored.py"
if intrinsic_file.exists() and not sygrid_file.exists():
    shutil.copy2(intrinsic_file, sygrid_file)

spec = importlib.util.spec_from_file_location("infer_base", str(BASE_SCRIPT))
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None

# 关键：先注册到 sys.modules，避免 py3.12 dataclass 报错
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# ----------------------------
# 运行时覆盖
# ----------------------------

# 新输出目录
mod.OUTPUT_DIR = OUTPUT_DIR

# 新 spalling 权重
mod.SPALLING_RENDER_LORA_PATH = str(V13_WEIGHT)

# 强制只跑 Spalling
_orig_choice = mod.np.random.choice
def _forced_choice(a, *args, **kwargs):
    try:
        if isinstance(a, (list, tuple)) and list(a) == ["Spalling", "Crack"]:
            return "Spalling"
    except Exception:
        pass
    return _orig_choice(a, *args, **kwargs)
mod.np.random.choice = _forced_choice

# 覆盖 GenerationConfig 默认值
_orig_gc_init = mod.GenerationConfig.__init__
def _gc_init_patched(self, *args, **kwargs):
    _orig_gc_init(self, *args, **kwargs)
    self.target_total_samples = 20
    self.resume_from_existing = False
    self.max_consecutive_failures = 200
    self.progress_print_every = 5
mod.GenerationConfig.__init__ = _gc_init_patched

# ----------------------------
# single-pass Spalling:
# 直接把 first_pass 作为最终结果
# ----------------------------

def _render_spalling_material_refine_candidate_bypass(
    pipe,
    first_pass_u8,
    structure,
    cfg,
):
    # 不再做第二轮 refine
    core_mask_u8 = mod.build_core_refine_mask(structure.causal_mask_u8)
    return first_pass_u8.copy(), core_mask_u8, None, "bypass_refine", "bypass_refine"

def _apply_material_residual_bypass(
    base_u8,
    refine_candidate_u8,
    core_mask_u8,
    soft_render_mask_u8,
    cfg,
):
    # 不做 residual blend
    return refine_candidate_u8.copy(), core_mask_u8.copy()

def _harmonize_local_sharpness_bypass(
    final_u8,
    blend_mask_u8,
    cfg,
):
    # 不做 sharpness harmonization
    return final_u8

mod.render_spalling_material_refine_candidate = _render_spalling_material_refine_candidate_bypass
mod.apply_material_residual = _apply_material_residual_bypass
mod.harmonize_local_sharpness = _harmonize_local_sharpness_bypass

# 清空旧输出
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 运行 Spalling single-pass v13 final")
print(f"   base script: {BASE_SCRIPT}")
print(f"   output dir:  {OUTPUT_DIR}")
print(f"   v13 weight:  {V13_WEIGHT}")
print("   mode: Spalling only / single-pass / no refine / no residual / no sharpness")

if hasattr(mod, "main"):
    mod.main()
else:
    raise RuntimeError("基础推理脚本没有 main()，请把脚本末尾发我。")
