
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("TORCH_HOME", "/root/autodl-tmp/hf_cache/torch_hub")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from attention_sniffer_v3_patched import AttentionSniffer
from causal_dataset_v20_sygrid_refactored import GearCausalDataset
from pipeline_contracts import (
    GeneratedSample,
    StructuralState,
    build_intent_from_pixel_measurements,
    compute_component_bounds,
    ensure_binary_mask,
)

MODEL_ID = "/root/autodl-tmp/hf_cache/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CONTROLNET_NORMAL_ID = "/root/autodl-tmp/hf_cache/hub/models--lllyasviel--sd-controlnet-normal/snapshots/1cbed9b3ca84422e4a2f23c14b9f5a114742b31d"
CONTROLNET_DEPTH_ID = "/root/autodl-tmp/hf_cache/hub/models--lllyasviel--sd-controlnet-depth/snapshots/35e42a3ea49845b3c76f202f145f257b9fb1b7d4"

# 主权重：负责背景、sniffer、shape、Crack
MAIN_LORA_PATH = "/root/autodl-tmp/Gear_Causal_Project_v2/weights/gear_and_defect_multiconcept_v2.safetensors"
# v10 权重：负责 Spalling 的整条绘制链（第一遍 + 第二遍材质 refine）
SPALLING_RENDER_LORA_PATH = "/root/autodl-tmp/Gear_Causal_Project_v2/weights/gear_and_defect_from_scratch_explainable_v12_1_spallspecial_balanced_soft.safetensors"

OUTPUT_DIR = Path("/root/autodl-tmp/Gear_Causal_Project_v2/output/causal_v15_2d_material_residual_soft_spalling_all_v12_1_fillclamp_relaxed_refactored")
CLASS_NAME_TO_ID = {"Spalling": 0, "Crack": 1}  # 如需单类缺陷，可改成两个都映射到 0


@dataclass
class GenerationConfig:
    # 兼容旧逻辑：num_samples 不再作为主控制项，保留仅为回退字段。
    num_samples: int = 10
    max_attempts_per_sample: int = 5

    # ===== 生产导出主控制项 =====
    target_total_samples: int = 2000
    # 若为 True，会统计 export_root/images 下已有样本并继续补齐到 target_total_samples。
    resume_from_existing: bool = True
    # 连续全局失败上限，避免生成器异常时无限循环。
    max_consecutive_failures: int = 300
    stop_on_total_failures: bool = False
    # 每成功导出多少张打印一次阶段进度。
    progress_print_every: int = 25

    # ===== 大框筛选：宽或高超过 200 px 直接重绘 =====
    enable_conservative_big_box_redraw: bool = True
    enable_crack_big_box_redraw: bool = True
    max_box_width_px: int = 200
    max_box_height_px: int = 200

    save_sample_overview: bool = False
    save_run_contact_sheet: bool = False
    summary_thumb_w: int = 220
    summary_thumb_h: int = 220
    summary_sample_limit: int | None = None

    background_steps: int = 20
    render_steps: int = 25
    refine_steps: int = 12

    background_strength: float = 0.99
    background_guidance: float = 7.5

    sniff_steps: int = 8
    sniff_threshold_candidates: tuple[int, ...] = (145, 160, 175, 190)

    control_scales_background: tuple[float, float] = (0.8, 0.8)

    render_strength_crack: float = 0.96
    render_guidance_crack: float = 8.5
    control_scales_render_crack: tuple[float, float] = (0.90, 0.90)

    render_strength_spalling: float = 0.86
    render_guidance_spalling: float = 8.0
    control_scales_render_spalling: tuple[float, float] = (0.74, 0.96)

    roi_supersample_factor_spalling: float = 2.0
    roi_padding_spalling: int = 72
    roi_min_side: int = 160
    roi_max_side_after_upscale: int = 512

    # Residual material refinement (Spalling only) —— 这里只是换 pipe，不改参数
    refine_strength_spalling: float = 0.28
    refine_guidance_spalling: float = 6.2
    control_scales_refine_spalling: tuple[float, float] = (0.48, 0.74)
    refine_roi_padding_spalling: int = 24
    refine_roi_supersample_factor_spalling: float = 1.35
    refine_roi_min_side: int = 112
    refine_roi_max_side_after_upscale: int = 384

    # Residual / photometric blending gains
    residual_luma_gain_pos: float = 0.11
    residual_luma_gain_neg: float = 0.24
    residual_chroma_gain: float = 0.045
    residual_blur_sigma: float = 3.2
    residual_mask_gamma: float = 1.20
    residual_mask_blur_sigma: float = 3.6
    residual_positive_clip: float = 10.0
    residual_negative_clip: float = 18.0
    residual_soft_render_weight: float = 0.55
    residual_core_weight: float = 1.00
    residual_ring_guard_sigma: float = 1.2
    cavity_darkening_gain: float = 3.4

    # ROI feather / local sharpness harmonization
    render_paste_feather_spalling: float = 3.8
    refine_paste_feather_spalling: float = 2.8
    sharpness_match_enabled: bool = True
    sharpness_ring_dilate_px: int = 12
    sharpness_ring_gap_px: int = 3
    sharpness_target_ratio: float = 1.08
    sharpness_sigma_candidates: tuple[float, ...] = (0.0, 0.45, 0.75, 1.05, 1.35, 1.70)
    sharpness_mask_blur_sigma: float = 2.8

    min_mask_area_for_accept: int = 80
    min_roi_mean_abs_diff: float = 4.5
    min_roi_p95_abs_diff: float = 10.0


def load_pipeline(device: str, lora_path: str):
    cn_normal = ControlNetModel.from_pretrained(
        CONTROLNET_NORMAL_ID,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    cn_depth = ControlNetModel.from_pretrained(
        CONTROLNET_DEPTH_ID,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        MODEL_ID,
        controlnet=[cn_normal, cn_depth],
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    state_dict = load_file(lora_path)
    unet_only_sd = {k: v for k, v in state_dict.items() if k.startswith("unet.")}
    te_only_sd = {k.replace("text_encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.")}

    pipe.load_lora_weights(unet_only_sd)

    text_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
    )
    pipe.text_encoder = get_peft_model(pipe.text_encoder, text_lora_config)
    pipe.text_encoder.load_state_dict(te_only_sd, strict=False)
    pipe.text_encoder.to(device, dtype=torch.float16)
    return pipe


def render_background(pipe, dataset: GearCausalDataset, cfg: GenerationConfig) -> np.ndarray:
    baseline = dataset.build_baseline_bundle()
    bg = pipe(
        prompt="a macro photo of cqut-metal-surface, highly detailed, raw industrial texture",
        image=Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8)),
        mask_image=Image.new("L", (512, 512), 255),
        control_image=[
            Image.fromarray(baseline["baseline_normal_u8"]),
            Image.fromarray(baseline["baseline_depth_u8"]),
        ],
        controlnet_conditioning_scale=list(cfg.control_scales_background),
        num_inference_steps=cfg.background_steps,
        strength=cfg.background_strength,
        guidance_scale=cfg.background_guidance,
    ).images[0]
    return np.array(bg, dtype=np.uint8)


def choose_render_prompt(defect_type: str) -> tuple[str, str]:
    if defect_type == "Spalling":
        positive_pool = [
            "a macro photo of recessed metallic spall, exposed fractured steel interior, rough broken inner wall, oxidized metallic grain, unfilled cavity, realistic industrial damage",
            "a macro photo of broken spall cavity on steel surface, exposed inner material texture, recessed fractured metallic interior, rough cavity wall, industrial damage",
            "a macro photo of pitted metal loss with visible inner texture, recessed broken cavity, fractured metallic substrate, unfilled pit, realistic industrial surface damage",
        ]
        negative_pool = [
            "filled hole, flat filled patch, painted patch, smooth interior, uniform black blob, glossy intact metal, polished surface, defect-free",
            "sealed cavity, flat dark patch, surface-only rust texture, intact smooth metal, shiny glossy finish",
            "uniform filled region, fake flat blob, clean polished metal, no defect, perfect surface",
        ]
    else:
        positive_pool = [
            "a macro photo of damaged metal fracture texture, sks-defect texture, hairline crack, raw industrial damage, sharp details",
            "a macro photo of narrow broken metal fissure, oxidized crack on steel surface, industrial defect, high detail",
            "a macro photo of thin fractured metallic seam, dark industrial crack texture, raw damage, sharp detail",
        ]
        negative_pool = [
            "flat painted patch, smooth sealed line, intact glossy metal, polished defect-free surface",
            "perfect metal surface, shiny, glossy, no defect, clean industrial finish",
            "flat intact metallic surface, polished, clean texture, defect-free",
        ]
    return str(np.random.choice(positive_pool)), str(np.random.choice(negative_pool))


def choose_material_refine_prompt() -> tuple[str, str]:
    positive_pool = [
        "a macro photo of exposed fractured metallic substrate, rough inner wall microtexture, non-uniform cavity interior, subtle material variation, realistic industrial macro detail",
        "a macro photo of pitted steel interior, exposed metallic grain, irregular cavity floor texture, fractured inner wall, subtle material variation, realistic industrial macro detail",
        "a macro photo of broken steel cavity interior, rough exposed substrate, non-uniform inner texture, localized metallic grain, irregular inner wall shading",
    ]
    negative_pool = [
        "flat dark patch, uniform black blob, smooth painted cavity, filled hole, surface-only texture, glossy intact metal, raised bump, protrusion, blister, outward bulge, bead-like highlight",
        "sealed cavity, flat blob, fake sticker texture, uniform stain, polished smooth metal, welded bead, convex lump, outward highlight",
        "featureless dark region, flat filled region, no inner texture, no exposed substrate, raised nodule, protruding metallic bead",
    ]
    return str(np.random.choice(positive_pool)), str(np.random.choice(negative_pool))


def get_render_params(defect_type: str, cfg: GenerationConfig):
    if defect_type == "Spalling":
        return cfg.render_strength_spalling, cfg.render_guidance_spalling, list(cfg.control_scales_render_spalling)
    return cfg.render_strength_crack, cfg.render_guidance_crack, list(cfg.control_scales_render_crack)


def _safe_center_for_size(cx: float, cy: float, sx: float, sy: float, defect_type: str) -> tuple[float, float]:
    if defect_type == "Spalling":
        margin_x = max(68.0, sx * 1.7)
        margin_y = max(64.0, sy * 1.7)
    else:
        major = max(sx, sy)
        minor = min(sx, sy)
        margin_x = max(60.0, major * 1.45)
        margin_y = max(52.0, minor * 2.0)
    return float(np.clip(cx, margin_x, 511.0 - margin_x)), float(np.clip(cy, margin_y, 511.0 - margin_y))


def jitter_intent(intent, defect_type: str, max_dx_px: float = 90, max_dy_px: float = 70, max_theta_deg: float = 25):
    dx = float(np.random.uniform(-max_dx_px, max_dx_px))
    dy = float(np.random.uniform(-max_dy_px, max_dy_px))
    raw_cx = intent.pixel_center.x + dx
    raw_cy = intent.pixel_center.y + dy

    if defect_type == "Spalling":
        sx = float(np.clip(intent.pixel_scale.sigma_x * np.random.uniform(1.10, 1.32), 18, 104))
        sy = float(np.clip(intent.pixel_scale.sigma_y * np.random.uniform(1.08, 1.28), 14, 90))
    else:
        raw_sx = intent.pixel_scale.sigma_x
        raw_sy = intent.pixel_scale.sigma_y
        if raw_sx >= raw_sy:
            sx = raw_sx * np.random.uniform(1.16, 1.45)
            sy = raw_sy * np.random.uniform(1.00, 1.14)
        else:
            sy = raw_sy * np.random.uniform(1.16, 1.45)
            sx = raw_sx * np.random.uniform(1.00, 1.14)
        sx = float(np.clip(sx, 16, 120))
        sy = float(np.clip(sy, 8, 58))

    new_cx, new_cy = _safe_center_for_size(raw_cx, raw_cy, sx, sy, defect_type)
    theta = float(intent.theta_rad + np.deg2rad(np.random.uniform(-max_theta_deg, max_theta_deg)))
    return build_intent_from_pixel_measurements(new_cx, new_cy, sx, sy, theta, source=f"{intent.source}+jitter+tangentpatch")


def augment_prototype_mask(mask_u8: np.ndarray, defect_type: str) -> np.ndarray:
    mask = ensure_binary_mask(mask_u8).copy()
    if np.random.rand() < 0.65:
        h, w = mask.shape
        angle_deg = float(np.random.uniform(-18, 18))
        scale = float(np.random.uniform(0.98, 1.16))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    if defect_type == "Spalling":
        if np.random.rand() < 0.70:
            k = np.random.randint(5, 11)
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        if np.random.rand() < 0.35:
            k = np.random.randint(3, 7)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    else:
        if np.random.rand() < 0.65:
            kx = np.random.randint(6, 11)
            ky = np.random.randint(1, 3)
            mask = cv2.dilate(mask, np.ones((ky, kx), np.uint8), iterations=1)
        if np.random.rand() < 0.20:
            kx = np.random.randint(3, 5)
            ky = np.random.randint(1, 2)
            mask = cv2.erode(mask, np.ones((ky, kx), np.uint8), iterations=1)
    return ensure_binary_mask(mask)


def build_soft_render_mask(structure: StructuralState, defect_type: str) -> np.ndarray:
    causal = ensure_binary_mask(structure.causal_mask_u8)
    if defect_type == "Spalling":
        core = cv2.dilate(causal, np.ones((11, 11), np.uint8), iterations=1)
        halo = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0) / 255.0
        render_f = np.maximum(causal.astype(np.float32) / 255.0, 0.92 * halo)
    else:
        core = cv2.dilate(causal, np.ones((9, 5), np.uint8), iterations=1)
        halo = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=2.4, sigmaY=2.0) / 255.0
        render_f = np.maximum(causal.astype(np.float32) / 255.0, 0.96 * halo)
    return np.clip(render_f * 255.0, 0, 255).astype(np.uint8)


def build_core_refine_mask(causal_mask_u8: np.ndarray) -> np.ndarray:
    core = ensure_binary_mask(causal_mask_u8)
    core = cv2.erode(core, np.ones((7, 7), np.uint8), iterations=1)
    core = cv2.erode(core, np.ones((5, 5), np.uint8), iterations=1)
    if int(core.sum()) == 0:
        core = cv2.erode(ensure_binary_mask(causal_mask_u8), np.ones((3, 3), np.uint8), iterations=1)
    if int(core.sum()) == 0:
        core = ensure_binary_mask(causal_mask_u8)
    soft = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=2.8, sigmaY=2.8)
    return np.clip(soft, 0, 255).astype(np.uint8)


def build_residual_blend_mask(core_mask_u8: np.ndarray, soft_render_mask_u8: np.ndarray, cfg: GenerationConfig) -> np.ndarray:
    core_f = np.clip(core_mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    soft_f = np.clip(soft_render_mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    blend = np.maximum(cfg.residual_core_weight * core_f, cfg.residual_soft_render_weight * soft_f)
    blend = np.power(np.clip(blend, 0.0, 1.0), cfg.residual_mask_gamma)
    blend = cv2.GaussianBlur(
        blend.astype(np.float32),
        (0, 0),
        sigmaX=cfg.residual_mask_blur_sigma,
        sigmaY=cfg.residual_mask_blur_sigma,
    )
    return np.clip(blend * 255.0, 0, 255).astype(np.uint8)


def extract_roi(img_u8: np.ndarray, mask_u8: np.ndarray, padding: int, min_side: int):
    ys, xs = np.where(mask_u8 > 10)
    if len(xs) == 0:
        return img_u8.copy(), (0, 0, img_u8.shape[1], img_u8.shape[0])

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    w = max(x1 - x0 + 1 + 2 * padding, min_side)
    h = max(y1 - y0 + 1 + 2 * padding, min_side)
    side = max(w, h)

    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(img_u8.shape[1], x0 + side)
    y1 = min(img_u8.shape[0], y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)

    return img_u8[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def upscale_square(arr_u8: np.ndarray, factor: float, max_side: int, interp: int) -> np.ndarray:
    h, w = arr_u8.shape[:2]
    new_side = int(min(max_side, max(16, round(max(h, w) * factor))))
    return cv2.resize(arr_u8, (new_side, new_side), interpolation=interp)


def paste_roi(base_u8: np.ndarray, roi_u8: np.ndarray, bbox, alpha_mask_u8: np.ndarray | None = None, feather_sigma: float = 2.0):
    x0, y0, x1, y1 = bbox
    out = base_u8.copy()
    roi_resized = cv2.resize(roi_u8, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)

    if alpha_mask_u8 is None:
        out[y0:y1, x0:x1] = roi_resized
        return out

    alpha = cv2.resize(alpha_mask_u8, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    if feather_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma)
        alpha = np.clip(alpha, 0.0, 1.0)

    base_roi = out[y0:y1, x0:x1].astype(np.float32)
    roi_resized = roi_resized.astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(base_roi * (1.0 - alpha[..., None]) + roi_resized * alpha[..., None], 0, 255).astype(np.uint8)
    return out


def render_spalling_roi_supersampled(
    pipe,
    baseline_img_u8: np.ndarray,
    structure: StructuralState,
    positive_prompt: str,
    negative_prompt: str,
    cfg: GenerationConfig,
):
    render_strength, render_guidance, render_scales = get_render_params("Spalling", cfg)
    soft_render_mask_u8 = build_soft_render_mask(structure, "Spalling")

    img_roi, bbox = extract_roi(baseline_img_u8, soft_render_mask_u8, cfg.roi_padding_spalling, cfg.roi_min_side)
    mask_roi, _ = extract_roi(soft_render_mask_u8, soft_render_mask_u8, cfg.roi_padding_spalling, cfg.roi_min_side)
    normal_roi, _ = extract_roi(structure.defect_normal_u8, soft_render_mask_u8, cfg.roi_padding_spalling, cfg.roi_min_side)
    depth_roi, _ = extract_roi(structure.defect_depth_u8, soft_render_mask_u8, cfg.roi_padding_spalling, cfg.roi_min_side)

    img_up = upscale_square(img_roi, cfg.roi_supersample_factor_spalling, cfg.roi_max_side_after_upscale, cv2.INTER_CUBIC)
    mask_up = upscale_square(mask_roi, cfg.roi_supersample_factor_spalling, cfg.roi_max_side_after_upscale, cv2.INTER_CUBIC)
    normal_up = upscale_square(normal_roi, cfg.roi_supersample_factor_spalling, cfg.roi_max_side_after_upscale, cv2.INTER_CUBIC)
    depth_up = upscale_square(depth_roi, cfg.roi_supersample_factor_spalling, cfg.roi_max_side_after_upscale, cv2.INTER_CUBIC)

    rendered_up = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray(img_up),
        mask_image=Image.fromarray(mask_up).convert("L"),
        control_image=[Image.fromarray(normal_up), Image.fromarray(depth_up)],
        controlnet_conditioning_scale=render_scales,
        num_inference_steps=cfg.render_steps,
        strength=render_strength,
        guidance_scale=render_guidance,
    ).images[0]

    rendered_up_u8 = np.array(rendered_up, dtype=np.uint8)
    return paste_roi(baseline_img_u8, rendered_up_u8, bbox, alpha_mask_u8=mask_roi, feather_sigma=cfg.render_paste_feather_spalling), soft_render_mask_u8, bbox


def render_spalling_material_refine_candidate(
    pipe,
    first_pass_u8: np.ndarray,
    structure: StructuralState,
    cfg: GenerationConfig,
):
    refine_prompt, refine_negative = choose_material_refine_prompt()
    core_mask_u8 = build_core_refine_mask(structure.causal_mask_u8)

    img_roi, bbox = extract_roi(first_pass_u8, core_mask_u8, cfg.refine_roi_padding_spalling, cfg.refine_roi_min_side)
    mask_roi, _ = extract_roi(core_mask_u8, core_mask_u8, cfg.refine_roi_padding_spalling, cfg.refine_roi_min_side)
    normal_roi, _ = extract_roi(structure.defect_normal_u8, core_mask_u8, cfg.refine_roi_padding_spalling, cfg.refine_roi_min_side)
    depth_roi, _ = extract_roi(structure.defect_depth_u8, core_mask_u8, cfg.refine_roi_padding_spalling, cfg.refine_roi_min_side)

    img_up = upscale_square(img_roi, cfg.refine_roi_supersample_factor_spalling, cfg.refine_roi_max_side_after_upscale, cv2.INTER_CUBIC)
    mask_up = upscale_square(mask_roi, cfg.refine_roi_supersample_factor_spalling, cfg.refine_roi_max_side_after_upscale, cv2.INTER_CUBIC)
    normal_up = upscale_square(normal_roi, cfg.refine_roi_supersample_factor_spalling, cfg.refine_roi_max_side_after_upscale, cv2.INTER_CUBIC)
    depth_up = upscale_square(depth_roi, cfg.refine_roi_supersample_factor_spalling, cfg.refine_roi_max_side_after_upscale, cv2.INTER_CUBIC)

    rendered_up = pipe(
        prompt=refine_prompt,
        negative_prompt=refine_negative,
        image=Image.fromarray(img_up),
        mask_image=Image.fromarray(mask_up).convert("L"),
        control_image=[Image.fromarray(normal_up), Image.fromarray(depth_up)],
        controlnet_conditioning_scale=list(cfg.control_scales_refine_spalling),
        num_inference_steps=cfg.refine_steps,
        strength=cfg.refine_strength_spalling,
        guidance_scale=cfg.refine_guidance_spalling,
    ).images[0]

    rendered_up_u8 = np.array(rendered_up, dtype=np.uint8)
    refine_candidate_u8 = paste_roi(first_pass_u8, rendered_up_u8, bbox, alpha_mask_u8=mask_roi, feather_sigma=cfg.refine_paste_feather_spalling)
    return refine_candidate_u8, core_mask_u8, bbox, refine_prompt, refine_negative


def apply_material_residual(
    base_u8: np.ndarray,
    refine_candidate_u8: np.ndarray,
    core_mask_u8: np.ndarray,
    soft_render_mask_u8: np.ndarray,
    cfg: GenerationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    base_lab = cv2.cvtColor(base_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(refine_candidate_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    ref_blur = cv2.GaussianBlur(ref_lab, (0, 0), sigmaX=cfg.residual_blur_sigma, sigmaY=cfg.residual_blur_sigma)
    residual = ref_lab - ref_blur

    residual[..., 0] = cv2.GaussianBlur(
        residual[..., 0], (0, 0), sigmaX=cfg.residual_ring_guard_sigma, sigmaY=cfg.residual_ring_guard_sigma
    )
    residual[..., 1] = cv2.GaussianBlur(
        residual[..., 1], (0, 0), sigmaX=cfg.residual_ring_guard_sigma, sigmaY=cfg.residual_ring_guard_sigma
    )
    residual[..., 2] = cv2.GaussianBlur(
        residual[..., 2], (0, 0), sigmaX=cfg.residual_ring_guard_sigma, sigmaY=cfg.residual_ring_guard_sigma
    )

    blend_mask_u8 = build_residual_blend_mask(core_mask_u8, soft_render_mask_u8, cfg)
    mask = np.clip(blend_mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)

    res_l = residual[..., 0]
    pos_l = np.clip(res_l, 0.0, cfg.residual_positive_clip)
    neg_l = np.clip(res_l, -cfg.residual_negative_clip, 0.0)
    res_a = np.clip(residual[..., 1], -cfg.residual_negative_clip, cfg.residual_positive_clip)
    res_b = np.clip(residual[..., 2], -cfg.residual_negative_clip, cfg.residual_positive_clip)

    core_f = np.clip(core_mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    core_f = cv2.GaussianBlur(core_f, (0, 0), sigmaX=2.2, sigmaY=2.2)

    out_lab = base_lab.copy()
    out_lab[..., 0] = (
        out_lab[..., 0]
        + cfg.residual_luma_gain_pos * pos_l * mask
        + cfg.residual_luma_gain_neg * neg_l * mask
        - cfg.cavity_darkening_gain * core_f
    )
    out_lab[..., 1] = out_lab[..., 1] + cfg.residual_chroma_gain * res_a * mask
    out_lab[..., 2] = out_lab[..., 2] + cfg.residual_chroma_gain * res_b * mask

    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB), blend_mask_u8


def _masked_sharpness(gray_f32: np.ndarray, mask_bool: np.ndarray) -> float:
    if int(mask_bool.sum()) < 24:
        return 0.0
    lap = cv2.Laplacian(gray_f32, cv2.CV_32F, ksize=3)
    vals = np.abs(lap[mask_bool])
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def harmonize_local_sharpness(
    final_u8: np.ndarray,
    blend_mask_u8: np.ndarray,
    cfg: GenerationConfig,
) -> np.ndarray:
    if not cfg.sharpness_match_enabled:
        return final_u8

    mask_bin = ensure_binary_mask(blend_mask_u8)
    defect_bool = mask_bin > 0
    if int(defect_bool.sum()) < 40:
        return final_u8

    k_outer = np.ones((2 * cfg.sharpness_ring_dilate_px + 1, 2 * cfg.sharpness_ring_dilate_px + 1), np.uint8)
    k_gap = np.ones((2 * cfg.sharpness_ring_gap_px + 1, 2 * cfg.sharpness_ring_gap_px + 1), np.uint8)

    outer = cv2.dilate(mask_bin, k_outer, iterations=1)
    inner = cv2.dilate(mask_bin, k_gap, iterations=1)
    ring_bool = (outer > 0) & (~(inner > 0))
    if int(ring_bool.sum()) < 60:
        return final_u8

    gray = cv2.cvtColor(final_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sharp_bg = _masked_sharpness(gray, ring_bool)
    sharp_def = _masked_sharpness(gray, defect_bool)
    if sharp_bg <= 1e-6 or sharp_def <= cfg.sharpness_target_ratio * sharp_bg:
        return final_u8

    alpha = np.clip(mask_bin.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=cfg.sharpness_mask_blur_sigma, sigmaY=cfg.sharpness_mask_blur_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    best = final_u8.copy()
    for sigma in cfg.sharpness_sigma_candidates:
        if sigma <= 1e-6:
            candidate = final_u8.copy()
        else:
            blurred = cv2.GaussianBlur(final_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
            candidate = np.clip(
                final_u8.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)

        cand_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
        cand_sharp = _masked_sharpness(cand_gray, defect_bool)
        best = candidate
        if cand_sharp <= cfg.sharpness_target_ratio * sharp_bg:
            break
    return best


def render_crack_fullframe(
    pipe,
    baseline_img_u8: np.ndarray,
    structure: StructuralState,
    positive_prompt: str,
    negative_prompt: str,
    cfg: GenerationConfig,
):
    render_strength, render_guidance, render_scales = get_render_params("Crack", cfg)
    soft_mask_u8 = build_soft_render_mask(structure, "Crack")
    rendered = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        image=Image.fromarray(baseline_img_u8),
        mask_image=Image.fromarray(soft_mask_u8).convert("L"),
        control_image=[Image.fromarray(structure.defect_normal_u8), Image.fromarray(structure.defect_depth_u8)],
        controlnet_conditioning_scale=render_scales,
        num_inference_steps=cfg.render_steps,
        strength=render_strength,
        guidance_scale=render_guidance,
    ).images[0]
    return np.array(rendered, dtype=np.uint8), soft_mask_u8, None, None, None, None


def detect_visible_defect_generated(
    baseline_img_u8: np.ndarray,
    final_img_u8: np.ndarray,
    soft_mask_u8: np.ndarray,
    cfg: GenerationConfig,
) -> tuple[bool, dict]:
    mask = soft_mask_u8 > 20
    area = int(mask.sum())
    if area < cfg.min_mask_area_for_accept:
        return False, {"reason": "mask_area_too_small", "area": area}

    baseline_gray = cv2.cvtColor(baseline_img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    final_gray = cv2.cvtColor(final_img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = np.abs(final_gray - baseline_gray)
    vals = diff[mask]

    mean_abs_diff = float(vals.mean()) if vals.size > 0 else 0.0
    p95_abs_diff = float(np.percentile(vals, 95)) if vals.size > 0 else 0.0

    ok = (mean_abs_diff >= cfg.min_roi_mean_abs_diff) and (p95_abs_diff >= cfg.min_roi_p95_abs_diff)
    return ok, {
        "reason": "ok" if ok else "visible_change_too_weak",
        "area": area,
        "mean_abs_diff": mean_abs_diff,
        "p95_abs_diff": p95_abs_diff,
    }


def generate_one_sample_with_retry(
    main_pipe,
    spalling_render_pipe,
    dataset: GearCausalDataset,
    sniffer: AttentionSniffer,
    sample_index: int,
    cfg: GenerationConfig,
):
    last_debug = {}

    for attempt in range(1, cfg.max_attempts_per_sample + 1):
        baseline_img_u8 = render_background(main_pipe, dataset, cfg)
        baseline_pil = Image.fromarray(baseline_img_u8)
        defect_type = str(np.random.choice(["Spalling", "Crack"]))

        sniff_prompt = f"a macro photo of sks-defect texture on uniform background, {defect_type.lower()}"
        sniff_result = sniffer.sniff_intent(
            image=baseline_pil,
            prompt=sniff_prompt,
            target_words=["sks", "defect", defect_type.lower()],
            steps=cfg.sniff_steps,
            threshold_candidates=cfg.sniff_threshold_candidates,
        )

        effective_intent = jitter_intent(sniff_result.intent, defect_type)
        prototype_mask_u8 = augment_prototype_mask(sniff_result.prototype.connected_component_mask_u8, defect_type)

        structure = dataset.build_structural_state(
            intent=effective_intent,
            defect_type=defect_type,
            prototype_mask_u8=prototype_mask_u8,
        )

        positive_prompt, negative_prompt = choose_render_prompt(defect_type)

        refine_prompt = None
        refine_negative = None
        refine_bbox = None
        residual_mode = None

        if defect_type == "Spalling":
            first_pass_u8, soft_render_mask_u8, roi_bbox = render_spalling_roi_supersampled(
                spalling_render_pipe,
                baseline_img_u8,
                structure,
                positive_prompt,
                negative_prompt,
                cfg,
            )
            # 仅这一处切到旧权重：Spalling 内部材质 refine
            refine_candidate_u8, core_mask_u8, refine_bbox, refine_prompt, refine_negative = render_spalling_material_refine_candidate(
                spalling_render_pipe,
                first_pass_u8,
                structure,
                cfg,
            )
            final_u8, blend_mask_u8 = apply_material_residual(
                base_u8=first_pass_u8,
                refine_candidate_u8=refine_candidate_u8,
                core_mask_u8=core_mask_u8,
                soft_render_mask_u8=soft_render_mask_u8,
                cfg=cfg,
            )
            final_u8 = harmonize_local_sharpness(
                final_u8=final_u8,
                blend_mask_u8=blend_mask_u8,
                cfg=cfg,
            )
            diag_mask_u8 = blend_mask_u8
            accept_mask_u8 = np.maximum(soft_render_mask_u8, core_mask_u8)
            residual_mode = "spalling_material_pipe_only+sharpness_harmonized"
        else:
            final_u8, soft_render_mask_u8, roi_bbox, _, _, core_mask_u8 = render_crack_fullframe(
                main_pipe,
                baseline_img_u8,
                structure,
                positive_prompt,
                negative_prompt,
                cfg,
            )
            diag_mask_u8 = soft_render_mask_u8
            accept_mask_u8 = soft_render_mask_u8
            residual_mode = "none"

        ok, dbg = detect_visible_defect_generated(
            baseline_img_u8=baseline_img_u8,
            final_img_u8=final_u8,
            soft_mask_u8=accept_mask_u8,
            cfg=cfg,
        )
        dbg.update({
            "attempt": attempt,
            "defect_type": defect_type,
            "roi_bbox": roi_bbox,
            "refine_bbox": refine_bbox,
            "refine_prompt": refine_prompt,
            "refine_negative": refine_negative,
            "residual_mode": residual_mode,
        })
        last_debug = dbg

        if ok:
            structure = StructuralState(
                defect_type=structure.defect_type,
                baseline_normal_u8=structure.baseline_normal_u8,
                defect_normal_u8=structure.defect_normal_u8,
                defect_depth_u8=structure.defect_depth_u8,
                field_strength_u8=structure.field_strength_u8,
                causal_mask_u8=structure.causal_mask_u8,
                render_mask_u8=diag_mask_u8,
                factual_img_u8=structure.factual_img_u8,
                prototype_mask_u8=structure.prototype_mask_u8,
                defect_mask_bool=structure.defect_mask_bool,
                defect_depth_map=structure.defect_depth_map,
                normal_perturbation=structure.normal_perturbation,
                total_z=structure.total_z,
                base_normal_field=structure.base_normal_field,
                defect_normal_field=structure.defect_normal_field,
                row_profile_y_px=structure.row_profile_y_px,
                intent=structure.intent,
            )

            sample = GeneratedSample(
                defect_type=defect_type,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                baseline_image_u8=baseline_img_u8,
                final_image_u8=final_u8,
                sniff_result=sniff_result,
                structure=structure,
            )
            return sample, last_debug

    raise RuntimeError(f"生成失败：连续 {cfg.max_attempts_per_sample} 次都未检测到足够可见的缺陷。最后一次调试信息: {last_debug}")


def save_profile_plot(structure: StructuralState, save_path: Path, dataset: GearCausalDataset):
    row_idx = structure.row_profile_y_px
    X_slice = dataset.X[row_idx, :]
    Z_base = dataset.Z[row_idx, :]
    Z_defect = structure.total_z[row_idx, :]
    N_defect = structure.defect_normal_field[row_idx, :]

    plt.figure(figsize=(15, 6))
    plt.plot(X_slice, Z_base, label="Base Profile", alpha=0.3)
    plt.plot(X_slice, Z_defect, label="With Defect", linewidth=2)
    step = 8
    plt.quiver(
        X_slice[::step],
        Z_defect[::step],
        N_defect[::step, 0],
        N_defect[::step, 2],
        scale=15,
        width=0.003,
        label="Normal Vectors",
    )
    plt.title(f"3D Topology Analysis (row={row_idx}, defect={structure.defect_type})")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(save_path, dpi=300)
    plt.close()



SUMMARY_KEYS = [
    ("05_canonical_prototype_patch.png", "05 prototype"),
    ("06_projected_strength.png", "06 projected"),
    ("07_structural_mask.png", "07 structural"),
    ("09_defect_normal.png", "09 normal"),
    ("11_final.png", "11 final"),
]


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fit_rgb(img: Image.Image, target_w: int, target_h: int, bg=(255, 255, 255)) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def save_sample_overview(save_dir: Path, thumb_w: int = 220, thumb_h: int = 220):
    title_font = _load_font(20, bold=True)
    text_font = _load_font(16, bold=False)

    margin = 16
    gap = 12
    label_h = 24
    width = margin * 2 + len(SUMMARY_KEYS) * thumb_w + (len(SUMMARY_KEYS) - 1) * gap
    height = margin * 2 + 28 + label_h + thumb_h
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"{save_dir.name} overview", font=title_font, fill=(30, 30, 30))

    x = margin
    y = margin + 34
    for fname, label in SUMMARY_KEYS:
        p = save_dir / fname
        if p.exists():
            tile = _fit_rgb(Image.open(p), thumb_w, thumb_h)
        else:
            tile = Image.new("RGB", (thumb_w, thumb_h), (235, 235, 235))
            d = ImageDraw.Draw(tile)
            d.text((10, 10), "missing", font=text_font, fill=(120, 120, 120))
        draw.text((x + 4, y), label, font=text_font, fill=(50, 50, 50))
        canvas.paste(tile, (x, y + label_h))
        draw.rectangle([x, y + label_h, x + thumb_w - 1, y + label_h + thumb_h - 1], outline=(180, 180, 180), width=1)
        x += thumb_w + gap

    canvas.save(save_dir / "00_sample_overview.png")


def save_run_contact_sheet(run_dir: Path, thumb_w: int = 220, thumb_h: int = 220, sample_limit: int | None = None):
    sample_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    if sample_limit is not None:
        sample_dirs = sample_dirs[:sample_limit]
    if not sample_dirs:
        return

    title_font = _load_font(24, bold=True)
    header_font = _load_font(18, bold=True)
    text_font = _load_font(16, bold=False)

    margin = 18
    gap_x = 14
    gap_y = 16
    row_title_w = 120
    width = margin * 2 + row_title_w + len(SUMMARY_KEYS) * thumb_w + (len(SUMMARY_KEYS) - 1) * gap_x
    height = margin * 2 + 40 + 28 + len(sample_dirs) * thumb_h + len(sample_dirs) * 24 + (len(sample_dirs) - 1) * gap_y + 28

    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"Run Summary: {run_dir.name}", font=title_font, fill=(30, 30, 30))

    x = margin + row_title_w
    y = margin + 44
    for _, label in SUMMARY_KEYS:
        draw.text((x + 4, y), label, font=header_font, fill=(45, 45, 45))
        x += thumb_w + gap_x

    y += 28
    for sample_dir in sample_dirs:
        draw.text((margin, y + thumb_h // 2 - 8), sample_dir.name, font=text_font, fill=(50, 50, 50))
        x = margin + row_title_w
        for fname, _ in SUMMARY_KEYS:
            p = sample_dir / fname
            if p.exists():
                tile = _fit_rgb(Image.open(p), thumb_w, thumb_h)
            else:
                tile = Image.new("RGB", (thumb_w, thumb_h), (235, 235, 235))
                d = ImageDraw.Draw(tile)
                d.text((10, 10), "missing", font=text_font, fill=(120, 120, 120))
            canvas.paste(tile, (x, y + 24))
            draw.rectangle([x, y + 24, x + thumb_w - 1, y + 24 + thumb_h - 1], outline=(180, 180, 180), width=1)
            x += thumb_w + gap_x
        y += thumb_h + 24 + gap_y

    canvas.save(run_dir / "00_run_summary.png")

def _bbox_to_yolo_line(x_min: int, y_min: int, x_max: int, y_max: int, class_id: int, image_w: int = 512, image_h: int = 512) -> str:
    xc = ((x_min + x_max) / 2.0) / image_w
    yc = ((y_min + y_max) / 2.0) / image_h
    w = (x_max - x_min) / image_w
    h = (y_max - y_min) / image_h
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    mask = ensure_binary_mask(mask_u8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _mask_bounds(mask_u8: np.ndarray):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _clip_box_xyxy(x_min: int, y_min: int, x_max: int, y_max: int, image_w: int = 512, image_h: int = 512):
    x_min = int(np.clip(x_min, 0, image_w - 1))
    y_min = int(np.clip(y_min, 0, image_h - 1))
    x_max = int(np.clip(x_max, 0, image_w - 1))
    y_max = int(np.clip(y_max, 0, image_h - 1))
    if x_max <= x_min:
        x_max = min(image_w - 1, x_min + 1)
    if y_max <= y_min:
        y_max = min(image_h - 1, y_min + 1)
    return x_min, y_min, x_max, y_max


def _expand_rect(bounds, margin: int = 2, image_w: int = 512, image_h: int = 512):
    x_min, y_min, x_max, y_max = bounds
    return _clip_box_xyxy(x_min - margin, y_min - margin, x_max + margin, y_max + margin, image_w, image_h)


def _square_from_bounds(bounds, margin: int = 3, image_w: int = 512, image_h: int = 512):
    x_min, y_min, x_max, y_max = bounds
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    side = max(w, h) + 2 * margin
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))
    return _clip_box_xyxy(x0, y0, x1, y1, image_w, image_h)


def build_export_bbox(sample: GeneratedSample, image_w: int = 512, image_h: int = 512):
    """构建更紧、且按形状自适应的导出框。

    原则：
    - 不再直接使用 render_mask，因为它包含 halo，框通常偏大。
    - 先从 causal_mask 中取最大连通域，尽量贴近真正缺陷本体。
    - Spalling 不再一律外切正方形；先看长宽比：
        * 若近似团块（长宽比 <= 1.8），使用外切正方形；
        * 若明显细长（长宽比 > 1.8），直接使用紧矩形，避免左右过宽。
    - Crack：使用普通外接矩形。

    保留旧逻辑作为兜底：如果 causal_mask 为空，再退回 render_mask。
    """
    # 旧逻辑（保留备查，不直接删除）
    # legacy_bounds = compute_component_bounds(sample.structure.render_mask_u8, padding=2)

    primary_mask = _largest_component(sample.structure.causal_mask_u8)
    bounds = _mask_bounds(primary_mask)

    fallback_source = "causal_mask"
    if bounds is None:
        fallback_source = "render_mask_fallback"
        fallback_mask = _largest_component(sample.structure.render_mask_u8)
        bounds = _mask_bounds(fallback_mask)

    if bounds is None:
        raise RuntimeError("无有效缺陷框，causal_mask 与 render_mask 均为空。")

    x_min, y_min, x_max, y_max = bounds
    comp_w = x_max - x_min + 1
    comp_h = y_max - y_min + 1
    aspect_ratio = max(comp_w, comp_h) / max(min(comp_w, comp_h), 1)

    if sample.defect_type == "Spalling":
        if aspect_ratio <= 1.8:
            box = _square_from_bounds(bounds, margin=3, image_w=image_w, image_h=image_h)
            box_mode = "largest_component_square_aspectaware"
        else:
            box = _expand_rect(bounds, margin=2, image_w=image_w, image_h=image_h)
            box_mode = "largest_component_rect_aspectaware_spalling"
    else:
        box = _expand_rect(bounds, margin=2, image_w=image_w, image_h=image_h)
        box_mode = "largest_component_rect"

    component_area = int(np.count_nonzero(primary_mask if fallback_source == "causal_mask" else fallback_mask))

    return box, {
        "box_mode": box_mode,
        "box_source": fallback_source,
        "component_bounds_xyxy": list(bounds),
        "component_w": int(comp_w),
        "component_h": int(comp_h),
        "component_area": int(component_area),
        "component_aspect_ratio": float(aspect_ratio),
    }


def evaluate_export_candidate(sample: GeneratedSample, bbox_xyxy, box_meta: dict, cfg: GenerationConfig, image_w: int = 512, image_h: int = 512):
    x_min, y_min, x_max, y_max = bbox_xyxy
    box_w = int(x_max - x_min + 1)
    box_h = int(y_max - y_min + 1)
    box_area = int(box_w * box_h)
    box_area_ratio = float(box_area / float(image_w * image_h))

    component_area = int(box_meta.get("component_area", 0))
    component_w = int(box_meta.get("component_w", 0))
    component_h = int(box_meta.get("component_h", 0))
    fill_ratio = float(component_area / max(box_area, 1))
    aspect_ratio = float(max(component_w, component_h) / max(min(component_w, component_h), 1))

    reject = False
    reasons: list[str] = []
    suspect_big_box = False

    if cfg.enable_conservative_big_box_redraw:
        if box_w > int(cfg.max_box_width_px):
            reject = True
            reasons.append("box_width_over_limit")
        if box_h > int(cfg.max_box_height_px):
            reject = True
            reasons.append("box_height_over_limit")
        if (box_w > int(cfg.max_box_width_px * 0.85)) or (box_h > int(cfg.max_box_height_px * 0.85)):
            suspect_big_box = True

    return reject, {
        "box_w": box_w,
        "box_h": box_h,
        "box_area": box_area,
        "box_area_ratio": box_area_ratio,
        "fill_ratio": fill_ratio,
        "component_area": component_area,
        "component_aspect_ratio": aspect_ratio,
        "suspect_big_box": bool(suspect_big_box),
        "reject_reasons": reasons,
    }


REQUIRED_SAMPLE_FILES = (
    "factual.png",
    "defect_hq.png",
    "defect_lq.png",
    "mask.png",
    "meta.json",
)


def _iter_sample_dirs(export_root: Path) -> list[Path]:
    if not export_root.exists():
        return []
    return sorted([p for p in export_root.iterdir() if p.is_dir() and p.name.startswith("sample_")])


def _is_complete_sample_dir(sample_dir: Path) -> bool:
    return all((sample_dir / name).exists() for name in REQUIRED_SAMPLE_FILES)


def _existing_export_stems(export_root: Path) -> list[str]:
    return [p.name for p in _iter_sample_dirs(export_root) if _is_complete_sample_dir(p)]


def _all_export_stems(export_root: Path) -> list[str]:
    return [p.name for p in _iter_sample_dirs(export_root)]


def _next_sample_index_from_existing(stems: list[str]) -> int:
    max_idx = -1
    for stem in stems:
        if stem.startswith("sample_"):
            try:
                max_idx = max(max_idx, int(stem.split("_")[-1]))
            except Exception:
                pass
    return max_idx + 1


def _build_sample_id(index: int) -> str:
    return f"sample_{index:06d}"


def _count_existing_by_class(export_root: Path) -> dict[str, int]:
    counts = {"Spalling": 0, "Crack": 0}
    for sample_dir in _iter_sample_dirs(export_root):
        meta_path = sample_dir / "meta.json"
        if not _is_complete_sample_dir(sample_dir) or (not meta_path.exists()):
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            defect_type = str(data.get("defect_type", ""))
            if defect_type in counts:
                counts[defect_type] += 1
        except Exception:
            continue
    return counts


def _normalize_subtype_name(defect_type: str) -> str:
    mapping = {
        "Spalling": "spalling",
        "Crack": "crack",
    }
    return mapping.get(str(defect_type), str(defect_type).lower())


def _derive_region_id_from_bbox(bbox_xyxy: tuple[int, int, int, int], image_size: int = 512, grid_size: int = 4) -> str:
    x_min, y_min, x_max, y_max = bbox_xyxy
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cell = image_size / float(grid_size)
    col = int(np.clip(cx // cell, 0, grid_size - 1))
    row = int(np.clip(cy // cell, 0, grid_size - 1))
    return f"region_r{row:02d}_c{col:02d}"


def _degrade_defect_image(defect_hq_u8: np.ndarray, level: int = 1) -> np.ndarray:
    img = np.asarray(defect_hq_u8, dtype=np.uint8)
    h, w = img.shape[:2]
    if level <= 1:
        down = cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(up, (3, 3), 0.8)
        bgr = cv2.cvtColor(blur, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return blur

    down = cv2.resize(img, (max(1, w // 3), max(1, h // 3)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(up, (5, 5), 1.2)
    return blur


def save_production_sample(
    sample: GeneratedSample,
    sample_id: str,
    export_root: Path,
    retry_debug: dict | None = None,
    precomputed_bbox: tuple[int, int, int, int] | None = None,
    precomputed_box_meta: dict | None = None,
    screening_meta: dict | None = None,
):
    sample_dir = export_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    if precomputed_bbox is None or precomputed_box_meta is None:
        bbox_xyxy, box_meta = build_export_bbox(sample)
    else:
        bbox_xyxy = precomputed_bbox
        box_meta = dict(precomputed_box_meta)

    factual_u8 = np.asarray(sample.baseline_image_u8, dtype=np.uint8)
    defect_hq_u8 = np.asarray(sample.final_image_u8, dtype=np.uint8)
    defect_lq_u8 = _degrade_defect_image(defect_hq_u8, level=1)
    mask_u8 = ensure_binary_mask(sample.structure.causal_mask_u8)

    subtype = _normalize_subtype_name(sample.defect_type)
    region_id = _derive_region_id_from_bbox(bbox_xyxy)

    factual_path = sample_dir / "factual.png"
    defect_hq_path = sample_dir / "defect_hq.png"
    defect_lq_path = sample_dir / "defect_lq.png"
    mask_path = sample_dir / "mask.png"
    meta_path = sample_dir / "meta.json"

    Image.fromarray(factual_u8).save(factual_path)
    Image.fromarray(defect_hq_u8).save(defect_hq_path)
    Image.fromarray(defect_lq_u8).save(defect_lq_path)
    Image.fromarray(mask_u8).save(mask_path)

    meta = {
        "sample_id": sample_id,
        "class": "defect",
        "subtype": subtype,
        "defect_type": sample.defect_type,
        "base_id": f"base_{sample_id}",
        "region_id": region_id,
        "view_id": "view_00",
        "lighting_id": "light_00",
        "bbox_xyxy": list(bbox_xyxy),
        "bbox_meta": box_meta,
        "screening_meta": screening_meta or {},
        "mask_area": int(np.count_nonzero(mask_u8)),
        "pixel_center": [sample.structure.intent.pixel_center.x, sample.structure.intent.pixel_center.y],
        "pixel_scale": [sample.structure.intent.pixel_scale.sigma_x, sample.structure.intent.pixel_scale.sigma_y],
        "theta_deg": sample.structure.intent.theta_deg,
        "used_fallback": sample.sniff_result.prototype.used_fallback,
        "paths": {
            "factual": str(factual_path),
            "defect_hq": str(defect_hq_path),
            "defect_lq": str(defect_lq_path),
            "mask": str(mask_path),
        },
        "retry_debug": retry_debug or {},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def save_diagnostics(sample: GeneratedSample, save_dir: Path, dataset: GearCausalDataset, retry_debug: dict | None = None):
    save_dir.mkdir(parents=True, exist_ok=True)

    # 生产版默认不再保存中间图，以下代码保留为论文/排查用途，按需取消注释。
    # Image.fromarray(sample.baseline_image_u8).save(save_dir / "01_background.png")
    # Image.fromarray(sample.sniff_result.prototype.heatmap_u8).save(save_dir / "02_attention_heatmap.png")
    # Image.fromarray(sample.sniff_result.prototype.binary_mask_u8).save(save_dir / "03_attention_binary.png")
    # Image.fromarray(sample.sniff_result.prototype.connected_component_mask_u8).save(save_dir / "04_attention_component.png")
    # Image.fromarray(sample.structure.prototype_mask_u8).save(save_dir / "05_canonical_prototype_patch.png")
    # Image.fromarray(sample.structure.field_strength_u8).save(save_dir / "06_projected_strength.png")
    # Image.fromarray(sample.structure.causal_mask_u8).save(save_dir / "07_structural_mask.png")
    # Image.fromarray(sample.structure.render_mask_u8).save(save_dir / "08_render_mask.png")
    # Image.fromarray(sample.structure.defect_normal_u8).save(save_dir / "09_defect_normal.png")
    # Image.fromarray(sample.structure.defect_depth_u8).save(save_dir / "10_defect_depth.png")
    # Image.fromarray(sample.final_image_u8).save(save_dir / "11_final.png")
    #
    # diag_img = sample.final_image_u8[:, :, ::-1].copy()
    # bounds = compute_component_bounds(sample.structure.render_mask_u8, padding=15)
    # if bounds is not None:
    #     x_min, y_min, x_max, y_max = bounds
    #     cv2.rectangle(diag_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    #     cv2.putText(
    #         diag_img,
    #         f"{sample.defect_type} | source={sample.sniff_result.intent.source}",
    #         (x_min, max(15, y_min - 10)),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.45,
    #         (0, 0, 255),
    #         1,
    #     )
    # cv2.imwrite(str(save_dir / "12_diagnostic_box.png"), diag_img)
    # save_profile_plot(sample.structure, save_dir / "13_vector_profile.png", dataset)
    # meta = {
    #     "defect_type": sample.defect_type,
    #     "pixel_center": [sample.structure.intent.pixel_center.x, sample.structure.intent.pixel_center.y],
    #     "physical_center": [sample.structure.intent.physical_center.x, sample.structure.intent.physical_center.y],
    #     "pixel_scale": [sample.structure.intent.pixel_scale.sigma_x, sample.structure.intent.pixel_scale.sigma_y],
    #     "physical_scale": [sample.structure.intent.physical_scale.sigma_x, sample.structure.intent.physical_scale.sigma_y],
    #     "theta_rad": sample.structure.intent.theta_rad,
    #     "theta_deg": sample.structure.intent.theta_deg,
    #     "used_fallback": sample.sniff_result.prototype.used_fallback,
    #     "retry_debug": retry_debug or {},
    # }
    # (save_dir / "14_metadata.txt").write_text(str(meta), encoding="utf-8")
    # save_sample_overview(save_dir)
    pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_root = OUTPUT_DIR / "defect"
    export_root.mkdir(parents=True, exist_ok=True)
    cfg = GenerationConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    existing_stems = _existing_export_stems(export_root) if cfg.resume_from_existing else []
    existing_count = len(existing_stems)
    all_existing_stems = _all_export_stems(export_root) if cfg.resume_from_existing else []
    next_index = _next_sample_index_from_existing(all_existing_stems) if cfg.resume_from_existing else 0
    remaining = max(int(cfg.target_total_samples) - existing_count, 0)
    existing_class_counts = _count_existing_by_class(export_root) if cfg.resume_from_existing else {"Spalling": 0, "Crack": 0}

    print("🚀 启动 sy-grid intrinsic refactored 版：true (s,y) state + local-normal depth")
    print("   - Spalling/Crack 几何：support/depth/core 全程保留在 intrinsic (s,y) grid，最后才导出 visible/control")
    print("   - 主权重：背景、sniffer、shape、Crack")
    print("   - v12.1 权重：Spalling 第一遍 render + 第二遍材质 refine")
    print("   - Spalling 仅在明显过大时才 shrink clamp，正常样本不过度缩小")
    print(f"   - 目标导出总数: {cfg.target_total_samples}")
    print(f"   - 已有样本数: {existing_count}")
    print(f"   - 还需补齐: {remaining}")
    incomplete_count = max(len(all_existing_stems) - existing_count, 0) if cfg.resume_from_existing else 0
    print(f"   - 当前类别计数: {existing_class_counts}")
    print(f"   - 不完整样本组: {incomplete_count}")

    if remaining <= 0:
        print("✅ 当前 export_root 已达到目标数量，无需继续生成。")
        print("   如有手动筛选后留下的不完整样本组，先运行清理脚本再继续。")
        return

    dataset = GearCausalDataset(res=512)
    main_pipe = load_pipeline(device, MAIN_LORA_PATH)
    spalling_render_pipe = load_pipeline(device, SPALLING_RENDER_LORA_PATH)
    sniffer = AttentionSniffer(main_pipe)

    success_count = 0
    total_failure_count = 0
    consecutive_failure_count = 0
    class_counts = dict(existing_class_counts)

    while success_count < remaining:
        absolute_idx = next_index + success_count + total_failure_count
        sample_id = _build_sample_id(absolute_idx)
        print(f"\n==== 生成样本 {success_count + 1}/{remaining} | sample_id={sample_id} ====")
        try:
            sample, retry_debug = generate_one_sample_with_retry(
                main_pipe,
                spalling_render_pipe,
                dataset,
                sniffer,
                absolute_idx,
                cfg,
            )
            bbox_xyxy, box_meta = build_export_bbox(sample)
            reject, screening_meta = evaluate_export_candidate(sample, bbox_xyxy, box_meta, cfg)
            if reject:
                total_failure_count += 1
                consecutive_failure_count += 1
                print(
                    f"↻ 筛选重绘 | sample_id={sample_id} | defect={sample.defect_type} | "
                    f"box={screening_meta.get('box_w')}x{screening_meta.get('box_h')} | "
                    f"fill={screening_meta.get('fill_ratio', 0.0):.3f} | "
                    f"area_ratio={screening_meta.get('box_area_ratio', 0.0):.4f} | "
                    f"reasons={screening_meta.get('reject_reasons', [])} | "
                    f"consecutive_failures={consecutive_failure_count}/{cfg.max_consecutive_failures} | "
                    f"total_failures={total_failure_count}"
                )
                if consecutive_failure_count >= int(cfg.max_consecutive_failures):
                    raise RuntimeError(
                        f"连续失败次数已达到上限 {cfg.max_consecutive_failures}，已停止生成。累计失败次数为 {total_failure_count}。"
                    )
                continue

            save_production_sample(
                sample,
                sample_id,
                export_root,
                retry_debug=retry_debug,
                precomputed_bbox=bbox_xyxy,
                precomputed_box_meta=box_meta,
                screening_meta=screening_meta,
            )
            class_counts[sample.defect_type] = class_counts.get(sample.defect_type, 0) + 1
            success_count += 1
            consecutive_failure_count = 0
            if (success_count % max(int(cfg.progress_print_every), 1) == 0) or (success_count == remaining):
                print(
                    f"📦 已导出 {success_count}/{remaining} | 总计={existing_count + success_count} | "
                    f"Spalling={class_counts.get('Spalling', 0)} | Crack={class_counts.get('Crack', 0)}"
                )
            print(f"✅ 完成 {sample_id} | defect={sample.defect_type} | attempts={retry_debug.get('attempt', 1)}")
        except Exception as exc:
            total_failure_count += 1
            consecutive_failure_count += 1
            print(
                f"⚠️ 生成失败，自动重绘下一张 | "
                f"consecutive_failures={consecutive_failure_count}/{cfg.max_consecutive_failures} | "
                f"total_failures={total_failure_count} | error={exc}"
            )
            if consecutive_failure_count >= int(cfg.max_consecutive_failures):
                raise RuntimeError(
                    f"连续失败次数已达到上限 {cfg.max_consecutive_failures}，已停止生成。累计失败次数为 {total_failure_count}。"
                ) from exc
            continue

    print("\n🎉 导出完成")
    print(f"   - 目标总数: {cfg.target_total_samples}")
    print(f"   - 实际总数: {existing_count + success_count}")
    print(f"   - 本轮新增: {success_count}")
    print(f"   - 本轮累计失败/重绘次数: {total_failure_count}")
    print(f"   - 最终类别计数: {class_counts}")


if __name__ == "__main__":
    main()
