from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def ensure_bin(mask: np.ndarray) -> np.ndarray:
    return ((mask > 127).astype(np.uint8) * 255)


def make_boundary_band(mask: np.ndarray, dilate_px: int = 6, erode_px: int = 1) -> np.ndarray:
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    dil = cv2.dilate(mask, kd, iterations=1)
    ero = cv2.erode(mask, ke, iterations=1)
    band = ((dil > 0) & (ero == 0)).astype(np.uint8) * 255
    return band


def preview_band(img_rgb: np.ndarray, band: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    sel = band > 0
    out[sel] = (0.7 * out[sel] + 0.3 * np.array([255, 120, 0])).astype(np.uint8)
    return out


def to_gray3(img_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[:, :, None], 3, axis=2)


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_rgb(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="initial_composite.png")
    ap.add_argument("--strict-mask", required=True, help="strict_mask.png")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="/root/autodl-tmp/models/sd15_inpaint_single/sd-v1-5-inpainting.ckpt")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--strength", type=float, default=0.22)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--band-dilate", type=int, default=6)
    ap.add_argument("--band-erode", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.array(pil_rgb(Path(args.image)))
    mask = cv2.imread(str(Path(args.strict_mask)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(args.strict_mask)
    mask = ensure_bin(mask)

    band = make_boundary_band(mask, dilate_px=args.band_dilate, erode_px=args.band_erode)
    band_preview = preview_band(img, band)

    save_rgb(out_dir / "input_image.png", img)
    cv2.imwrite(str(out_dir / "strict_mask.png"), mask)
    cv2.imwrite(str(out_dir / "boundary_band.png"), band)
    save_rgb(out_dir / "boundary_band_preview.png", band_preview)

    pipe = StableDiffusionInpaintPipeline.from_single_file(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    prompt = (
        "grayscale industrial macro photo, preserve the exact defect body and exact defect shape, "
        "only refine the local boundary transition near the mask band, "
        "make the pasted defect blend naturally into the surrounding metal surface, "
        "keep the defect clearly visible, keep the interior mostly unchanged, "
        "simple realistic defect texture, crisp but natural edge, no extra defects, no color"
    )
    negative_prompt = (
        "change defect shape, move defect, resize defect, remove defect, blur whole image, "
        "colorful stain, orange rust, blue rust, extra scratches, extra pits, extra defects, "
        "heavy texture everywhere, overpainted smooth patch"
    )

    init_pil = Image.fromarray(to_gray3(img))
    mask_pil = Image.fromarray(band).convert("L")

    for i in range(args.variants):
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed + i)
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_pil,
            mask_image=mask_pil,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.strength,
            generator=gen,
        ).images[0]

        arr = np.array(out.convert("RGB"))
        arr = to_gray3(arr)  # 强制回灰度
        save_rgb(out_dir / f"variant_{i:02d}.png", arr)

    print(f"[OK] done: {out_dir}")


if __name__ == "__main__":
    main()
