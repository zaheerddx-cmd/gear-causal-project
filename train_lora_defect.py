from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/hf_cache/torch_hub"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm


MODEL_ID = "/root/autodl-tmp/hf_cache/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
OUTPUT_DIR = "/root/autodl-tmp/Gear_Causal_Project_v2/weights"

BACKGROUND_DIR = "/root/autodl-tmp/Gear_Causal_Project_v2/data/aligned_baseline"
DEFECT_DIR = "/root/autodl-tmp/Gear_Causal_Project_v2/data/train_defects_cropped_512_plain_v1"

BACKGROUND_PROMPT = "a macro photo of cqc matte granular industrial metal surface"
DEFECT_PROMPT = (
    "a macro photo of sks industrial metal defect, recessed metallic cavity, "
    "spall and crack texture, fractured inner wall, damaged surface on uniform matte background"
)

WEIGHT_NAME = "gear_and_defect_from_scratch_explainable_v13_defectonly_testsource_v8_plainscale.safetensors"


class DefectOnlyDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        bg_dir: str,
        defect_dir: str,
        size: int = 512,
        bg_repeats: int = 12,
        defect_repeats: int = 8,
        seed: int = 42,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.instances = []
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.transform_bg = transforms.Compose([
            transforms.Lambda(lambda img: img.transpose(Image.ROTATE_270)),
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # defect-only：不做几何大扰动，不做 flip/crop
        self.transform_defect = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.03, contrast=0.03),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        bg_prompt_ids = self.tokenizer(
            BACKGROUND_PROMPT,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        defect_prompt_ids = self.tokenizer(
            DEFECT_PROMPT,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        self._append_dir(bg_dir, bg_prompt_ids, "bg", bg_repeats)
        self._append_dir(defect_dir, defect_prompt_ids, "defect", defect_repeats)

        random.shuffle(self.instances)
        print(f"📦 数据集加载完成，总样本数: {len(self.instances)}")
        print(f"   - background prompt: {BACKGROUND_PROMPT}")
        print(f"   - defect prompt:     {DEFECT_PROMPT}")
        print(f"   - repeats(bg/defect)=({bg_repeats}/{defect_repeats})")

    def _append_dir(self, data_dir: str, prompt_ids: torch.Tensor, kind: str, repeats: int):
        d = Path(data_dir)
        if not d.exists():
            raise FileNotFoundError(f"❌ 找不到数据目录: {d}")

        img_paths = sorted([
            p for p in d.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        ])
        if not img_paths:
            raise RuntimeError(f"❌ 目录中没有图片: {d}")

        for path in img_paths:
            for _ in range(repeats):
                self.instances.append({
                    "path": str(path),
                    "prompt_ids": prompt_ids,
                    "kind": kind,
                })

    def _make_random_matte_neutral_bg(self, w: int, h: int) -> Image.Image:
        base = self.rng.integers(112, 146, size=(h, w), dtype=np.uint8).astype(np.float32)
        noise_low = self.rng.normal(0, 1, (h, w)).astype(np.float32)
        noise_low = cv2.GaussianBlur(noise_low, (0, 0), sigmaX=7.0, sigmaY=7.0)
        noise_hi = self.rng.normal(0, 1, (h, w)).astype(np.float32)
        noise_hi = cv2.GaussianBlur(noise_hi, (0, 0), sigmaX=1.0, sigmaY=3.0)
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        gentle_grad = 4.5 * yy + 2.5 * xx
        img = base + 5.0 * noise_low + 2.0 * noise_hi + gentle_grad
        img = np.clip(img, 96, 168).astype(np.uint8)
        rgb = np.stack([img, img, img], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        item = self.instances[idx]
        img = Image.open(item["path"])

        # 若有透明通道，自动合成到中性磨砂背景；普通 RGB 灰底图则直接原样训练
        if item["kind"] == "defect" and (
            img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        ):
            img = img.convert("RGBA")
            background = self._make_random_matte_neutral_bg(*img.size).convert("RGBA")
            img = Image.alpha_composite(background, img).convert("RGB")
        else:
            img = img.convert("RGB")

        if item["kind"] == "bg":
            pixel_values = self.transform_bg(img)
        else:
            pixel_values = self.transform_defect(img)

        return {
            "pixel_values": pixel_values,
            "input_ids": item["prompt_ids"],
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    print("🚀 启动 defect-only LoRA 训练")
    print(f"   background dir: {BACKGROUND_DIR}")
    print(f"   defect dir:     {DEFECT_DIR}")
    print(f"   output weight:  {Path(OUTPUT_DIR) / WEIGHT_NAME}")
    print(f"   device:         {device}")

    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        local_files_only=True,
    ).to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID,
        subfolder="unet",
        local_files_only=True,
    ).to(device, dtype=weight_dtype)
    scheduler = DDPMScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler",
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        local_files_only=True,
    ).to(device, dtype=torch.float32)
    vae.requires_grad_(False)

    unet.requires_grad_(False)
    unet_lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    text_encoder.requires_grad_(False)
    text_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "v_proj"],
    )
    text_encoder = get_peft_model(text_encoder, text_lora_config)

    unet_lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in unet_lora_params:
        p.data = p.data.to(torch.float32)

    text_lora_params = [p for p in text_encoder.parameters() if p.requires_grad]
    for p in text_lora_params:
        p.data = p.data.to(torch.float32)

    optimizer = torch.optim.AdamW([
        {"params": unet_lora_params, "lr": args.lr_unet},
        {"params": text_lora_params, "lr": args.lr_text},
    ])

    scaler = torch.amp.GradScaler("cuda")

    dataset = DefectOnlyDataset(
        tokenizer=tokenizer,
        bg_dir=BACKGROUND_DIR,
        defect_dir=DEFECT_DIR,
        size=args.resolution,
        bg_repeats=args.bg_repeats,
        defect_repeats=args.defect_repeats,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    unet.train()
    text_encoder.train()

    progress_bar = tqdm(total=args.num_steps)
    global_step = 0

    while global_step < args.num_steps:
        for batch in dataloader:
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda"):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1
            if global_step >= args.num_steps:
                break

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    unet_lora_state_dict = get_peft_model_state_dict(unet)
    text_encoder_lora_state_dict = get_peft_model_state_dict(text_encoder)

    StableDiffusionPipeline.save_lora_weights(
        save_directory=OUTPUT_DIR,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_state_dict,
        weight_name=WEIGHT_NAME,
    )

    print(f"\n✅ 训练完成，权重已保存到: {Path(OUTPUT_DIR) / WEIGHT_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lr_unet", type=float, default=1e-4)
    parser.add_argument("--lr_text", type=float, default=1e-5)
    parser.add_argument("--bg_repeats", type=int, default=12)
    parser.add_argument("--defect_repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)