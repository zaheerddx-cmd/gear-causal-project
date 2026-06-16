from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from pipeline_contracts import PIXEL_RESOLUTION, SpatialIntent, StructuralState, ensure_binary_mask, physical_to_pixel


class GearCausalDataset(Dataset):
    """Physics engine with explicit, frame-safe interfaces.

    All public methods now accept a `SpatialIntent`. That removes the ambiguous
    `(cx, cy, sigma_x, sigma_y, angle)` tuple contract that previously mixed
    pixel-space and physical-space values across modules.
    """

    def __init__(self, res: int = 512, intensity: float = 2.496, polar: float = 0.8956, virtual_size: int = 10000):
        self.res = res
        self.intensity = intensity
        self.polar = polar
        self.virtual_size = virtual_size

        x, y = np.linspace(-100, 100, res), np.linspace(0, 120, res)
        z = np.zeros_like(x)
        for i, xi in enumerate(x):
            d = abs(xi)
            if d <= 10:
                z[i] = -120.0
            elif d <= 20:
                z[i] = -120.0 + 20.0 * (1 - np.cos(((d - 10) / 10.0) * np.pi / 2))
            elif d <= 60:
                z[i] = -100.0 + 100.0 * (((d - 20) / 40.0) ** 0.6)
            elif d <= 62:
                z[i] = 0.0 + 2.0 * ((d - 60) / 2.0)
            else:
                z[i] = 2.0

        self.X, self.Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        self.Z_grad_x = np.tile(np.convolve(np.gradient(z, dx), np.ones(3) / 3, mode="same"), (res, 1))
        self.Z = np.tile(z, (res, 1))
        self.Points_3D = np.stack([self.X, self.Y, self.Z], axis=-1)

        self.L_POS = np.array([93.9, 60, 72])
        self.C_POS = np.array([-36, 60, 89.7])

        shad_1d = np.ones_like(x)
        for i in range(len(x)):
            if self.L_POS[0] < x[i]:
                ray_z = z[i] + (x[:i] - x[i]) * (self.L_POS[2] - z[i]) / (self.L_POS[0] - x[i] + 1e-8)
                if np.any(z[:i] > ray_z):
                    shad_1d[i] = 0.0
        self.shadow = np.tile(np.convolve(shad_1d, np.ones(15) / 15, mode="same"), (res, 1))

        self.right_idx = np.where((x >= 0) & (x <= 80))[0]
        dz_r = np.gradient(z[self.right_idx], dx)
        self.s_arc_raw = np.cumsum(np.sqrt(dx**2 + (dz_r * dx) ** 2))
        self.s_arc_raw -= self.s_arc_raw[0]
        self.s_uni = np.linspace(0, self.s_arc_raw[-1], len(self.right_idx))

    def __len__(self):
        return self.virtual_size

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / (norm + 1e-8)

    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return np.clip(np.sum(v1 * v2, axis=-1), 0.0, 1.0)

    def render_physics_light(self, N, V_matrix, L_matrix, alpha, F0, shadow_mask):
        ambient = 0.05
        NdotL = self.dot_product(N, L_matrix)
        diffuse = NdotL * 0.3
        H = self.normalize(L_matrix + V_matrix)
        NdotH = self.dot_product(N, H)
        alpha2 = alpha**2
        D = alpha2 / (np.pi * (NdotH**2 * (alpha2 - 1.0) + 1.0) ** 2 + 1e-8)
        VdotH = self.dot_product(V_matrix, H)
        F = F0 + (1.0 - F0) * ((1.0 - VdotH) ** 5)
        NdotV = self.dot_product(N, V_matrix)
        specular = (D * F) / (4.0 * np.clip(NdotL * NdotV, 0.01, 1.0)) * NdotL
        specular_filtered = specular * (1.0 - self.polar)
        direct_light = (diffuse + specular_filtered) * shadow_mask.squeeze() * self.intensity
        return np.clip(ambient + direct_light, 0.0, 1.0)

    def compute_bounced_light(self, I_direct, bounces: int = 2, attenuation: float = 0.6):
        I_total = np.copy(I_direct)
        I_current = np.copy(I_direct)
        valley_mask = np.clip((0.0 - self.Z) / 100.0, 0.0, 1.0)
        blur_kernel = np.ones(15) / 15.0
        for _ in range(bounces):
            I_bounced_raw = np.fliplr(I_current) * attenuation * valley_mask
            I_bounced = np.zeros_like(I_bounced_raw)
            for r in range(I_bounced_raw.shape[0]):
                I_bounced[r, :] = np.convolve(I_bounced_raw[r, :].ravel(), blur_kernel, mode="same")
            I_total += I_bounced
            I_current = I_bounced
        return np.clip(I_total, 0.0, 1.0)

    def render_sample(self, N_field, F0_field):
        L_mat = self.normalize(self.L_POS - self.Points_3D)
        V_mat = self.normalize(self.C_POS - self.Points_3D)
        I_direct = self.render_physics_light(N_field, V_mat, L_mat, 0.1, F0_field, self.shadow)
        I_final = self.compute_bounced_light(I_direct)

        img_r = I_final[:, self.right_idx]
        rect = np.zeros((self.res, len(self.right_idx)))
        for i in range(self.res):
            rect[i, :] = np.interp(self.s_uni, self.s_arc_raw.ravel(), img_r[i, :].ravel())
        return cv2.resize(rect, (PIXEL_RESOLUTION, PIXEL_RESOLUTION), interpolation=cv2.INTER_LINEAR)

    def get_normal_map(self, N_field):
        N_rgb = (N_field + 1.0) / 2.0
        img_r = N_rgb[:, self.right_idx, :]
        rect = np.zeros((self.res, len(self.right_idx), 3))
        for i in range(self.res):
            for c in range(3):
                rect[i, :, c] = np.interp(self.s_uni, self.s_arc_raw.ravel(), img_r[i, :, c].ravel())
        rect_resized = cv2.resize(rect, (PIXEL_RESOLUTION, PIXEL_RESOLUTION), interpolation=cv2.INTER_LINEAR)
        return (rect_resized * 255).astype(np.uint8)

    def build_baseline_bundle(self) -> dict:
        unnormalized_base_N = np.stack([-self.Z_grad_x, np.zeros_like(self.X), np.ones_like(self.X)], axis=-1)
        peak_mask = np.abs(self.X) > 60
        stripe_freq = 2.0
        stripe_amp = 0.5
        micro_stripes = np.sin(self.Y * stripe_freq) * stripe_amp
        unnormalized_base_N[peak_mask, 0] += micro_stripes[peak_mask]
        N_base = self.normalize(unnormalized_base_N)

        base_metal = np.full((self.res, self.res), 0.55) + np.random.randn(self.res, self.res) * 0.03
        base_metal = np.clip(base_metal, 0.0, 1.0)
        factual_img = self.render_sample(N_base, np.copy(base_metal))
        smooth_normal = self.get_normal_map(N_base)
        flat_depth = np.full((PIXEL_RESOLUTION, PIXEL_RESOLUTION), 128, dtype=np.uint8)
        return {
            "factual_img_u8": (factual_img * 255).astype(np.uint8),
            "baseline_normal_u8": smooth_normal,
            "baseline_depth_u8": flat_depth,
            "unnormalized_base_N": unnormalized_base_N,
        }

    def generate_causal_defect(
        self,
        intent: SpatialIntent,
        defect_type: str,
        prototype_mask_u8: Optional[np.ndarray] = None,
    ):
        cx = intent.physical_center.x
        cy = intent.physical_center.y
        sigma_x = max(intent.physical_scale.sigma_x, 1.0)
        sigma_y = max(intent.physical_scale.sigma_y, 1.0)
        angle = intent.theta_rad

        defect_depth = np.zeros((self.res, self.res), dtype=np.float32)
        x_1d = self.X[0, :]
        z_1d = self.Z[0, :]
        dx_val = x_1d[1] - x_1d[0]
        dz_1d = np.gradient(z_1d, dx_val)
        s_1d = np.cumsum(np.sqrt(dx_val**2 + dz_1d**2))

        cx_idx = np.argmin(np.abs(x_1d - cx))
        S_cx = s_1d[cx_idx]

        S_X_2d = np.tile(s_1d, (self.res, 1))
        diff_x = S_X_2d - S_cx
        diff_y = self.Y - cy

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        X_rot = diff_x * cos_a + diff_y * sin_a
        Y_rot = -diff_x * sin_a + diff_y * cos_a

        if prototype_mask_u8 is not None:
            prototype_mask_u8 = ensure_binary_mask(prototype_mask_u8)
            u = (X_rot / (sigma_x * 2.0) + 0.5) * prototype_mask_u8.shape[1]
            v = (Y_rot / (sigma_y * 2.0) + 0.5) * prototype_mask_u8.shape[0]
            u_map = np.clip(u, 0, prototype_mask_u8.shape[1] - 1).astype(np.float32)
            v_map = np.clip(v, 0, prototype_mask_u8.shape[0] - 1).astype(np.float32)
            manifold_sticker = cv2.remap(prototype_mask_u8.astype(np.float32), u_map, v_map, interpolation=cv2.INTER_LINEAR)
            boundary_mask = (np.abs(X_rot) < sigma_x) & (np.abs(Y_rot) < sigma_y)
            final_mask = (manifold_sticker > 80) & boundary_mask
            if np.any(final_mask):
                sticker_max = manifold_sticker[final_mask].max() + 1e-8
                normalized_sticker = manifold_sticker[final_mask] / sticker_max
                max_depth = np.random.uniform(15.0, 20.0)
                defect_depth[final_mask] = -np.power(normalized_sticker, 0.5) * max_depth
        else:
            normalized_dist = np.sqrt((X_rot / sigma_x) ** 2 + (Y_rot / sigma_y) ** 2)
            final_mask = normalized_dist < 1.0
            max_depth = np.clip((sigma_x + sigma_y) / 2.0, 5.0, 10.0)
            defect_depth[final_mask] = -max_depth * (1.0 - normalized_dist[final_mask] ** 2)

        if not np.any(final_mask):
            final_mask = ((X_rot / sigma_x) ** 2 + (Y_rot / sigma_y) ** 2) < 1.0
            max_depth = np.clip((sigma_x + sigma_y) / 2.0, 5.0, 10.0)
            defect_depth[final_mask] = -max_depth * (1.0 - ((X_rot[final_mask] / sigma_x) ** 2 + (Y_rot[final_mask] / sigma_y) ** 2))

        micro_noise = gaussian_filter(np.random.randn(self.res, self.res), sigma=1.0) * 0.8
        defect_depth[final_mask] += micro_noise[final_mask]

        dz_dy, dz_dx = np.gradient(defect_depth)
        normal_perturbation = np.stack([-dz_dx, -dz_dy, np.zeros_like(dz_dx)], axis=-1)
        return final_mask, normal_perturbation, defect_type, defect_depth

    def build_structural_state(self, intent: SpatialIntent, defect_type: str, prototype_mask_u8: Optional[np.ndarray] = None) -> StructuralState:
        baseline = self.build_baseline_bundle()
        unnormalized_base_N = baseline["unnormalized_base_N"]
        N_base = self.normalize(unnormalized_base_N)

        defect_mask, normal_perturbation, d_type, defect_depth_map = self.generate_causal_defect(
            intent=intent,
            defect_type=defect_type,
            prototype_mask_u8=prototype_mask_u8,
        )

        unnormalized_defect_N = unnormalized_base_N.copy()
        gx = normal_perturbation[:, :, 0] * 5.0
        gy = normal_perturbation[:, :, 1] * 5.0
        Bx = unnormalized_base_N[:, :, 0]
        unnormalized_defect_N[defect_mask, 0] += gx[defect_mask]
        unnormalized_defect_N[defect_mask, 1] += gy[defect_mask]
        unnormalized_defect_N[defect_mask, 2] = 1.0 - (gx[defect_mask] * Bx[defect_mask])
        N_defect = self.normalize(unnormalized_defect_N)

        defect_normal_map = self.get_normal_map(N_defect)
        baseline_normal_map = self.get_normal_map(N_base)

        mask_float = defect_mask.astype(np.float32)
        mask_r = mask_float[:, self.right_idx]
        mask_rect = np.zeros((self.res, len(self.right_idx)))
        for i in range(self.res):
            mask_rect[i, :] = np.interp(self.s_uni, self.s_arc_raw.ravel(), mask_r[i, :].ravel())
        mask_rect_binary = (mask_rect > 0.5).astype(np.uint8) * 255
        mask_binary_resized = cv2.resize(mask_rect_binary, (PIXEL_RESOLUTION, PIXEL_RESOLUTION), interpolation=cv2.INTER_NEAREST)
        causal_mask_u8 = cv2.dilate(mask_binary_resized, np.ones((9, 9), np.uint8), iterations=1)

        total_z = self.Z.copy()
        total_z[defect_mask] += defect_depth_map[defect_mask] * 2.5
        depth_r = total_z[:, self.right_idx]
        depth_rect = np.zeros((self.res, len(self.right_idx)))
        for i in range(self.res):
            depth_rect[i, :] = np.interp(self.s_uni, self.s_arc_raw.ravel(), depth_r[i, :].ravel())
        depth_min, depth_max = -120.0, 5.0
        depth_normalized = np.clip((depth_rect - depth_min) / (depth_max - depth_min), 0.0, 1.0)
        depth_resized = cv2.resize(depth_normalized, (PIXEL_RESOLUTION, PIXEL_RESOLUTION), interpolation=cv2.INTER_LINEAR)
        depth_img = (depth_resized * 255).astype(np.uint8)

        row_profile_y_px = int(np.clip(physical_to_pixel(intent.physical_center.x, intent.physical_center.y).y, 0, PIXEL_RESOLUTION - 1))

        return StructuralState(
            defect_type=d_type,
            baseline_normal_u8=baseline_normal_map,
            defect_normal_u8=defect_normal_map,
            defect_depth_u8=depth_img,
            causal_mask_u8=causal_mask_u8,
            factual_img_u8=baseline["factual_img_u8"],
            prototype_mask_u8=ensure_binary_mask(prototype_mask_u8) if prototype_mask_u8 is not None else np.zeros((PIXEL_RESOLUTION, PIXEL_RESOLUTION), dtype=np.uint8),
            defect_mask_bool=defect_mask,
            defect_depth_map=defect_depth_map,
            normal_perturbation=normal_perturbation,
            total_z=total_z,
            row_profile_y_px=row_profile_y_px,
            intent=intent,
        )

    # Backward-compatible wrapper for older experiments.
    def get_deterministic_sample(self, cx, cy, sigma_x, sigma_y, angle, defect_type, external_mask=None):
        pixel = physical_to_pixel(cx, cy)
        if abs(cx) > 100 or cy > 120:
            pixel = type(pixel)(x=cx, y=cy)
        if pixel.x == cx and pixel.y == cy and (abs(cx) > 100 or cy > 120):
            from pipeline_contracts import build_intent_from_pixel_measurements
            intent = build_intent_from_pixel_measurements(cx, cy, sigma_x, sigma_y, angle, source="legacy_pixel_guess")
        else:
            from pipeline_contracts import PhysicalScale, PixelScale, PixelPoint, PhysicalPoint, SpatialIntent, physical_sigma_to_pixel
            pixel_scale = physical_sigma_to_pixel(sigma_x, sigma_y)
            intent = SpatialIntent(
                pixel_center=pixel,
                physical_center=PhysicalPoint(float(cx), float(cy)),
                pixel_scale=pixel_scale,
                physical_scale=PhysicalScale(float(sigma_x), float(sigma_y)),
                theta_rad=float(angle),
                source="legacy_physical_tuple",
            )
        structure = self.build_structural_state(intent=intent, defect_type=defect_type, prototype_mask_u8=external_mask)
        return {
            "factual_img": structure.factual_img_u8,
            "smooth_normal": structure.baseline_normal_u8,
            "defect_normal": structure.defect_normal_u8,
            "causal_mask": structure.causal_mask_u8,
            "defect_depth": structure.defect_depth_u8,
            "defect_type": structure.defect_type,
            "debug_albedo": np.zeros((PIXEL_RESOLUTION, PIXEL_RESOLUTION), dtype=np.uint8),
            "debug_noise": np.zeros((PIXEL_RESOLUTION, PIXEL_RESOLUTION), dtype=np.uint8),
        }

    def __getitem__(self, idx):
        from pipeline_contracts import build_intent_from_pixel_measurements

        cx_rand = np.random.uniform(100.0, 360.0)
        cy_rand = np.random.uniform(120.0, 390.0)
        sigma_x_rand = np.random.uniform(20.0, 60.0)
        sigma_y_rand = np.random.uniform(12.0, 30.0)
        angle_rand = np.random.uniform(-1.0, 1.0)
        dtype_rand = np.random.choice(["Spalling", "Crack"])
        intent = build_intent_from_pixel_measurements(cx_rand, cy_rand, sigma_x_rand, sigma_y_rand, angle_rand, source="dataset_random")
        structure = self.build_structural_state(intent=intent, defect_type=dtype_rand)
        return {
            "factual_img": structure.factual_img_u8,
            "smooth_normal": structure.baseline_normal_u8,
            "defect_normal": structure.defect_normal_u8,
            "causal_mask": structure.causal_mask_u8,
            "defect_depth": structure.defect_depth_u8,
            "defect_type": structure.defect_type,
        }
