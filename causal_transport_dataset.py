from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from pipeline_contracts import PIXEL_RESOLUTION, SpatialIntent, StructuralState, ensure_binary_mask, physical_to_pixel

PHYS_X_MIN = -100.0
PHYS_X_MAX = 100.0
PHYS_Y_MIN = 0.0
PHYS_Y_MAX = 120.0


class GearCausalDataset(Dataset):
    """Physics engine v20: intrinsic point-cloud transport integrated into the main dataset.

    What changes relative to v17:
    1) Prototype points are no longer placed by a single affine warp in the visible chart.
    2) The canonical prototype is discretized into foreground points.
    3) These points are transported in the intrinsic (arc-length, physical-y) plane around the
       intent center, then rasterized back to the visible chart.
    4) The downstream depth / normal / render stages stay the same as v17.

    This version is intended as the first "back to the mainline" integration of the probe-validated
    transport logic. It mainly upgrades the shape transport stage; it does not yet claim a fully
    solved region-wise differential transport model for all shapes.
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
                z[i] = 2.0 * ((d - 60) / 2.0)
            else:
                z[i] = 2.0

        self.x_1d = x
        self.y_1d = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.X, self.Y = np.meshgrid(x, y)
        self.Z_grad_x = np.tile(np.convolve(np.gradient(z, self.dx), np.ones(3) / 3, mode="same"), (res, 1))
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
        dz_r = np.gradient(z[self.right_idx], self.dx)
        self.s_arc_raw = np.cumsum(np.sqrt(self.dx**2 + (dz_r * self.dx) ** 2))
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

    @staticmethod
    def _smoothstep(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _normalize_signed_field(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        std = float(np.std(x))
        if std < eps:
            return np.zeros_like(x, dtype=np.float32)
        y = x / std
        y = np.clip(y, -1.8, 1.8) / 1.8
        return y.astype(np.float32)

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
        micro_stripes = np.sin(self.Y * 2.0) * 0.5
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
            "base_normal_field": N_base,
        }

    @staticmethod
    def _scale_intent_for_defect(intent: SpatialIntent, defect_type: str) -> tuple[float, float]:
        sx = max(intent.pixel_scale.sigma_x, 1.0)
        sy = max(intent.pixel_scale.sigma_y, 1.0)
        if defect_type == "Spalling":
            sx *= float(np.random.uniform(1.10, 1.30))
            sy *= float(np.random.uniform(1.08, 1.28))
        else:
            if sx >= sy:
                sx *= float(np.random.uniform(1.12, 1.40))
                sy *= float(np.random.uniform(0.98, 1.12))
            else:
                sy *= float(np.random.uniform(1.12, 1.40))
                sx *= float(np.random.uniform(0.98, 1.12))
        return sx, sy

    @staticmethod
    def _build_render_mask(structural_mask_u8: np.ndarray, defect_type: str) -> np.ndarray:
        structural_mask_u8 = ensure_binary_mask(structural_mask_u8)
        structural_f = structural_mask_u8.astype(np.float32) / 255.0
        if defect_type == "Spalling":
            core = cv2.dilate(structural_mask_u8, np.ones((11, 11), np.uint8), iterations=1)
            halo = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0) / 255.0
            render_f = np.maximum(structural_f, 0.92 * halo)
        else:
            core = cv2.dilate(structural_mask_u8, np.ones((9, 5), np.uint8), iterations=1)
            halo = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=2.4, sigmaY=2.0) / 255.0
            render_f = np.maximum(structural_f, 0.96 * halo)
        return (np.clip(render_f, 0.0, 1.0) * 255).astype(np.uint8)

    @staticmethod
    def _canonicalize_prototype(prototype_mask_u8: np.ndarray, defect_type: str, canvas_size: int = 192) -> np.ndarray:
        mask = ensure_binary_mask(prototype_mask_u8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = mask[y0:y1 + 1, x0:x1 + 1]
        h, w = crop.shape[:2]
        max_dim = max(h, w)
        target_fill = 0.46 if defect_type == "Spalling" else 0.40
        target_dim = max(40, int(round(canvas_size * target_fill)))
        scale = target_dim / max(max_dim, 1)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_LINEAR if min(h, w) > 2 else cv2.INTER_NEAREST
        resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        x_start = (canvas_size - new_w) // 2
        y_start = (canvas_size - new_h) // 2
        canvas[y_start:y_start + new_h, x_start:x_start + new_w] = resized
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=1.0 if defect_type == "Spalling" else 0.8)
        return canvas


    def _chart_x_px_to_physical_x(self, chart_x_px: np.ndarray | float) -> np.ndarray:
        chart_x_px = np.asarray(chart_x_px, dtype=np.float32)
        chart_idx = np.clip(chart_x_px / max(PIXEL_RESOLUTION - 1, 1) * (len(self.right_idx) - 1), 0, len(self.right_idx) - 1)
        s_val = np.interp(chart_idx, np.arange(len(self.right_idx)), self.s_arc_raw).astype(np.float32)
        x_phys = np.interp(s_val, self.s_arc_raw, self.x_1d[self.right_idx]).astype(np.float32)
        return x_phys.astype(np.float32)

    def _physical_x_to_chart_x_px(self, x_phys: np.ndarray | float) -> np.ndarray:
        x_phys = np.asarray(x_phys, dtype=np.float32)
        x_clamped = np.clip(x_phys, self.x_1d[self.right_idx][0], self.x_1d[self.right_idx][-1])
        s_val = np.interp(x_clamped, self.x_1d[self.right_idx], self.s_arc_raw).astype(np.float32)
        chart_idx = np.interp(s_val, self.s_arc_raw, np.arange(len(self.right_idx))).astype(np.float32)
        chart_x_px = chart_idx / max(len(self.right_idx) - 1, 1) * (PIXEL_RESOLUTION - 1)
        return chart_x_px.astype(np.float32)

    @staticmethod
    def _chart_y_px_to_surface_y_phys(chart_y_px: np.ndarray | float) -> np.ndarray:
        return ((np.asarray(chart_y_px, dtype=np.float32) / PIXEL_RESOLUTION) * (PHYS_Y_MAX - PHYS_Y_MIN) + PHYS_Y_MIN).astype(np.float32)

    @staticmethod
    def _surface_y_phys_to_chart_y_px(y_phys: np.ndarray | float) -> np.ndarray:
        return (((np.asarray(y_phys, dtype=np.float32) - PHYS_Y_MIN) / (PHYS_Y_MAX - PHYS_Y_MIN)) * PIXEL_RESOLUTION).astype(np.float32)

    @staticmethod
    def _extract_local_coordinates(mask_u8: np.ndarray, mode: str = "raw_centered") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ys, xs = np.where(mask_u8 > 10)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(2, dtype=np.float32), np.eye(2, dtype=np.float32), np.ones(2, dtype=np.float32)

        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        ctr = pts.mean(axis=0, keepdims=True)
        cen = pts - ctr

        cov = (cen.T @ cen) / max(len(pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order].astype(np.float32)
        local_pca = cen @ eigvecs

        if mode == "pca":
            local = local_pca
        else:
            local = cen

        max_abs = np.max(np.abs(local), axis=0)
        max_abs = np.maximum(max_abs, 1.0).astype(np.float32)
        return pts.astype(np.float32), local.astype(np.float32), eigvecs, max_abs

    @staticmethod
    def _bilinear_splat_to_canvas(points_xy: np.ndarray, values: np.ndarray, H: int = PIXEL_RESOLUTION, W: int = PIXEL_RESOLUTION) -> np.ndarray:
        acc = np.zeros((H, W), dtype=np.float32)
        wacc = np.zeros((H, W), dtype=np.float32)

        xs = points_xy[:, 0]
        ys = points_xy[:, 1]
        vals = values.astype(np.float32)

        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        dx = xs - x0
        dy = ys - y0

        for ox, oy, w in [
            (0, 0, (1.0 - dx) * (1.0 - dy)),
            (1, 0, dx * (1.0 - dy)),
            (0, 1, (1.0 - dx) * dy),
            (1, 1, dx * dy),
        ]:
            xi = x0 + ox
            yi = y0 + oy
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            np.add.at(acc, (yi[valid], xi[valid]), vals[valid] * w[valid])
            np.add.at(wacc, (yi[valid], xi[valid]), w[valid])

        out = np.zeros_like(acc)
        valid = wacc > 1e-8
        out[valid] = acc[valid] / wacc[valid]
        return np.clip(out, 0.0, 1.0)

    def _project_patch_in_visible_chart(self, intent: SpatialIntent, defect_type: str, prototype_mask_u8: np.ndarray):
        """Intrinsic point-cloud transport into chart space.

        The prototype is discretized into foreground points. These points are centered in their
        canonical local coordinates, scaled by the intent, rotated in the intrinsic (arc-length, y)
        plane, and then rasterized back into the visible chart.
        """
        sx_px, sy_px = self._scale_intent_for_defect(intent, defect_type)
        cx = float(intent.pixel_center.x)
        cy = float(intent.pixel_center.y)
        theta = float(intent.theta_rad)

        canonical_patch = self._canonicalize_prototype(prototype_mask_u8, defect_type=defect_type, canvas_size=192)
        pts, local, _, max_abs = self._extract_local_coordinates(canonical_patch, mode="raw_centered")
        if len(pts) == 0:
            return np.zeros((PIXEL_RESOLUTION, PIXEL_RESOLUTION), dtype=np.float32), canonical_patch

        local_norm = local / max_abs[None, :]

        # Keep the same empirically validated intrinsic-plane scaling used in the clean probe.
        scale_x = (1.20 if defect_type == "Spalling" else 1.30)
        scale_y = (1.20 if defect_type == "Spalling" else 1.05)

        ds = local_norm[:, 0] * (sx_px * scale_x)
        dy_phys = local_norm[:, 1] * (((sy_px * scale_y) / PIXEL_RESOLUTION) * (PHYS_Y_MAX - PHYS_Y_MIN))

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        ds_r = c * ds - s * dy_phys * 4.0
        dy_r = s * ds / 4.0 + c * dy_phys

        x_center_phys = self._chart_x_px_to_physical_x(np.array([cx], dtype=np.float32))[0]
        y_center_phys = self._chart_y_px_to_surface_y_phys(np.array([cy], dtype=np.float32))[0]
        s_center = float(np.interp(np.array([x_center_phys], dtype=np.float32), self.x_1d[self.right_idx], self.s_arc_raw)[0])

        s_target = np.clip(s_center + ds_r.astype(np.float32), self.s_arc_raw[0], self.s_arc_raw[-1])
        x_target_phys = np.interp(s_target, self.s_arc_raw, self.x_1d[self.right_idx]).astype(np.float32)
        y_target_phys = np.clip(y_center_phys + dy_r.astype(np.float32), PHYS_Y_MIN, PHYS_Y_MAX).astype(np.float32)

        chart_x = self._physical_x_to_chart_x_px(x_target_phys)
        chart_y = self._surface_y_phys_to_chart_y_px(y_target_phys)

        values = np.ones((len(chart_x),), dtype=np.float32)
        canvas = self._bilinear_splat_to_canvas(np.stack([chart_x, chart_y], axis=1), values)
        visible = self._smoothstep(np.clip(canvas, 0.0, 1.0))

        support = cv2.GaussianBlur(visible.astype(np.float32), (0, 0), sigmaX=2.2 if defect_type == "Spalling" else 1.4)
        visible_strength = np.clip(0.76 * visible + 0.24 * support, 0.0, 1.0)
        return visible_strength.astype(np.float32), canonical_patch

    def _lift_visible_field_to_surface(self, visible_strength: np.ndarray) -> np.ndarray:
        visible_strength = np.clip(visible_strength, 0.0, 1.0).astype(np.float32)
        chart_strip = cv2.resize(visible_strength, (len(self.right_idx), self.res), interpolation=cv2.INTER_LINEAR)
        field_surface = np.zeros((self.res, self.res), dtype=np.float32)
        for i in range(self.res):
            row_strip = chart_strip[i, :]
            field_surface[i, self.right_idx] = np.interp(self.s_arc_raw, self.s_uni, row_strip.ravel())
        return np.clip(field_surface, 0.0, 1.0)

    def _build_visible_field(self, field_strength: np.ndarray) -> np.ndarray:
        return np.clip(field_strength, 0.0, 1.0).astype(np.float32)

    def generate_causal_defect(self, intent: SpatialIntent, defect_type: str, prototype_mask_u8: Optional[np.ndarray] = None):
        if prototype_mask_u8 is None:
            prototype_mask_u8 = np.zeros((PIXEL_RESOLUTION, PIXEL_RESOLUTION), dtype=np.uint8)

        visible_strength, canonical_patch = self._project_patch_in_visible_chart(intent, defect_type, prototype_mask_u8)
        field_strength = self._lift_visible_field_to_surface(visible_strength)

        t_x = np.stack([np.ones_like(self.X), np.zeros_like(self.X), self.Z_grad_x], axis=-1)
        t_y = np.stack([np.zeros_like(self.X), np.ones_like(self.X), np.zeros_like(self.X)], axis=-1)
        n_field = self.normalize(np.cross(t_x, t_y))
        t_x = self.normalize(t_x)
        t_y = self.normalize(t_y)
        theta = intent.theta_rad
        tangent_major_field = self.normalize(np.cos(theta) * t_x + np.sin(theta) * t_y)
        tangent_minor_field = self.normalize(-np.sin(theta) * t_x + np.cos(theta) * t_y)

        if defect_type == "Spalling":
            threshold = 0.14
            max_depth = float(np.random.uniform(4.6, 6.9))
            power = 1.55
            tilt_major = float(np.random.uniform(0.04, 0.10))
            tilt_minor = float(np.random.uniform(-0.03, 0.03))
            blur_sigma = 0.42
        else:
            threshold = 0.10
            max_depth = float(np.random.uniform(1.6, 3.0))
            power = 1.75
            tilt_major = float(np.random.uniform(0.08, 0.18))
            tilt_minor = float(np.random.uniform(-0.05, 0.05))
            blur_sigma = 0.45

        defect_mask = field_strength > threshold
        if not np.any(defect_mask):
            defect_mask = field_strength > (threshold * 0.75)

        base_strength = np.power(np.clip(field_strength, 0.0, 1.0), power)

        if defect_type == "Spalling":
            # bowl in visible chart, then lift already happened via field_strength; add mild internal relief on surface field
            low = gaussian_filter(np.random.randn(self.res, self.res), sigma=2.2)
            mid = gaussian_filter(np.random.randn(self.res, self.res), sigma=0.9)
            relief = self._normalize_signed_field(0.65 * low + 0.35 * mid)
            etch_strength = base_strength * (1.0 + 0.12 * relief * defect_mask.astype(np.float32))
            etch_strength = np.clip(etch_strength, 0.0, None)
            etch_strength = cv2.GaussianBlur(etch_strength.astype(np.float32), (0, 0), sigmaX=blur_sigma)
            etch_magnitude = max_depth * etch_strength
            etch_magnitude += gaussian_filter(np.random.randn(self.res, self.res), sigma=1.0) * 0.035 * defect_mask.astype(np.float32)
        else:
            etch_strength = cv2.GaussianBlur(base_strength.astype(np.float32), (0, 0), sigmaX=blur_sigma)
            etch_magnitude = max_depth * etch_strength
            etch_magnitude += gaussian_filter(np.random.randn(self.res, self.res), sigma=1.0) * 0.05 * defect_mask.astype(np.float32)

        etch_magnitude = np.clip(etch_magnitude, 0.0, None)

        etch_dir = self.normalize(n_field + tilt_major * tangent_major_field + tilt_minor * tangent_minor_field)
        displaced_points = self.Points_3D - etch_magnitude[..., None] * etch_dir
        dP_dx = np.gradient(displaced_points, self.dx, axis=1)
        dP_dy = np.gradient(displaced_points, self.dy, axis=0)
        defect_normal_field = self.normalize(np.cross(dP_dx, dP_dy))
        normal_perturbation = defect_normal_field - n_field
        total_z = displaced_points[:, :, 2]
        defect_depth_map = total_z - self.Z

        return defect_mask, visible_strength, field_strength, normal_perturbation, defect_normal_field, total_z, defect_depth_map, canonical_patch

    def build_structural_state(self, intent: SpatialIntent, defect_type: str, prototype_mask_u8: Optional[np.ndarray] = None) -> StructuralState:
        baseline = self.build_baseline_bundle()
        base_normal_field = baseline["base_normal_field"]

        defect_mask, visible_strength, field_strength_surface, normal_perturbation, defect_normal_field, total_z, defect_depth_map, canonical_patch = self.generate_causal_defect(
            intent=intent, defect_type=defect_type, prototype_mask_u8=prototype_mask_u8
        )

        baseline_normal_map = self.get_normal_map(base_normal_field)
        defect_normal_map = self.get_normal_map(defect_normal_field)

        strength_u8 = np.clip(visible_strength * 255.0, 0, 255).astype(np.uint8)
        strength_blur = cv2.GaussianBlur(visible_strength, (0, 0), sigmaX=1.0)
        if defect_type == "Spalling":
            causal_mask_u8 = ensure_binary_mask((strength_blur > 0.13).astype(np.uint8) * 255)
            causal_mask_u8 = cv2.morphologyEx(causal_mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        else:
            causal_mask_u8 = ensure_binary_mask((strength_blur > 0.10).astype(np.uint8) * 255)
            causal_mask_u8 = cv2.morphologyEx(causal_mask_u8, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        render_mask_u8 = self._build_render_mask(causal_mask_u8, defect_type)

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
            defect_type=defect_type,
            baseline_normal_u8=baseline_normal_map,
            defect_normal_u8=defect_normal_map,
            defect_depth_u8=depth_img,
            field_strength_u8=strength_u8,
            causal_mask_u8=causal_mask_u8,
            render_mask_u8=render_mask_u8,
            factual_img_u8=baseline["factual_img_u8"],
            prototype_mask_u8=canonical_patch,
            defect_mask_bool=defect_mask,
            defect_depth_map=defect_depth_map,
            normal_perturbation=normal_perturbation,
            total_z=total_z,
            base_normal_field=base_normal_field,
            defect_normal_field=defect_normal_field,
            row_profile_y_px=row_profile_y_px,
            intent=intent,
        )
