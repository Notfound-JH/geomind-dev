"""Helpers for explicit IsaacLab render-quality presets."""

from __future__ import annotations

DEFAULT_HIGH_QUALITY_SAMPLES_PER_PIXEL = 16


def apply_high_quality_render_settings(sim_cfg, *, samples_per_pixel: int = DEFAULT_HIGH_QUALITY_SAMPLES_PER_PIXEL) -> None:
    """Mutate an IsaacLab ``SimulationCfg`` to favor image quality over frame rate."""
    render_cfg = sim_cfg.render

    render_cfg.enable_reflections = True
    render_cfg.enable_translucency = True
    render_cfg.enable_global_illumination = True
    render_cfg.enable_ambient_occlusion = True
    render_cfg.enable_shadows = True
    render_cfg.enable_direct_lighting = True
    render_cfg.enable_dlssg = False
    render_cfg.samples_per_pixel = max(1, int(samples_per_pixel))
    render_cfg.enable_dl_denoiser = True
    render_cfg.antialiasing_mode = "DLAA"
    # Keep the DLSS quality mode explicit in case the AA mode is switched back to DLSS.
    render_cfg.dlss_mode = 2
    render_cfg.dome_light_upper_lower_strategy = 4
    render_cfg.rendering_mode = "quality"
