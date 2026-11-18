from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .codec import plane_png, rgba_png
from .domain import ANGLES, Box, Offset
from .ingest import MOSAIC_PRESETS, prepare_mosaic, prepare_planes
from .registration import footprints


@dataclass(frozen=True)
class DemoScene:
    key: str
    name: str
    description: str
    width: int
    height: int
    format: str

    def as_dict(self):
        return dict(self.__dict__)


SCENES = {
    scene.key: scene for scene in (
        DemoScene('coated-panel', 'Coated panel', 'Specular coating, a low-signal corner and a native coverage gap.', 560, 380, 'planes'),
        DemoScene('shifted-laminate', 'Shifted laminate', 'A textured coupon captured sequentially with a small translation.', 480, 330, 'planes'),
        DemoScene('sensor-glass', 'Glass mosaic', 'A 2 × 2 micro-polarizer raw tile with two glass seams.', 416, 304, 'mosaic'),
    )
}


def scenes():
    return [scene.as_dict() for scene in SCENES.values()]


def scene_spec(key):
    from .domain import enum_value
    return SCENES[enum_value(key, SCENES, 'Demo scene')]


def _grid(width, height):
    y, x = np.meshgrid(np.linspace(0, 1, height, dtype=np.float32), np.linspace(0, 1, width, dtype=np.float32), indexing='ij')
    return x, y


def _noise(shape, seed, sigma=1):
    rng = np.random.default_rng(seed)
    surface = ndimage.gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma)
    return surface / max(float(surface.std()), 1e-6)


def _smooth_band(values, midpoint, spread):
    return np.exp(-((values - midpoint) / spread) ** 2)


def _holes(x, y):
    central = ((x - 0.52) / 0.074) ** 2 + ((y - 0.58) / 0.091) ** 2 < 1
    corner = x + y > 1.84
    thin = (x > 0.072) & (x < 0.078) & (y > 0.32) & (y < 0.7)
    return central | corner | thin


def _coating(scene):
    x, y = _grid(scene.width, scene.height)
    matte = _noise(x.shape, 40, 2.5)
    coarse = _noise(x.shape, 41, 22)
    taper = _smooth_band(y - 0.32 * x, 0.12, 0.13)
    ribbon = taper * _smooth_band(x, 0.57, 0.64)
    fingers = 0.5 + 0.5 * np.sin(x * 38 * np.pi - y * 2 * np.pi)
    blemish = _smooth_band(x + y * 0.19, 0.63, 0.006) * _smooth_band(y, 0.72, 0.14)
    intensity = 0.33 + 0.30 * y + 0.17 * coarse / 3 + ribbon * (0.72 + 0.06 * fingers) + 0.11 * blemish + 0.02 * matte
    intensity *= np.where((x < 0.2) & (y > 0.8), 0.048, 1)
    degree = 0.055 + ribbon * (0.55 + 0.025 * matte) + 0.16 * _smooth_band(x, 0.78, 0.21)
    direction = 173 + 18 * x + 12 * y + 28 * _smooth_band(x + y * 0.8, 0.94, 0.18) + 1.8 * coarse
    return intensity, degree, direction


def _laminate(scene):
    x, y = _grid(scene.width, scene.height)
    grain = _noise(x.shape, 113, 1.2)
    fiber1, fiber2 = np.sin(92 * (x + y * 0.34)), np.sin(123 * (y - x * 0.22))
    nodes = 0.04 * fiber1 * fiber2
    mottled = 0.075 * _noise(x.shape, 117, 6)
    scratch = _smooth_band(y - x * 0.55, 0.2, 0.016) * _smooth_band(x, 0.68, 0.25)
    notch = _smooth_band(x + y * 0.25, 0.486, 0.011) * _smooth_band(y, 0.25, 0.18)
    intensity = 0.48 + 0.08 * fiber1 + 0.10 * fiber2 + nodes + mottled + 0.016 * grain + scratch * 0.29
    degree = 0.15 + 0.18 * _smooth_band(x, 0.72, 0.24) + 0.09 * fiber1 + 0.08 * scratch
    direction = np.where(x > 0.47 + 0.03 * fiber2, 123, 33) + 2.2 * grain + 38 * notch
    return intensity, degree, direction


def _glass(scene):
    x, y = _grid(scene.width, scene.height)
    mottled = _noise(x.shape, 33, 8)
    seams = _smooth_band(x + y * 0.16, 0.35, 0.014) + _smooth_band(x + y * 0.16, 0.72, 0.021)
    rainbow = _smooth_band(y - x * 0.23, 0.26, 0.1)
    folds = np.sin(22 * x + 16 * y) * np.cos(17 * y + 5 * x)
    intensity = 0.57 + 0.07 * folds + 0.3 * seams + 0.32 * rainbow + 0.045 * mottled
    degree = 0.14 + 0.28 * rainbow + 0.19 * seams + 0.10 * (folds + 1)
    direction = 104 + 73 * x - 35 * y + 14 * np.sin(y * 8) + 3.1 * mottled
    return intensity, degree, direction


def _analyzers(scene, model):
    intensity, degree, axis = model(scene)
    x, y = _grid(scene.width, scene.height)
    rgb_axes = np.deg2rad(2 * axis)
    polarized = degree * intensity
    q, u = polarized * np.cos(rgb_axes), polarized * np.sin(rgb_axes)
    analysis = {0: q, 45: u, 90: -q, 135: -u}
    gains = np.sin(1.6 * np.pi * x) * np.cos(1.3 * np.pi * y) * 0.07 + 0.92
    levels = {}
    for n, a in enumerate(ANGLES):
        random = _noise(x.shape, 870 + n, 0.7)
        image = (intensity + analysis[a]) * 0.5 * gains + 0.015 + random * 0.002
        levels[a] = np.clip(image, 0, 1)
    return levels


def _with_shift(level, dx, dy):
    shape = level.shape
    src, dst = footprints(*shape, Offset(dx, dy))
    values = np.zeros(shape, dtype=np.float32)
    values[dst.slices] = level[src.slices]
    support = np.zeros(shape, dtype=bool)
    support[dst.slices] = True
    return values, support


def _alpha_png(values, valid):
    levels = np.round(np.clip(values, 0, 1) * 255).astype(np.uint8)
    return rgba_png(levels, valid)


def _box(scene, x, y, width, height):
    return Box(round(x * scene.width), round(y * scene.height), round(width * scene.width), round(height * scene.height)).as_dict()


def sample_regions(scene):
    if scene.key == 'coated-panel':
        return [
            ('Specular band', _box(scene, 0.32, 0.22, 0.27, 0.13)),
            ('Matte coating', _box(scene, 0.56, 0.59, 0.25, 0.18)),
            ('Native coverage gap', _box(scene, 0.40, 0.46, 0.20, 0.20)),
        ]
    if scene.key == 'shifted-laminate':
        return [
            ('Laminate A', _box(scene, 0.15, 0.25, 0.22, 0.28)),
            ('Laminate B', _box(scene, 0.57, 0.33, 0.23, 0.30)),
        ]
    return [
        ('Left glass seam', _box(scene, 0.20, 0.30, 0.13, 0.27)),
        ('Right glass seam', _box(scene, 0.56, 0.21, 0.13, 0.28)),
    ]


def demo_payload(key):
    scene = scene_spec(key)
    x, y = _grid(scene.width, scene.height)
    if key == 'coated-panel':
        levels = _analyzers(scene, _coating)
        highlight_clip = (x > 0.45) & (x < 0.52) & (y > 0.24) & (y < 0.29)
        levels[0][highlight_clip] = 1.0
        levels[45][(x > 0.7) & (x < 0.78) & (y > 0.40) & (y < 0.43)] *= 1.45
        mask = (x > 0.015) & (x < 0.985) & (y > 0.018) & (y < 0.982)
        mask &= ~((x > 0.26) & (x < 0.30) & (y > 0.55) & (y < 0.83))
        payloads = {
            angle: (plane_png(image), f'coated-panel-{angle}.png') for angle, image in levels.items()
        }
        payloads[45] = (_alpha_png(levels[45], ~_holes(x, y)), 'coated-panel-45-alpha.png')
        return prepare_planes(payloads, (plane_png(mask, 8), 'coated-panel-inclusion.png'))
    if key == 'shifted-laminate':
        levels = _analyzers(scene, _laminate)
        shifts = {0: (0, 0), 45: (8, -5), 90: (-6, 4), 135: (4, 7)}
        files = {}
        for angle in ANGLES:
            source, valid = _with_shift(levels[angle], *shifts[angle])
            files[angle] = (_alpha_png(source, valid), f'shifted-laminate-{angle}.png')
        return prepare_planes(files)
    levels = _analyzers(scene, _glass)
    layout = '90-45-135-0'
    mosaic = np.empty((2 * scene.height, 2 * scene.width), dtype=np.float32)
    for index, angle in enumerate(MOSAIC_PRESETS[layout]):
        row, column = divmod(index, 2)
        mosaic[row::2, column::2] = levels[angle]
    return prepare_mosaic((plane_png(mosaic), 'glass-detector-mosaic.png'), layout)


def demo_reference(key, kind):
    scene = scene_spec(key)
    x, y = _grid(scene.width, scene.height)
    vignetting = np.sin(1.6 * np.pi * x) * np.cos(1.3 * np.pi * y) * 0.07 + 0.92
    level = np.full_like(x, 0.015) if kind == 'dark' else vignetting * 0.65 + 0.015
    level += _noise(x.shape, 9754, 1.4) * 0.00065
    return plane_png(level), f'{key}-{kind}-reference.png'
