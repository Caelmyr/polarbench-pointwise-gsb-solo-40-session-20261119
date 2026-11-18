from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .domain import Box
from .quality import dominant_image
from .settings import LAYERS

COLOR_STOPS = {
    'gray': ((13, 19, 29), (250, 250, 245)),
    'viridis': ((35, 20, 74), (53, 58, 117), (32, 126, 146), (51, 184, 139), (166, 214, 97), (249, 235, 142)),
    'magma': ((24, 14, 46), (56, 30, 100), (135, 49, 124), (204, 68, 104), (245, 122, 101), (252, 218, 152)),
    'ice': ((16, 21, 38), (22, 63, 112), (31, 121, 170), (68, 190, 208), (220, 248, 241)),
    'balance': ((23, 68, 143), (84, 169, 217), (29, 37, 52), (239, 162, 92), (222, 76, 70)),
    'axial': ((249, 122, 105), (218, 184, 83), (113, 201, 138), (74, 177, 220), (128, 136, 241), (226, 110, 190), (249, 122, 105)),
}


@lru_cache(maxsize=8)
def lut(palette):
    stops = np.asarray(COLOR_STOPS[palette], dtype=np.float32)
    locations = np.linspace(0, 1, len(stops))
    points = np.linspace(0, 1, 1024)
    values = np.column_stack([np.interp(points, locations, stops[:, channel]) for channel in range(3)])
    return np.rint(values).astype(np.uint8)


def map_colors(signal, low, high, palette, gamma=1.0):
    clip = np.clip((signal - low) / (high - low), 0, 1)
    clip = np.nan_to_num(clip, nan=0.0)
    if gamma != 1.0:
        clip **= 1.0 / gamma
    return lut(palette)[np.rint(clip * 1023).astype(np.int16)]


def _checker(bbox):
    y, x = np.ogrid[bbox.y:bbox.bottom, bbox.x:bbox.right]
    alternate = ((x // 10) + (y // 10)) % 2 == 0
    return np.where(alternate[:, :, None], (36, 43, 56), (26, 33, 46)).astype(np.uint8)


def apply_background(rgb, mask, bbox, style):
    if style == 'transparent':
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgb[~mask] = 0
        return np.dstack((rgb, alpha))
    if style == 'checker':
        checker = _checker(bbox)
        rgb[~mask] = checker[~mask]
    else:
        rgb[~mask] = 0
    return rgb


@dataclass(frozen=True)
class Rendered:
    payload: bytes
    viewport: Box
    width: int
    height: int
    layer: str

    def metadata(self):
        return {
            'layer': self.layer,
            'viewport': self.viewport.as_dict(),
            'output_width': self.width,
            'output_height': self.height,
        }


def scaled_size(box, max_side):
    ratio = min(1.0, max_side / max(box.width, box.height))
    return max(1, round(box.width * ratio)), max(1, round(box.height * ratio))


def _render_vectors(image, results, box, view):
    draw = ImageDraw.Draw(image)
    field = results.field
    spacing = view.vector_spacing
    first_x = max(spacing // 2, ((box.x + spacing - 1 - spacing // 2) // spacing) * spacing + spacing // 2)
    first_y = max(spacing // 2, ((box.y + spacing - 1 - spacing // 2) // spacing) * spacing + spacing // 2)
    for y in range(first_y, box.bottom, spacing):
        for x in range(first_x, box.right, spacing):
            if not field.oriented[y, x]:
                continue
            axis = np.deg2rad(field.direction[y, x] + results.processing.zero_angle)
            length = spacing * min(0.38, 0.12 + 0.30 * float(field.degree[y, x]))
            dx, dy = np.cos(axis) * length, np.sin(axis) * length
            px, py = x - box.x, y - box.y
            endpoints = (round(px - dx), round(py - dy), round(px + dx), round(py + dy))
            draw.line(endpoints, fill=(12, 20, 31), width=3)
            draw.line(endpoints, fill=(249, 251, 235), width=1)


def _render_grid(image, box):
    draw = ImageDraw.Draw(image)
    spacing = max(10, 10 * round(max(box.width, box.height) / 100))
    for x in range((box.x // spacing + 1) * spacing, box.right, spacing):
        local = x - box.x
        draw.line((local, 0, local, box.height - 1), fill=(100, 116, 133), width=1)
    for y in range((box.y // spacing + 1) * spacing, box.bottom, spacing):
        local = y - box.y
        draw.line((0, local, box.width - 1, local), fill=(100, 116, 133), width=1)


def raster(results, view, full_size=False):
    height, width = results.field.shape
    selected = view.crop or Box.entire(width, height)
    layer = view.layer
    subset = selected.slices
    if layer == 'quality':
        pixels = dominant_image(results.quality, focus=view.quality_focus)[subset].copy()
    else:
        signal, support = results.channel(layer, view.analyzer_angle)
        if layer == 'direction':
            pixels = map_colors(signal[subset] % 180, 0, 180, 'axial')
        else:
            pixels = map_colors(signal[subset], view.display.low, view.display.high, view.palette, view.gamma)
        pixels = apply_background(pixels, support[subset], selected, view.invalid_style)
    image = Image.fromarray(pixels)
    if view.grid:
        _render_grid(image, selected)
    if view.vectors and not layer.startswith('raw-') and layer != 'quality':
        _render_vectors(image, results, selected, view)
    output_size = image.size if full_size else scaled_size(selected, view.max_side)
    if output_size != image.size:
        method = Image.Resampling.NEAREST if layer in ('quality', 'direction') else Image.Resampling.BOX
        image = image.resize(output_size, method)
    return Rendered(rgba_png(np.asarray(image)), selected, *output_size, layer)


def legend(view):
    schema = LAYERS[view.layer]
    if view.layer == 'quality':
        from .quality import legends
        return {'type': 'category', 'entries': legends()}
    if view.layer == 'direction':
        samples = np.array([0, 30, 60, 90, 120, 150, 180])
        colors = map_colors(samples, 0, 180, 'axial')
        return {'type': 'cyclic', 'values': samples.tolist(), 'colors': colors.tolist(), 'unit': schema.unit}
    values = np.linspace(view.display.low, view.display.high, 9)
    colors = map_colors(values, view.display.low, view.display.high, view.palette, view.gamma)
    return {
        'type': 'gradient', 'values': values.tolist(), 'colors': colors.tolist(), 'unit': schema.unit,
        'min': view.display.low, 'max': view.display.high,
        'description': schema.description,
    }


def _chart_frame(title, description):
    image = Image.new('RGB', (780, 420), (23, 32, 44))
    draw = ImageDraw.Draw(image)
    draw.text((27, 24), title, fill=(237, 243, 248), font=ImageFont.load_default(size=18))
    draw.text((27, 53), description, fill=(164, 182, 202), font=ImageFont.load_default(size=13))
    left, top, right, bottom = 72, 99, 748, 363
    draw.line((left, top, left, bottom, right, bottom), fill=(116, 135, 158), width=1)
    return image, draw, (left, top, right, bottom)


def _y_marks(draw, box, latest, *, start=0):
    left, top, right, bottom = box
    for part in np.linspace(0, 1, 5):
        y = round(bottom - part * (bottom - top))
        draw.line((left, y, right, y), fill=(50, 67, 88), width=1)
        draw.text((left - 6, y), f'{start + part * (latest - start):.3g}', anchor='rm', fill=(170, 188, 208))


def _x_marks(draw, box, lo, hi, unit):
    left, _, right, bottom = box
    for part in np.linspace(0, 1, 7):
        x = round(left + part * (right - left))
        draw.text((x, bottom + 14), f'{lo + part * (hi - lo):.3g}', anchor='mt', fill=(170, 188, 208))
    draw.text(((left + right) // 2, bottom + 36), unit, anchor='mt', fill=(170, 188, 208))


def histogram_chart(result):
    data = result['histogram']
    buckets = data['counts']
    image, draw, box = _chart_frame(
        f'{LAYERS[result["layer"]].label} distribution',
        f'{result["layer_usable"]:,} of {result["pixels"]:,} sampled detector pixels',
    )
    left, top, right, bottom = box
    high = max(1, max(buckets))
    _y_marks(draw, box, high)
    length = len(buckets)
    for idx, count in enumerate(buckets):
        if count:
            x0 = round(left + (right - left) * idx / length)
            x1 = max(x0, round(left + (right - left) * (idx + 1) / length) - 2)
            bar_top = min(bottom - 1, round(bottom - count * (bottom - top) / high))
            draw.rectangle((x0, bar_top, x1, bottom - 1), fill=(81, 190, 167))
    _x_marks(draw, box, data['edges'][0], data['edges'][-1], LAYERS[result['layer']].unit)
    return rgba_png(np.asarray(image))


def profile_chart(data):
    values = data['value']
    present = [value for value in values if value is not None]
    image, draw, box = _chart_frame(
        f'{LAYERS[data["layer"]].label} scan line',
        f'{data["valid_samples"]} of {len(values)} samples usable. Gaps represent excluded detector samples.',
    )
    left, top, right, bottom = box
    if not present:
        _x_marks(draw, box, 0, data['length'], 'Detector distance (px)')
        return rgba_png(np.asarray(image))
    if data['layer'] == 'direction':
        lo, hi = 0, 180
    else:
        lo, hi = min(present), max(present)
        if hi <= lo:
            lo, hi = lo - 0.01, hi + 0.01
    _y_marks(draw, box, hi, start=lo)
    last, previous = None, None
    for distance, value in zip(data['distance'], values):
        if value is None:
            last, previous = None, None
            continue
        x = round(left + (right - left) * distance / data['length'])
        y = round(bottom - (bottom - top) * (value - lo) / (hi - lo))
        wrap = data['layer'] == 'direction' and previous is not None and abs(previous - value) > 90
        if last is not None and not wrap:
            draw.line((*last, x, y), fill=(87, 207, 184), width=2)
        else:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(87, 207, 184))
        last, previous = (x, y), value
    _x_marks(draw, box, 0, data['length'], 'Detector distance (px)')
    return rgba_png(np.asarray(image))
