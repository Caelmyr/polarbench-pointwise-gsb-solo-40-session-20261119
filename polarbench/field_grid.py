from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .codec import rgba_png
from .domain import Box, InputError, enum_value, integer, number, object_value
from .regions import csv_text
from .render import map_colors
from .stokes import axial_mean


@dataclass(frozen=True)
class GridStudy:
    box: Box
    tile: int
    fraction: float
    vector_weighting: str

    @classmethod
    def parse(cls, raw, width, height):
        data = object_value(raw, 'Orientation grid')
        box = Box.parse(data['box'], width, height) if data.get('box') else Box.entire(width, height)
        tile = integer(data.get('tile', max(12, math.ceil(max(box.width, box.height) / 14))), 'Cell width', 4, 600)
        count = math.ceil(box.width / tile) * math.ceil(box.height / tile)
        if count > 900:
            raise InputError('Choose larger cells or a smaller inspection window to keep the grid below 900 cells.')
        return cls(
            box, tile, number(data.get('min_fraction', 0.45), 'Minimum orientation coverage', 0.01, 1),
            enum_value(data.get('weighting', 'polarized'), ('equal', 'polarized'), 'Grid weighting'),
        )

    def cells(self):
        for row, y in enumerate(range(self.box.y, self.box.bottom, self.tile)):
            for column, x in enumerate(range(self.box.x, self.box.right, self.tile)):
                yield column, row, Box(x, y, min(self.tile, self.box.right - x), min(self.tile, self.box.bottom - y))


def _one_cell(field, scope, cfg):
    area = scope.slices
    axes = field.oriented[area]
    count = int(axes.sum())
    fraction = count / scope.area
    base = {
        'box': scope.as_dict(),
        'measured': int(field.measured[area].sum()),
        'oriented': count,
        'fraction': fraction,
        'degree_mean': None,
        'axis': None,
        'coherence': None,
        'state': 'uncovered' if count == 0 else 'insufficient',
    }
    if count == 0 or fraction < cfg.fraction:
        return base
    q, u, confidence = (layer[area][axes].astype(np.float64) for layer in (field.q, field.u, field.polarized))
    degree = field.degree[area][axes]
    if cfg.vector_weighting == 'equal':
        q, u = q / confidence, u / confidence
        mass = count
    else:
        mass = float(confidence.sum())
    a, b = float(q.sum() / mass), float(u.sum() / mass)
    coherence = min(1.0, math.hypot(a, b))
    return dict(base, degree_mean=float(degree.mean()), axis=axial_mean(a, b) if coherence >= 0.01 else None,
                coherence=coherence, state='directional' if coherence >= 0.01 else 'balanced')


def orientation_grid(results, raw):
    cfg = GridStudy.parse(raw, results.decoded.width, results.decoded.height)
    cells = [dict(_one_cell(results.field, bounds, cfg), column=column, row=row) for column, row, bounds in cfg.cells()]
    return {
        'window': cfg.box.as_dict(),
        'columns': math.ceil(cfg.box.width / cfg.tile),
        'rows': math.ceil(cfg.box.height / cfg.tile),
        'tile': cfg.tile,
        'min_fraction': cfg.fraction,
        'weighting': cfg.vector_weighting,
        'usable_cells': sum(cell['state'] == 'directional' for cell in cells),
        'covered_cells': sum(cell['state'] in ('directional', 'balanced') for cell in cells),
        'cells': cells,
    }


def _axis_glyph(draw, left, top, right, bottom, angle, confidence):
    x, y = (left + right) / 2, (top + bottom) / 2
    radians = math.radians(angle)
    bar = min(right - left, bottom - top) * (0.17 + confidence * 0.25)
    dx, dy = math.cos(radians) * bar, math.sin(radians) * bar
    endpoints = (x - dx, y - dy, x + dx, y + dy)
    draw.line(endpoints, fill=(19, 30, 45), width=3)
    draw.line(endpoints, fill=(253, 252, 230), width=1)


def orientation_grid_png(data, zero_angle):
    window = data['window']
    scale = min(1.6, max(0.25, 850 / max(window['width'], window['height'])))
    width = max(1, round(window['width'] * scale))
    height = max(1, round(window['height'] * scale))
    canvas = Image.new('RGB', (width, height), (29, 39, 54))
    draw = ImageDraw.Draw(canvas)
    for cell in data['cells']:
        b = cell['box']
        left = round((b['x'] - window['x']) * scale)
        top = round((b['y'] - window['y']) * scale)
        right = max(left, round((b['x'] + b['width'] - window['x']) * scale) - 1)
        bottom = max(top, round((b['y'] + b['height'] - window['y']) * scale) - 1)
        if cell['axis'] is None:
            color = (47, 66, 92) if cell['state'] == 'insufficient' else (29, 39, 54)
            if cell['state'] == 'balanced':
                color = (149, 151, 123)
            draw.rectangle((left, top, right, bottom), fill=color, outline=(23, 33, 49))
            continue
        signal = np.array([cell['axis']])
        hue = map_colors(signal, 0, 180, 'axial')[0].astype(np.float64)
        strength = 0.52 + 0.47 * cell['coherence']
        background = np.array([49, 58, 74])
        color = tuple(int(v) for v in hue * strength + background * (1 - strength))
        draw.rectangle((left, top, right, bottom), fill=color, outline=(23, 33, 49))
        _axis_glyph(draw, left, top, right, bottom, cell['axis'] + zero_angle, cell['coherence'])
    return rgba_png(np.asarray(canvas))


def orientation_csv(result):
    return csv_text(('column', 'row', 'x', 'y', 'width', 'height', 'measured', 'oriented', 'fraction', 'mean_degree', 'axis', 'coherence', 'state'), (
        (c['column'], c['row'], *[c['box'][k] for k in ('x', 'y', 'width', 'height')], c['measured'], c['oriented'], c['fraction'],
         c['degree_mean'], c['axis'], c['coherence'], c['state']) for c in result['cells']
    ))
