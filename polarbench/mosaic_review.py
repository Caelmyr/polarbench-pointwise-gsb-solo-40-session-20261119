from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .domain import Box, InputError, enum_value, integer, object_value
from .ingest import MOSAIC_PRESETS, source_download
from .codec import read_image
from .render import map_colors

ANGLE_COLORS = {0: (121, 219, 194), 45: (119, 173, 242), 90: (244, 181, 111), 135: (226, 132, 180)}


@dataclass(frozen=True)
class MosaicStudy:
    key: str
    layout: tuple[int, ...]
    window: Box
    raw: object
    mode: str

    def raw_slice(self, row, col):
        box = self.window
        return np.s_[box.y * 2 + row:box.bottom * 2:2, box.x * 2 + col:box.right * 2:2]


def mosaic_study(store, record, raw):
    if record['capture']['type'] != 'mosaic':
        raise InputError('The tile layout is available for raw micro-polarizer mosaics only.')
    data = object_value(raw, 'Sensor tile review')
    cap = record['capture']
    width, height = cap['width'], cap['height']
    box = Box.parse(data['box'], width, height) if data.get('box') else Box.entire(width, height)
    mode = enum_value(data.get('mode', 'map'), ('map', 'tile'), 'Sensor map')
    if mode == 'tile':
        cols = integer(data.get('columns', 6), 'Tile preview columns', 1, 14)
        rows = integer(data.get('rows', 5), 'Tile preview rows', 1, 14)
        box = Box(box.x, box.y, min(box.width, cols), min(box.height, rows))
    payload, _ = source_download(store, record, 'mosaic')
    pattern = cap['mosaic_layout']
    return MosaicStudy(pattern, MOSAIC_PRESETS[pattern], box, read_image(payload), mode)


def _lane_reading(study, position, angle):
    row, col = divmod(position, 2)
    crop = study.raw_slice(row, col)
    valid = study.raw.valid[crop]
    codes = study.raw.values[crop][valid]
    measurement = None
    maximum = (1 << study.raw.bit_depth) - 1
    if codes.size:
        lo, mid, hi = np.percentile(codes, (2, 50, 98))
        measurement = {'min': int(codes.min()), 'p02': float(lo), 'median': float(mid), 'p98': float(hi),
                       'max': int(codes.max()), 'raw_full_scale': int(np.count_nonzero(codes == maximum))}
    return {
        'angle': angle, 'position': {'row': row, 'column': col}, 'pixels': int(valid.size),
        'native_valid': int(valid.sum()), 'bit_depth': study.raw.bit_depth, 'measurement': measurement,
        'legend': ANGLE_COLORS[angle],
    }


def describe_mosaic(study):
    rows = [_lane_reading(study, p, angle) for p, angle in enumerate(study.layout)]
    raw_width, raw_height = study.raw.width, study.raw.height
    return {
        'pattern': study.key,
        'cells': study.window.as_dict(),
        'sensor': {'width': raw_width, 'height': raw_height},
        'sensor_window': {'x': study.window.x * 2, 'y': study.window.y * 2,
                          'width': study.window.width * 2, 'height': study.window.height * 2},
        'lanes': rows,
    }


def _partial_samples(study, position):
    row, column = divmod(position, 2)
    cut = study.raw_slice(row, column)
    points = study.raw.values[cut].astype(np.float32) / ((1 << study.raw.bit_depth) - 1)
    return points, study.raw.valid[cut]


def _preview_font(size):
    return ImageFont.load_default(size=size)


def _picture_cells(study):
    window = study.window
    tile_size = min(37, max(17, 600 // max(2 * window.height, 2 * window.width)))
    width, height = 2 * window.width * tile_size + 48, 2 * window.height * tile_size + 80
    view = Image.new('RGB', (width, height), (20, 28, 40))
    draw = ImageDraw.Draw(view)
    x0, y0 = 24, 55
    draw.text((x0, 19), f'{window.width} × {window.height} detector cells', fill=(236, 245, 252), font=_preview_font(14))
    for position, angle in enumerate(study.layout):
        dy, dx = divmod(position, 2)
        data, usable = _partial_samples(study, position)
        for y in range(window.height):
            line_y = y0 + (y * 2 + dy) * tile_size
            for x in range(window.width):
                point_x = x0 + (x * 2 + dx) * tile_size
                chip = ANGLE_COLORS[angle]
                keep = usable[y, x]
                strength = 0.2 + 0.8 * float(data[y, x])
                rgb = tuple(round(c * strength) for c in chip) if keep else (45, 47, 57)
                draw.rectangle((point_x, line_y, point_x + tile_size - 1, line_y + tile_size - 1), fill=rgb, outline=(22, 31, 43))
                if tile_size >= 20:
                    msg = str(angle) if keep else '–'
                    draw.text((point_x + tile_size / 2, line_y + tile_size / 2), msg,
                              anchor='mm', font=_preview_font(10), fill=(237, 246, 250))
    return rgba_png(np.asarray(view))


def _picture_lanes(study):
    width, height = 860, 687
    board = Image.new('RGBA', (width, height), (23, 32, 46, 255))
    draw = ImageDraw.Draw(board)
    draw.text((26, 19), 'Native micro-polarizer lanes', fill=(244, 249, 251), font=_preview_font(20))
    draw.text((26, 52), f'Pattern: {study.key} · upper left, upper right / lower left, lower right',
              fill=(164, 190, 214), font=_preview_font(12))
    for position, angle in enumerate(study.layout):
        y, x = divmod(position, 2)
        x0, y0 = 26 + x * 408, 79 + y * 293
        draw.rounded_rectangle((x0, y0, x0 + 391, y0 + 275), radius=9, fill=(14, 23, 35), outline=(56, 75, 97))
        caption = f'{angle}°   ·   Sensor parity ({x}, {y})'
        draw.text((x0 + 12, y0 + 12), caption, fill=ANGLE_COLORS[angle], font=_preview_font(14))
        levels, support = _partial_samples(study, position)
        pixels = map_colors(levels, 0, 1, 'gray')
        alpha = np.where(support, 255, 0).astype(np.uint8)
        layer = Image.fromarray(np.dstack((pixels, alpha)))
        layer.thumbnail((365, 223), Image.Resampling.BOX)
        board.alpha_composite(layer, (x0 + 12 + (365 - layer.width) // 2, y0 + 42 + (223 - layer.height) // 2))
    return rgba_png(np.asarray(board))


def preview_mosaic(study):
    return _picture_cells(study) if study.mode == 'tile' else _picture_lanes(study)
