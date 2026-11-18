from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .codec import rgba_png
from .domain import enum_value, integer, number, object_value
from .regions import csv_text, select_region
from .render import raster
from .settings import SCALAR_LAYERS

LOCATABLE = tuple(layer for layer in SCALAR_LAYERS if layer != 'direction')


@dataclass(frozen=True)
class MarkerSpec:
    layer: str
    kind: str
    minimum: float | None
    spacing: int
    count: int

    @classmethod
    def parse(cls, raw, layer):
        data = object_value(raw, 'Signal markers')
        target = enum_value(data.get('layer', layer if layer in LOCATABLE else 'intensity'), LOCATABLE, 'Marker layer')
        mode = enum_value(data.get('kind', 'maxima'), ('maxima', 'minima'), 'Marker type')
        cutoff = data.get('limit')
        return cls(
            target, mode, None if cutoff is None else number(cutoff, 'Marker threshold', -16, 16),
            integer(data.get('spacing', 16), 'Minimum spacing', 2, 200),
            integer(data.get('count', 12), 'Maximum markers', 1, 60),
        )


def _peak_choices(numbers, valid, spec):
    normalized = numbers if spec.kind == 'maxima' else -numbers
    raw_valid = valid.copy()
    if spec.minimum is not None:
        if spec.kind == 'maxima':
            raw_valid &= numbers >= spec.minimum
        else:
            raw_valid &= numbers <= spec.minimum
    if not raw_valid.any():
        return []
    strength = np.where(raw_valid, normalized, -np.inf)
    size = spec.spacing * 2 + 1
    exact = ndimage.maximum_filter(strength, size=size, mode='constant', cval=-np.inf)
    local_peaks = raw_valid & (strength == exact)
    chunks, count = ndimage.label(local_peaks, structure=ndimage.generate_binary_structure(2, 2))
    if count == 0:
        return []
    # Assign one representative to a flat peak at its center. Equal and detached peaks remain separate.
    candidates = []
    for index, obj in enumerate(ndimage.find_objects(chunks), start=1):
        local = (chunks[obj] == index)
        row, col = np.nonzero(local)
        values = strength[obj][local]
        strongest = values.max()
        peak_envelope = values == strongest
        distance = np.hypot(row - row.mean(), col - col.mean())
        idx = int(np.argmin(np.where(peak_envelope, distance, np.inf)))
        pos = int(row[idx] + obj[0].start), int(col[idx] + obj[1].start)
        candidates.append((float(-strongest), pos, int(local.sum())))
    candidates.sort()
    return candidates[:spec.count]


def locate_markers(results, record, raw, view):
    data = object_value(raw, 'Signal markers')
    spec = MarkerSpec.parse(data, view.layer)
    height, width = results.field.shape
    selected, site = select_region(record, data, width, height)
    values, usable = results.channel(spec.layer, view.analyzer_angle)
    crop = selected.box
    cut = crop.slices
    valid = usable[cut] & selected.local_mask()
    peaks = _peak_choices(values[cut], valid, spec)
    markers = []
    for index, (_, (row, col), plateau) in enumerate(peaks, start=1):
        x, y = col + crop.x, row + crop.y
        markers.append({'id': index, 'x': x, 'y': y, 'value': float(values[y, x]),
                        'plateau_area': plateau, 'issues': results.quality.issues_at(x, y)})
    return {'kind': spec.kind, 'layer': spec.layer, 'region': site, 'region_geometry': selected.as_dict(),
            'usable': int(valid.sum()), 'minimum_spacing': spec.spacing, 'markers': markers}


def marker_image(results, view, report):
    rendered = raster(results, view)
    with Image.open(io.BytesIO(rendered.payload)) as src:
        source = src.convert('RGBA')
    draw = ImageDraw.Draw(source)
    viewport = rendered.viewport
    sx, sy = (rendered.width / viewport.width, rendered.height / viewport.height)
    for marker in report['markers']:
        x, y = marker['x'], marker['y']
        if not viewport.contains(x, y):
            continue
        x, y = round((x - viewport.x) * sx), round((y - viewport.y) * sy)
        tip = (244, 208, 124)
        outline = (21, 31, 43)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=outline, width=3)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=tip, width=2)
        text_x = min(max(x + 9, 16), source.width - 8)
        text_y = max(y, 7)
        draw.text((text_x, text_y), str(marker['id']), fill=tip, anchor='lm', stroke_width=2, stroke_fill=outline,
                  font=ImageFont.load_default(size=13))
    return rgba_png(np.asarray(source))


def markers_csv(report):
    return csv_text(('id', 'layer', 'detector_x', 'detector_y', 'value', 'plateau_area', 'flags'), (
        (item['id'], report['layer'], item['x'], item['y'], item['value'], item['plateau_area'], '|'.join(item['issues']))
        for item in report['markers']
    ))
