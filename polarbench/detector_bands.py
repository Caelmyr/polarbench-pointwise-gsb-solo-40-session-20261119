from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .domain import enum_value, integer, number, object_value
from .regions import csv_text, select_region
from .settings import LAYERS, SCALAR_LAYERS

BAND_LAYERS = tuple(layer for layer in SCALAR_LAYERS if layer != 'direction')


@dataclass(frozen=True)
class BandSpec:
    axis: str
    layer: str
    step: int
    min_fraction: float

    @classmethod
    def parse(cls, raw, default_layer):
        data = object_value(raw, 'Detector bands')
        return cls(
            enum_value(data.get('axis', 'rows'), ('rows', 'columns'), 'Band direction'),
            enum_value(data.get('layer', default_layer if default_layer in BAND_LAYERS else 'intensity'), BAND_LAYERS, 'Band layer'),
            integer(data.get('step', 8), 'Band thickness', 1, 160),
            number(data.get('min_fraction', 0.3), 'Minimum usable fraction', 0.01, 1),
        )


def _bands(geometry, config):
    box = geometry.box
    rows = config.axis == 'rows'
    count = box.height if rows else box.width
    for start in range(0, count, config.step):
        end = min(start + config.step, count)
        section = (slice(start, end), slice(None)) if rows else (slice(None), slice(start, end))
        offset = box.y if rows else box.x
        yield start + offset, end + offset, section


def detect_bands(results, record, raw, view):
    data = object_value(raw, 'Detector bands')
    config = BandSpec.parse(data, view.layer)
    height, width = results.field.shape
    shape, name = select_region(record, data, width, height)
    array, valid = results.channel(config.layer, view.analyzer_angle)
    cut = shape.box.slices
    inside = shape.local_mask()
    readings = array[cut]
    usable = inside & valid[cut]
    output = []
    for start, end, region in _bands(shape, config):
        count = int(inside[region].sum())
        valid_count = int(usable[region].sum())
        support = valid_count / count if count else None
        measurement = None
        if valid_count and support >= config.min_fraction:
            values = readings[region][usable[region]]
            lower, center, upper = map(float, np.percentile(values, (10, 50, 90)))
            measurement = {'p10': lower, 'median': center, 'p90': upper, 'mean': float(values.mean(dtype=np.float64))}
        output.append({'first': start, 'after_last': end, 'inside': count, 'usable': valid_count,
                       'fraction': support, 'measurement': measurement})
    return {
        'region': name,
        'geometry': shape.as_dict(),
        'axis': config.axis,
        'layer': config.layer,
        'step': config.step,
        'threshold': config.min_fraction,
        'usable_bands': sum(row['measurement'] is not None for row in output),
        'bands': output,
    }


def band_chart(report):
    image = Image.new('RGB', (760, 408), (23, 32, 44))
    draw = ImageDraw.Draw(image)
    rows = report['bands']
    layer = LAYERS[report['layer']]
    draw.text((27, 22), f'{layer.label} across detector {report["axis"]}',
              fill=(237, 243, 248), font=ImageFont.load_default(size=17))
    draw.text((27, 52), f'{report["usable_bands"]} of {len(rows)} bands meet required coverage.',
              fill=(166, 185, 206), font=ImageFont.load_default(size=13))
    x0, x1, y0, y1, support_top, support_floor = 70, 718, 88, 276, 302, 348
    draw.line((x0, y0, x0, y1, x1, y1), fill=(94, 119, 148))
    draw.line((x0, support_floor, x1, support_floor), fill=(94, 119, 148))
    draw.text((x0 - 8, support_top), 'Coverage', anchor='rm', fill=(165, 185, 205))
    highlighted = [row['measurement'] for row in rows if row['measurement'] is not None]
    low = min((s['p10'] for s in highlighted), default=0.0)
    high = max((s['p90'] for s in highlighted), default=1.0)
    if high - low < 1e-8:
        low, high = low - 0.01, high + 0.01
    for piece in np.linspace(0, 1, 5):
        tick = round(y1 - piece * (y1 - y0))
        draw.line((x0, tick, x1, tick), fill=(55, 73, 95))
        draw.text((x0 - 8, tick), f'{low + piece * (high - low):.3g}', anchor='rm', fill=(164, 182, 200))
    extent = rows[-1]['after_last'] - rows[0]['first']
    cutoff_y = round(support_floor - report['threshold'] * (support_floor - support_top))
    draw.line((x0, cutoff_y, x1, cutoff_y), fill=(238, 165, 102))
    last = None
    for band in rows:
        left = round(x0 + (band['first'] - rows[0]['first']) / extent * (x1 - x0))
        right = round(x0 + (band['after_last'] - rows[0]['first']) / extent * (x1 - x0))
        x = (right + left) // 2
        coverage = band['fraction']
        if coverage is not None and coverage > 0:
            top = min(support_floor - 1, round(support_floor - coverage * (support_floor - support_top)))
            draw.rectangle((left, top, max(left, right - 1), support_floor - 1), fill=(91, 181, 164))
        measurement = band['measurement']
        if not measurement:
            last = None
            continue
        y = lambda sample: round(y1 - (sample - low) / (high - low) * (y1 - y0))
        draw.line((x, y(measurement['p10']), x, y(measurement['p90'])), fill=(101, 139, 197))
        position = x, y(measurement['median'])
        if last is not None:
            draw.line((*last, *position), fill=(110, 219, 192), width=2)
        last = position
    draw.text((x0, 364), str(rows[0]['first']), anchor='mt', fill=(194, 207, 223))
    draw.text((x1, 364), str(rows[-1]['after_last'] - 1), anchor='mt', fill=(194, 207, 223))
    draw.text(((x1 + x0) // 2, 369), 'Detector coordinate (px)', anchor='mt', fill=(194, 207, 223))
    return rgba_png(np.asarray(image))


def bands_csv(report):
    return csv_text(('first', 'after_last', 'pixels', 'usable', 'coverage', 'mean', 'p10', 'median', 'p90'), (
        (r['first'], r['after_last'], r['inside'], r['usable'], r['fraction'],
         *((r['measurement'] or {}).get(k) for k in ('mean', 'p10', 'median', 'p90')))
        for r in report['bands']
    ))
