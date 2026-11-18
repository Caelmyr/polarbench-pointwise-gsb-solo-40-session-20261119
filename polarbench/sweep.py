from __future__ import annotations

import dataclasses
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .domain import InputError, integer, number, object_value
from .regions import csv_text, select_region
from .render import map_colors
from .stokes import virtual_analyzer


@dataclasses.dataclass(frozen=True)
class SweepRequest:
    start: float
    stop: float
    count: int
    level: float

    @classmethod
    def parse(cls, raw):
        data = object_value(raw, 'Analyzer sweep')
        start = number(data.get('start', 0), 'First analyzer angle', 0, 180)
        stop = number(data.get('stop', 165), 'Last analyzer angle', 0, 180)
        if start >= stop:
            raise InputError('The sweep needs increasing endpoints within one 180° period.')
        return cls(start, stop, integer(data.get('count', 12), 'Number of analyzer angles', 2, 36),
                   number(data.get('signal_level', 0.2), 'Signal threshold', 0, 8))

    def positions(self):
        return np.linspace(self.start, self.stop, self.count)


def _curves(s0, q, u, cfg):
    records = []
    for angle in cfg.positions():
        theta = np.deg2rad(angle * 2)
        values = np.maximum(0, (s0 + q * np.cos(theta) + u * np.sin(theta)) * 0.5)
        if values.size:
            lo, median, hi = np.percentile(values, (10, 50, 90))
            measurement = {'mean': float(values.mean(dtype=np.float64)), 'p10': float(lo), 'median': float(median),
                           'p90': float(hi), 'under_target': int(np.count_nonzero(values < cfg.level))}
        else:
            measurement = None
        records.append({'analyzer_angle': float(angle), 'metrics': measurement})
    return records


def measure_sweep(results, record, raw):
    spec = SweepRequest.parse(raw)
    height, width = results.field.shape
    shape, name = select_region(record, raw, width, height)
    selected = shape.local_mask()
    detector = results.field
    cut = shape.box.slices
    valid = detector.measured[cut] & selected
    parts = [component[cut][valid] for component in (detector.s0, detector.q, detector.u)]
    return {
        'region': name,
        'bounds': shape.box.as_dict(),
        'pixels': int(selected.sum()),
        'usable': int(valid.sum()),
        'minimum_signal': spec.level,
        'angles': _curves(*parts, spec),
        'basis': 'Selected reference axis; each angle operates on an identical valid detector support.',
    }


def sweep_csv(result):
    fields = ('angle_deg', 'selected', 'usable', 'mean', 'p10', 'median', 'p90', 'under_signal_threshold')
    return csv_text(fields, (
        (sample['analyzer_angle'], result['pixels'], result['usable'],
         *((sample['metrics'] or {}).get(name) for name in ('mean', 'p10', 'median', 'p90', 'under_target')))
        for sample in result['angles']
    ))


def _render_samples(results, geometry, view, sampled):
    target_extent = (286, 184)
    selection = geometry.box.slices
    local_support = results.field.measured[selection] & geometry.local_mask()
    for angle in sampled:
        response = virtual_analyzer(results.field, float(angle))[selection]
        mapped = rgba_png(map_colors(response, view.display.low, view.display.high, 'gray', view.gamma), local_support)
        with Image.open(io.BytesIO(mapped)) as frame:
            copy = frame.convert('RGBA')
            copy.thumbnail(target_extent, Image.Resampling.LANCZOS)
            yield copy


def sweep_sheet(results, record, view, raw):
    definition = SweepRequest.parse(raw)
    if definition.count > 12:
        raise InputError('The contact sheet supports up to 12 angles. Use the CSV for larger sweeps.')
    data = object_value(raw, 'Sweep sheet')
    shape, name = select_region(record, data, results.decoded.width, results.decoded.height)
    angles = definition.positions()
    columns = min(3, definition.count)
    rows = (definition.count + columns - 1) // columns
    canvas = Image.new('RGBA', (50 + columns * 316, 94 + rows * 226), (25, 35, 50, 255))
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.load_default(size=19), ImageFont.load_default(size=13)
    draw.text((25, 21), 'Virtual analyzer sweep', fill=(240, 247, 251), font=font)
    draw.text((25, 52), f'{name[:54]} · {shape.box.width} × {shape.box.height} detector pixels', fill=(153, 180, 204), font=small)
    for i, (angle, frame) in enumerate(zip(angles, _render_samples(results, shape, view, angles))):
        x, y = 25 + i % columns * 316, 86 + i // columns * 226
        draw.rounded_rectangle((x, y, x + 301, y + 211), 9, fill=(12, 21, 34), outline=(58, 78, 102))
        draw.text((x + 10, y + 9), f'{angle:.1f}°', fill=(234, 244, 249), font=small)
        canvas.alpha_composite(frame, (x + 7 + (286 - frame.width) // 2, y + 26 + (184 - frame.height) // 2))
    return rgba_png(np.asarray(canvas))


def _response_axes(output, report):
    draw = ImageDraw.Draw(output)
    title, small = ImageFont.load_default(size=18), ImageFont.load_default(size=13)
    draw.text((25, 20), f'Analyzer response · {report["region"][:50]}', fill=(235, 243, 250), font=title)
    caption = f'{report["usable"]:,} / {report["pixels"]:,} detector pixels usable at each angle'
    draw.text((25, 50), caption, fill=(158, 181, 204), font=small)
    left, top, right, bottom = 70, 88, 734, 308
    axis, grid = (113, 133, 156), (59, 76, 98)
    draw.line((left, top, left, bottom, right, bottom), fill=axis)
    angles = report['angles']
    start, end = angles[0]['analyzer_angle'], angles[-1]['analyzer_angle']
    target = report['minimum_signal']
    high = max((max(sample['metrics']['p90'], target) for sample in angles if sample['metrics']), default=1.0)
    high = max(0.001, high * 1.06)
    for part in np.linspace(0, 1, 6):
        row = round(bottom - part * (bottom - top))
        col = round(left + part * (right - left))
        draw.line((left, row, right, row), fill=grid)
        draw.text((left - 7, row), f'{high * part:.3g}', anchor='rm', fill=(188, 205, 224))
        draw.text((col, bottom + 9), f'{start + part * (end - start):.1f}°', anchor='mt', fill=(188, 205, 224))
    return draw, (left, top, right, bottom), high


def sweep_curve(report):
    canvas = Image.new('RGB', (760, 380), (23, 33, 47))
    draw, (left, top, right, bottom), high = _response_axes(canvas, report)
    samples = report['angles']
    start, end = samples[0]['analyzer_angle'], samples[-1]['analyzer_angle']
    level = report['minimum_signal']
    cut = round(bottom - level / high * (bottom - top))
    valid = [sample for sample in samples if sample['metrics'] is not None]
    for x in range(left, right, 12):
        if top <= cut <= bottom:
            draw.line((x, cut, min(right, x + 6), cut), fill=(229, 164, 101))
    if valid:
        def mapped(sample, metric):
            pos = sample['analyzer_angle']
            val = sample['metrics'][metric]
            return (round(left + (pos - start) / (end - start) * (right - left)),
                    round(bottom - val / high * (bottom - top)))
        floor = [mapped(point, 'p10') for point in valid]
        cap = [mapped(point, 'p90') for point in valid]
        median = [mapped(point, 'median') for point in valid]
        draw.polygon([*floor, *reversed(cap)], fill=(45, 90, 111))
        if len(median) > 1:
            draw.line(median, width=3, fill=(119, 225, 201))
        for x, y in median:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(132, 240, 211))
    for i, (text, color) in enumerate((('10–90% interval', (45, 90, 111)), ('Median', (119, 225, 201)), ('Signal threshold', (229, 164, 101)))):
        xx = left + i * 160
        draw.rounded_rectangle((xx, 345, xx + 12, 356), radius=2, fill=color)
        draw.text((xx + 17, 343), text, fill=(201, 214, 226), font=ImageFont.load_default(size=13))
    return rgba_png(np.asarray(canvas))
