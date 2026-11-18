from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .domain import InputError, number, object_value
from .quality import BY_ISSUE
from .regions import csv_text, select_region
from .stokes import analyzer_response

# One analyzer angle applies to the entire detector image. A candidate keeps the protected area bright
# where at least the requested fraction of its measurable pixels stay above the requested share of their
# brightness at the angle currently being previewed. Glare is ranked by a high percentile per region so a
# handful of isolated specks cannot decide the proposal for the whole image.
DEFAULT_ANGLE_STEP = 1.0
DEFAULT_RETENTION = 0.85
DEFAULT_RETAINED_FRACTION = 0.90
GLARE_PERCENTILE = 90.0
MAX_ANGLES = 3600
MAX_ANGLE_PIXELS = 40_000_000
PROTECT_KEYS = ('protect', 'preserve', 'retain', 'bright')
GLARE_KEYS = ('glare', 'reflection', 'highlight')
EXCLUSION_KEYS = ('outside', 'alpha', 'reference', 'mask', 'saturation', 'dark', 'closure', 'physical')


@dataclasses.dataclass(frozen=True)
class BalanceRequest:
    protect: dict
    glare: dict
    baseline: float
    angle_step: float
    retention: float
    retained_fraction: float

    @classmethod
    def parse(cls, raw, view):
        data = object_value(raw, 'Analyzer balance')
        protect = _selection(data, PROTECT_KEYS, 'protected')
        glare = _selection(data, GLARE_KEYS, 'glare')
        if protect.get('region_id') and protect.get('region_id') == glare.get('region_id'):
            raise InputError('The protected and glare areas must be different saved regions.')
        step = number(data.get('angle_step', DEFAULT_ANGLE_STEP), 'Analyzer angle step', 0.05, 10)
        if 180 / step > MAX_ANGLES:
            raise InputError(f'The analyzer angle step leaves more than {MAX_ANGLES} candidates.')
        return cls(
            protect, glare,
            number(data.get('baseline_angle', view.analyzer_angle), 'Current analyzer angle', 0, 180) % 180,
            step,
            number(data.get('retention_level', DEFAULT_RETENTION), 'Protected brightness level', 0.05, 1),
            number(data.get('retained_fraction', DEFAULT_RETAINED_FRACTION), 'Protected pixel fraction', 0.05, 1),
        )

    def positions(self):
        grid = np.arange(0, 180, self.angle_step, dtype=np.float64)
        values = np.unique(np.round(np.append(grid, self.baseline), 6))
        return values[(values >= 0) & (values < 180)]

    def check_work(self, pixels):
        if self.positions().size * pixels > MAX_ANGLE_PIXELS:
            raise InputError('The analyzer angle step is too fine for the size of these regions. Use a coarser step.')


def _selection(data, keys, role):
    for key in keys:
        value = data.get(f'{key}_region_id')
        if value:
            return {'region_id': value}
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return {'region_id': value}
        if isinstance(value, dict) and (value.get('region_id') or value.get('geometry') is not None):
            return {'region_id': value.get('region_id'), 'geometry': value.get('geometry')}
    raise InputError(f'Select a saved region for the {role} area.')


def _rounded(value):
    return None if value is None else round(float(value), 7)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    p10, p50, p90, p95 = np.percentile(values, (10, 50, 90, 95))
    return {
        'mean': _rounded(values.mean()),
        'median': _rounded(p50),
        'p10': _rounded(p10),
        'p90': _rounded(p90),
        'p95': _rounded(p95),
        'max': _rounded(values.max()),
    }


def _ratio(value, reference):
    if value is None or reference is None or reference == 0:
        return None
    return _rounded(value / reference)


def _support(field, quality, geometry):
    cut = geometry.box.slices
    local = geometry.local_mask()
    flags = quality.flags[cut]
    usable = local & field.measured[cut]
    if not usable.any():
        raise InputError('A selected region has no measurable detector pixels after coverage and mask checks.')
    excluded = {key: int(np.count_nonzero(local & ((flags & BY_ISSUE[key].bit) != 0))) for key in EXCLUSION_KEYS}
    return {
        'cut': cut,
        'pixels': int(np.count_nonzero(local)),
        'usable_pixels': int(np.count_nonzero(usable)),
        'excluded': excluded,
        'not_measured': int(np.count_nonzero(local & ~field.measured[cut])),
        's0': field.s0[cut][usable],
        'q': field.q[cut][usable],
        'u': field.u[cut][usable],
    }


def _region_report(name, geometry, support, selected):
    return {
        'region': name,
        'bounds': geometry.box.as_dict(),
        'pixels': support['pixels'],
        'usable': support['usable_pixels'],
        'excluded': support['excluded'],
        'excluded_total': support['not_measured'],
        'baseline': _summary(support['baseline']),
        'suggested': _summary(selected),
    }


def _curve(protect, glare, cfg):
    rows = []
    for angle in cfg.positions():
        angle = float(angle)
        retained = analyzer_response(protect['s0'], protect['q'], protect['u'], angle)
        kept = int(np.count_nonzero(retained >= cfg.retention * protect['baseline']))
        response = analyzer_response(glare['s0'], glare['q'], glare['u'], angle)
        p50, p90 = np.percentile(response, (50, GLARE_PERCENTILE))
        rows.append({
            'angle': round(angle, 6),
            'protect_retained': kept,
            'protect_retained_fraction': _rounded(kept / protect['usable_pixels']),
            'glare_p50': _rounded(p50),
            'glare_p90': _rounded(p90),
            'glare_max': _rounded(response.max()),
            'feasible': bool(kept / protect['usable_pixels'] >= cfg.retained_fraction - 1e-12),
        })
    return rows


def _choose(rows, cfg):
    feasible = [row for row in rows if row['feasible']]
    if not feasible:
        return None
    # Equal glare scores favour the angle nearest the current preview, so a negligible gain never shifts it far.
    return min(feasible, key=lambda row: (row['glare_p90'], _separation(row['angle'], cfg.baseline), row['angle']))


def _separation(a, b):
    distance = abs(a - b) % 180
    return min(distance, 180 - distance)


def suggest_balance(results, record, raw, view):
    cfg = BalanceRequest.parse(raw, view)
    field, quality = results.field, results.quality
    height, width = field.shape
    protect_geometry, protect_name = select_region(record, cfg.protect, width, height)
    glare_geometry, glare_name = select_region(record, cfg.glare, width, height)
    protect = _support(field, quality, protect_geometry)
    glare = _support(field, quality, glare_geometry)
    protect['baseline'] = analyzer_response(protect['s0'], protect['q'], protect['u'], cfg.baseline)
    glare['baseline'] = analyzer_response(glare['s0'], glare['q'], glare['u'], cfg.baseline)
    cfg.check_work(protect['usable_pixels'] + glare['usable_pixels'])
    rows = _curve(protect, glare, cfg)
    best = _choose(rows, cfg)
    if best is None:
        raise InputError(
            f'No analyzer angle keeps {cfg.retained_fraction * 100:g}% of the protected region above '
            f'{cfg.retention * 100:g}% of its current brightness. Widen the step or relax the requirement.'
        )
    angle = best['angle']
    protect_at = analyzer_response(protect['s0'], protect['q'], protect['u'], angle)
    glare_at = analyzer_response(glare['s0'], glare['q'], glare['u'], angle)
    protect_baseline, glare_baseline = _summary(protect['baseline']), _summary(glare['baseline'])
    protect_suggested, glare_suggested = _summary(protect_at), _summary(glare_at)
    glare_ratio = _ratio(glare_suggested['p90'], glare_baseline['p90'])
    return {
        'baseline_angle': _rounded(cfg.baseline),
        'suggested_angle': _rounded(angle),
        'changed': bool(abs(angle - cfg.baseline) > 1e-9),
        'feasible': True,
        'angle_step': cfg.angle_step,
        'candidates': len(rows),
        'feasible_angles': sum(row['feasible'] for row in rows),
        'glare_statistic': f'{GLARE_PERCENTILE:g}th percentile analyzer response',
        'requirements': {
            'retention_level': cfg.retention,
            'retained_fraction': cfg.retained_fraction,
        },
        'protect': dict(_region_report(protect_name, protect_geometry, protect, protect_at),
                        retained=best['protect_retained'],
                        retained_fraction=best['protect_retained_fraction'],
                        change={
                            'mean': _rounded(protect_suggested['mean'] - protect_baseline['mean']),
                            'median': _rounded(protect_suggested['median'] - protect_baseline['median']),
                            'mean_ratio': _ratio(protect_suggested['mean'], protect_baseline['mean']),
                            'median_ratio': _ratio(protect_suggested['median'], protect_baseline['median']),
                        }),
        'glare': dict(_region_report(glare_name, glare_geometry, glare, glare_at),
                      change={
                          'mean': _rounded(glare_suggested['mean'] - glare_baseline['mean']),
                          'p90': _rounded(glare_suggested['p90'] - glare_baseline['p90']),
                          'mean_ratio': _ratio(glare_suggested['mean'], glare_baseline['mean']),
                          'p90_ratio': glare_ratio,
                          'p90_reduction': _rounded(1 - glare_ratio) if glare_ratio is not None else None,
                      }),
        'angles': rows,
        'basis': ('One analyzer angle for the entire detector image. Brightness loss is counted only on measurable '
                  'pixels inside the protected region; positions outside source coverage or removed by the detector '
                  'mask are excluded from both numerator and denominator.'),
    }


def _axes(draw, box, x_span, y_span, y_decimals, x_decimals=0):
    left, top, right, bottom = box
    start, end = x_span
    low, high = y_span
    label = ImageFont.load_default(size=12)
    for part in np.linspace(0, 1, 5):
        row = round(bottom - part * (bottom - top))
        draw.line((left, row, right, row), fill=(59, 76, 98))
        draw.text((left - 8, row), f'{low + (high - low) * part:.{y_decimals}f}', anchor='rm', fill=(188, 205, 224), font=label)
        if part in (0, 0.5, 1):
            draw.text((round(left + part * (right - left)), bottom + 8),
                      f'{start + (end - start) * part:.{x_decimals}f}°', anchor='mt', fill=(188, 205, 224), font=label)
    draw.line((left, top, left, bottom, right, bottom), fill=(113, 133, 156))


def _plot(draw, box, start, end, values, y_high, color):
    left, top, right, bottom = box
    points = [
        (round(left + (angle - start) / (end - start) * (right - left)),
         round(bottom - min(max(value, 0), y_high) / y_high * (bottom - top)))
        for angle, value in values
    ]
    if len(points) > 1:
        draw.line(points, width=3, fill=color)
    return points


def balance_chart(report):
    canvas = Image.new('RGB', (780, 470), (23, 33, 47))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=13)
    draw.text((27, 20), f'Analyzer balance · {report["suggested_angle"]:.1f}°', fill=(238, 246, 250), font=title)
    draw.text((27, 48), f'{report["protect"]["region"][:34]} keeps brightness · {report["glare"]["region"][:34]} suppresses glare',
              fill=(158, 181, 204), font=small)
    rows = report['angles']
    start, end = rows[0]['angle'], rows[-1]['angle']
    left_box, right_box = (72, 92, 386, 344), (466, 92, 754, 344)
    _axes(draw, left_box, (start, end), (0, 1), 2)
    glare_high = max(0.001, max(row['glare_p90'] for row in rows) * 1.08)
    _axes(draw, right_box, (start, end), (0, glare_high), 3)
    draw.text((72, 66), 'Protected brightness retained', fill=(202, 226, 220), font=small)
    draw.text((466, 66), 'Glare region p90 response', fill=(238, 205, 170), font=small)
    _plot(draw, left_box, start, end, ((row['angle'], row['protect_retained_fraction']) for row in rows), 1, (119, 225, 201))
    _plot(draw, right_box, start, end, ((row['angle'], row['glare_p90']) for row in rows), glare_high, (239, 177, 111))
    y = round(left_box[3] - report['requirements']['retained_fraction'] * (left_box[3] - left_box[1]))
    for x in range(left_box[0], left_box[2], 12):
        draw.line((x, y, min(left_box[2], x + 6), y), fill=(229, 164, 101))
    draw.text((left_box[0] + 4, y - 17), f'{report["requirements"]["retained_fraction"] * 100:.0f}% requirement',
              fill=(229, 164, 101), font=small)
    for box in (left_box, right_box):
        for angle, color in ((report['baseline_angle'], (229, 164, 101)), (report['suggested_angle'], (119, 225, 201))):
            x = round(box[0] + (angle - start) / (end - start) * (box[2] - box[0]))
            draw.line((x, box[1], x, box[3]), fill=color, width=2)
    draw.text((left_box[0], left_box[3] + 34), f'Current {report["baseline_angle"]:.1f}°', fill=(229, 164, 101), font=small)
    draw.text((left_box[0] + 150, left_box[3] + 34), f'Suggested {report["suggested_angle"]:.1f}°', fill=(119, 225, 201), font=small)
    protect, glare = report['protect'], report['glare']['change']
    reduction = '—' if glare['p90_reduction'] is None else f'{glare["p90_reduction"] * 100:+.1f}%'
    draw.text((left_box[0], left_box[3] + 56),
              f'Protected kept {protect["retained_fraction"] * 100:.1f}% · glare p90 {reduction} · '
              f'{report["feasible_angles"]} of {report["candidates"]} angles meet the requirement',
              fill=(171, 193, 214), font=small)
    return rgba_png(np.asarray(canvas))


def balance_csv(report):
    fields = ('angle_deg', 'protect_retained', 'protect_usable', 'protect_retained_fraction',
              'glare_p50', 'glare_p90', 'glare_max', 'meets_requirement')
    return csv_text(fields, (
        (row['angle'], row['protect_retained'], report['protect']['usable'], row['protect_retained_fraction'],
         row['glare_p50'], row['glare_p90'], row['glare_max'], row['feasible'])
        for row in report['angles']
    ))
