from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Calibration, reference_maps
from .codec import raw_fraction, rgba_png
from .domain import ANGLES, Box, InputError, enum_value, integer, object_value
from .regions import csv_text, select_region
from .render import map_colors


@dataclass(frozen=True)
class ReferenceStudy:
    kind: str
    values: np.ndarray
    valid: np.ndarray
    box: Box
    normalized_gains: np.ndarray | None
    width: int
    height: int


def _strict_metrics(values):
    if not values.size:
        return None
    median = float(np.median(values))
    return {
        'mean': float(values.mean(dtype=np.float64)),
        'median': median,
        'min': float(values.min()),
        'max': float(values.max()),
        'mad': float(np.median(np.abs(values - median))),
        'stddev': float(values.std(dtype=np.float64)),
        'percentile_5': float(np.percentile(values, 5)),
        'percentile_95': float(np.percentile(values, 95)),
    }


def _review_hist(values, bins):
    if values.size:
        minimum, maximum = map(float, (values.min(), values.max()))
        if maximum == minimum:
            minimum -= 0.005
            maximum += 0.005
    else:
        minimum, maximum = 0, 1
    hist, edges = np.histogram(values, bins, range=(minimum, maximum))
    return {'counts': hist.tolist(), 'edges': edges.tolist()}


def reference_study(decoded, record, processing, raw):
    request = object_value(raw, 'Calibration reference study')
    kind = enum_value(request.get('kind', 'flat'), ('dark', 'flat'), 'Reference type')
    image = decoded.dark if kind == 'dark' else decoded.flat
    if image is None:
        raise InputError(f'Upload a {kind} reference to inspect it.')
    width, height = decoded.width, decoded.height
    chosen = Box.parse(request['box'], width, height) if request.get('box') is not None else Box.entire(width, height)
    values = raw_fraction(image)
    flags = image.valid & (values < (0.9 if kind == 'dark' else 0.995))
    gains = None
    if kind == 'flat':
        config = Calibration(processing.calibration.use_dark, True, processing.calibration.tuning)
        maps = reference_maps(decoded, record, config)
        gains, flags = maps.scale, maps.flat_valid
    return ReferenceStudy(kind, values, flags, chosen, gains, width, height)


def summarize_reference(study, raw):
    data = object_value(raw, 'Calibration statistics')
    bins = integer(data.get('bins', 32), 'Reference histogram bins', 6, 160)
    cut = study.box.slices
    usable = study.valid[cut]
    intensity = study.values[cut][usable]
    signals = _strict_metrics(intensity)
    gain = _strict_metrics(study.normalized_gains[cut][usable]) if study.normalized_gains is not None else None
    row_values = np.sum(np.where(usable, study.values[cut], 0), axis=1, dtype=np.float64)
    row_counts = np.count_nonzero(usable, axis=1)
    col_values = np.sum(np.where(usable, study.values[cut], 0), axis=0, dtype=np.float64)
    col_counts = np.count_nonzero(usable, axis=0)
    return {
        'kind': study.kind,
        'window': study.box.as_dict(),
        'usable': int(usable.sum()),
        'window_pixels': int(usable.size),
        'signal': signals,
        'gain': gain,
        'histogram': _review_hist(intensity, bins),
        'row_signal': [float(s / c) if c else None for s, c in zip(row_values, row_counts)],
        'column_signal': [float(s / c) if c else None for s, c in zip(col_values, col_counts)],
    }


def reference_map_png(study, raw):
    data = object_value(raw, 'Calibration map')
    layer = enum_value(data.get('layer', 'signal'), ('signal', 'gain', 'support'), 'Reference map layer')
    values = study.values[study.box.slices]
    valid = study.valid[study.box.slices]
    if layer == 'support':
        return rgba_png(np.where(valid, 255, 0).astype(np.uint8))
    if layer == 'gain':
        if study.normalized_gains is None:
            raise InputError('A gain map requires a flat reference.')
        values = study.normalized_gains[study.box.slices]
    subset = values[valid]
    low, high = tuple(np.percentile(subset, (2, 98))) if subset.size else (0.0, 1.0)
    if high - low < 1e-5:
        low, high = low - 0.005, high + 0.005
    return rgba_png(map_colors(values, low, high, 'ice' if layer == 'signal' else 'balance'), valid)


def reference_csv(summary):
    sections = (
        ('row', summary['window']['y'], summary['row_signal']),
        ('column', summary['window']['x'], summary['column_signal']),
    )
    return csv_text(('axis', 'detector_index', 'mean_signal'), (
        (axis, start + i, value) for axis, start, readings in sections for i, value in enumerate(readings)
    ))


def suggest_neutral_balance(results, record, raw):
    data = object_value(raw, 'Neutral balance')
    if not data.get('region_id') and not data.get('geometry'):
        raise InputError('Select a neutral, diffusely lit target before estimating channel balance.')
    geometry, label = select_region(record, data, results.decoded.width, results.decoded.height)
    cut = geometry.box.slices
    chosen = geometry.local_mask() & results.field.base_support[cut]
    minimum = integer(data.get('min_pixels', 24), 'Minimum neutral samples', 8, 100000)
    count = int(chosen.sum())
    if count < minimum:
        raise InputError(f'The neutral target has {count} usable common samples. Select a larger neutral area.')
    limits = results.processing.limits
    channels = {a: results.aligned[a].data.values[cut][chosen] for a in ANGLES}
    medians = {a: float(np.median(samples)) for a, samples in channels.items()}
    if min(medians.values()) < limits.min_intensity / 2:
        raise InputError('The neutral target is below the useful signal threshold for at least one angle.')
    anchor = float(np.median(list(medians.values())))
    after = {a: results.processing.calibration.tuning[a].gain * anchor / medians[a] for a in ANGLES}
    changes = {}
    for angle, gain in after.items():
        usable_gain = 0.25 <= gain <= 4
        changes[str(angle)] = {
            'before': results.processing.calibration.tuning[angle].gain,
            'sample_median': medians[angle],
            'after': gain if usable_gain else None,
            'in_range': usable_gain,
        }
    predicted = [channel * anchor / medians[a] for a, channel in channels.items()]
    q, u = predicted[0] - predicted[2], predicted[1] - predicted[3]
    s0 = 0.5 * sum(predicted)
    remaining = np.hypot(q, u) / np.maximum(s0, 1e-4)
    return {
        'region': label,
        'target': anchor,
        'common_samples': count,
        'predicted_median_degree': float(np.median(remaining)),
        'channels': changes,
        'available': all(value['in_range'] for value in changes.values()),
    }
