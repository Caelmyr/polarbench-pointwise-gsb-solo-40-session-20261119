from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import rgba_png
from .domain import Box, enum_value, integer, number, object_value
from .regions import csv_text
from .render import map_colors

PAIR_LABELS = ('0° + 90°', '45° + 135°')
SUPPORT_COLORS = ((31, 41, 56), (217, 114, 182), (106, 184, 234), (88, 180, 141))


@dataclass(frozen=True)
class PairAudit:
    box: Box
    first: np.ndarray
    second: np.ndarray
    first_valid: np.ndarray
    second_valid: np.ndarray
    limit: float
    mode: str
    view_limit: float
    bins: int

    @property
    def shared(self):
        return self.first_valid & self.second_valid

    @property
    def residual(self):
        return self.first - self.second


def _sum_pair(results, pair, window):
    a, b = (results.aligned[angle] for angle in pair)
    crop = window.slices
    mask = a.valid[crop] & b.valid[crop]
    if results.decoded.mask is not None:
        mask &= results.decoded.mask[crop]
    return a.data.values[crop] + b.data.values[crop], mask


def pair_audit(results, raw):
    request = object_value(raw, 'Opposite analyzer pairs')
    w, h = results.decoded.width, results.decoded.height
    window = Box.parse(request['box'], w, h) if request.get('box') else Box.entire(w, h)
    first, fm = _sum_pair(results, (0, 90), window)
    second, sm = _sum_pair(results, (45, 135), window)
    return PairAudit(window, first, second, fm, sm,
                     number(request.get('threshold', 0.10), 'Pair error threshold', 0.001, 1),
                     enum_value(request.get('mode', 'coverage'), ('coverage', 'signed-difference', 'first', 'second'), 'Pair evidence'),
                     number(request.get('display_max', 0.4), 'Difference display range', 0.001, 8),
                     integer(request.get('bins', 30), 'Pair histogram bins', 6, 180))


def _metrics(values, support):
    data = values[support]
    return {
        'usable': int(data.size),
        'mean': float(data.mean(dtype=np.float64)) if data.size else None,
        'median': float(np.median(data)) if data.size else None,
        'p90': float(np.percentile(data, 90)) if data.size else None,
    }


def _ratio(audit):
    intensity = (audit.first + audit.second) * 0.5
    return audit.residual / np.maximum(intensity, 0.0001)


def pair_summary(audit):
    shared = audit.shared
    differences = _ratio(audit)[shared]
    counts, edges = np.histogram(differences, bins=audit.bins, range=(-1, 1))
    coverage = {
        'both': int(shared.sum()),
        'first_only': int((audit.first_valid & ~audit.second_valid).sum()),
        'second_only': int((audit.second_valid & ~audit.first_valid).sum()),
        'unusable': int((~audit.first_valid & ~audit.second_valid).sum()),
    }
    return {
        'window': audit.box.as_dict(),
        'coverage': coverage,
        'pairs': {label: _metrics(values, support) for label, values, support in zip(
            PAIR_LABELS, (audit.first, audit.second), (audit.first_valid, audit.second_valid))
        },
        'threshold': audit.limit,
        'above_threshold': int((abs(differences) >= audit.limit).sum()),
        'median_signed_error': float(np.median(differences)) if differences.size else None,
        'histogram': {'counts': counts.tolist(), 'edges': edges.tolist(), 'outside': differences.size - int(counts.sum())},
    }


def _coverage_palette(audit):
    indexes = audit.first_valid.astype(np.uint8) + 2 * audit.second_valid.astype(np.uint8)
    return np.asarray(SUPPORT_COLORS, dtype=np.uint8)[indexes]


def audit_image(audit):
    if audit.mode == 'coverage':
        return rgba_png(_coverage_palette(audit))
    if audit.mode == 'signed-difference':
        values = map_colors(_ratio(audit), -audit.view_limit, audit.view_limit, 'balance')
        return rgba_png(values, audit.shared)
    values, support = (audit.first, audit.first_valid) if audit.mode == 'first' else (audit.second, audit.second_valid)
    return rgba_png(map_colors(values, 0, 1.3, 'gray'), support)


def pair_histogram_csv(report):
    histogram = report['histogram']
    return csv_text(('lower_relative_error', 'upper_relative_error', 'shared_samples'), (
        (histogram['edges'][i], histogram['edges'][i + 1], value)
        for i, value in enumerate(histogram['counts'])
    ))
