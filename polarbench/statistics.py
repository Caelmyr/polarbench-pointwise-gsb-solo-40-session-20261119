from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import enum_value, integer, number, object_value
from .quality import ISSUES
from .regions import csv_text, select_region
from .settings import SCALAR_LAYERS
from .stokes import analyzer_response, axial_mean


@dataclass(frozen=True)
class AnalysisRequest:
    layer: str
    bins: int
    weighting: str
    percentiles: tuple[float, ...]

    @classmethod
    def parse(cls, raw, default_layer):
        data = object_value(raw, 'Region measurement')
        layer = enum_value(data.get('layer', default_layer), SCALAR_LAYERS, 'Measurement layer')
        bins = integer(data.get('bins', 36 if layer == 'direction' else 40), 'Histogram bins', 6, 180)
        weighting = enum_value(data.get('weighting', 'equal'), ('equal', 'polarized', 'intensity'), 'Direction weights')
        return cls(layer, bins, weighting, (2.0, 25.0, 50.0, 75.0, 98.0))


def _rounded(value):
    return None if value is None else round(float(value), 7)


def _linear_statistics(values, quantiles):
    if not values.size:
        return None
    row = np.percentile(values, quantiles)
    return {
        'min': _rounded(values.min()),
        'max': _rounded(values.max()),
        'mean': _rounded(np.mean(values, dtype=np.float64)),
        'stddev': _rounded(np.std(values, dtype=np.float64)),
        'percentiles': {f'p{p:g}': _rounded(val) for p, val in zip(quantiles, row)},
    }


def _axial_statistics(field, geometry, oriented, weight_kind):
    if not oriented.any():
        return None
    cut = geometry.box.slices
    axis = field.direction[cut][oriented].astype(np.float64)
    angle = np.deg2rad(axis * 2)
    sin, cos = np.sin(angle), np.cos(angle)
    if weight_kind == 'equal':
        weights = np.ones(axis.shape, dtype=np.float64)
    elif weight_kind == 'intensity':
        weights = field.s0[cut][oriented].astype(np.float64)
    else:
        weights = field.polarized[cut][oriented].astype(np.float64)
    mass = float(weights.sum())
    if mass == 0:
        return None
    q = np.dot(cos, weights) / mass
    u = np.dot(sin, weights) / mass
    coherence = min(1.0, float(np.hypot(q, u)))
    axis_mean = axial_mean(q, u) if coherence > 1e-5 else None
    spread = np.rad2deg(np.sqrt(-2 * np.log(max(1e-12, coherence)))) / 2
    return {
        'mean_axis': _rounded(axis_mean),
        'coherence': _rounded(coherence),
        'spread_deg': _rounded(spread),
        'valid_pixels': int(axis.size),
        'weighting': weight_kind,
        'weight_sum': _rounded(mass),
    }


def _histogram(values, bins, extent):
    if extent is None and values.size:
        low, high = float(values.min()), float(values.max())
        if high <= low:
            pad = max(1e-4, abs(low) * 0.02)
            low, high = low - pad, high + pad
        extent = low, high
    elif extent is None:
        extent = 0.0, 1.0
    counts, edges = np.histogram(values, bins=bins, range=extent)
    return {
        'edges': [_rounded(edge) for edge in edges],
        'counts': counts.tolist(),
        'underflow': int(np.count_nonzero(values < extent[0])),
        'overflow': int(np.count_nonzero(values > extent[1])),
    }


def _polar_response(field, geometry, measured):
    if not measured.any():
        return None
    box = geometry.box.slices
    s0, q, u = [float(np.mean(layer[box][measured], dtype=np.float64)) for layer in (field.s0, field.q, field.u)]
    angles = np.linspace(0, 180, num=37)
    high = axial_mean(q, u)
    low = ((high + 90) % 180) if high is not None else None
    return {
        'angles': angles.tolist(),
        'signals': [_rounded(value) for value in analyzer_response(s0, q, u, angles)],
        'mean_intensity': _rounded(s0),
        'strongest_angle': _rounded(high),
        'weakest_angle': _rounded(low),
    }


def measure(results, record, request, view):
    raw = object_value(request, 'Region measurement')
    cfg = AnalysisRequest.parse(raw, view.layer if view.layer != 'quality' else 'intensity')
    height, width = results.field.shape
    geometry, name = select_region(record, raw, width, height)
    values, valid = results.channel(cfg.layer, view.analyzer_angle)
    local = geometry.local_mask()
    cut = geometry.box.slices
    selected = local & valid[cut]
    measured = local & results.field.measured[cut]
    oriented = local & results.field.oriented[cut]
    values_inside = values[cut][selected]
    extent = (0, 180) if cfg.layer == 'direction' else None
    if 'hist_low' in raw or 'hist_high' in raw:
        if cfg.layer != 'direction':
            lo, hi = number(raw.get('hist_low'), 'Histogram lower limit', -8, 8), number(raw.get('hist_high'), 'Histogram upper limit', -8, 8)
            from .domain import InputError
            if lo >= hi:
                raise InputError('Histogram lower limit must be smaller than upper limit.')
            extent = lo, hi
    flags = results.quality.flags[cut][local]
    return {
        'region': name,
        'geometry': geometry.as_dict(),
        'layer': cfg.layer,
        'pixels': int(local.sum()),
        'layer_usable': int(selected.sum()),
        'measurement_usable': int(measured.sum()),
        'orientation_usable': int(oriented.sum()),
        'linear': None if cfg.layer == 'direction' else _linear_statistics(values_inside, cfg.percentiles),
        'direction': _axial_statistics(results.field, geometry, oriented, cfg.weighting),
        'histogram': _histogram(values_inside, cfg.bins, extent),
        'issue_counts': {issue.key: int(np.count_nonzero(flags & issue.bit)) for issue in ISSUES},
        'mean_analyzer_response': _polar_response(results.field, geometry, measured),
    }


def histogram_csv(result):
    hist = result['histogram']
    return csv_text(('layer', 'edge_low', 'edge_high', 'count'), (
        (result['layer'], hist['edges'][i], hist['edges'][i + 1], count)
        for i, count in enumerate(hist['counts'])
    ))


def region_table(results, record, view, request):
    data = object_value(request, 'Region table')
    names = data.get('region_ids')
    if names is not None:
        from .domain import list_value
        names = list_value(names, 'Region table selection', 24)
    else:
        names = [region['id'] for region in record['regions']]
    if not names:
        names = [None]
    return [measure(results, record, dict(data, region_id=region_id), view) for region_id in names]


def table_csv(rows):
    fields = ('region', 'layer', 'pixels', 'usable', 'mean', 'median', 'mean_axis', 'coherence', 'dark', 'cropped', 'mask')
    return csv_text(fields, (
        (r['region'], r['layer'], r['pixels'], r['layer_usable'], (r['linear'] or {}).get('mean'),
         (r['linear'] or {}).get('percentiles', {}).get('p50'), (r['direction'] or {}).get('mean_axis'),
         (r['direction'] or {}).get('coherence'), r['issue_counts']['dark'], r['issue_counts']['outside'], r['issue_counts']['mask'])
        for r in rows
    ))
