from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import rgba_png
from .domain import Box, InputError, enum_value, number, object_value
from .render import apply_background, map_colors
from .settings import LAYERS
from .stokes import polarity_difference

MODES = ('intensity', 'degree', 'direction', 'polarized', 'pair-error')


@dataclass(frozen=True)
class CompareRequest:
    layer: str
    box: Box
    alignment_axis: str
    threshold: float
    limit: float
    map_style: str

    @classmethod
    def parse(cls, raw, width, height):
        data = object_value(raw, 'Capture comparison')
        layer = enum_value(data.get('layer', 'degree'), MODES, 'Comparison layer')
        direction = layer == 'direction'
        max_delta = 90 if direction else 8
        view_default = 30 if direction else 0.35
        scope = Box.parse(data['box'], width, height) if data.get('box') else Box.entire(width, height)
        return cls(
            layer=layer,
            box=scope,
            alignment_axis=enum_value(data.get('axis', 'detector'), ('detector', 'selected-reference'), 'Comparison direction reference'),
            threshold=number(data.get('threshold', 10 if direction else 0.1), 'Change threshold', 0, max_delta),
            limit=number(data.get('limit', view_default), 'Display difference', 0.0001, max_delta),
            map_style=enum_value(data.get('map_style', 'difference'), ('difference', 'support', 'exceeds-threshold'), 'Comparison map'),
        )


@dataclass(frozen=True)
class Compared:
    config: CompareRequest
    baseline_name: str
    current_name: str
    baseline_id: str
    current_id: str
    delta: np.ndarray
    reference_valid: np.ndarray
    current_valid: np.ndarray

    @property
    def shared(self):
        return self.reference_valid & self.current_valid

    def summary(self):
        shared = self.shared
        a, b = self.reference_valid, self.current_valid
        differences = self.delta[shared]
        if differences.size:
            absolute = np.abs(differences)
            stats = {
                'signed_mean': float(np.mean(differences, dtype=np.float64)),
                'absolute_mean': float(np.mean(absolute, dtype=np.float64)),
                'absolute_p95': float(np.percentile(absolute, 95)),
                'absolute_max': float(np.max(absolute)),
                'above_threshold': int(np.count_nonzero(absolute >= self.config.threshold)),
            }
        else:
            stats = None
        return {
            'layer': self.config.layer,
            'unit': LAYERS[self.config.layer].unit,
            'current': {'id': self.current_id, 'name': self.current_name},
            'reference': {'id': self.baseline_id, 'name': self.baseline_name},
            'axis': self.config.alignment_axis,
            'scope': self.config.box.as_dict(),
            'total_pixels': int(shared.size),
            'shared_pixels': int(shared.sum()),
            'only_current': int(np.count_nonzero(b & ~a)),
            'only_reference': int(np.count_nonzero(a & ~b)),
            'excluded_both': int(np.count_nonzero(~a & ~b)),
            'statistics': stats,
            'threshold': self.config.threshold,
        }


SUPPORT_COLORS = {
    'both': (70, 174, 138),
    'current': (123, 160, 241),
    'reference': (237, 159, 91),
    'none': (37, 45, 60),
}


def compare_fields(current, reference, current_record, reference_record, raw):
    if current.field.shape != reference.field.shape:
        raise InputError('A comparison needs equal detector dimensions. Align or crop the sources externally before importing.')
    height, width = current.field.shape
    cfg = CompareRequest.parse(raw, width, height)
    cut = cfg.box.slices
    now, support_now = current.channel(cfg.layer)
    before, support_before = reference.channel(cfg.layer)
    now, before = now[cut], before[cut]
    if cfg.layer == 'direction':
        if cfg.alignment_axis == 'detector':
            now = now + current.processing.zero_angle
            before = before + reference.processing.zero_angle
        delta = polarity_difference(now, before)
    else:
        delta = now - before
    return Compared(
        cfg,
        reference_record['name'], current_record['name'], reference_record['id'], current_record['id'],
        delta, support_before[cut], support_now[cut],
    )


def comparison_png(result):
    cfg = result.config
    a, b = result.reference_valid, result.current_valid
    shared = a & b
    if cfg.map_style == 'support':
        rgb = np.empty(shared.shape + (3,), dtype=np.uint8)
        rgb[...] = SUPPORT_COLORS['none']
        rgb[a & ~b] = SUPPORT_COLORS['reference']
        rgb[b & ~a] = SUPPORT_COLORS['current']
        rgb[shared] = SUPPORT_COLORS['both']
    elif cfg.map_style == 'exceeds-threshold':
        rgb = np.full(shared.shape + (3,), 44, dtype=np.uint8)
        rgb[shared & (np.abs(result.delta) >= cfg.threshold)] = (243, 175, 104)
    else:
        rgb = map_colors(result.delta, -cfg.limit, cfg.limit, 'balance')
    if cfg.map_style != 'support':
        rgb = apply_background(rgb, shared, cfg.box, 'transparent')
    return rgba_png(rgb)


def compare_legend(result):
    cfg = result.config
    if cfg.map_style == 'support':
        return {'type': 'category', 'entries': [
            {'label': 'Both usable', 'color': SUPPORT_COLORS['both']},
            {'label': 'Current only', 'color': SUPPORT_COLORS['current']},
            {'label': 'Reference only', 'color': SUPPORT_COLORS['reference']},
            {'label': 'Neither usable', 'color': SUPPORT_COLORS['none']},
        ]}
    if cfg.map_style == 'exceeds-threshold':
        return {'type': 'category', 'entries': [
            {'label': 'Within threshold', 'color': [44, 44, 44]},
            {'label': 'Meets change threshold', 'color': [243, 175, 104]},
        ]}
    signals = np.linspace(-cfg.limit, cfg.limit, 9)
    return {
        'type': 'gradient', 'values': signals.tolist(),
        'colors': map_colors(signals, -cfg.limit, cfg.limit, 'balance').tolist(),
        'unit': LAYERS[cfg.layer].unit,
        'min': -cfg.limit,
        'max': cfg.limit,
    }
