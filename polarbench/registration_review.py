from __future__ import annotations

import dataclasses

import numpy as np

from .codec import rgba_png
from .domain import ANGLES, Box, enum_value, integer, number, object_value
from .registration import _texture
from .render import map_colors

_VIEWS = ('overlay', 'checkerboard', 'edge-residual', 'coverage')


@dataclasses.dataclass(frozen=True)
class Evaluation:
    angle: int
    bounds: Box
    mode: str
    thickness: int
    low: float
    high: float
    reference: np.ndarray
    moving: np.ndarray
    reference_valid: np.ndarray
    moving_valid: np.ndarray

    @property
    def common(self):
        return self.reference_valid & self.moving_valid



def registration_view(results, raw):
    data = object_value(raw, 'Registration audit')
    angle = enum_value(integer(data.get('angle', 45), 'Moving analyzer', 0, 135), ANGLES[1:], 'Moving analyzer')
    view = enum_value(data.get('mode', 'overlay'), _VIEWS, 'Registration evidence map')
    cutoff = number(data.get('level', 0.8), 'Full-scale viewing level', 0.05, 6)
    width, height = results.decoded.width, results.decoded.height
    bounds = Box.parse(data['box'], width, height) if data.get('box') else Box.entire(width, height)
    a, b = results.aligned[0], results.aligned[angle]
    valid_a, valid_b = a.valid[bounds.slices], b.valid[bounds.slices]
    pix_a, pix_b = a.data.values[bounds.slices], b.data.values[bounds.slices]
    return Evaluation(angle, bounds, view, integer(data.get('tile', 30), 'Checkerboard tile size', 4, 240),
                      0.0, cutoff, pix_a, pix_b, valid_a, valid_b)


def _estimate_edges(evaluation):
    if min(evaluation.reference.shape) < 3:
        return None
    ag, am = _texture(evaluation.reference, evaluation.reference_valid)
    bg, bm = _texture(evaluation.moving, evaluation.moving_valid)
    scope = am & bm
    return ag, bg, scope


def _quality_score(evaluation):
    edge = _estimate_edges(evaluation)
    if edge is None or not edge[2].any():
        return {'edge_samples': 0, 'correlation': None, 'mean_absolute_normalized_residual': None}
    a, b, keep = edge
    xa, xb = (field[keep].astype(np.float64) for field in (a, b))
    da, db = xa - xa.mean(), xb - xb.mean()
    norm = np.linalg.norm(da) * np.linalg.norm(db)
    correlation = float(np.dot(da, db) / norm) if norm > 1e-10 else None
    scale_a, scale_b = np.percentile(xa, 94), np.percentile(xb, 94)
    residual = None
    if scale_a > 1e-6 and scale_b > 1e-6:
        normalized_a = np.clip(xa / scale_a, 0, 3)
        normalized_b = np.clip(xb / scale_b, 0, 3)
        residual = float(np.abs(normalized_a - normalized_b).mean())
    return {'edge_samples': int(xa.size), 'correlation': correlation, 'mean_absolute_normalized_residual': residual}


def summarize_evaluation(results, evaluation):
    a, b = evaluation.reference_valid, evaluation.moving_valid
    shared = a & b
    counts = {
        'common': int(shared.sum()),
        'reference_only': int(np.count_nonzero(a & ~b)),
        'moving_only': int(np.count_nonzero(b & ~a)),
        'neither': int(np.count_nonzero(~a & ~b)),
    }
    signed = evaluation.moving[shared] - evaluation.reference[shared]
    return {
        'angle': evaluation.angle,
        'shift': results.processing.registration.offsets[evaluation.angle].as_dict(),
        'window': evaluation.bounds.as_dict(),
        'pixels': int(shared.size),
        'support': counts,
        'mean_analyzer_signal_difference': float(np.mean(signed, dtype=np.float64)) if signed.size else None,
        'texture_agreement': _quality_score(evaluation),
    }


def _coverage_image(evaluation):
    a, b = evaluation.reference_valid, evaluation.moving_valid
    pix = np.zeros(a.shape + (3,), dtype=np.uint8)
    pix[...] = (27, 36, 51)
    pix[a & ~b] = (222, 107, 163)
    pix[b & ~a] = (86, 188, 201)
    pix[a & b] = (74, 173, 128)
    return pix


def _overlay(evaluation):
    scale = evaluation.high - evaluation.low
    a, b = (np.rint(np.clip((v - evaluation.low) / scale, 0, 1) * 255).astype(np.uint8)
            for v in (evaluation.reference, evaluation.moving))
    a = np.where(evaluation.reference_valid, a, 0)
    b = np.where(evaluation.moving_valid, b, 0)
    return np.stack((a, b, np.maximum(a, b)), axis=-1)


def _edge_overlay(evaluation):
    evidence = _estimate_edges(evaluation)
    if evidence is None:
        return _coverage_image(evaluation)
    a, b, keep = evidence
    cut = (a[keep], b[keep])
    if not keep.any():
        return _coverage_image(evaluation)
    norm_a, norm_b = (max(1e-4, float(np.percentile(item, 94))) for item in cut)
    difference = (b / norm_b - a / norm_a)
    output = map_colors(difference, -1, 1, 'balance')
    output[~keep] = (27, 36, 51)
    return output


def _checkerboard(evaluation):
    box = evaluation.bounds
    dy, dx = np.ogrid[box.y:box.bottom, box.x:box.right]
    use_reference = ((dx // evaluation.thickness) + (dy // evaluation.thickness)) % 2 == 0
    display = np.where(use_reference, evaluation.reference, evaluation.moving)
    support = np.where(use_reference, evaluation.reference_valid, evaluation.moving_valid)
    pix = map_colors(display, evaluation.low, evaluation.high, 'gray')
    pix[~support] = (27, 36, 51)
    return pix


def evaluation_image(evaluation):
    render = {'coverage': _coverage_image, 'overlay': _overlay, 'edge-residual': _edge_overlay, 'checkerboard': _checkerboard}
    return rgba_png(render[evaluation.mode](evaluation))
