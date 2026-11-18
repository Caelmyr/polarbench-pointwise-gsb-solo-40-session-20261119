from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import ANGLES, number, object_value


@dataclass(frozen=True)
class Limits:
    min_intensity: float = 0.045
    max_pair_error: float = 0.12
    saturation: float = 0.995
    min_direction: float = 0.045
    max_degree: float = 1.04
    mask_threshold: int = 1

    @classmethod
    def parse(cls, raw):
        from .domain import integer
        data = object_value(raw, 'Quality limits')
        return cls(
            min_intensity=number(data.get('min_intensity', cls.min_intensity), 'Darkness threshold', 0.0001, 4),
            max_pair_error=number(data.get('max_pair_error', cls.max_pair_error), 'Pair discrepancy', 0.001, 1),
            saturation=number(data.get('saturation', cls.saturation), 'Raw clipping threshold', 0.4, 1),
            min_direction=number(data.get('min_direction', cls.min_direction), 'Direction confidence cutoff', 0.001, 1),
            max_degree=number(data.get('max_degree', cls.max_degree), 'Physical degree tolerance', 1, 1.3),
            mask_threshold=integer(data.get('mask_threshold', cls.mask_threshold), 'Mask threshold', 1, 255),
        )

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class PolarField:
    s0: np.ndarray
    q: np.ndarray
    u: np.ndarray
    polarized: np.ndarray
    degree: np.ndarray
    direction: np.ndarray
    pair_error: np.ndarray
    pair_difference: np.ndarray
    base_support: np.ndarray
    measured: np.ndarray
    oriented: np.ndarray

    @property
    def shape(self):
        return self.s0.shape

    def channel(self, layer, *, azimuth=0):
        if layer == 'intensity':
            return self.s0, self.measured
        if layer == 'q':
            return self.q, self.measured
        if layer == 'u':
            return self.u, self.measured
        if layer == 'polarized':
            return self.polarized, self.measured
        if layer == 'degree':
            return self.degree, self.measured
        if layer == 'direction':
            return self.direction, self.oriented
        if layer == 'pair-error':
            return self.pair_error, self.base_support
        if layer == 'pair-difference':
            return self.pair_difference, self.base_support
        if layer == 'analyzer':
            return virtual_analyzer(self, azimuth), self.measured
        if layer == 'glare-min':
            return analyzer_extremum(self, -1), self.measured
        if layer == 'glare-max':
            return analyzer_extremum(self, 1), self.measured
        raise KeyError(layer)


def intersection_support(aligned, common, detector_mask):
    shape = aligned[0].data.values.shape
    geometry = np.zeros(shape, dtype=bool)
    geometry[common.slices] = True
    native = np.logical_and.reduce([aligned[angle].data.native_valid for angle in ANGLES])
    reference = np.logical_and.reduce([aligned[angle].data.reference_valid for angle in ANGLES])
    saturation = np.logical_or.reduce([aligned[angle].data.clipped for angle in ANGLES])
    mask = detector_mask if detector_mask is not None else np.ones(shape, dtype=bool)
    return geometry, native, reference, saturation, mask


def reconstruct(aligned, common, detector_mask, limits: Limits, zero_angle: float):
    source = {a: aligned[a].data.values for a in ANGLES}
    pair_a = source[0] + source[90]
    pair_b = source[45] + source[135]
    s0 = (pair_a + pair_b) * np.float32(0.5)
    q, u = source[0] - source[90], source[45] - source[135]
    pair_difference = pair_a - pair_b
    ratio = np.zeros_like(s0)
    np.divide(np.abs(pair_difference), np.maximum(s0, 0.0001), out=ratio)
    polarized = np.hypot(q, u)
    degree = np.zeros_like(s0)
    np.divide(polarized, s0, where=s0 > 0, out=degree)
    if zero_angle:
        # Coordinates increase to the right and down. Rotate the frame, not the raw image.
        q, u = rotate_components(q, u, zero_angle)
    direction = np.mod(np.arctan2(u, q) * np.float32(90 / np.pi), 180)
    geometry, native, reference, saturation, mask = intersection_support(aligned, common, detector_mask)
    base = geometry & native & reference & mask & ~saturation
    measured = base & (s0 >= limits.min_intensity) & (ratio <= limits.max_pair_error) & (degree <= limits.max_degree)
    oriented = measured & (degree >= limits.min_direction)
    return PolarField(s0, q, u, polarized, degree, direction, ratio, pair_difference, base, measured, oriented)


def virtual_analyzer(field, angle):
    radians = np.deg2rad(2 * angle)
    return np.maximum(0, 0.5 * (field.s0 + field.q * np.cos(radians) + field.u * np.sin(radians)))


def analyzer_extremum(field, sign):
    return np.maximum(0, 0.5 * (field.s0 + sign * field.polarized))


def polarity_difference(direction_a, direction_b):
    return (direction_a - direction_b + 90) % 180 - 90


def rotate_components(q, u, azimuth):
    radians = np.deg2rad(2 * azimuth)
    c, s = np.float32(np.cos(radians)), np.float32(np.sin(radians))
    return q * c + u * s, -q * s + u * c


def angle_display(value):
    if value is None:
        return None
    angle = float(value) % 180
    return 0.0 if angle > 179.999999 else angle


def axial_mean(q, u, fallback=None):
    if np.hypot(q, u) <= 1e-9:
        return fallback
    return angle_display(np.arctan2(u, q) * 90 / np.pi)


def analyzer_response(s0, q, u, angle):
    radii = np.deg2rad(2 * np.asarray(angle, dtype=np.float64))
    return np.maximum(0, 0.5 * (s0 + q * np.cos(radii) + u * np.sin(radii)))
