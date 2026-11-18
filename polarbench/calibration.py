from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import raw_fraction
from .domain import ANGLES, Box, InputError, as_angle_dict, number, object_value, per_angle


@dataclass(frozen=True)
class ChannelTuning:
    black: float = 0.0
    gain: float = 1.0

    @classmethod
    def parse(cls, value, angle):
        state = object_value({} if value is None else value, f'{angle}° intensity settings')
        return cls(
            black=number(state.get('black', 0), f'{angle}° black level', 0, 0.25),
            gain=number(state.get('gain', 1), f'{angle}° gain', 0.25, 4),
        )

    def as_dict(self):
        return {'black': self.black, 'gain': self.gain}


@dataclass(frozen=True)
class Calibration:
    use_dark: bool
    use_flat: bool
    tuning: dict[int, ChannelTuning]

    @classmethod
    def parse(cls, value, references):
        state = object_value(value, 'Intensity calibration')
        use_dark, use_flat = state.get('use_dark', False), state.get('use_flat', False)
        if not isinstance(use_dark, bool) or not isinstance(use_flat, bool):
            raise InputError('Reference activation must be true or false.')
        if use_dark and not references.get('dark'):
            raise InputError('Upload a dark detector reference before enabling dark subtraction.')
        if use_flat and not references.get('flat'):
            raise InputError('Upload a flat detector reference before enabling shading correction.')
        return cls(use_dark, use_flat, per_angle(state.get('channels', {}), 'Intensity settings', ChannelTuning.parse))

    def as_dict(self):
        return {
            'use_dark': self.use_dark,
            'use_flat': self.use_flat,
            'channels': as_angle_dict({a: self.tuning[a].as_dict() for a in ANGLES}),
        }


@dataclass(frozen=True)
class ReferenceMaps:
    dark: np.ndarray | None
    scale: np.ndarray | None
    dark_valid: np.ndarray | None
    flat_valid: np.ndarray | None
    flat_anchor: float | None
    anchor_pixels: int | None

    def summary(self):
        return {
            'dark': self.dark is not None,
            'flat': self.scale is not None,
            'flat_reference_level': self.flat_anchor,
            'flat_reference_pixels': self.anchor_pixels,
            'unusable_dark_pixels': None if self.dark_valid is None else int(self.dark_valid.size - self.dark_valid.sum()),
            'unusable_flat_pixels': None if self.flat_valid is None else int(self.flat_valid.size - self.flat_valid.sum()),
        }


@dataclass(frozen=True)
class CalibratedPlane:
    values: np.ndarray
    native_valid: np.ndarray
    reference_valid: np.ndarray
    clipped: np.ndarray
    below_black: np.ndarray

    @property
    def valid(self):
        return self.native_valid & self.reference_valid & ~self.clipped


def _flat_box(record, shape):
    height, width = shape
    ref = record.get('references', {}).get('flat', {})
    if ref.get('box'):
        return Box.parse(ref['box'], width, height)
    return Box.entire(width, height)


def reference_maps(decoded, record, setting: Calibration):
    dark, dark_valid = None, None
    if setting.use_dark:
        dark = raw_fraction(decoded.dark)
        dark_valid = decoded.dark.valid & (dark < 0.9)
    if not setting.use_flat:
        return ReferenceMaps(dark, None, dark_valid, None, None, None)
    flat = raw_fraction(decoded.flat)
    flat_valid = decoded.flat.valid & (flat < 0.995)
    if dark is not None:
        flat = flat - dark
        flat_valid &= dark_valid
    flat_valid &= flat > 0.025
    selection = _flat_box(record, flat.shape).slices
    anchor_samples = flat[selection][flat_valid[selection]]
    if anchor_samples.size < 16:
        raise InputError('Flat reference needs at least 16 usable pixels in the selected normalization box.')
    anchor = float(np.median(anchor_samples))
    gains = np.zeros_like(flat, dtype=np.float32)
    np.divide(anchor, flat, where=flat_valid, out=gains)
    flat_valid &= (gains >= 0.25) & (gains <= 4)
    gains[~flat_valid] = 1.0
    return ReferenceMaps(dark, gains, dark_valid, flat_valid, anchor, int(anchor_samples.size))


def prepare_channel(image, tuning: ChannelTuning, refs: ReferenceMaps, upper: float):
    raw = raw_fraction(image)
    clipped = raw >= upper
    value = raw - tuning.black
    reference_valid = np.ones(raw.shape, dtype=bool)
    if refs.dark is not None:
        value -= refs.dark
        reference_valid &= refs.dark_valid
    below_black = value < 0
    np.maximum(value, 0, out=value)
    if refs.scale is not None:
        value *= refs.scale
        reference_valid &= refs.flat_valid
    value *= tuning.gain
    return CalibratedPlane(value, image.valid, reference_valid, clipped, below_black)


def channel_evidence(source, prepared):
    raw = raw_fraction(source)
    selection = source.valid
    supported = prepared.valid
    reported = prepared.values[supported]
    distribution = None
    if reported.size:
        distribution = {
            'min': float(reported.min()),
            'median': float(np.median(reported)),
            'max': float(reported.max()),
        }
    return {
        'raw_min': float(np.min(raw[selection])) if selection.any() else None,
        'raw_max': float(np.max(raw[selection])) if selection.any() else None,
        'native_excluded': int(np.count_nonzero(~selection)),
        'reference_excluded': int(np.count_nonzero(selection & ~prepared.reference_valid)),
        'saturated': int(np.count_nonzero(selection & prepared.clipped)),
        'black_clamped': int(np.count_nonzero(selection & prepared.below_black)),
        'usable': int(np.count_nonzero(supported)),
        'corrected_distribution': distribution,
    }

