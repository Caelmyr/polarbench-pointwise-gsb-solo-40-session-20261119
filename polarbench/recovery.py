from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import calibration, ingest, registration
from .calibration import CalibratedPlane
from .domain import ANGLES, InputError, bool_value, number, object_value
from .registration import AlignedPlane

SOURCE_MAIN = 0
SOURCE_SHORT = 1
SOURCE_MIXED = 2
SOURCE_LABELS = {SOURCE_MAIN: 'main', SOURCE_SHORT: 'short', SOURCE_MIXED: 'both'}


@dataclass(frozen=True)
class Recovery:
    enabled: bool
    short_id: str | None
    exposure_ratio: float
    min_short: float
    agreement: float

    @classmethod
    def parse(cls, raw):
        data = object_value({} if raw is None else raw, 'Short exposure recovery')
        enabled = bool_value(data.get('enabled', False), 'Recovery enabled flag')
        short_id = data.get('short_capture_id')
        if short_id is not None and (not isinstance(short_id, str) or not short_id.strip()):
            raise InputError('Choose a same-size short exposure capture for recovery.')
        return cls(
            enabled,
            short_id.strip() if isinstance(short_id, str) else None,
            number(data.get('exposure_ratio', 4), 'Normal / short exposure ratio', 1.0001, 100000),
            number(data.get('min_short', 0.02), 'Minimum reliable short exposure signal', 0.001, 0.5),
            number(data.get('agreement', 0.2), 'Short/main agreement tolerance', 0.001, 1),
        )

    def as_dict(self):
        return {
            'enabled': self.enabled,
            'short_capture_id': self.short_id,
            'exposure_ratio': self.exposure_ratio,
            'min_short': self.min_short,
            'agreement': self.agreement,
        }


@dataclass(frozen=True)
class RecoveryReport:
    config: Recovery
    short_name: str
    merged: dict[int, AlignedPlane]
    common: object
    provenance: dict[int, np.ndarray]
    conflicts: np.ndarray
    filled: np.ndarray
    recovered: np.ndarray
    angle_counts: dict[str, dict[str, int]]

    def angle_source(self, angle, y, x):
        return SOURCE_LABELS[int(self.provenance[angle][y, x])]

    def summary(self):
        return {
            'enabled': True,
            'short_capture_id': self.config.short_id,
            'short_capture_name': self.short_name,
            'exposure_ratio': self.config.exposure_ratio,
            'min_short': self.config.min_short,
            'agreement_tolerance': self.config.agreement,
            'recovered_pixels': int(self.recovered.sum()),
            'short_filled_pixels': int(self.filled.sum()),
            'agreement_conflict_pixels': int(self.conflicts.sum()),
            'angle_sources': self.angle_counts,
        }


def merge_recovery(decoded, main_aligned, main_common, refs, config: Recovery, limits,
                   tuning, registration_config, store, short_record):
    """Merge a same-canvas short exposure capture into the aligned main planes.

    Main-only usable samples are always preserved. Short exposure samples are scaled
    to the main exposure and accepted only where they are reliable and agree with any
    usable main sample. A pixel is recovered only when all four angles end with a
    trusted measurement and no angle shows a short/main disagreement.
    """
    short_decoded = ingest.decode(store, short_record, limits.mask_threshold)
    if short_decoded.shape != decoded.shape:
        raise InputError('The short exposure capture must have the same detector canvas size as the opened capture.')
    short_prepared = {
        angle: calibration.prepare_channel(short_decoded.planes[angle], tuning[angle], refs, limits.saturation)
        for angle in ANGLES
    }
    short_aligned = registration.align_planes(short_prepared, registration_config)[0]

    shape = decoded.shape
    provenance, merged_planes = {}, {}
    conflict = np.zeros(shape, dtype=bool)
    filled = np.zeros(shape, dtype=bool)
    angle_counts = {}

    for angle in ANGLES:
        main = main_aligned[angle].data
        short = short_aligned[angle].data
        main_ok = main.native_valid & main.reference_valid & ~main.clipped
        short_scaled = short.values * np.float32(config.exposure_ratio)
        short_ok = short.native_valid & short.reference_valid & ~short.clipped & (short.values >= config.min_short)

        both = main_ok & short_ok
        scale = np.maximum(main.values, np.float32(0.0001))
        angle_conflict = both & (np.abs(short_scaled - main.values) > config.agreement * scale)
        # A disagreement at one angle rejects the short exposure wherever the merge
        # relies on it for this pixel; the main sample itself stays untouched.
        conflict |= angle_conflict
        use_short = short_ok & ~angle_conflict & ~main_ok
        filled |= use_short

        values = np.where(main_ok, main.values, np.where(use_short, short_scaled, main.values)).astype(np.float32)
        native_valid = main.native_valid | (short.native_valid & use_short)
        reference_valid = main.reference_valid | (short.reference_valid & use_short)
        clipped = main.clipped & ~use_short
        below_black = main.below_black & ~use_short
        merged_planes[angle] = AlignedPlane(
            CalibratedPlane(values, native_valid, reference_valid, clipped, below_black),
            main_aligned[angle].footprint,
        )

        source = np.full(shape, SOURCE_MAIN, dtype=np.uint8)
        source[main_ok & short_ok] = SOURCE_MIXED
        source[use_short] = SOURCE_SHORT
        provenance[angle] = source
        angle_counts[str(angle)] = {
            'main_measured': int(main_ok.sum()),
            'short_measured': int(short_ok.sum()),
            'short_filled': int(use_short.sum()),
            'agreement_conflicts': int(angle_conflict.sum()),
            'short_clipped': int(short.clipped.sum()),
        }

    trusted = np.logical_and.reduce([merged_planes[a].valid for a in ANGLES])
    recovered = trusted & ~conflict
    return RecoveryReport(
        config,
        short_record['name'],
        merged_planes,
        main_common,
        provenance,
        conflict,
        filled,
        recovered,
        angle_counts,
    )
