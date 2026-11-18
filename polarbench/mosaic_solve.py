"""Infer the physical analyzer-angle order of a 2 x 2 micro-polarizer mosaic.

The camera only delivers the packed sensor image and the four tile angles are
unknown.  Two fixed-camera reference frames are available, each with a known
linear-polarization direction relative to the horizontal edge of the photo but
captured with different exposure settings, and some pixels carry native
occlusion (alpha coverage).

The estimator never compares brightness across the two references and never
compares different tiles at different pixel positions:

* every tile is summarized by the median over the detector positions that are
  valid in *all four* tiles of the same reference frame (a robust,
  position-matched statistic; occluded samples are dropped);
* within each reference frame a Stokes-like contrast vector is built from
  geometrically opposite tile positions.  Exposure, black level and gain
  change the vector's length, never its direction, so its doubled angle can be
  matched to the known polarization direction without any cross-frame
  intensity comparison;
* all 24 assignments of {0, 45, 90, 135} to the four tile positions are
  scored.  When the data cannot separate the leading candidates the result is
  reported as uncertain instead of forcing an order on the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .codec import ImagePlane, raw_fraction, read_image
from .domain import ANGLES, Box, InputError, number, object_value

# A tile needs a minimum number of common, covered detector positions before
# its median can represent the frame.
MIN_COMMON_PIXELS = 32
# Modulation (|(Q, U)| relative to the mean level) below this is dominated by
# noise and cannot identify an angle order.
MIN_MODULATION = 0.02
# Minimum separation (degrees on the doubled-angle circle) between the two
# known directions.  Parallel or perpendicular references constrain the same
# doubled angle and cannot separate the remaining permutations.
MIN_DIRECTION_SEPARATION = 4.0
# Angular mean residual (degrees, doubled-angle circle) at or below which a
# permutation is considered to explain the references.
FIT_ANGLE_TOLERANCE = 8.0
# The winning permutation must separate itself from the runner-up by at least
# this many degrees of angular residual.
DECISION_MARGIN = 6.0


@dataclass(frozen=True)
class ReferenceObservation:
    direction: float
    medians: tuple[float, float, float, float]
    modulation: float
    common_pixels: int
    informative: bool
    reason: str | None


@dataclass(frozen=True)
class OrderCandidate:
    order: tuple[int, int, int, int]
    key: str
    residuals: tuple[float | None, ...]
    angular_mean: float


def order_key(order) -> str:
    return '-'.join(str(angle) for angle in order)


def wrapped_angle(degrees: float) -> float:
    """Map an angle to the half-open (-180, 180] interval."""
    return (degrees + 180.0) % 360.0 - 180.0


def _tile_planes(raw: ImagePlane):
    if raw.width % 2 or raw.height % 2:
        raise InputError('A micro-polarizer mosaic requires even width and height with complete 2 × 2 cells.')
    return [
        ImagePlane(raw.values[row::2, col::2], raw.valid[row::2, col::2],
                   raw.bit_depth, raw.format, raw.original_mode, raw.sha256)
        for row in (0, 1)
        for col in (0, 1)
    ]


def observe_reference(raw: ImagePlane, direction: float, window: Box | None) -> ReferenceObservation:
    """Reduce one reference frame to its exposure-independent tile medians."""
    tiles = _tile_planes(raw)
    height, width = tiles[0].shape
    if window is None:
        window = Box.entire(width, height)
    section = window.slices
    support = np.logical_and.reduce([tile.valid[section] for tile in tiles])
    common_pixels = int(support.sum())
    if common_pixels < MIN_COMMON_PIXELS:
        raise InputError(
            f'Only {common_pixels} detector positions are covered in all four tiles; '
            f'at least {MIN_COMMON_PIXELS} common positions are needed to infer the tile order.'
        )
    fractions = [raw_fraction(tile)[section][support] for tile in tiles]
    medians = tuple(float(np.median(values)) for values in fractions)
    # Geometrically opposite tile positions form the modulation contrast pair.
    contrast = np.hypot(medians[0] - medians[2], medians[1] - medians[3])
    modulation = float(contrast / max(float(np.mean(medians)), 1e-12))
    informative = modulation >= MIN_MODULATION
    reason = None if informative else (
        f'Polarization modulation is only {modulation:.4f} of the mean signal; '
        f'a frame needs at least {MIN_MODULATION:.3f} to reveal an angle order.'
    )
    return ReferenceObservation(float(direction) % 180.0, medians, modulation,
                                common_pixels, informative, reason)


def _candidate_residual(order, medians, direction):
    """Doubled-angle residual (degrees) of one frame under one tile order.

    Position ``p`` is assumed to carry analyzer ``order[p]``.  With Q = I(0) -
    I(90) and U = I(45) - I(135), linearly polarized light at direction theta
    predicts (Q, U) pointing at 2*theta; the frame exposure only scales the
    vector, so this is a pure angle difference and never crosses frames.
    """
    position = {angle: index for index, angle in enumerate(order)}
    measured_q = medians[position[0]] - medians[position[90]]
    measured_u = medians[position[45]] - medians[position[135]]
    measured_angle = np.rad2deg(np.arctan2(measured_u, measured_q))
    return abs(wrapped_angle(measured_angle - 2.0 * direction))


def rank_orders(observations) -> list[OrderCandidate]:
    ranked = []
    for order in permutations(ANGLES):
        residuals = tuple(
            None if not item.informative
            else _candidate_residual(order, item.medians, item.direction)
            for item in observations
        )
        scored = [value for value in residuals if value is not None]
        if scored:
            ranked.append(OrderCandidate(tuple(order), order_key(order), residuals, float(np.mean(scored))))
    ranked.sort(key=lambda item: item.angular_mean)
    return ranked


def _direction_redundancy(observations):
    """Return a reason when the two informative frames share one angle constraint."""
    informative = [item for item in observations if item.informative]
    if len(informative) != 2:
        return None
    axial_gap = abs(wrapped_angle(informative[0].direction - informative[1].direction))
    axial_gap = min(axial_gap, 180.0 - axial_gap)
    # On the doubled-angle circle (period 180 degrees) an axial gap of 0
    # degrees (parallel) or 90 degrees (perpendicular) gives the same line.
    if 2.0 * min(axial_gap, abs(90.0 - axial_gap)) < MIN_DIRECTION_SEPARATION:
        return (
            f'The two reference directions differ by {axial_gap:.1f}° (axial). '
            'Parallel or perpendicular references constrain the same doubled angle and '
            'cannot distinguish the remaining tile permutations.'
        )
    return None


def solve_mosaic(reference_files, directions, window: Box | None = None):
    """Infer the tile order from two reference frames with known directions."""
    if len(reference_files) != 2 or len(directions) != 2:
        raise InputError('Provide exactly two reference mosaic images and their polarization directions.')
    observations = []
    sensor_shapes = set()
    for index, (file, raw_direction) in enumerate(zip(reference_files, directions), start=1):
        direction = number(raw_direction, f'Reference {index} polarization direction', 0, 180)
        payload = file[0] if isinstance(file, tuple) else file
        raw = read_image(payload)
        sensor_shapes.add(raw.shape)
        observations.append(observe_reference(raw, direction, window))
    if len(sensor_shapes) != 1:
        raise InputError('Both reference frames must come from the same sensor and share their mosaic dimensions.')
    if not any(item.informative for item in observations):
        raise InputError(
            'Neither reference frame shows measurable polarization modulation. '
            'Use references containing linearly polarized light so the tile order can be identified.'
        )
    ranked = rank_orders(observations)
    best, runner_up = ranked[0], ranked[1]
    margin = runner_up.angular_mean - best.angular_mean
    redundancy_reason = _direction_redundancy(observations)
    single_frame = sum(item.informative for item in observations) == 1
    if single_frame and redundancy_reason is None:
        weak = next((item for item in observations if not item.informative), None)
        detail = f' The other reference is too weak: {weak.reason}' if weak is not None else ''
        redundancy_reason = (
            'Only one reference frame shows measurable polarization. A single polarized frame '
            'constrains the doubled angle alone and cannot separate the remaining tile permutations.'
            + detail
        )
    independent = not single_frame and redundancy_reason is None
    confident = (
        best.angular_mean <= FIT_ANGLE_TOLERANCE
        and margin >= DECISION_MARGIN
        and independent
    )
    alternatives = [
        item for item in ranked[1:]
        if item.angular_mean <= FIT_ANGLE_TOLERANCE
        or item.angular_mean - best.angular_mean < DECISION_MARGIN
        or (not confident and item.angular_mean - best.angular_mean <= 1.0)
    ]
    return {
        'confident': confident,
        'status': 'confident' if confident else 'uncertain',
        'inferred_order': list(best.order),
        'inferred_layout': best.key,
        'best_angular_residual': best.angular_mean,
        'decision_margin': margin,
        'independent_directions': independent,
        'uncertainty_reason': redundancy_reason,
        'fit_tolerance': FIT_ANGLE_TOLERANCE,
        'decision_margin_required': DECISION_MARGIN,
        'min_modulation': MIN_MODULATION,
        'tiles': [
            {'position': index, 'row': index // 2, 'column': index % 2, 'angle': angle}
            for index, angle in enumerate(best.order)
        ],
        'candidates': [
            {
                'layout': item.key,
                'order': list(item.order),
                'angular_residual': item.angular_mean,
                'residuals': [None if value is None else float(value) for value in item.residuals],
                'within_tolerance': item.angular_mean <= FIT_ANGLE_TOLERANCE,
            }
            for item in ranked[:6]
        ],
        'alternatives': [
            {'layout': item.key, 'order': list(item.order), 'angular_residual': item.angular_mean}
            for item in alternatives
        ],
        'references': [
            {
                'direction': item.direction,
                'common_pixels': item.common_pixels,
                'medians': list(item.medians),
                'modulation': item.modulation,
                'informative': item.informative,
                'reason': item.reason,
            }
            for item in observations
        ],
    }


def solve_from_request(form, files):
    """Parse a multipart solver request into the inference payload."""
    payload = object_value(form.get('payload'), 'Mosaic calibration request') if form.get('payload') else {}

    def field(name):
        return payload.get(name, form.get(name))

    directions = [
        number(field('direction_1'), 'Reference 1 polarization direction', 0, 180),
        number(field('direction_2'), 'Reference 2 polarization direction', 0, 180),
    ]
    reference_files = []
    for index in (1, 2):
        upload = files.get(f'reference_{index}')
        if upload is None or not getattr(upload, 'filename', ''):
            raise InputError(f'Reference mosaic {index} image is required.')
        data = upload.stream.read()
        if not data:
            raise InputError(f'Reference mosaic {index} image is empty.')
        reference_files.append((data, upload.filename))
    window = None
    if payload.get('box') is not None:
        first = read_image(reference_files[0][0])
        window = Box.parse(payload['box'], first.width // 2, first.height // 2)
    return solve_mosaic(reference_files, directions, window)
