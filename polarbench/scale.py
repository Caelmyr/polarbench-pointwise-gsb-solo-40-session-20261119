from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import ImageFont

from .domain import InputError, enum_value, number, object_value
from .profiles import ScanLine
from .regions import select_region

UNITS = ('mm', 'cm', 'um', 'in')
UNIT_LABELS = {'mm': 'mm', 'cm': 'cm', 'um': 'µm', 'in': 'in'}

# A fitted scale is offered directly when every line converts within this fraction of its measured length.
READY_TOLERANCE = 0.02
# A line is treated as suspect only when it misses by more than this fraction.
OUTLIER_TOLERANCE = 0.05
# A suspect line is excluded only when its miss is at least this many times the second-worst residual.
OUTLIER_RATIO = 2.0
# The two-line fit must stay within this fraction of the three-line fit before one line is blamed.
PITCH_DRIFT_LIMIT = 0.15
# SVD singular values below this fraction of the largest mean the lines cannot separate X and Y.
DIRECTION_RATIO = 0.15
CALIBRATION_LINE_COUNT = 3


@dataclass(frozen=True)
class SpatialScale:
    unit: str
    pitch_x: float
    pitch_y: float
    basis: str
    witness: dict | None = None

    @classmethod
    def from_input(cls, raw, width, height):
        data = object_value(raw, 'Detector scale')
        unit = enum_value(data.get('unit', 'mm'), UNITS, 'Length unit')
        basis = enum_value(data.get('basis', 'pitch'), ('pitch', 'measured-line'), 'Scale method')
        if basis == 'pitch':
            x = number(data.get('pitch_x'), 'Detector pitch X', 1e-8, 1000)
            y = number(data.get('pitch_y', x), 'Detector pitch Y', 1e-8, 1000)
            return cls(unit, x, y, basis)
        line = ScanLine.parse(data.get('line'), width, height)
        real_length = number(data.get('length'), 'Known reference length', 1e-6, 1e6)
        physical_per_px = real_length / line.length
        if not 1e-8 <= physical_per_px <= 1000:
            raise InputError('Measured detector pitch falls outside the supported range.')
        return cls(unit, physical_per_px, physical_per_px, basis, {'line': line.as_dict(), 'length': real_length})

    @classmethod
    def from_record(cls, raw):
        if raw is None:
            return cls('px', 1.0, 1.0, 'detector')
        return cls(**raw)

    def as_dict(self):
        return dict(self.__dict__)

    @property
    def symbol(self):
        return UNIT_LABELS.get(self.unit, self.unit)

    def distance(self, dx, dy):
        return math.hypot(self.pitch_x * dx, self.pitch_y * dy)

    def area(self, pixels):
        return pixels * self.pitch_x * self.pitch_y


def update_scale(record, raw):
    request = object_value(raw, 'Image scale')
    cap = record['capture']
    data = SpatialScale.from_input(request, cap['width'], cap['height'])
    record['scale'] = data.as_dict()
    return record['scale']


def clear_scale(record):
    record['scale'] = None


def _calibration_entries(raw, width, height):
    body = object_value(raw, 'Line calibration')
    entries = body.get('lines')
    if not isinstance(entries, list) or len(entries) != CALIBRATION_LINE_COUNT:
        raise InputError(f'Line calibration needs exactly {CALIBRATION_LINE_COUNT} calibration lines.')
    unit = enum_value(body.get('unit', 'mm'), UNITS, 'Length unit')
    parsed = []
    for index, item in enumerate(entries, start=1):
        data = object_value(item, f'Calibration line {index}')
        line = ScanLine.parse(data.get('line'), width, height)
        measured = number(data.get('length'), f'Calibration line {index} measured length', 1e-6, 1e6)
        parsed.append((line, float(measured)))
    return unit, parsed


def _line_vectors(entries):
    vectors, measured = [], []
    for line, length in entries:
        vectors.append((abs(line.end.x - line.start.x), abs(line.end.y - line.start.y)))
        measured.append(length)
    return (np.asarray(vectors, dtype=np.float64),
            np.asarray(measured, dtype=np.float64))


def _residuals(vectors, measured, pitches):
    converted = np.hypot(vectors[:, 0] * pitches[0], vectors[:, 1] * pitches[1])
    return converted, (converted - measured) / measured


def _fit_pitches(vectors, measured, indexes=None):
    """Weighted nonlinear least squares for (pitch_x, pitch_y).

    Each line contributes a relative residual (converted - measured) / measured, so a short and a long
    calibration line carry the same weight and the longest line cannot drag the estimate. Gauss-Newton
    starts from the linearized guess, which is close because the model is smooth and convex in the pitches.
    """
    if indexes is None:
        indexes = np.arange(len(vectors))
    vec = vectors[indexes]
    obs = measured[indexes]
    # Linearized seed: w * (dx*px + dy*py) = w * L with w = 1/L.
    seed_weights = 1.0 / obs
    seed_design = vec * seed_weights[:, None]
    singular = np.linalg.svd(seed_design, compute_uv=False)
    if singular[-1] < DIRECTION_RATIO * singular[0]:
        return None
    guess, *_ = np.linalg.lstsq(seed_design, np.ones_like(obs), rcond=None)
    if not np.isfinite(guess).all() or (guess <= 0).any():
        return None

    pitches = guess
    for _ in range(50):
        converted, residual = _residuals(vec, obs, pitches)
        cost = float(residual @ residual)
        jacobian = np.column_stack((
            vec[:, 0] * pitches[0] / np.maximum(converted, 1e-12),
            vec[:, 1] * pitches[1] / np.maximum(converted, 1e-12),
        )) / obs[:, None]
        try:
            step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if float(np.abs(step).max()) < 1e-10:
            break
        # Backtracking keeps both pitches positive and only accepts steps that reduce the relative cost.
        damping = 1.0
        next_pitches = None
        for _ in range(40):
            candidate = pitches + damping * step
            if np.isfinite(candidate).all() and (candidate > 0).all():
                _, candidate_residual = _residuals(vec, obs, candidate)
                if float(candidate_residual @ candidate_residual) < cost:
                    next_pitches = candidate
                    break
            damping *= 0.5
        if next_pitches is None:
            # No improving step remains: measurement noise dominates, so the current estimate is the fit.
            break
        pitches = next_pitches
    return [float(value) for value in pitches]


def _line_report(entries, pitches):
    report, residuals = [], []
    for line, measured in entries:
        dx = line.end.x - line.start.x
        dy = line.end.y - line.start.y
        converted = math.hypot(pitches[0] * dx, pitches[1] * dy)
        error = (converted - measured) / measured
        residuals.append(abs(error))
        report.append({
            'line': line.as_dict(),
            'measured_length': measured,
            'converted_length': float(converted),
            'relative_error': float(error),
            'detector_length': float(line.length),
        })
    return report, residuals


def calibrate_from_lines(record, raw):
    """Estimate separate X/Y pitches from three measured reference lines without changing the record."""
    cap = record['capture']
    unit, entries = _calibration_entries(raw, cap['width'], cap['height'])
    current = SpatialScale.from_record(record.get('scale'))
    vectors, measured = _line_vectors(entries)
    pitches = _fit_pitches(vectors, measured)

    if pitches is None:
        # All lines point essentially the same way: horizontal and vertical pitches cannot be separated.
        # Report converted lengths under the existing scale so each line can still be compared by hand.
        lines, _ = _line_report(entries, [current.pitch_x, current.pitch_y])
        return {
            'status': 'underdetermined',
            'unit': unit,
            'warnings': ['The calibration lines run in nearly the same direction, so horizontal and vertical '
                         'scales cannot be separated. The existing scale was left untouched.'],
            'lines': lines,
            'current_scale': current.as_dict(),
            'proposed_scale': None,
            'excluded_indexes': [],
        }

    lines, residuals = _line_report(entries, pitches)
    status, warnings, excluded = 'ready', [], []
    if max(residuals) > READY_TOLERANCE:
        ordered = np.argsort(residuals)[::-1]
        worst, second_worst = int(ordered[0]), int(ordered[1])
        keep = np.asarray([index for index in range(len(entries)) if index != worst])
        pair_pitches = _fit_pitches(vectors, measured, keep)
        drift = (max(abs(pair_pitches[0] / pitches[0] - 1), abs(pair_pitches[1] / pitches[1] - 1))
                 if pair_pitches is not None else math.inf)
        standout = residuals[worst] >= OUTLIER_RATIO * max(residuals[second_worst], 1e-12)
        if (residuals[worst] > OUTLIER_TOLERANCE and standout and pair_pitches is not None
                and drift <= PITCH_DRIFT_LIMIT):
            # One line clearly disagrees while the other two stay mutually consistent; it cannot drag the fit.
            pitches = pair_pitches
            lines, residuals = _line_report(entries, pitches)
            status, excluded = 'review', [worst]
            warnings.append(
                f'Line {worst + 1} disagrees with the other two by {residuals[worst]:.1%}; it was left out of '
                'the estimate. Recheck that line before replacing the existing scale.')
        else:
            # No single line can be blamed confidently: keep the existing scale and ask for a manual recheck.
            status = 'underdetermined'
            warnings.append('The converted lengths do not agree tightly enough to identify one reliable scale. '
                            'Recheck the calibration lines and measured lengths; the existing scale was kept.')

    if pitches is not None and not all(1e-8 <= pitch <= 1000 for pitch in pitches):
        status = 'underdetermined'
        pitches = None
        warnings.append('The estimated detector pitch falls outside the supported range. '
                        'Recheck the measured lengths; the existing scale was kept.')

    if status == 'underdetermined':
        # No replacement may be applied while the evidence is ambiguous or directionally inseparable.
        pitches = None

    return {
        'status': status,
        'unit': unit,
        'warnings': warnings,
        'lines': lines,
        'current_scale': current.as_dict(),
        'proposed_scale': (SpatialScale(unit, pitches[0], pitches[1], 'pitch',
                                        witness={'method': 'calibration-lines',
                                                 'lines': [item['line'] for item in lines],
                                                 'measured_lengths': [item['measured_length'] for item in lines],
                                                 'excluded_indexes': excluded}).as_dict()
                           if pitches is not None else None),
        'excluded_indexes': excluded,
    }


def apply_calibrated_scale(record, raw):
    proposal = calibrate_from_lines(record, raw)
    if proposal['status'] == 'underdetermined' or proposal['proposed_scale'] is None:
        raise InputError('The calibration lines cannot determine a replacement scale. '
                         'Keep the existing scale or recheck the lines.')
    record['scale'] = proposal['proposed_scale']
    return proposal


def measure_extent(record, raw):
    body = object_value(raw, 'Spatial measurement')
    cap = record['capture']
    pitch = SpatialScale.from_record(record.get('scale'))
    if body.get('line'):
        line = ScanLine.parse(body['line'], cap['width'], cap['height'])
        dx, dy = (line.end.x - line.start.x), (line.end.y - line.start.y)
        return {
            'length': pitch.distance(dx, dy), 'detector_length': line.length,
            'dx': dx * pitch.pitch_x, 'dy': dy * pitch.pitch_y,
            'unit': pitch.symbol, 'line': line.as_dict(), 'scale': pitch.as_dict(),
        }
    selection, name = select_region(record, body, cap['width'], cap['height'])
    count = int(selection.local_mask().sum())
    bounds = selection.box
    return {
        'name': name, 'geometry': selection.as_dict(), 'pixels': count,
        'width': bounds.width * pitch.pitch_x, 'height': bounds.height * pitch.pitch_y,
        'area': pitch.area(count), 'area_unit': f'{pitch.symbol}²',
        'unit': pitch.symbol, 'scale': pitch.as_dict(),
    }


def _nice_length(target):
    order = 10 ** math.floor(math.log10(target))
    selection = max((sample for sample in (1, 2, 2.5, 5, 10) if (sample * order) <= target), default=1)
    return selection * order


def scale_bar_for(record, viewport, output_width):
    scale = SpatialScale.from_record(record.get('scale'))
    units_per_output = scale.pitch_x * viewport.width / output_width
    target = _nice_length(units_per_output * output_width * 0.22)
    pixels = max(1, round(target / units_per_output))
    if scale.unit == 'px' and target < 1:
        pixels = max(1, round(1 / units_per_output))
        target = 1
    return {'pixels': pixels, 'unit': scale.symbol, 'length': target, 'direction': 'detector x'}


def draw_scale(draw, position, scale_bar, *, tau=1):
    x, y = position
    stop = x + round(scale_bar['pixels'] * tau)
    cast = (230, 241, 247)
    shadow = (16, 23, 31)
    bar_note = f'{scale_bar["length"]:g} {scale_bar["unit"]}'
    base = ImageFont.load_default(size=14)
    for offset in (-1, 1):
        draw.line((x, y + offset, stop, y + offset), fill=shadow, width=3)
    draw.line((x, y, stop, y), fill=cast, width=2)
    draw.line((x, y - 4, x, y + 4), fill=cast, width=2)
    draw.line((stop, y - 4, stop, y + 4), fill=cast, width=2)
    draw.text(((x + stop) // 2, y - 8), bar_note, anchor='mb', font=base, fill=cast, stroke_width=2, stroke_fill=shadow)


def spatial_settings(record):
    config = SpatialScale.from_record(record.get('scale'))
    return {'scale': config.as_dict(), 'unit_label': config.symbol, 'calibrated': config.unit != 'px'}
