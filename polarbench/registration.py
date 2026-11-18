from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from .calibration import CalibratedPlane
from .domain import ANGLES, Box, InputError, Offset, as_angle_dict, integer, object_value, per_angle

MAX_SHIFT = 160

MIN_CORRELATION = 0.18
PEAK_SEPARATION = 0.015
PLAUSIBLE_MARGIN = 0.05
PLAUSIBLE_LIMIT = 5


@dataclass(frozen=True)
class Registration:
    offsets: dict[int, Offset]

    @classmethod
    def parse(cls, raw):
        offsets = per_angle(raw, 'Registration offsets', lambda value, angle: Offset.parse({} if value is None else value, MAX_SHIFT))
        if offsets[0] != Offset():
            raise InputError('The 0° reference plane always defines detector coordinates.')
        return cls(offsets)

    def as_dict(self):
        return as_angle_dict({angle: self.offsets[angle].as_dict() for angle in ANGLES})


def footprints(height, width, offset: Offset):
    if abs(offset.dx) >= width or abs(offset.dy) >= height:
        return None
    x0, y0 = max(0, offset.dx), max(0, offset.dy)
    dst = Box(x0, y0, width - abs(offset.dx), height - abs(offset.dy))
    src = Box(x0 - offset.dx, y0 - offset.dy, dst.width, dst.height)
    return src, dst


def _mapped(array, source, target, fill):
    if source == target:
        return array
    moved = np.full_like(array, fill)
    moved[target.slices] = array[source.slices]
    return moved


@dataclass(frozen=True)
class AlignedPlane:
    data: CalibratedPlane
    footprint: Box

    @property
    def valid(self):
        return self.data.valid


def align_plane(plane: CalibratedPlane, offset: Offset):
    height, width = plane.values.shape
    overlap = footprints(height, width, offset)
    if overlap is None:
        raise InputError('Registration translates an entire analyzer plane out of the detector image.')
    src, dst = overlap
    moved = CalibratedPlane(
        _mapped(plane.values, src, dst, 0),
        _mapped(plane.native_valid, src, dst, False),
        _mapped(plane.reference_valid, src, dst, False),
        _mapped(plane.clipped, src, dst, False),
        _mapped(plane.below_black, src, dst, False),
    )
    return AlignedPlane(moved, dst)


def align_planes(planes, config: Registration):
    aligned = {a: align_plane(planes[a], config.offsets[a]) for a in ANGLES}
    common = aligned[0].footprint
    for angle in ANGLES[1:]:
        common = common.intersection(aligned[angle].footprint)
        if common is None:
            raise InputError('These offsets leave no common detector area across the four angles.')
    return aligned, common


def _texture(value, allowed):
    work = np.where(allowed, value, 0)
    safe = ndimage.binary_erosion(allowed, iterations=1, border_value=0)
    gx = ndimage.sobel(work, axis=1, mode='nearest')
    gy = ndimage.sobel(work, axis=0, mode='nearest')
    gradients = np.hypot(gx, gy)
    gradients[~safe] = 0
    if not safe.any():
        return gradients, safe
    limit = np.quantile(gradients[safe], 0.95)
    np.minimum(gradients, max(0.001, limit * 1.5), out=gradients)
    return gradients, safe


def _candidate_offsets(reference, moving, valid_a, valid_b, radius):
    a = np.where(valid_a, reference - np.mean(reference[valid_a]), 0)
    b = np.where(valid_b, moving - np.mean(moving[valid_b]), 0)
    grid = signal.correlate(a, b, mode='full', method='fft')
    y0, x0 = reference.shape[0] - 1, reference.shape[1] - 1
    reach_y, reach_x = min(y0, radius), min(x0, radius)
    local = grid[y0 - reach_y:y0 + reach_y + 1, x0 - reach_x:x0 + reach_x + 1].copy()
    candidates = []
    for _ in range(min(8, local.size)):
        index = int(local.argmax())
        y, x = np.unravel_index(index, local.shape)
        if not np.isfinite(local[y, x]):
            break
        candidates.append(Offset(int(x - reach_x), int(y - reach_y)))
        local[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = -np.inf
    return candidates


def _similarity(a, b, valid_a, valid_b, offset, step=1):
    overlap = footprints(*a.shape, offset)
    if overlap is None:
        return None
    src, dst = overlap
    x_region = a[dst.slices][::step, ::step]
    y_region = b[src.slices][::step, ::step]
    keep = valid_a[dst.slices][::step, ::step] & valid_b[src.slices][::step, ::step]
    count = np.count_nonzero(keep)
    if count < max(25, int(0.2 * keep.size)):
        return None
    x, y = x_region[keep], y_region[keep]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom <= 1e-8:
        return None
    return float(np.dot(x, y) / denom), int(count * step**2)


def _rank(candidates, a, b, safe_a, safe_b, step):
    scored = []
    for pos in candidates:
        score = _similarity(a, b, safe_a, safe_b, pos, step)
        if score is not None:
            scored.append((pos, *score))
    return sorted(scored, key=lambda entry: (-entry[1], abs(entry[0].dx) + abs(entry[0].dy), entry[0].dy, entry[0].dx))


def _scan_window(reference, moving, region: Box, radius: int):
    scale = max(1, (max(reference.shape) + 479) // 480)
    a, b = reference[region.slices], moving[region.slices]
    a_values, a_valid = _texture(a, np.isfinite(a))
    b_values, b_valid = _texture(b, np.isfinite(b))
    if min(a_valid.sum(), b_valid.sum()) < 100:
        return {'scored': False, 'reason': 'Too little common usable texture in the registration window.'}
    reduced = np.s_[::scale, ::scale]
    a_coarse, b_coarse = a_valid[reduced], b_valid[reduced]
    if not a_coarse.any() or not b_coarse.any():
        return {'scored': False, 'reason': 'Texture disappears at the coarse search scale. Choose a smaller window.'}
    offsets = _candidate_offsets(
        a_values[reduced], b_values[reduced], a_coarse, b_coarse, (radius + scale - 1) // scale,
    )
    local_candidates = {Offset()}
    for guess in offsets:
        center_x, center_y = guess.dx * scale, guess.dy * scale
        for dy in range(max(-radius, center_y - scale + 1), min(radius, center_y + scale - 1) + 1):
            for dx in range(max(-radius, center_x - scale + 1), min(radius, center_x + scale - 1) + 1):
                local_candidates.add(Offset(dx, dy))
    ranked = _rank(local_candidates, a_values, b_values, a_valid, b_valid, max(1, scale))
    if not ranked:
        return {'scored': False, 'reason': 'No shift has enough textured overlap to score.'}
    best, correlation, pixels = ranked[0]
    alternatives = [entry for entry in ranked[1:] if max(abs(entry[0].dx - best.dx), abs(entry[0].dy - best.dy)) > 1]
    gap = (correlation - alternatives[0][1]) if alternatives else None
    return {
        'scored': True,
        'ranked': ranked,
        'best': best,
        'correlation': correlation,
        'overlap_samples': pixels,
        'distinct_peak_margin': gap,
        'at_search_limit': max(abs(best.dx), abs(best.dy)) == radius,
    }


def _confident(scan):
    if not scan['scored']:
        return False
    gap = scan['distinct_peak_margin']
    return scan['correlation'] >= MIN_CORRELATION and (gap is None or gap >= PEAK_SEPARATION)


def _find_offset(reference, moving, region: Box, radius: int):
    scan = _scan_window(reference, moving, region, radius)
    if not scan['scored']:
        return {'usable': False, 'reason': scan['reason']}
    return {
        'usable': _confident(scan),
        'offset': scan['best'].as_dict(),
        'correlation': scan['correlation'],
        'overlap_samples': scan['overlap_samples'],
        'distinct_peak_margin': scan['distinct_peak_margin'],
        'at_search_limit': scan['at_search_limit'],
        'reason': 'Low confidence when overlap, texture match, or separation between peaks is weak.',
    }


def suggest_registration(planes, raw, width, height):
    state = object_value(raw, 'Registration request')
    box = Box.parse(state['box'], width, height) if state.get('box') else Box.entire(width, height)
    if min(box.width, box.height) < 16:
        raise InputError('Registration needs an inspection window at least 16 × 16 pixels.')
    radius = integer(state.get('radius', 16), 'Search radius', 1, min(MAX_SHIFT, min(box.width, box.height) // 3))
    available = {a: np.where(p.valid, p.values, np.nan) for a, p in planes.items()}
    output = {str(angle): _find_offset(available[0], available[angle], box, radius) for angle in ANGLES[1:]}
    return {'reference_angle': 0, 'window': box.as_dict(), 'search_radius': radius, 'channels': output}


def _close(first: Offset, second: Offset):
    return max(abs(first.dx - second.dx), abs(first.dy - second.dy)) <= 1


def _distinct_peaks(ranked):
    if not ranked:
        return []
    cutoff = max(MIN_CORRELATION, ranked[0][1] - PLAUSIBLE_MARGIN)
    peaks = []
    for offset, correlation, _pixels in ranked:
        if correlation < cutoff:
            break
        if any(_close(offset, kept) for kept, _ in peaks):
            continue
        peaks.append((offset, correlation))
        if len(peaks) >= PLAUSIBLE_LIMIT:
            break
    return peaks


def _shared_peaks(first, second):
    supported = [
        (offset, correlation)
        for offset, correlation in (*first, *second)
        if any(_close(offset, other) for other, _ in first)
        and any(_close(offset, other) for other, _ in second)
    ]
    clusters = []
    for entry in supported:
        for cluster in clusters:
            if any(_close(entry[0], other) for other, _ in cluster):
                cluster.append(entry)
                break
        else:
            clusters.append([entry])
    merged = True
    while merged:
        merged = False
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                if any(_close(a[0], b[0]) for a in clusters[left] for b in clusters[right]):
                    clusters[left] += clusters.pop(right)
                    merged = True
                    break
            if merged:
                break
    return [max(cluster, key=lambda entry: entry[1]) for cluster in clusters]


def _window_evidence(scan, peaks):
    if not scan['scored']:
        return {'scored': False, 'confident': False, 'reason': scan['reason']}
    return {
        'scored': True,
        'confident': _confident(scan),
        'best': scan['best'].as_dict(),
        'correlation': scan['correlation'],
        'overlap_samples': scan['overlap_samples'],
        'distinct_peak_margin': scan['distinct_peak_margin'],
        'at_search_limit': scan['at_search_limit'],
        'plausible': [{'offset': offset.as_dict(), 'correlation': correlation} for offset, correlation in peaks],
    }


def _combine_evidence(scan_a, scan_b):
    peaks_a = _distinct_peaks(scan_a['ranked']) if scan_a['scored'] else []
    peaks_b = _distinct_peaks(scan_b['ranked']) if scan_b['scored'] else []
    confident_a, confident_b = _confident(scan_a), _confident(scan_b)
    result = {'evidence': [_window_evidence(scan_a, peaks_a), _window_evidence(scan_b, peaks_b)], 'shared': []}
    if not peaks_a and not peaks_b:
        return dict(result, status='unusable', usable=False, offset=None, correlation=None,
                    reason='Neither reference region offers usable texture in this channel.')
    if not peaks_a or not peaks_b:
        scan, confident = (scan_a, confident_a) if peaks_a else (scan_b, confident_b)
        silent = 'second' if peaks_a else 'first'
        if not confident:
            return dict(result, status='review', usable=False, offset=None, correlation=None,
                        reason=f'The {silent} region offers no usable texture and the other is not conclusive on its own.')
        return dict(result, status='proposed', usable=True, offset=scan['best'].as_dict(), correlation=scan['correlation'],
                    reason=f'The {silent} region offers no usable texture; the other region is conclusive on its own.')
    if confident_a and confident_b:
        if _close(scan_a['best'], scan_b['best']):
            scan = scan_a if scan_a['correlation'] >= scan_b['correlation'] else scan_b
            return dict(result, status='proposed', usable=True, offset=scan['best'].as_dict(), correlation=scan['correlation'],
                        reason='Both regions are conclusive and agree on the shift.')
        return dict(result, status='conflict', usable=False, offset=None, correlation=None,
                    reason='Both regions are conclusive but point to different shifts. No offset was filled in.')
    if confident_a != confident_b:
        scan, other_peaks = (scan_a, peaks_b) if confident_a else (scan_b, peaks_a)
        support = ' and the other region supports it' if any(_close(scan['best'], other) for other, _ in other_peaks) \
            else '; the other region is inconclusive'
        return dict(result, status='proposed', usable=True, offset=scan['best'].as_dict(), correlation=scan['correlation'],
                    reason=f'One region is conclusive{support}.')
    shared = _shared_peaks(peaks_a, peaks_b)
    if len(shared) == 1:
        offset, correlation = shared[0]
        return dict(result, status='proposed', usable=True, offset=offset.as_dict(), correlation=correlation,
                    reason='Only one candidate shift is plausible in both regions.')
    if not shared:
        return dict(result, status='review', usable=False, offset=None, correlation=None,
                    reason='Each region alone is ambiguous and they share no candidate shift.')
    return dict(result, status='ambiguous', usable=False, offset=None, correlation=None,
                shared=[offset.as_dict() for offset, _ in shared],
                reason='More than one candidate shift is plausible in both regions.')


def _search_radius(state, box: Box):
    limit = min(MAX_SHIFT, min(box.width, box.height) // 3)
    if state.get('radius') is None:
        return max(1, min(16, limit))
    return integer(state['radius'], 'Search radius', 1, limit)


def suggest_registration_pair(planes, windows, raw):
    state = object_value(raw, 'Registration request')
    radii = [_search_radius(state, box) for _name, box in windows]
    available = {a: np.where(p.valid, p.values, np.nan) for a, p in planes.items()}
    channels = {}
    for angle in ANGLES[1:]:
        scans = [
            _scan_window(available[0], available[angle], box, radius)
            for (_name, box), radius in zip(windows, radii)
        ]
        channels[str(angle)] = _combine_evidence(*scans)
    return {
        'reference_angle': 0,
        'mode': 'two-region',
        'regions': [
            {'name': name, 'window': box.as_dict(), 'search_radius': radius}
            for (name, box), radius in zip(windows, radii)
        ],
        'channels': channels,
    }
