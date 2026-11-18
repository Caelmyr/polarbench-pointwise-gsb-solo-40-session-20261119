from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .domain import ANGLES, Box, InputError, integer, object_value
from .stokes import intersection_support


@dataclass(frozen=True)
class Issue:
    key: str
    bit: int
    label: str
    color: tuple[int, int, int]
    explanation: str


ISSUES = (
    Issue('outside', 1, 'Outside common overlap', (57, 68, 87), 'An aligned source plane does not cover this position.'),
    Issue('alpha', 2, 'Native coverage gap', (105, 78, 170), 'At least one source has incomplete alpha coverage.'),
    Issue('reference', 4, 'Reference unusable', (177, 87, 187), 'Dark or flat reference could not support correction here.'),
    Issue('mask', 8, 'Masked by inspector', (90, 111, 136), 'The detector-space inclusion mask excludes this position.'),
    Issue('saturation', 16, 'Analyzer clipped', (246, 159, 64), 'At least one original sample reached the clipping cutoff.'),
    Issue('dark', 32, 'Low intensity', (86, 135, 211), 'Total intensity is too weak to be trusted.'),
    Issue('closure', 64, 'Analyzer pairs disagree', (224, 91, 79), 'Opposite analyzer pairs estimate different total intensities.'),
    Issue('physical', 128, 'Degree exceeds tolerance', (235, 122, 159), 'Estimated polarized intensity is physically too large.'),
    Issue('weak-axis', 256, 'Direction too weak', (215, 200, 113), 'Intensity and degree are usable but orientation is too weak.'),
)
BY_ISSUE = {issue.key: issue for issue in ISSUES}
HEALTHY = (65, 167, 135)


@dataclass(frozen=True)
class QualityField:
    flags: np.ndarray
    valid: np.ndarray
    oriented: np.ndarray
    footprints: dict

    @property
    def shape(self):
        return self.flags.shape

    def reason(self, name):
        if not isinstance(name, str) or name not in BY_ISSUE:
            raise InputError('Unknown quality issue.')
        return (self.flags & BY_ISSUE[name].bit) != 0

    def issues_at(self, x, y):
        code = int(self.flags[y, x])
        return [issue.key for issue in ISSUES if code & issue.bit]


def audit(aligned, common, mask, field, limits):
    geometry, native, reference, saturation, keep = intersection_support(aligned, common, mask)
    flags = np.zeros(field.shape, dtype=np.uint16)
    base = field.base_support
    conditions = (
        ~geometry,
        geometry & ~native,
        geometry & ~reference,
        ~keep,
        geometry & native & saturation,
        base & (field.s0 < limits.min_intensity),
        base & (field.pair_error > limits.max_pair_error),
        base & (field.degree > limits.max_degree),
        field.measured & ~field.oriented,
    )
    for issue, condition in zip(ISSUES, conditions):
        flags[condition] |= issue.bit
    return QualityField(
        flags, field.measured, field.oriented,
        {str(a): aligned[a].footprint.as_dict() for a in ANGLES},
    )


def legends():
    return [
        {'key': 'good', 'label': 'Usable measurement', 'color': list(HEALTHY), 'explanation': 'All measurements and their orientation are usable.'},
        *[{
            'key': issue.key,
            'label': issue.label,
            'color': list(issue.color),
            'explanation': issue.explanation,
        } for issue in ISSUES],
    ]


def summary(quality, box: Box | None = None):
    selection = box.slices if box else np.s_[:, :]
    flags = quality.flags[selection]
    counts = {issue.key: int(np.count_nonzero(flags & issue.bit)) for issue in ISSUES}
    overlap = int(np.count_nonzero(np.bitwise_and(flags, flags - 1)))
    return {
        'pixels': int(flags.size),
        'usable': int(np.count_nonzero(quality.valid[selection])),
        'orientation_usable': int(np.count_nonzero(quality.oriented[selection])),
        'no_flags': int(np.count_nonzero(flags == 0)),
        'overlapping_issues': overlap,
        'issues': counts,
        'counts_can_overlap': True,
    }


def dominant_image(quality, *, focus=None):
    rgb = np.empty(quality.shape + (3,), dtype=np.uint8)
    rgb[...] = HEALTHY
    if focus:
        found = quality.reason(focus)
        rgb[...] = (40, 49, 62)
        rgb[found] = BY_ISSUE[focus].color
        return rgb
    for issue in reversed(ISSUES):
        rgb[(quality.flags & issue.bit) != 0] = issue.color
    return rgb


def one_plane_coverage(aligned, x, y):
    evidence = {}
    for angle in ANGLES:
        plane = aligned[angle]
        exists = plane.footprint.contains(x, y)
        evidence[str(angle)] = {
            'inside': exists,
            'native_alpha': bool(plane.data.native_valid[y, x]) if exists else None,
            'valid_reference': bool(plane.data.reference_valid[y, x]) if exists else None,
            'clipped': bool(plane.data.clipped[y, x]) if exists else None,
            'black_clamped': bool(plane.data.below_black[y, x]) if exists else None,
            'corrected': float(plane.data.values[y, x]) if exists and plane.valid[y, x] else None,
        }
    return evidence


def _patch_row(mask_labels, idx, cut, footprint, destination):
    yy, xx = np.nonzero(mask_labels[cut] == idx)
    height = cut[0].stop - cut[0].start
    width = cut[1].stop - cut[1].start
    x0, y0 = cut[1].start + footprint.x, cut[0].start + footprint.y
    box = Box(x0, y0, width, height)
    destination.append({
        'pixels': int(xx.size),
        'box': box.as_dict(),
        'centroid': {'x': float(x0 + xx.mean()), 'y': float(y0 + yy.mean())},
        'fills_box': float(xx.size / box.area),
        'touches_window': bool(box.x == footprint.x or box.y == footprint.y or box.right == footprint.right or box.bottom == footprint.bottom),
    })


def issue_patches(quality, raw):
    request = object_value(raw, 'Quality patch report')
    issue = request.get('issue')
    matches = quality.reason(issue)
    height, width = quality.shape
    footprint = Box.parse(request['box'], width, height) if 'box' in request and request['box'] is not None else Box.entire(width, height)
    min_area = integer(request.get('min_area', 12), 'Minimum patch size', 1, footprint.area)
    subset = matches[footprint.slices]
    structure = ndimage.generate_binary_structure(2, 1)
    mask_labels, count = ndimage.label(subset, structure)
    cuts = ndimage.find_objects(mask_labels)
    counts = np.bincount(mask_labels.ravel())
    # Filter and rank first to keep the response bounded on noisy detector masks.
    retained = [idx for idx in range(1, count + 1) if counts[idx] >= min_area]
    retained.sort(key=lambda idx: (-counts[idx], idx))
    rows = []
    for idx in retained[:40]:
        _patch_row(mask_labels, idx, cuts[idx - 1], footprint, rows)
    return {
        'issue': issue,
        'box': footprint.as_dict(),
        'minimum_size': min_area,
        'all_components': int(count),
        'qualified_components': len(retained),
        'shown': len(rows),
        'patches': rows,
    }
