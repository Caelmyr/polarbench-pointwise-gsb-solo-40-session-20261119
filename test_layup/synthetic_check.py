"""Synthetic checks for the layup direction patch algorithm.

Run from the repository root with: python -I -m test_layup.synthetic_check
The fixtures here are hand-built arrays, not captured material data.
"""
from __future__ import annotations

import sys

sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, '/workspace')

import numpy as np

from polarbench.layup import review_layup
from polarbench.quality import BY_ISSUE, QualityField
from polarbench.stokes import PolarField

WEAK_BIT = BY_ISSUE['weak-axis'].bit
MASK_BIT = BY_ISSUE['mask'].bit
H, W = 36, 48


def build_case(direction, oriented, weak=None, masked=None, baseline_box=None):
    weak = weak if weak is not None else np.zeros_like(oriented)
    masked = masked if masked is not None else np.zeros_like(oriented)
    measured = oriented | weak
    radians = np.deg2rad(direction * 2.0)
    degree = np.where(oriented, 0.4, np.where(weak, 0.01, 0.0))
    s0 = np.ones_like(direction, dtype=np.float32)
    polarized = s0 * degree
    q = polarized * np.cos(radians).astype(np.float32)
    u = polarized * np.sin(radians).astype(np.float32)
    field = PolarField(
        s0=s0, q=q, u=u, polarized=polarized, degree=degree,
        direction=direction.astype(np.float32),
        pair_error=np.zeros_like(s0), pair_difference=np.zeros_like(s0),
        base_support=measured | masked, measured=measured, oriented=oriented,
    )
    flags = np.zeros(direction.shape, dtype=np.uint16)
    flags[weak] |= WEAK_BIT
    flags[masked] |= MASK_BIT
    quality = QualityField(flags=flags, valid=measured, oriented=oriented, footprints={})
    record = {'scale': None, 'regions': []}
    raw = {
        'reference_box': {'x': baseline_box[0], 'y': baseline_box[1],
                          'width': baseline_box[2], 'height': baseline_box[3]},
        'deviation': 20.0, 'split': 25.0, 'bridge_gap': 2, 'min_area': 4,
    }

    class Results:
        pass
    results = Results()
    results.field = field
    results.quality = quality
    return results, record, raw


def base_maps():
    direction = np.zeros((H, W), dtype=np.float32)
    oriented = np.ones((H, W), dtype=bool)
    return direction, oriented


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {name} {detail}')
    if not condition:
        check.failures += 1
check.failures = 0


def test_circular_angle():
    direction, oriented = base_maps()
    direction[10, 10] = 179.0
    direction[10, 11] = 1.0
    direction[10, 12] = 25.0
    results, record, raw = build_case(direction, oriented, baseline_box=(1, 1, 6, 6))
    report = review_layup(results, record, raw)
    # 179 vs baseline 0 and 1 vs 0 are ~1 degree apart; only the 25 degree pixel is anomalous.
    check('circular seam 179/1 not split', report['anomaly_pixels'] == 1,
          f"(anomaly_pixels={report['anomaly_pixels']})")
    check('reference direction near zero', abs(report['reference_direction']) < 0.001,
          f"(ref={report['reference_direction']})")


def test_weak_bridge_merges_and_excludes_area():
    direction, oriented = base_maps()
    direction[10:14, 5:9] = 40.0      # blob A (16 px)
    direction[10:14, 11:15] = 40.0   # blob B (16 px); columns 9 and 10 are the two-pixel gap
    weak = np.zeros_like(oriented)
    weak[10:14, 9:11] = True         # two weak columns between the blobs
    oriented[weak] = False
    results, record, raw = build_case(direction, oriented, weak=weak, baseline_box=(1, 1, 6, 6))
    report = review_layup(results, record, raw)
    check('two blobs joined across 2 weak pixels', report['all_components'] == 1,
          f"(components={report['all_components']})")
    patch = report['patches'][0]
    check('measured area excludes weak pixels', patch['measured_pixels'] == 32,
          f"(measured={patch['measured_pixels']})")
    check('bridge weak pixels reported separately', patch['bridge_weak_pixels'] == 8,
          f"(bridge={patch['bridge_weak_pixels']})")
    check('patch median direction', abs(patch['median_direction'] - 40.0) < 0.01)


def test_masked_strip_blocks_connection():
    direction, oriented = base_maps()
    direction[10:14, 5:9] = 40.0
    direction[10:14, 11:15] = 40.0
    masked = np.zeros_like(oriented)
    masked[10:14, 9:11] = True
    oriented[masked] = False
    results, record, raw = build_case(direction, oriented, masked=masked, baseline_box=(1, 1, 6, 6))
    report = review_layup(results, record, raw)
    check('masked strip never bridges', report['all_components'] == 2,
          f"(components={report['all_components']})")
    check('no bridge pixels counted through mask', all(p['bridge_weak_pixels'] == 0 for p in report['patches']))


def test_direction_split_keeps_distinct_layups_apart():
    direction, oriented = base_maps()
    direction[10:14, 8:12] = 40.0    # blob at +40
    direction[10:14, 12:16] = 80.0   # adjacent blob at +80; mutual gap 40 > split 25
    results, record, raw = build_case(direction, oriented, baseline_box=(1, 1, 6, 6))
    report = review_layup(results, record, raw)
    check('adjacent but differently laid blobs stay separate', report['all_components'] == 2,
          f"(components={report['all_components']})")


def test_split_tolerance_merges_consistent_blobs():
    direction, oriented = base_maps()
    direction[10:14, 8:12] = 40.0
    direction[10:14, 12:16] = 44.0   # 4 degree mutual difference, within split tolerance
    results, record, raw = build_case(direction, oriented, baseline_box=(1, 1, 6, 6))
    report = review_layup(results, record, raw)
    check('consistent adjacent blobs merge', report['all_components'] == 1,
          f"(components={report['all_components']})")


def test_single_weak_pixel_bridge():
    direction, oriented = base_maps()
    direction[20, 5:9] = 40.0
    direction[20, 10:14] = 40.0
    weak = np.zeros_like(oriented)
    weak[20, 9] = True
    oriented[weak] = False
    results, record, raw = build_case(direction, oriented, weak=weak, baseline_box=(1, 1, 6, 6))
    raw['min_area'] = 2
    report = review_layup(results, record, raw)
    check('single weak pixel joins a streak', report['all_components'] == 1,
          f"(components={report['all_components']})")
    check('one bridge pixel, area still measured-only',
          report['patches'][0]['bridge_weak_pixels'] == 1 and report['patches'][0]['measured_pixels'] == 8)


def test_bridge_disagreement_blocks():
    direction, oriented = base_maps()
    direction[20, 5:9] = 40.0
    direction[20, 10:14] = -40.0 % 180  # 140 degrees; opposite deviation side
    weak = np.zeros_like(oriented)
    weak[20, 9] = True
    oriented[weak] = False
    results, record, raw = build_case(direction, oriented, weak=weak, baseline_box=(1, 1, 6, 6))
    raw['min_area'] = 2
    report = review_layup(results, record, raw)
    check('weak bridge rejected when ends disagree', report['all_components'] == 2,
          f"(components={report['all_components']})")


def test_min_area_filter():
    direction, oriented = base_maps()
    direction[30, 30] = 40.0  # single isolated anomaly pixel
    direction[10:14, 5:9] = 40.0
    results, record, raw = build_case(direction, oriented, baseline_box=(1, 1, 6, 6))
    raw['min_area'] = 4
    report = review_layup(results, record, raw)
    check('min area keeps large blob only', report['qualified_components'] == 1
          and report['all_components'] == 2)


def test_reference_box_requires_direction():
    direction, oriented = base_maps()
    weak = np.zeros_like(oriented)
    weak[1:5, 1:5] = True
    oriented[weak] = False
    results, record, raw = build_case(direction, oriented, weak=weak, baseline_box=(1, 1, 4, 4))
    try:
        review_layup(results, record, raw)
        check('reference box without directions rejected', False)
    except Exception as exc:
        check('reference box without directions rejected', 'usable layup direction' in str(exc))


def main():
    test_circular_angle()
    test_weak_bridge_merges_and_excludes_area()
    test_masked_strip_blocks_connection()
    test_direction_split_keeps_distinct_layups_apart()
    test_split_tolerance_merges_consistent_blobs()
    test_single_weak_pixel_bridge()
    test_bridge_disagreement_blocks()
    test_min_area_filter()
    test_reference_box_requires_direction()
    print(f"\n{check.failures} failure(s)")
    sys.exit(1 if check.failures else 0)


if __name__ == '__main__':
    main()
