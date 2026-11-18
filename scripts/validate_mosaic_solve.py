"""Standalone validation for the reference-driven mosaic order inference.

Run with: python3 -I scripts/validate_mosaic_solve.py
Synthetic frames are written under .scratch/fixtures and never imported.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# -I keeps CWD off sys.path; the interpreter's site-packages live on PYTHONPATH
# in this image and have to be re-added explicitly.
for extra in os.environ.get('PYTHONPATH', '').split(os.pathsep):
    if extra and extra not in sys.path:
        sys.path.append(extra)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from polarbench.codec import rgba_png  # noqa: E402
from polarbench.domain import ANGLES, InputError  # noqa: E402
from polarbench.mosaic_solve import solve_mosaic  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / '.scratch' / 'fixtures'
FIXTURES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20261118)


def make_frame(true_order, theta_deg, exposure, black, occlusion=False,
               degree=0.55, width=120, height=80, texture=True):
    """Build a packed 2x2 mosaic for a uniformly linearly polarized scene."""
    theta = np.deg2rad(theta_deg)
    s0 = 0.55
    if texture:
        y, x = np.mgrid[0:height, 0:width]
        s0_field = s0 + 0.10 * np.sin(x / 9.0) * np.cos(y / 7.0)
    else:
        s0_field = np.full((height, width), s0)
    pol = degree * s0_field
    q = pol * np.cos(2 * theta)
    u = pol * np.sin(2 * theta)
    analyzer = {0: q, 45: u, 90: -q, 135: -u}
    sensor = np.zeros((2 * height, 2 * width), dtype=np.float32)
    valid_full = np.ones((2 * height, 2 * width), dtype=bool)
    for position, angle in enumerate(true_order):
        row, col = divmod(position, 2)
        signal = exposure * (0.5 * (s0_field + analyzer[angle]) + black)
        signal += rng.normal(0, 0.0015, size=signal.shape)
        signal = np.clip(signal, 0, 1)
        sensor[row::2, col::2] = signal
    if occlusion:
        # A native occluded band in one corner plus random dead samples.
        valid_full[0:24, 0:30] = False
        drop = rng.random(sensor.shape) < 0.02
        valid_full &= ~drop
    levels = np.round(sensor * 255).astype(np.uint8)
    path = FIXTURES / f'ref-{theta_deg:g}-{exposure:g}.png'
    path.write_bytes(rgba_png(levels, valid_full))
    return path


def load(path):
    return path.read_bytes(), path.name


def case(title, true_order, directions, exposures, expected_confident=True,
         expected_order=None, occlusion=False, **kwargs):
    files = [
        load(make_frame(true_order, theta, exposure, black=0.03, occlusion=occlusion, **kwargs))
        for theta, exposure in zip(directions, exposures)
    ]
    result = solve_mosaic(files, directions)
    status = 'OK ' if result['confident'] == expected_confident else 'FAIL'
    inferred = tuple(result['inferred_order'])
    order_ok = expected_order is None or inferred == tuple(expected_order)
    print(f'[{status}] {title}: status={result["status"]} inferred={inferred} '
          f'residual={result["best_angular_residual"]:.3f} margin={result["decision_margin"]:.3f}')
    for ref in result['references']:
        print(f'        theta={ref["direction"]:>5.1f} mod={ref["modulation"]:.4f} '
              f'common={ref["common_pixels"]} medians={[round(m, 4) for m in ref["medians"]]}')
    if not result['confident']:
        print(f'        alternatives={[a["layout"] for a in result["alternatives"]]}')
    assert result['confident'] == expected_confident, f'{title}: confidence mismatch'
    if expected_confident:
        assert result['independent_directions'] and result['uncertainty_reason'] is None
    if expected_order is not None:
        assert order_ok, f'{title}: expected {expected_order}, got {inferred}'
    return result


def main():
    unknown = (90, 135, 45, 0)  # an order absent from the five presets

    # 1. Two well-separated directions, different exposure and black levels,
    #    with native occlusion -> must recover the true order confidently.
    case('different exposures + occlusion', unknown, [20, 75], [0.9, 0.45],
         expected_confident=True, expected_order=unknown, occlusion=True)

    # 2. Two references whose directions are 90 deg apart: the doubled angles
    #    coincide, so orders genuinely cannot be separated -> uncertain, and
    #    the tied permutations must be offered as alternatives.
    perpendicular = case('perpendicular input directions', unknown, [20, 110], [1.0, 0.6],
                         expected_confident=False)
    assert perpendicular['alternatives'], 'perpendicular references must expose tied orders'
    assert perpendicular['uncertainty_reason']

    # 3. Parallel directions -> no independent second constraint -> uncertain.
    parallel = case('parallel input directions', unknown, [30, 30], [1.0, 0.6],
                    expected_confident=False)
    assert parallel['uncertainty_reason']
    assert parallel['independent_directions'] is False

    # 4. One weakly polarized (nearly unpolarized) reference: only one
    #    informative frame remains -> must stay uncertain.
    files = [
        load(make_frame(unknown, 20, 1.0, 0.03, degree=0.5)),
        load(make_frame(unknown, 80, 0.7, 0.03, degree=0.005)),
    ]
    result = solve_mosaic(files, [20, 80])
    print(f'[{"OK " if not result["confident"] else "FAIL"}] weak second reference: '
          f'status={result["status"]} mods={[round(r["modulation"], 4) for r in result["references"]]}')
    assert not result['confident']
    assert result['references'][1]['informative'] is False
    assert result['independent_directions'] is False and result['uncertainty_reason']

    # 5. Both references weak -> hard error, never a guessed order.
    try:
        files = [
            load(make_frame(unknown, 20, 1.0, 0.03, degree=0.004)),
            load(make_frame(unknown, 80, 0.7, 0.03, degree=0.004)),
        ]
        solve_mosaic(files, [20, 80])
        raise AssertionError('expected InputError for two weak frames')
    except InputError as error:
        print(f'[OK ] both weak rejected: {error}')

    # 6. Direction values that are not on the 45-degree grid still work.
    case('arbitrary input direction', (45, 0, 135, 90), [12.5, 100], [1.1, 0.5],
         expected_confident=True, expected_order=(45, 0, 135, 90))

    # 7. A preset order is also recovered.
    case('known preset order', (0, 45, 135, 90), [0, 60], [0.8, 1.2],
         expected_confident=True, expected_order=(0, 45, 135, 90))

    # 8. Occlusion so heavy that too few common pixels remain -> hard error.
    sparse_valid = np.zeros((160, 240), dtype=bool)
    sparse_valid[0:6, 0:6] = True  # 9 detector cells after 2x2, below the minimum
    dense = rgba_png(np.full((160, 240), 128, dtype=np.uint8), sparse_valid)
    path = FIXTURES / 'tiny-valid.png'
    path.write_bytes(dense)
    try:
        solve_mosaic([(dense, 'a.png'), (dense, 'b.png')], [10, 70])
        raise AssertionError('expected InputError for insufficient common pixels')
    except InputError as error:
        assert 'common positions' in str(error)
        print(f'[OK ] sparse coverage rejected: {error}')

    # 9. Different sensor shapes are rejected.
    other = make_frame(unknown, 75, 0.5, 0.03, width=100, height=80)
    try:
        solve_mosaic([load(FIXTURES / 'ref-20-0.9.png'), load(other)], [20, 75])
        raise AssertionError('expected InputError for shape mismatch')
    except InputError as error:
        print(f'[OK ] shape mismatch rejected: {error}')

    print('\nAll solver validation cases passed.')


if __name__ == '__main__':
    main()
