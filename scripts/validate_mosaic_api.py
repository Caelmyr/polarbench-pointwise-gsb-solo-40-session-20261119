"""End-to-end checks for reference-driven mosaic calibration via the Flask API.

Run with: python3 -I scripts/validate_mosaic_api.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

for extra in os.environ.get('PYTHONPATH', '').split(os.pathsep):
    if extra and extra not in sys.path:
        sys.path.append(extra)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from werkzeug.datastructures import MultiDict  # noqa: E402

from polarbench.codec import rgba_png  # noqa: E402
from polarbench.ingest import MOSAIC_LAYOUTS  # noqa: E402
from polarbench.server import create_app  # noqa: E402

rng = np.random.default_rng(77)


def build_mosaic(order, theta, exposure, h=64, w=96, degree=0.5, occlude=False):
    th = np.deg2rad(theta)
    s0 = 0.5
    pol = degree * s0
    q, u = pol * np.cos(2 * th), pol * np.sin(2 * th)
    analyzer = {0: q, 45: u, 90: -q, 135: -u}
    sensor = np.zeros((2 * h, 2 * w), np.float32)
    for position, angle in enumerate(order):
        row, col = divmod(position, 2)
        signal = exposure * 0.5 * (s0 + analyzer[angle]) + 0.02
        sensor[row::2, col::2] = np.clip(signal + rng.normal(0, 0.002, (h, w)), 0, 1)
    valid = np.ones_like(sensor, dtype=bool)
    if occlude:
        valid[0:18, 0:24] = False
    return rgba_png((sensor * 255).astype(np.uint8), valid)


def png_file(payload, name):
    return (io.BytesIO(payload), name)


def main():
    data_dir = Path(__file__).resolve().parent.parent / '.scratch' / 'api-data'
    app = create_app(data_dir)
    client = app.test_client()
    true_order = (135, 0, 90, 45)  # not among the five named presets
    ref1 = build_mosaic(true_order, 25, 1.0, occlude=True)
    ref2 = build_mosaic(true_order, 80, 0.4)
    workpiece = build_mosaic(true_order, 50, 0.8)

    # 1. All 24 canonical layouts are accepted by the configuration metadata.
    config = client.get('/api/configuration').get_json()
    assert set(config['mosaic_patterns']) <= set(MOSAIC_LAYOUTS)
    print(f'[OK ] configuration exposes {len(config["mosaic_patterns"])} named presets; '
          f'backend accepts {len(MOSAIC_LAYOUTS)} canonical layouts')

    # 2. Calibration endpoint recovers the unknown order.
    response = client.post(
        '/api/mosaic-calibration',
        data=MultiDict([
            ('reference_1', png_file(ref1, 'ref-a.png')),
            ('reference_2', png_file(ref2, 'ref-b.png')),
            ('direction_1', '25'),
            ('direction_2', '80'),
        ]),
        content_type='multipart/form-data',
    )
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert result['confident'] is True
    assert tuple(result['inferred_order']) == true_order
    print(f'[OK ] calibration confident: layout={result["inferred_layout"]} '
          f'residual={result["best_angular_residual"]:.3f} margin={result["decision_margin"]:.3f}')

    # 3. Missing direction is a 400, never a guessed order.
    bad = client.post(
        '/api/mosaic-calibration',
        data=MultiDict([
            ('reference_1', png_file(ref1, 'a.png')),
            ('reference_2', png_file(ref2, 'b.png')),
            ('direction_1', '25'),
        ]),
        content_type='multipart/form-data',
    )
    assert bad.status_code == 400
    print(f'[OK ] missing direction rejected: {bad.get_json()["error"]}')

    # 4. Import the workpiece using the inferred (non-preset) layout key.
    imported = client.post(
        '/api/projects',
        data=MultiDict([
            ('kind', 'mosaic'),
            ('name', 'Inferred-order workpiece'),
            ('mosaic', png_file(workpiece, 'workpiece.png')),
            ('mosaic_layout', result['inferred_layout']),
        ]),
        content_type='multipart/form-data',
    )
    assert imported.status_code == 201, imported.get_json()
    project = imported.get_json()
    assert project['capture']['mosaic_layout'] == result['inferred_layout']
    print(f'[OK ] workpiece imported with inferred layout {project["capture"]["mosaic_layout"]}')

    # 5. Decode and verify tile->angle mapping matches the physical truth.
    with app.app_context():
        bench = app.config['BENCH']
        record = bench.store.get(project['id'])
        from polarbench.ingest import decode
        capture = decode(bench.store, record)
    positions = capture.sensor_positions
    for angle, (row, col) in positions.items():
        expected_position = true_order.index(angle)
        assert (row, col) == divmod(expected_position, 2)
    print(f'[OK ] decoded tile positions match physical order {true_order}')

    # 6. Uncertain inputs (perpendicular directions) report status explicitly.
    ref_perp = build_mosaic(true_order, 115, 0.6)
    uncertain = client.post(
        '/api/mosaic-calibration',
        data=MultiDict([
            ('reference_1', png_file(ref1, 'ref-a.png')),
            ('reference_2', png_file(ref_perp, 'ref-c.png')),
            ('direction_1', '25'),
            ('direction_2', '115'),
        ]),
        content_type='multipart/form-data',
    ).get_json()
    assert uncertain['confident'] is False and uncertain['uncertainty_reason']
    print(f'[OK ] perpendicular references flagged uncertain: {uncertain["uncertainty_reason"]}')

    print('\nAll API validation cases passed.')


if __name__ == '__main__':
    main()
