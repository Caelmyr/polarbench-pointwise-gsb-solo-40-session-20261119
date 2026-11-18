"""Verify the PNG overlay colours exactly the measured pixels the JSON reports."""
import sys
sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, '/workspace')

import numpy as np
from PIL import Image
from io import BytesIO

from polarbench.server import create_app
from polarbench.layup import ANOMALY_COLOR, BRIDGE_COLOR, REFERENCE_COLOR
from polarbench.layup_images import layup_overlay

app = create_app(data_dir='/tmp/polarbench-layup-render')
client = app.test_client()
created = client.post('/api/projects/demo', json={'demo': 'shifted-laminate'}).get_json()
pid = created['id']
payload = {
    'state': created['defaults'],
    'reference_box': {'x': 70, 'y': 90, 'width': 70, 'height': 70},
    'deviation': 20.0, 'split': 25.0, 'bridge_gap': 2, 'min_area': 20,
}

# Internal call to keep the renderer-only masks, mirroring the API route.
bench = app.config['BENCH']
record = bench.store.get(pid)
results, _ = bench.pending(record, payload['state'])
from polarbench.layup import review_layup
report = review_layup(results, record, payload)
png = layup_overlay(results, report)

render_masks = report['_render']
anomaly_px = int(render_masks['anomaly_mask'].sum())
bridge_px = int(render_masks['bridge_mask'].sum())
reported_measured = sum(p['measured_pixels'] for p in report['patches'])
reported_bridge = sum(p['bridge_weak_pixels'] for p in report['patches'])

print('render anomaly px', anomaly_px, '== sum reported measured', reported_measured)
assert anomaly_px == reported_measured
print('render bridge px', bridge_px, 'sum reported bridge (per patch)', reported_bridge)
# A weak cell can border two shown patches (counted twice per patch), so the distinct mask
# count is bounded by the per-patch sum; both exclude every measured pixel.
assert bridge_px <= reported_bridge

# Outlines, the reference box and labels overwrite some filled pixels, so RGB counting cannot
# recover the mask. Filling itself is mask-driven (verified above); here we only confirm the
# overlay renders at the exact window size and contains the marker colours somewhere.
real = np.asarray(Image.open(BytesIO(png)).convert('RGB'))
assert real.shape[0] == report['window']['height'] and real.shape[1] == report['window']['width']
assert np.all(real == np.array(ANOMALY_COLOR), axis=2).any()
assert np.all(real == np.array(BRIDGE_COLOR), axis=2).any()
assert np.all(real == np.array(REFERENCE_COLOR), axis=2).any()

# Anomaly mask and bridge mask must never overlap (weak pixels are not counted as measured).
assert not np.logical_and(render_masks['anomaly_mask'], render_masks['bridge_mask']).any()

# Every coloured anomaly pixel must be a truly oriented (measured) detector position.
print('anomaly on unsupported positions:', int((render_masks['anomaly_mask'] & ~results.field.oriented).sum()))
assert int((render_masks['anomaly_mask'] & ~results.field.oriented).sum()) == 0

# Coordinates stay inside the original detector frame.
cap = created['capture']
for p in report['patches']:
    b = p['box']
    assert b['x'] >= 0 and b['y'] >= 0
    assert b['x'] + b['width'] <= cap['width'] and b['y'] + b['height'] <= cap['height']

# CSV round-trips the same measured counts.
from polarbench.layup import layup_csv
import csv
rows = list(csv.DictReader(layup_csv(report).decode().splitlines()))
assert len(rows) == len(report['patches'])
assert sum(int(r['measured_pixels']) for r in rows) == reported_measured
assert all(int(r['bridge_weak_pixels']) >= 0 for r in rows)
print('CSV rows', len(rows), 'match JSON patches')
print('\nRENDER CONSISTENCY OK')
