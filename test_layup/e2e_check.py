"""End-to-end checks for the layup review API using the synthetic laminate demo."""
from __future__ import annotations

import io
import sys

sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, '/workspace')

from polarbench.server import create_app

app = create_app(data_dir='/tmp/polarbench-layup-e2e')
client = app.test_client()


def post(path, payload):
    response = client.post(path, json=payload)
    assert response.status_code in (200, 201), (path, response.status_code, response.get_json())
    return response


created = post('/api/projects/demo', {'demo': 'shifted-laminate'}).get_json()
project_id = created['id']
# Laminate A occupies the left portion of the frame and carries the 33 degree family.
reference_box = {'x': 70, 'y': 90, 'width': 70, 'height': 70}
payload = {
    'state': created['defaults'],
    'reference_box': reference_box,
    'deviation': 20.0,
    'split': 25.0,
    'bridge_gap': 2,
    'min_area': 20,
}
report = post(f'/api/projects/{project_id}/layup/patches', payload).get_json()
assert '_render' not in report, 'renderer arrays must not leak into JSON'
print('reference_direction', report['reference_direction'])
print('reference_pixels', report['reference_pixels'])
print('all_components', report['all_components'], 'qualified', report['qualified_components'])
print('anomaly_pixels', report['anomaly_pixels'], 'weak', report['weak_pixels'])
for row in report['patches'][:5]:
    print(' patch', row['patch'], row['box'], 'measured', row['measured_pixels'],
          'bridge', row['bridge_weak_pixels'], 'median', row['median_direction'],
          'mean_dev', row['mean_deviation'], 'side', row['deviation_side'],
          'area', row['measured_area'], row['area_unit'])
assert report['reference_pixels'] >= 4
assert report['patches'], 'expected at least one deviating laminate patch'
for row in report['patches']:
    assert row['measured_pixels'] >= 20
    assert row['mean_deviation'] >= 20.0 - 0.01
    assert row['measured_area'] == row['measured_pixels']  # no calibrated scale -> pixel units
    # Every patch sits in original detector coordinates inside the frame.
    box = row['box']
    assert 0 <= box['x'] and box['x'] + box['width'] <= created['capture']['width']
    assert 0 <= box['y'] and box['y'] + box['height'] <= created['capture']['height']

png = post(f'/api/projects/{project_id}/layup/patches', {**payload, 'output': 'png'})
assert png.headers['Content-Type'] == 'image/png'
assert png.data[:8] == b'\x89PNG\r\n\x1a\n'
print('png bytes', len(png.data))

csv_response = post(f'/api/projects/{project_id}/layup/patches', {**payload, 'output': 'csv'})
assert csv_response.headers['Content-Type'].startswith('text/csv')
text = csv_response.data.decode('utf-8')
print('csv header:', text.splitlines()[0])
assert 'measured_pixels' in text and 'bridge_weak_pixels' in text

# Validation: missing reference box must be a clean 400.
bad = client.post(f'/api/projects/{project_id}/layup/patches',
                  json={'state': created['defaults'], 'deviation': 20})
assert bad.status_code == 400 and 'reference' in bad.get_json()['error']
print('validation 400:', bad.get_json()['error'])

# Validation: reference box over an unsupported area.
off = client.post(f'/api/projects/{project_id}/layup/patches',
                  json={**payload, 'reference_box': {'x': 0, 'y': 0, 'width': 4, 'height': 4}})
print('corner box status', off.status_code, off.get_json())

print('\nE2E OK')
