from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .domain import ANGLES, ConflictError, InputError, MissingError, Point, enum_value, identifier, integer, label, note, now, object_value
from .quality import one_plane_coverage
from .regions import csv_text
from .settings import SCALAR_LAYERS
from .stokes import axial_mean
from .store import new_id

MAX_PROFILES = 16
MAX_SAMPLES = 1024


@dataclass(frozen=True)
class ScanLine:
    start: Point
    end: Point

    @classmethod
    def parse(cls, raw, width, height):
        data = object_value(raw, 'Scan line')
        start = Point.parse(data.get('start'), 'Start', width, height)
        end = Point.parse(data.get('end'), 'End', width, height)
        if math.hypot(end.x - start.x, end.y - start.y) < 1:
            raise InputError('The scan line needs to span at least one detector pixel.')
        return cls(start, end)

    @property
    def length(self):
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)

    def sample(self, count):
        t = np.linspace(0, 1, count, dtype=np.float64)
        return self.start.x + t * (self.end.x - self.start.x), self.start.y + t * (self.end.y - self.start.y), t * self.length

    def as_dict(self):
        return dataclasses.asdict(self)


def get_profile(record, profile_id):
    identifier(profile_id, 'Scan line')
    for item in record['profiles']:
        if item['id'] == profile_id:
            return item
    raise MissingError('Scan line could not be found.')


def add_profile(record, raw):
    if len(record['profiles']) >= MAX_PROFILES:
        raise ConflictError(f'Remove an existing line before adding another. The limit is {MAX_PROFILES}.')
    request = object_value(raw, 'Save scan line')
    cap = record['capture']
    geometry = ScanLine.parse(request.get('geometry'), cap['width'], cap['height'])
    item = {
        'id': new_id(),
        'name': label(request.get('name'), 'Scan line name'),
        'geometry': geometry.as_dict(),
        'note': note(request.get('note', '')),
        'created_at': now(),
    }
    record['profiles'].append(item)
    return item


def remove_profile(record, profile_id):
    record['profiles'].remove(get_profile(record, profile_id))


def _neighborhood(x, y, width, height):
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(width - 1, x0 + 1), np.minimum(height - 1, y0 + 1)
    dx, dy = x - x0, y - y0
    return (
        (x0, y0, (1 - dx) * (1 - dy)),
        (x1, y0, dx * (1 - dy)),
        (x0, y1, (1 - dx) * dy),
        (x1, y1, dx * dy),
    )


def _interp(value, support, neighborhood):
    n = neighborhood[0][0].shape[0]
    weighted = np.zeros(n, dtype=np.float64)
    usable = np.ones(n, dtype=bool)
    for x, y, w in neighborhood:
        weighted += value[y, x] * w
        usable &= (w <= 1e-10) | support[y, x]
    return weighted, usable


def _interpolate_axis(field, neighborhood):
    # Interpolate components before taking an axis to preserve the wrap seam.
    q, support = _interp(field.q, field.oriented, neighborhood)
    u, _ = _interp(field.u, field.oriented, neighborhood)
    norm = np.hypot(q, u)
    support &= norm > 1e-7
    return (np.rad2deg(np.arctan2(u, q)) * 0.5) % 180, support


def _sample_series(results, x, y, layer, analyzer, interpolation):
    arr, support = results.channel(layer, analyzer)
    height, width = arr.shape
    if interpolation == 'nearest':
        pos_x, pos_y = np.minimum(width - 1, np.floor(x + 0.5).astype(int)), np.minimum(height - 1, np.floor(y + 0.5).astype(int))
        return arr[pos_y, pos_x], support[pos_y, pos_x]
    neighborhood = _neighborhood(x, y, width, height)
    if layer == 'direction':
        return _interpolate_axis(results.field, neighborhood)
    return _interp(arr, support, neighborhood)


def _gaps(kept):
    indexes = np.flatnonzero(np.diff(np.r_[False, ~kept, False]))
    return [{'first': int(start), 'last': int(end - 1), 'samples': int(end - start)} for start, end in indexes.reshape((-1, 2))]


def scan(results, record, raw, view):
    data = object_value(raw, 'Scan line analysis')
    line_id, coordinates = data.get('profile_id'), data.get('geometry')
    if line_id and coordinates:
        raise InputError('Choose a saved line or draw a new line.')
    saved = get_profile(record, line_id) if line_id else None
    if saved:
        coordinates = saved['geometry']
    height, width = results.field.shape
    line = ScanLine.parse(coordinates, width, height)
    mode = enum_value(data.get('interpolation', 'bilinear'), ('nearest', 'bilinear'), 'Sampling method')
    default_layer = view.layer if view.layer != 'quality' else 'intensity'
    layer = enum_value(data.get('layer', default_layer), SCALAR_LAYERS, 'Scan line layer')
    count = integer(data.get('samples', min(MAX_SAMPLES, math.ceil(line.length) + 1)), 'Scan samples', 2, MAX_SAMPLES)
    x, y, distance = line.sample(count)
    values, valid = _sample_series(results, x, y, layer, view.analyzer_angle, mode)
    return {
        'name': saved['name'] if saved else 'Scan line',
        'geometry': line.as_dict(),
        'length': line.length,
        'interpolation': mode,
        'layer': layer,
        'distance': distance.tolist(),
        'position': np.stack((x, y), axis=1).tolist(),
        'value': [float(value) if keep else None for value, keep in zip(values, valid)],
        'valid_samples': int(valid.sum()),
        'gaps': _gaps(valid),
    }


def scan_csv(data):
    return csv_text(('x', 'y', 'distance_px', data['layer'], 'usable'), (
        (pos[0], pos[1], t, value, value is not None)
        for pos, t, value in zip(data['position'], data['distance'], data['value'])
    ))


def _raw_evidence(results, x, y):
    raw_coords = {}
    native_pattern = results.decoded.sensor_positions
    for angle in ANGLES:
        image = results.decoded.planes[angle]
        offset = results.processing.registration.offsets[angle]
        sx, sy = x - offset.dx, y - offset.dy
        if 0 <= sx < image.width and 0 <= sy < image.height:
            sample = {'x': sx, 'y': sy, 'code': int(image.values[sy, sx]), 'bits': image.bit_depth}
            sample['native_alpha'] = bool(image.valid[sy, sx])
            if native_pattern is not None:
                row, column = native_pattern[angle]
                sample['sensor_x'] = 2 * sx + column
                sample['sensor_y'] = 2 * sy + row
            raw_coords[str(angle)] = sample
        else:
            raw_coords[str(angle)] = None
    return raw_coords


def probe(results, raw, view):
    request = object_value(raw, 'Pixel probe')
    height, width = results.field.shape
    x = integer(request.get('x'), 'Detector x', 0, width - 1)
    y = integer(request.get('y'), 'Detector y', 0, height - 1)
    field = results.field
    point = (y, x)
    visible = None
    if view.layer != 'quality':
        plane, support = results.channel(view.layer, view.analyzer_angle)
        if support[point]:
            visible = float(plane[point])
    masks = {
        'detector_mask_included': bool(results.decoded.mask[y, x]) if results.decoded.mask is not None else True,
        'common_footprint_included': results.common.contains(x, y),
        'all_analyzers_available': bool(field.base_support[point]),
    }
    recovery_info = None
    if results.recovery is not None:
        report = results.recovery
        recovery_info = {
            'short_capture_name': report.short_name,
            'exposure_ratio': report.config.exposure_ratio,
            'recovered': bool(report.recovered[point]),
            'agreement_conflict': bool(report.conflicts[point]),
            'angle_source': {str(angle): report.angle_source(angle, y, x) for angle in ANGLES},
        }
    return {
        'x': x,
        'y': y,
        'coverage': masks,
        'raw_coordinates': 'sensor-parity' if results.decoded.sensor_positions is not None else 'direct-plane',
        'status': results.quality.issues_at(x, y),
        'shown_value': visible,
        'layer': view.layer,
        'passed_s0': bool(field.measured[point]),
        'passed_direction': bool(field.oriented[point]),
        'recovery': recovery_info,
        'combined': {
            'intensity': float(field.s0[point]) if field.measured[point] else None,
            'degree': float(field.degree[point]) if field.measured[point] else None,
            'q': float(field.q[point]) if field.measured[point] else None,
            'u': float(field.u[point]) if field.measured[point] else None,
            'direction': axial_mean(field.q[point], field.u[point]) if field.oriented[point] else None,
            'pair_error': float(field.pair_error[point]) if field.base_support[point] else None,
        },
        'aligned_planes': one_plane_coverage(results.aligned, x, y),
        'raw_planes': _raw_evidence(results, x, y),
    }
