from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass

import numpy as np

from .domain import Box, ConflictError, InputError, MAX_REGIONS, MissingError, Point, enum_value, identifier, label, list_value, note, now, object_value
from .store import new_id

SHAPES = ('box', 'ellipse', 'polygon')
REGION_COLORS = ('#70d8bc', '#f1b275', '#87acf7', '#e49dbd', '#c6a1f2', '#d8d583')


@dataclass(frozen=True)
class Geometry:
    shape: str
    box: Box
    points: tuple[Point, ...] = ()

    @classmethod
    def parse(cls, raw, width, height):
        data = object_value(raw, 'Region geometry')
        shape = enum_value(data.get('shape', 'box'), SHAPES, 'Region shape')
        if shape != 'polygon':
            box = Box.parse(data.get('box', data), width, height)
            return cls(shape, box)
        raw_points = list_value(data.get('points'), 'Vertices', 40)
        points = tuple(Point.parse(item, f'Vertex {i + 1}', width, height) for i, item in enumerate(raw_points))
        if len(points) < 3:
            raise InputError('A polygon needs at least three vertices.')
        if points[0] == points[-1]:
            points = points[:-1]
        if len(set(points)) != len(points) or len(points) < 3:
            raise InputError('Polygon vertices must be distinct.')
        _validate_polygon(points)
        xs, ys = [p.x for p in points], [p.y for p in points]
        x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
        y0, y1 = math.floor(min(ys)), math.ceil(max(ys))
        result = cls(shape, Box(x0, y0, x1 - x0 + 1, y1 - y0 + 1), points)
        if not result.local_mask().any():
            raise InputError('The polygon contains no detector-pixel centers.')
        return result

    def local_mask(self):
        if self.shape == 'box':
            return np.ones((self.box.height, self.box.width), dtype=bool)
        y, x = np.ogrid[self.box.y:self.box.bottom, self.box.x:self.box.right]
        if self.shape == 'ellipse':
            cx = self.box.x + (self.box.width - 1) / 2
            cy = self.box.y + (self.box.height - 1) / 2
            return ((x - cx) / (self.box.width / 2)) ** 2 + ((y - cy) / (self.box.height / 2)) ** 2 <= 1
        return _polygon_mask(x, y, self.points)

    def full_mask(self, shape):
        blank = np.zeros(shape, dtype=bool)
        blank[self.box.slices] = self.local_mask()
        return blank

    def as_dict(self):
        data = {'shape': self.shape, 'box': self.box.as_dict()}
        if self.shape == 'polygon':
            data['points'] = [point.as_dict() for point in self.points]
        return data


def _polygon_mask(x, y, points):
    inside = np.zeros((y.shape[0], x.shape[1]), dtype=bool)
    edges = np.zeros_like(inside)
    for start, end in zip(points, (*points[1:], points[0])):
        dx, dy = end.x - start.x, end.y - start.y
        area = (x - start.x) * dy - (y - start.y) * dx
        on_segment = (
            (x >= min(start.x, end.x)) & (x <= max(start.x, end.x))
            & (y >= min(start.y, end.y)) & (y <= max(start.y, end.y))
        )
        edges |= (np.abs(area) <= 1e-7) & on_segment
        if dy != 0:
            crossed = (start.y > y) != (end.y > y)
            intercept = start.x + (y - start.y) * dx / dy
            inside ^= crossed & (x < intercept)
    return inside | edges


def _direction(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _within(a, b, p):
    return min(a.x, b.x) - 1e-10 <= p.x <= max(a.x, b.x) + 1e-10 and min(a.y, b.y) - 1e-10 <= p.y <= max(a.y, b.y) + 1e-10


def _intersects(a, b, c, d):
    one, two = _direction(a, b, c), _direction(a, b, d)
    three, four = _direction(c, d, a), _direction(c, d, b)
    if one * two < 0 and three * four < 0:
        return True
    return any(abs(v) <= 1e-10 and _within(e1, e2, p) for v, e1, e2, p in (
        (one, a, b, c), (two, a, b, d), (three, c, d, a), (four, c, d, b),
    ))


def _validate_polygon(points):
    edge_pairs = list(zip(points, (*points[1:], points[0])))
    shoelace = sum(a.x * b.y - b.x * a.y for a, b in edge_pairs)
    if abs(shoelace) <= 1e-7:
        raise InputError('Polygon needs a nonzero area.')
    for j, (a, b) in enumerate(edge_pairs):
        for k in range(j + 2, len(edge_pairs)):
            if k == len(edge_pairs) - 1 and j == 0:
                continue
            c, d = edge_pairs[k]
            if _intersects(a, b, c, d):
                raise InputError('Polygon edges must not cross.')


def region_item(raw, width, height, existing=None, index=0):
    data = object_value(raw, 'Region')
    prev = existing or {}
    geometry = Geometry.parse(data.get('geometry', prev.get('geometry')), width, height)
    return {
        'id': prev.get('id') or new_id(),
        'name': label(data.get('name', prev.get('name')), 'Region name'),
        'note': note(data.get('note', prev.get('note', '')), 'Region note'),
        'color': enum_value(data.get('color', prev.get('color', REGION_COLORS[index % len(REGION_COLORS)])), REGION_COLORS, 'Region color'),
        'geometry': geometry.as_dict(),
        'created_at': prev.get('created_at') or now(),
    }


def get_region(record, region_id):
    identifier(region_id, 'Region')
    for region in record['regions']:
        if region['id'] == region_id:
            return region
    raise MissingError('Region could not be found.')


def select_region(record, request, width, height):
    region_id, geometry = request.get('region_id'), request.get('geometry')
    if region_id and geometry:
        raise InputError('Choose a saved region or an unsaved detector selection.')
    if region_id:
        item = get_region(record, region_id)
        return Geometry.parse(item['geometry'], width, height), item['name']
    if geometry:
        return Geometry.parse(geometry, width, height), 'Selection'
    return Geometry('box', Box.entire(width, height)), 'Entire detector'


def add_region(record, data):
    saved = record['regions']
    if len(saved) >= MAX_REGIONS:
        raise ConflictError(f'A capture can hold at most {MAX_REGIONS} saved regions.')
    capture = record['capture']
    item = region_item(data, capture['width'], capture['height'], index=len(saved))
    saved.append(item)
    return item


def change_region(record, region_id, data):
    existing = get_region(record, region_id)
    capture = record['capture']
    item = region_item(data, capture['width'], capture['height'], existing)
    at = record['regions'].index(existing)
    record['regions'][at] = item
    return item


def remove_region(record, region_id):
    record['regions'].remove(get_region(record, region_id))


def csv_text(columns, rows):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(columns)
    for row in rows:
        sanitized = [
            "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
            for value in row
        ]
        writer.writerow(sanitized)
    return output.getvalue().encode('utf-8')
