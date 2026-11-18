from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import ImageFont

from .domain import InputError, enum_value, number, object_value
from .profiles import ScanLine
from .regions import select_region

UNITS = ('mm', 'cm', 'um', 'in')
UNIT_LABELS = {'mm': 'mm', 'cm': 'cm', 'um': 'µm', 'in': 'in'}


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
