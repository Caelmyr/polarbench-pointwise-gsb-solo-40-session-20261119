from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

ANGLES = (0, 45, 90, 135)
MAX_SIDE = 3200
MAX_PIXELS = 5_000_000
MAX_UPLOAD = 14 * 1024 * 1024
MAX_LABEL = 100
MAX_REGIONS = 24
MAX_SESSIONS = 24
MAX_PROJECTS = 40
IDENTIFIER = re.compile(r'[0-9a-f]{12}')
CHANNEL_FIELD = {angle: f'i{angle}' for angle in ANGLES}


class InputError(ValueError):
    pass


class MissingError(LookupError):
    pass


class ConflictError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def object_value(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise InputError(f'{name} must be a JSON object.')
    return value


def list_value(value, name: str, maximum: int) -> list:
    if not isinstance(value, list) or len(value) > maximum:
        raise InputError(f'{name} requires a list with at most {maximum} entries.')
    return value


def enum_value(value, options, name: str):
    if not isinstance(value, (str, int, float)) or value not in options:
        expected = ', '.join(map(str, options))
        raise InputError(f'{name} must be one of {expected}.')
    return value


def number(value, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise InputError(f'{name} must be a number from {low} to {high}.')
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError(f'{name} must be a number from {low} to {high}.') from error
    if not math.isfinite(num) or not low <= num <= high:
        raise InputError(f'{name} must be a number from {low} to {high}.')
    return num


def integer(value, name: str, low: int, high: int) -> int:
    num = number(value, name, low, high)
    if int(num) != num:
        raise InputError(f'{name} must be a whole number.')
    return int(num)


def label(value, name: str, *, limit: int = MAX_LABEL) -> str:
    if not isinstance(value, str):
        raise InputError(f'{name} must be text.')
    cleaned = value.strip()
    if not cleaned or len(cleaned) > limit or any(ord(char) < 32 for char in cleaned):
        raise InputError(f'{name} needs 1 to {limit} visible characters.')
    return cleaned


def note(value, name: str = 'Note') -> str:
    if not isinstance(value, str) or len(value) > 2000:
        raise InputError(f'{name} must be text of at most 2000 characters.')
    if any(ord(char) < 32 and char not in '\r\n\t' for char in value):
        raise InputError(f'{name} contains a control character.')
    return value.strip()


def identifier(value: str, item: str = 'Record') -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise MissingError(f'{item} could not be found.')
    return value


def check_shape(height: int, width: int) -> None:
    if min(height, width) < 8:
        raise InputError('Image dimensions must be at least 8 pixels in both directions.')
    if max(height, width) > MAX_SIDE or height * width > MAX_PIXELS:
        raise InputError(f'Images may contain up to {MAX_PIXELS:,} pixels and {MAX_SIDE}px per side.')


def bool_value(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise InputError(f'{name} must be true or false.')
    return value


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def parse(cls, raw, name: str, width: int, height: int):
        data = object_value(raw, name)
        return cls(
            number(data.get('x'), f'{name} x', 0, width - 1),
            number(data.get('y'), f'{name} y', 0, height - 1),
        )

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def entire(cls, width: int, height: int):
        return cls(0, 0, width, height)

    @classmethod
    def parse(cls, raw, width: int, height: int):
        data = object_value(raw, 'Region')
        x = integer(data.get('x'), 'Left edge', 0, width - 1)
        y = integer(data.get('y'), 'Top edge', 0, height - 1)
        return cls(
            x, y,
            integer(data.get('width'), 'Region width', 1, width - x),
            integer(data.get('height'), 'Region height', 1, height - y),
        )

    @property
    def area(self):
        return self.width * self.height

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def slices(self):
        return slice(self.y, self.bottom), slice(self.x, self.right)

    def contains(self, x, y):
        return self.x <= x < self.right and self.y <= y < self.bottom

    def intersection(self, other):
        left, right = max(self.x, other.x), min(self.right, other.right)
        top, bottom = max(self.y, other.y), min(self.bottom, other.bottom)
        if left >= right or top >= bottom:
            return None
        return Box(left, top, right - left, bottom - top)

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Offset:
    dx: int = 0
    dy: int = 0

    @classmethod
    def parse(cls, raw, limit: int = 160):
        data = object_value(raw, 'Registration offset')
        return cls(
            integer(data.get('dx', 0), 'Horizontal registration offset', -limit, limit),
            integer(data.get('dy', 0), 'Vertical registration offset', -limit, limit),
        )

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DisplayRange:
    low: float
    high: float

    @classmethod
    def parse(cls, raw, name='Display range', extent=(-8.0, 8.0)):
        data = object_value(raw, name)
        low = number(data.get('low'), f'{name} low', *extent)
        high = number(data.get('high'), f'{name} high', *extent)
        if low >= high:
            raise InputError(f'{name} low must be smaller than high.')
        return cls(low, high)

    def as_dict(self):
        return dataclasses.asdict(self)


def per_angle(raw, name, parser):
    data = object_value(raw, name)
    extra = set(data) - {str(angle) for angle in ANGLES}
    if extra:
        raise InputError(f'{name} has an unrecognized analyzer angle.')
    return {angle: parser(data.get(str(angle)), angle) for angle in ANGLES}


def as_angle_dict(values):
    return {str(angle): values[angle] for angle in ANGLES}
