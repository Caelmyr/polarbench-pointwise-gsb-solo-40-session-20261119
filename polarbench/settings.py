from __future__ import annotations

from dataclasses import dataclass

from .calibration import Calibration
from .domain import ANGLES, Box, DisplayRange, InputError, bool_value, enum_value, integer, number, object_value
from .quality import BY_ISSUE
from .registration import Registration
from .stokes import Limits


@dataclass(frozen=True)
class Layer:
    id: str
    label: str
    unit: str
    palette: str
    minimum: float
    maximum: float
    description: str

    def as_dict(self):
        return dict(self.__dict__)


LAYERS = {
    item.id: item for item in (
        Layer('intensity', 'Total intensity', 'Signal', 'gray', 0, 1.45, 'Mean total signal across the opposing analyzer pairs.'),
        Layer('degree', 'Linear degree', 'Ratio', 'viridis', 0, 0.75, 'Polarized intensity divided by total intensity.'),
        Layer('direction', 'Polarization direction', 'Degrees', 'axial', 0, 180, 'Cyclic 180° orientation relative to the current reference axis.'),
        Layer('polarized', 'Polarized intensity', 'Signal', 'magma', 0, 0.8, 'Length of the linear Stokes vector.'),
        Layer('analyzer', 'Virtual analyzer', 'Signal', 'gray', 0, 0.9, 'Reconstruction through a single user-selected linear analyzer.'),
        Layer('glare-min', 'Minimum analyzer signal', 'Signal', 'gray', 0, 0.65, 'Per-pixel lower theoretical analyzer bound.'),
        Layer('glare-max', 'Maximum analyzer signal', 'Signal', 'gray', 0, 0.95, 'Per-pixel upper theoretical analyzer bound.'),
        Layer('q', 'Stokes Q', 'Signal', 'balance', -0.7, 0.7, 'Opposing horizontal and vertical components after reference rotation.'),
        Layer('u', 'Stokes U', 'Signal', 'balance', -0.7, 0.7, 'Opposing diagonal components after reference rotation.'),
        Layer('pair-error', 'Pair closure ratio', 'Ratio', 'magma', 0, 0.24, 'Relative mismatch between opposite-angle intensity sums.'),
        Layer('pair-difference', 'Signed pair disagreement', 'Signal', 'balance', -0.35, 0.35, 'I0 + I90 minus I45 + I135.'),
        Layer('quality', 'Quality evidence', 'Category', 'flags', 0, 1, 'Common coverage and measurement confidence evidence.'),
        Layer('raw-0', 'Raw 0°', 'Full scale', 'gray', 0, 1, 'Original analyzer samples in native coordinates.'),
        Layer('raw-45', 'Raw 45°', 'Full scale', 'gray', 0, 1, 'Original analyzer samples in native coordinates.'),
        Layer('raw-90', 'Raw 90°', 'Full scale', 'gray', 0, 1, 'Original analyzer samples in native coordinates.'),
        Layer('raw-135', 'Raw 135°', 'Full scale', 'gray', 0, 1, 'Original analyzer samples in native coordinates.'),
        Layer('aligned-0', 'Aligned 0°', 'Signal', 'gray', 0, 0.95, 'Registered analyzer plane after intensity correction.'),
        Layer('aligned-45', 'Aligned 45°', 'Signal', 'gray', 0, 0.95, 'Registered analyzer plane after intensity correction.'),
        Layer('aligned-90', 'Aligned 90°', 'Signal', 'gray', 0, 0.95, 'Registered analyzer plane after intensity correction.'),
        Layer('aligned-135', 'Aligned 135°', 'Signal', 'gray', 0, 0.95, 'Registered analyzer plane after intensity correction.'),
    )
}
SCALAR_LAYERS = tuple(layer for layer in LAYERS if layer != 'quality')
PALETTES = ('gray', 'viridis', 'magma', 'ice', 'balance', 'axial')


@dataclass(frozen=True)
class Processing:
    calibration: Calibration
    registration: Registration
    limits: Limits
    zero_angle: float

    @classmethod
    def parse(cls, raw, references):
        data = object_value(raw, 'Processing settings')
        return cls(
            Calibration.parse(data.get('calibration', {}), references),
            Registration.parse(data.get('registration', {})),
            Limits.parse(data.get('limits', {})),
            number(data.get('zero_angle', 0), 'Reference direction', 0, 180) % 180,
        )

    def as_dict(self):
        return {
            'calibration': self.calibration.as_dict(),
            'registration': self.registration.as_dict(),
            'limits': self.limits.as_dict(),
            'zero_angle': self.zero_angle,
        }


@dataclass(frozen=True)
class View:
    layer: str
    display: DisplayRange
    palette: str
    analyzer_angle: float
    gamma: float
    invalid_style: str
    vectors: bool
    vector_spacing: int
    quality_focus: str | None
    crop: Box | None
    max_side: int
    grid: bool

    @classmethod
    def parse(cls, raw, width, height):
        data = object_value(raw, 'Viewer settings')
        layer = enum_value(data.get('layer', 'degree'), LAYERS, 'Map')
        spec = LAYERS[layer]
        limits = DisplayRange(spec.minimum, spec.maximum)
        if data.get('display') is not None:
            limits = DisplayRange.parse(data['display'])
        palette = data.get('palette', spec.palette)
        if layer in ('quality', 'direction'):
            palette = spec.palette
        else:
            enum_value(palette, PALETTES, 'Palette')
        focus = data.get('quality_focus')
        if focus is not None:
            enum_value(focus, BY_ISSUE, 'Quality focus')
        return cls(
            layer=layer,
            display=limits,
            palette=palette,
            analyzer_angle=number(data.get('analyzer_angle', 0), 'Virtual analyzer angle', 0, 180) % 180,
            gamma=number(data.get('gamma', 1), 'Display gamma', 0.35, 3),
            invalid_style=enum_value(data.get('invalid_style', 'transparent'), ('transparent', 'checker', 'black'), 'Invalid area style'),
            vectors=bool_value(data.get('vectors', False), 'Vector overlay'),
            vector_spacing=integer(data.get('vector_spacing', 30), 'Vector spacing', 12, 150),
            quality_focus=focus,
            crop=Box.parse(data['crop'], width, height) if data.get('crop') else None,
            max_side=integer(data.get('max_side', 1280), 'Preview edge', 80, 3200),
            grid=bool_value(data.get('grid', False), 'Detector grid'),
        )

    def as_dict(self):
        return {
            'layer': self.layer,
            'display': self.display.as_dict(),
            'palette': self.palette,
            'analyzer_angle': self.analyzer_angle,
            'gamma': self.gamma,
            'invalid_style': self.invalid_style,
            'vectors': self.vectors,
            'vector_spacing': self.vector_spacing,
            'quality_focus': self.quality_focus,
            'crop': self.crop.as_dict() if self.crop else None,
            'max_side': self.max_side,
            'grid': self.grid,
        }


def default_settings():
    config = Processing.parse({}, {})
    return {'processing': config.as_dict(), 'view': View.parse({}, 8, 8).as_dict()}


def parse_settings(data, record):
    raw = object_value(data, 'Inspection state')
    for field in set(raw) - {'processing', 'view'}:
        raise InputError(f'The inspection state has an unknown field ({field}).')
    capture = record['capture']
    previous = record['defaults']
    return (
        Processing.parse(raw.get('processing', previous['processing']), record['references']),
        View.parse(raw.get('view', previous['view']), capture['width'], capture['height']),
    )


def processing_schema():
    return {
        'layer_options': [layer.as_dict() for layer in LAYERS.values()],
        'scalar_layers': list(SCALAR_LAYERS),
        'palettes': list(PALETTES),
        'angles': list(ANGLES),
        'defaults': default_settings(),
    }
