from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .codec import plane_png, rgba_png
from .domain import Box, InputError, enum_value, integer, object_value
from .ingest import prepare_mask
from .regions import select_region

OPERATIONS = ('keep', 'exclude', 'restore', 'intersect')
CHANGE_PALETTE = {
    'retained': (68, 164, 133),
    'removed': (234, 101, 88),
    'restored': (125, 170, 245),
    'excluded': (34, 47, 66),
}


@dataclass(frozen=True)
class MaskEdit:
    before: np.ndarray
    after: np.ndarray
    selection_name: str
    mode: str
    footprint: Box

    def report(self):
        keep = self.after
        prior = self.before
        return {
            'effect': self.mode,
            'selection': self.selection_name,
            'bounds': self.footprint.as_dict(),
            'pixels': int(prior.size),
            'before': int(np.count_nonzero(prior)),
            'after': int(np.count_nonzero(keep)),
            'removed': int(np.count_nonzero(prior & ~keep)),
            'restored': int(np.count_nonzero(~prior & keep)),
            'empty': not bool(np.any(keep)),
        }

    def as_input(self):
        return prepare_mask((plane_png(self.after, 8), 'edited-detector-mask.png'), self.after.shape)


def edit_mask(record, decoded, raw):
    data = object_value(raw, 'Detector mask edit')
    mode = enum_value(data.get('mode', 'exclude'), OPERATIONS, 'Mask edit operation')
    if not data.get('geometry') and not data.get('region_id'):
        raise InputError('Draw a detector region or choose a saved one before editing the mask.')
    geometry, region_name = select_region(record, data, decoded.width, decoded.height)
    before = np.ones(decoded.shape, dtype=bool) if decoded.mask is None else decoded.mask
    chosen = geometry.full_mask(decoded.shape)
    if mode == 'keep':
        after = chosen
    elif mode == 'exclude':
        after = before & ~chosen
    elif mode == 'restore':
        after = before | chosen
    else:
        after = before & chosen
    return MaskEdit(before, after, region_name, mode, geometry.box)


def preview_edit(edit):
    keep, before = edit.after, edit.before
    rgb = np.empty(keep.shape + (3,), dtype=np.uint8)
    rgb[...] = CHANGE_PALETTE['excluded']
    rgb[keep & before] = CHANGE_PALETTE['retained']
    rgb[~keep & before] = CHANGE_PALETTE['removed']
    rgb[keep & ~before] = CHANGE_PALETTE['restored']
    return rgba_png(rgb)


def mask_edit_legend():
    return {'type': 'category', 'entries': [
        {'label': key.title(), 'color': list(color)} for key, color in CHANGE_PALETTE.items()
    ]}


def _hole_report(mask, min_pixels):
    blanks = ~mask
    height, width = mask.shape
    components, count = ndimage.label(blanks, structure=ndimage.generate_binary_structure(2, 1))
    counts = np.bincount(components.ravel())
    border = set(np.unique(np.r_[components[0], components[-1], components[:, 0], components[:, -1]])) - {0}
    ids = np.flatnonzero(counts[1:] >= min_pixels) + 1
    best = sorted(ids, key=lambda idx: (-counts[idx], idx))[:20]
    cuts = ndimage.find_objects(components)
    details = []
    for idx in best:
        ys, xs = cuts[idx - 1]
        crop = Box(xs.start, ys.start, xs.stop - xs.start, ys.stop - ys.start)
        details.append({'pixels': int(counts[idx]), 'box': crop.as_dict(), 'border': idx in border})
    return {
        'excluded': int(np.count_nonzero(blanks)),
        'usable': int(np.count_nonzero(mask)),
        'total': height * width,
        'components': int(count),
        'interior_holes': count - len(border),
        'border_components': len(border),
        'large_exclusions': details,
    }


def inspect_mask(decoded, raw):
    data = object_value(raw, 'Mask evidence')
    mask = decoded.mask if decoded.mask is not None else np.ones(decoded.shape, dtype=bool)
    minimum = integer(data.get('area', 12), 'Minimum excluded patch area', 1, mask.size)
    return dict(_hole_report(mask, minimum), has_uploaded_mask=decoded.mask is not None, minimum_area=minimum)


def mask_inspection_png(decoded, report):
    mask = decoded.mask if decoded.mask is not None else np.ones(decoded.shape, dtype=bool)
    colors = np.where(mask[:, :, None], (54, 133, 118), (29, 36, 56)).astype(np.uint8)
    image = Image.fromarray(colors)
    draw = ImageDraw.Draw(image)
    for i, exclusion in enumerate(report['large_exclusions'], 1):
        box = exclusion['box']
        x, y, w, h = (box[key] for key in ('x', 'y', 'width', 'height'))
        color = (238, 156, 78) if exclusion['border'] else (228, 110, 170)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=2)
        draw.text((x + 2, y + 2), str(i), fill=(251, 246, 234), stroke_width=1, stroke_fill=(19, 25, 34))
    return rgba_png(np.asarray(image))
