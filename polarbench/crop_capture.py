from __future__ import annotations

import copy
import io

import numpy as np
from PIL import Image

from .calibration import reference_maps
from .codec import read_image
from .domain import ANGLES, Box, InputError, label, note, object_value
from .ingest import DecodedCapture, prepare_mosaic, prepare_planes, prepare_reference, source_download
from .profiles import add_profile
from .regions import Geometry, region_item
from .registration import Registration, footprints
from .scale import SpatialScale
from .settings import parse_settings


def _native_crop(payload, box):
    plane = read_image(payload)
    selection = box.slices
    pixels = plane.values[selection]
    support = plane.valid[selection]
    if plane.bit_depth == 16:
        if not support.all():
            raise InputError('16-bit source pixels with incomplete alpha need a separate mask before creating a cropped capture.')
        image = Image.fromarray(pixels.astype(np.uint16))
    elif support.all():
        image = Image.fromarray(pixels.astype(np.uint8))
    else:
        alpha = np.where(support, 255, 0).astype(np.uint8)
        image = Image.fromarray(np.dstack((pixels.astype(np.uint8), alpha)))
    output = io.BytesIO()
    image.save(output, 'PNG', compress_level=3)
    return output.getvalue()


def _clip_source(store, record, field, box):
    payload, source = source_download(store, record, field)
    stem = source['name'].rsplit('.', 1)[0]
    return _native_crop(payload, box), f'{stem}-crop.png'


def _local_box(source, crop):
    return Box(source.x - crop.x, source.y - crop.y, source.width, source.height)


def _crop_regions(record, crop):
    width, height = record['capture']['width'], record['capture']['height']
    copied = []
    for item in record['regions']:
        current = Geometry.parse(item['geometry'], width, height)
        if current.box.intersection(crop) != current.box:
            continue
        translated = current.as_dict()
        translated['box'] = _local_box(current.box, crop).as_dict()
        if current.shape == 'polygon':
            translated['points'] = [{'x': p.x - crop.x, 'y': p.y - crop.y} for p in current.points]
        copied.append(region_item(dict(item, geometry=translated), crop.width, crop.height))
    return copied


def _crop_lines(record, crop, metadata):
    out = {'profiles': [], 'capture': metadata}
    for item in record['profiles']:
        coords = item['geometry']
        if not all(crop.x <= point['x'] <= crop.right - 1 and crop.y <= point['y'] <= crop.bottom - 1
                   for point in coords.values()):
            continue
        moved = {key: {'x': point['x'] - crop.x, 'y': point['y'] - crop.y} for key, point in coords.items()}
        add_profile(out, dict(item, geometry=moved))
    return out['profiles']


def _common_area(config, crop):
    allowable = Box.entire(crop.width, crop.height)
    for offset in config.offsets.values():
        overlap = footprints(crop.height, crop.width, offset)
        if overlap is None:
            return None
        allowable = allowable.intersection(overlap[1])
        if allowable is None:
            return None
    return allowable


def validate_crop(record, raw):
    data = object_value(raw, 'New cropped capture')
    cap = record['capture']
    crop = Box.parse(data.get('box'), cap['width'], cap['height'])
    if min(crop.width, crop.height) < 8:
        raise InputError('A cropped capture needs at least 8 × 8 detector pixels.')
    registration = Registration.parse(record['defaults']['processing']['registration'])
    common = _common_area(registration, crop)
    if common is None:
        raise InputError('The current stored registration leaves no shared overlap inside that crop.')
    relative = _local_box(crop, crop)
    if cap['type'] == 'mosaic':
        raw_window = Box(2 * crop.x, 2 * crop.y, 2 * crop.width, 2 * crop.height)
    else:
        raw_window = crop
    return {
        'detector_window': crop.as_dict(),
        'source_window': raw_window.as_dict(),
        'expected_overlap': common.as_dict(),
        'size': relative.as_dict(),
        'retained_regions': len(_crop_regions(record, crop)),
    }


def _native_crop_capture(store, record, crop, plan):
    capture = record['capture']
    mask = _clip_source(store, record, 'mask', crop) if record['mask'] else None
    if capture['type'] == 'mosaic':
        technical = plan['source_window']
        moved = _clip_source(store, record, 'mosaic', Box(**technical))
        new = prepare_mosaic(moved, capture['mosaic_layout'], mask)
    else:
        files = {a: _clip_source(store, record, f'i{a}', crop) for a in ANGLES}
        new = prepare_planes(files, mask)
    if capture.get('synthetic'):
        new.metadata['synthetic'] = True
    return new


def _cropped_references(store, record, crop):
    refs = {}
    for key, item in record['references'].items():
        data = _clip_source(store, record, key, crop)
        matrix, payload = prepare_reference(data, (crop.height, crop.width), key)
        window = item.get('box')
        translated = None
        if window:
            overlap = Box(**window).intersection(crop)
            if overlap and overlap.area >= 16:
                translated = _local_box(overlap, crop).as_dict()
        refs[key] = matrix, payload, translated
    return refs


def make_crop(store, record, raw):
    data = object_value(raw, 'New cropped capture')
    proposed = label(data.get('name', f'{record["name"][:90]} crop'), 'Cropped capture name')
    memo = note(data.get('note', f'Detector crop of {record["name"]}.'), 'Capture note')
    plan = validate_crop(record, data)
    crop = Box(**plan['detector_window'])
    prepared = _native_crop_capture(store, record, crop, plan)
    derived = copy.deepcopy(record['defaults'])
    derived['view']['crop'] = None
    refs = _cropped_references(store, record, crop)
    descriptions = {key: {'file': item, 'box': box} for key, (item, payload, box) in refs.items()}
    record_copy = {'defaults': derived, 'references': descriptions, 'capture': prepared.metadata}
    process, view = parse_settings({}, record_copy)
    if process.calibration.use_dark or process.calibration.use_flat:
        images = {key: read_image(payload) for key, (item, payload, box) in refs.items()}
        decoded = DecodedCapture({}, None, images.get('dark'), images.get('flat'))
        reference_maps(decoded, record_copy, process.calibration)
    previous_scale = record.get('scale')
    copied_scale = None
    if previous_scale:
        physical = SpatialScale.from_record(previous_scale)
        copied_scale = SpatialScale(physical.unit, physical.pitch_x, physical.pitch_y, 'pitch').as_dict()
    prepared.metadata['source_detector_window'] = crop.as_dict()
    return store.create(
        name=proposed, memo=memo, capture=prepared.metadata, payloads=prepared.payloads,
        defaults={'processing': process.as_dict(), 'view': view.as_dict()}, mask=prepared.mask,
        regions=_crop_regions(record, crop), profiles=_crop_lines(record, crop, prepared.metadata), references=refs, scale=copied_scale,
    )
