from __future__ import annotations

from dataclasses import dataclass

from .codec import ImagePlane, input_packet, read_image, read_mask, read_upload, slice_plane
from .domain import ANGLES, CHANNEL_FIELD, InputError, check_shape, enum_value

MOSAIC_PRESETS = {
    '0-45-135-90': (0, 45, 135, 90),
    '90-45-135-0': (90, 45, 135, 0),
    '90-135-45-0': (90, 135, 45, 0),
    '45-0-90-135': (45, 0, 90, 135),
    '0-135-45-90': (0, 135, 45, 90),
}
REFERENCE_FIELDS = ('dark', 'flat')


@dataclass(frozen=True)
class PreparedCapture:
    metadata: dict
    payloads: dict[str, bytes]
    mask: tuple[dict, bytes] | None = None

    @property
    def shape(self):
        return self.metadata['height'], self.metadata['width']


@dataclass(frozen=True)
class DecodedCapture:
    planes: dict[int, ImagePlane]
    mask: object | None
    dark: ImagePlane | None
    flat: ImagePlane | None
    sensor_positions: dict[int, tuple[int, int]] | None = None

    @property
    def shape(self):
        return self.planes[0].shape

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]


def _named_image(value):
    payload, filename = value
    plane = read_image(payload)
    return plane, input_packet(plane, name=filename, byte_count=len(payload)), payload


def _same_canvas(planes):
    shape = planes[0].shape
    for angle in ANGLES:
        if planes[angle].shape != shape:
            raise InputError(f'{angle}° needs the same detector canvas as 0°. Use aligned-size images.')
    return shape


def prepare_planes(source_files: dict[int, tuple[bytes, str]], mask_file=None):
    if set(source_files) != set(ANGLES):
        raise InputError('Provide one source plane each for 0°, 45°, 90° and 135°.')
    planes, files, payloads = {}, {}, {}
    for angle in ANGLES:
        field = CHANNEL_FIELD[angle]
        planes[angle], files[field], payloads[field] = _named_image(source_files[angle])
    height, width = _same_canvas(planes)
    metadata = {
        'type': 'planes',
        'width': width,
        'height': height,
        'sources': files,
        'plane_metadata': {str(angle): planes[angle].metadata() for angle in ANGLES},
    }
    mask = prepare_mask(mask_file, (height, width)) if mask_file else None
    return PreparedCapture(metadata, payloads, mask)


def mosaic_channels(plane: ImagePlane, order: tuple[int, int, int, int]):
    if len(order) != 4 or set(order) != set(ANGLES):
        raise InputError('A detector tile must assign 0°, 45°, 90° and 135° once each.')
    height, width = plane.shape
    if width % 2 or height % 2:
        raise InputError('A micro-polarizer mosaic requires even width and height with complete 2 × 2 cells.')
    check_shape(height // 2, width // 2)
    return {angle: slice_plane(plane, position // 2, position % 2) for position, angle in enumerate(order)}


def prepare_mosaic(source_file, order_key, mask_file=None):
    enum_value(order_key, MOSAIC_PRESETS, 'Mosaic pattern')
    order = MOSAIC_PRESETS[order_key]
    plane, info, payload = _named_image(source_file)
    channels = mosaic_channels(plane, order)
    height, width = channels[0].shape
    metadata = {
        'type': 'mosaic',
        'width': width,
        'height': height,
        'mosaic_layout': order_key,
        'mosaic_positions': [list(divmod(order.index(angle), 2)) for angle in ANGLES],
        'sensor_size': [plane.width, plane.height],
        'sources': {'mosaic': info},
        'plane_metadata': {str(angle): channels[angle].metadata() for angle in ANGLES},
    }
    mask = prepare_mask(mask_file, (height, width)) if mask_file else None
    return PreparedCapture(metadata, {'mosaic': payload}, mask)


def files_from_form(form, files):
    kind = enum_value(form.get('kind', 'planes'), ('planes', 'mosaic'), 'Capture input type')
    mask = read_upload(files['mask'], 'Inspection mask') if files.get('mask') else None
    if kind == 'planes':
        planes = {angle: read_upload(files.get(CHANNEL_FIELD[angle]), f'{angle}° analyzer') for angle in ANGLES}
        return prepare_planes(planes, mask)
    source = read_upload(files.get('mosaic'), 'Detector mosaic')
    return prepare_mosaic(source, form.get('mosaic_layout', '0-45-135-90'), mask)


def prepare_mask(file, shape):
    payload, filename = file
    keep, image = read_mask(payload, shape)
    return dict(
        input_packet(image, name=filename, byte_count=len(payload)),
        positive_pixels=int(keep.sum()),
        semantics='positive grayscale with full alpha means inspect',
    ), payload


def prepare_reference(file, shape, key):
    enum_value(key, REFERENCE_FIELDS, 'Reference type')
    image, item, payload = _named_image(file)
    if image.shape != shape:
        raise InputError(f'{key.title()} map needs the detector canvas size {shape[1]} × {shape[0]}.')
    if not image.valid.any():
        raise InputError(f'{key.title()} map has no fully covered pixels.')
    return item, payload


def _source_plane(store, project_id, source):
    payload = store.source_bytes(project_id, source)
    result = read_image(payload)
    if result.sha256 != source['sha256']:
        from .domain import ConflictError
        raise ConflictError('An image differs from its stored source hash.')
    return result


def decode(store, record, mask_threshold=1):
    project_id = record['id']
    capture = record['capture']
    items = capture['sources']
    positions = None
    if capture['type'] == 'planes':
        planes = {angle: _source_plane(store, project_id, items[CHANNEL_FIELD[angle]]) for angle in ANGLES}
    else:
        raw = _source_plane(store, project_id, items['mosaic'])
        tile = MOSAIC_PRESETS[capture['mosaic_layout']]
        positions = {angle: divmod(index, 2) for index, angle in enumerate(tile)}
        planes = mosaic_channels(raw, tile)
    canvas = _same_canvas(planes)
    mask = None
    if record['mask'] is not None:
        payload = store.source_bytes(project_id, record['mask'])
        mask, parsed = read_mask(payload, canvas, mask_threshold)
        if parsed.sha256 != record['mask']['sha256']:
            from .domain import ConflictError
            raise ConflictError('The detector mask differs from its stored source hash.')
    refs = {}
    for tag in REFERENCE_FIELDS:
        ref = record['references'].get(tag)
        refs[tag] = _source_plane(store, project_id, ref['file']) if ref else None
    return DecodedCapture(planes, mask, refs['dark'], refs['flat'], positions)


def source_download(store, record, key):
    if key == 'mask':
        item = record['mask']
    elif key in REFERENCE_FIELDS:
        entry = record['references'].get(key)
        item = entry['file'] if entry else None
    else:
        item = record['capture']['sources'].get(key)
    if item is None:
        from .domain import MissingError
        raise MissingError('That source image is unavailable for this capture.')
    return store.source_bytes(record['id'], item), item


def describe_inputs(record):
    capture = record['capture']
    return {
        'type': capture['type'],
        'detector_canvas': {'width': capture['width'], 'height': capture['height']},
        'sensor_size': capture.get('sensor_size'),
        'mosaic_layout': capture.get('mosaic_layout'),
        'planes': [
            dict(capture['plane_metadata'][str(angle)], angle=angle, field=CHANNEL_FIELD[angle])
            for angle in ANGLES
        ],
        'source_files': [
            {key: value for key, value in dict(item, field=field).items() if key != 'blob'}
            for field, item in capture['sources'].items()
        ],
        'mask': None if not record['mask'] else {
            key: value for key, value in record['mask'].items() if key != 'blob'
        },
        'references': {
            key: {'file': {k: v for k, v in value['file'].items() if k != 'blob'}, 'box': value.get('box')}
            for key, value in record['references'].items()
        },
    }
