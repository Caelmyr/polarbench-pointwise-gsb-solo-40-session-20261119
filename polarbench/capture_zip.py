from __future__ import annotations

import hashlib
import io
import json
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import PurePosixPath

from .artifacts import Download, _zip_entry, safe_stem
from .codec import meaningful_filename
from .domain import ANGLES, CHANNEL_FIELD, MAX_REGIONS, MAX_UPLOAD, ConflictError, InputError, list_value, object_value
from .ingest import prepare_mosaic, prepare_planes, prepare_reference
from .profiles import MAX_PROFILES, add_profile
from .regions import region_item
from .settings import default_settings, parse_settings
from .store import json_bytes

PACKET_FORMAT = 'polarbench-sources-1'
MANIFEST = 'capture.json'
MAX_UNPACKED = 95 * 1024 * 1024
MAX_MANIFEST = 400 * 1024
ALLOWED_FIELDS = {*CHANNEL_FIELD.values(), 'mosaic', 'mask', 'dark', 'flat'}


def _path_for(field, capture_item):
    extension = 'tif' if capture_item['format'] == 'TIFF' else 'png'
    return f'sources/{field}.{extension}'


def _write_input(store, record, packet, output, field, item):
    path = _path_for(field, item)
    content = store.source_bytes(record['id'], item)
    digest = hashlib.sha256(content).hexdigest()
    if digest != item['sha256'] or len(content) != item['bytes']:
        raise ConflictError(f'The {field} source does not match its recorded capture identity.')
    _zip_entry(output, path, content, stored=True)
    packet['files'][field] = {
        'path': path,
        'name': item['name'],
        'bytes': len(content),
        'sha256': digest,
    }


def source_package(store, record):
    source = record['capture']
    packet = {
        'format': PACKET_FORMAT,
        'name': record['name'],
        'note': record['note'],
        'kind': source['type'],
        'mosaic_layout': source.get('mosaic_layout'),
        'synthetic': bool(source.get('synthetic')),
        'files': {},
        'defaults': record['defaults'],
        'regions': record['regions'],
        'profiles': record['profiles'],
        'flat_box': record['references'].get('flat', {}).get('box'),
        'scale': record.get('scale'),
    }
    sources = list(source['sources'].values())
    if record['mask']:
        sources.append(record['mask'])
    sources.extend(item['file'] for item in record['references'].values())
    if sum(item['bytes'] for item in sources) + MAX_MANIFEST > MAX_UNPACKED:
        raise InputError('This capture exceeds the portable package size budget. Download its sources individually.')
    destination = io.BytesIO()
    with zipfile.ZipFile(destination, 'w', compresslevel=4) as output:
        for field, item in source['sources'].items():
            _write_input(store, record, packet, output, field, item)
        if record['mask'] is not None:
            _write_input(store, record, packet, output, 'mask', record['mask'])
        for field, item in record['references'].items():
            _write_input(store, record, packet, output, field, item['file'])
        _zip_entry(output, MANIFEST, json_bytes(packet))
    return Download(destination.getvalue(), f'{safe_stem(record["name"])}-sources.zip', 'application/zip')


def _upload_payload(upload):
    if upload is None:
        raise InputError('Select a PolarBench source package.')
    payload = upload.stream.read(MAX_UNPACKED + 1)
    if not payload:
        raise InputError('The selected source package is empty.')
    if len(payload) > MAX_UNPACKED:
        raise InputError('The source package exceeds the 95 MB import cap.')
    return payload


def _validate_members(archive):
    mapping = {}
    if len(archive.infolist()) > len(ALLOWED_FIELDS) + 1:
        raise InputError('The package has more files than a single capture can contain.')
    for member in archive.infolist():
        name = member.filename
        parsed = PurePosixPath(name)
        if name in mapping or parsed.is_absolute() or any(part in ('..', '.') for part in parsed.parts):
            raise InputError('The source package contains invalid paths or duplicate entries.')
        if (name != MANIFEST and (len(parsed.parts) != 2 or parsed.parts[0] != 'sources' or
                                 parsed.stem not in ALLOWED_FIELDS or parsed.suffix not in ('.png', '.tif'))):
            raise InputError('The source package contains an unexpected file.')
        max_size = MAX_MANIFEST if name == MANIFEST else MAX_UPLOAD
        if member.file_size > max_size or member.flag_bits & 1:
            raise InputError('The source package contains an encrypted or oversized member.')
        mapping[name] = member
    if MANIFEST not in mapping or sum(item.file_size for item in mapping.values()) > MAX_UNPACKED:
        raise InputError('The source package is missing its manifest or exceeds its unpacking budget.')
    return mapping


def _read_sources(archive, manifest, mapping):
    descriptors = object_value(manifest.get('files'), 'Source package file list')
    if set(descriptors) - ALLOWED_FIELDS:
        raise InputError('The package names unknown analyzer or reference files.')
    source_items, read = {}, {MANIFEST}
    for field, item in descriptors.items():
        descriptor = object_value(item, f'Package entry {field}')
        path = descriptor.get('path')
        if not isinstance(path, str) or path not in mapping or PurePosixPath(path).stem != field or path in read:
            raise InputError('Package manifest does not match the stored source paths.')
        payload = archive.read(mapping[path])
        if len(payload) != descriptor.get('bytes') or hashlib.sha256(payload).hexdigest() != descriptor.get('sha256'):
            raise InputError(f'Source package hash does not match the {field} image.')
        filename = meaningful_filename(descriptor.get('name') if isinstance(descriptor.get('name'), str) else None)
        source_items[field] = payload, filename
        read.add(path)
    if set(mapping) != read:
        raise InputError('The source package includes unlisted files.')
    return source_items


def _prepare_capture(manifest, inputs):
    kind = manifest.get('kind')
    if kind == 'planes':
        if not all(field in inputs for field in CHANNEL_FIELD.values()) or 'mosaic' in inputs:
            raise InputError('A plane package requires exactly four analyzer source images.')
        capture = prepare_planes({angle: inputs[CHANNEL_FIELD[angle]] for angle in ANGLES}, inputs.get('mask'))
    elif kind == 'mosaic':
        if any(field in inputs for field in CHANNEL_FIELD.values()) or 'mosaic' not in inputs:
            raise InputError('A mosaic package requires a single detector image.')
        capture = prepare_mosaic(inputs['mosaic'], manifest.get('mosaic_layout'), inputs.get('mask'))
    else:
        raise InputError('The packet uses an unknown detector capture format.')
    if manifest.get('synthetic') is True:
        capture.metadata['synthetic'] = True
    return capture


@dataclass(frozen=True)
class PortableCapture:
    name: str
    note: str
    capture: object
    settings: dict
    regions: list
    profiles: list
    references: dict
    scale: dict | None


def decode_archive(upload, name=None):
    payload = _upload_payload(upload)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            mapping = _validate_members(archive)
            manifest = object_value(json.loads(archive.read(mapping[MANIFEST])), 'Source package manifest')
            if manifest.get('format') != PACKET_FORMAT:
                raise InputError('This archive is not a supported PolarBench source package.')
            inputs = _read_sources(archive, manifest, mapping)
    except (zipfile.BadZipFile, OSError, UnicodeError, ValueError, NotImplementedError, RuntimeError, zlib.error) as error:
        if isinstance(error, InputError):
            raise
        raise InputError('The source archive could not be read.') from error
    capture = _prepare_capture(manifest, inputs)
    height, width = capture.shape
    refs = {key: prepare_reference(inputs[key], capture.shape, key) for key in ('dark', 'flat') if key in inputs}
    flat_box = manifest.get('flat_box')
    if flat_box is not None and 'flat' not in refs:
        raise InputError('A flat-reference box requires an imported flat reference.')
    if flat_box is not None:
        from .domain import Box
        box = Box.parse(flat_box, width, height)
        if box.area < 16:
            raise InputError('The flat normalization box must cover at least 16 detector pixels.')
        flat_box = box.as_dict()
    annotation = [
        region_item(raw, width, height, index=i)
        for i, raw in enumerate(list_value(manifest.get('regions', []), 'Package regions', MAX_REGIONS))
    ]
    scratch = {'profiles': [], 'capture': capture.metadata}
    for raw in list_value(manifest.get('profiles', []), 'Package scan lines', MAX_PROFILES):
        add_profile(scratch, raw)
    record = {'capture': capture.metadata, 'references': refs, 'defaults': default_settings()}
    processing, view = parse_settings(manifest.get('defaults', {}), record)
    scale = manifest.get('scale')
    if scale is not None:
        from .scale import SpatialScale
        raw = dict(object_value(scale, 'Package scale'))
        if raw.get('basis') == 'measured-line':
            witness = object_value(raw.get('witness'), 'Package scale witness')
            raw.update(witness)
        scale = SpatialScale.from_input(raw, width, height).as_dict()
    return PortableCapture(
        name or manifest.get('name'), manifest.get('note', ''), capture,
        {'processing': processing.as_dict(), 'view': view.as_dict()},
        annotation, scratch['profiles'],
        {key: (item, data, flat_box if key == 'flat' else None) for key, (item, data) in refs.items()},
        scale,
    )
