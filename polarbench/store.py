from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
from pathlib import Path

from .domain import ConflictError, InputError, MAX_PROJECTS, MissingError, identifier, label, note, now

SCHEMA = 1
PROJECT_QUOTA = 160 * 1024 * 1024
MIN_DISK_AVAILABLE = 32 * 1024 * 1024
_RECORD = 'project.json'
_BLOB_NAME = re.compile(r'([0-9a-f]{64})\.(png|tif)')


def new_id():
    return secrets.token_hex(6)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _atomic_bytes(path: Path, payload: bytes):
    temp = path.with_name(f'.{path.name}.{new_id()}.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _source_items(record):
    capture = record['capture']
    yield from capture['sources'].values()
    if record.get('mask'):
        yield record['mask']
    for reference in record.get('references', {}).values():
        yield reference['file']


def public_record(record):
    exported = copy.deepcopy(record)
    for item in _source_items(exported):
        item.pop('blob', None)
    from .sessions import session_list
    exported['sessions'] = session_list(record)
    return exported


def source_identity(record):
    capture = record['capture']
    return {
        'type': capture['type'],
        'size': [capture['width'], capture['height']],
        'sources': {key: item['sha256'] for key, item in capture['sources'].items()},
        'mosaic_layout': capture.get('mosaic_layout'),
        'mask': (record['mask']['sha256'] if record.get('mask') else None),
        'references': {
            key: {'sha256': reference['file']['sha256'], 'box': reference.get('box')}
            for key, reference in record.get('references', {}).items()
        },
    }


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.projects = self.root / 'projects'
        self.projects.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _project_dir(self, project_id):
        return self.projects / identifier(project_id, 'Capture')

    def _record_unlocked(self, project_id):
        record_file = self._project_dir(project_id) / _RECORD
        try:
            content = json.loads(record_file.read_text('utf-8'))
        except FileNotFoundError as error:
            raise MissingError('Capture could not be found.') from error
        except (OSError, json.JSONDecodeError) as error:
            raise ConflictError('Capture metadata could not be read.') from error
        if content.get('schema') != SCHEMA or content.get('id') != project_id:
            raise ConflictError('Capture metadata uses an unsupported format.')
        return content

    def get(self, project_id):
        with self._lock:
            return self._record_unlocked(project_id)

    def _dirs(self):
        return (p for p in self.projects.iterdir() if p.is_dir() and (p / _RECORD).is_file())

    def list(self):
        with self._lock:
            objects = []
            for project in self._dirs():
                try:
                    record = self._record_unlocked(project.name)
                except (MissingError, ConflictError):
                    continue
                objects.append({
                    'id': record['id'],
                    'name': record['name'],
                    'updated_at': record['updated_at'],
                    'type': record['capture']['type'],
                    'width': record['capture']['width'],
                    'height': record['capture']['height'],
                    'demo': record['capture'].get('demo'),
                    'region_count': len(record['regions']),
                    'session_count': len(record['sessions']),
                })
        return sorted(objects, key=lambda item: (item['updated_at'], item['id']), reverse=True)

    def _check_disk(self, additional=0):
        if shutil.disk_usage(self.root).free < MIN_DISK_AVAILABLE + additional:
            raise ConflictError('There is insufficient data-volume space to store this capture safely.')

    @staticmethod
    def _project_bytes(directory):
        return sum(item.stat().st_size for item in (directory / 'source').iterdir() if item.is_file())

    @staticmethod
    def _blob_name(item):
        extension = 'tif' if item['format'] == 'TIFF' else 'png'
        return f'{item["sha256"]}.{extension}'

    def _write_blob(self, folder, item, payload):
        if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ConflictError('Input bytes and source metadata do not match.')
        filename = self._blob_name(item)
        path = folder / filename
        if not path.exists():
            _atomic_bytes(path, payload)
        return dict(item, blob=filename)

    def create(self, *, name, memo, capture, payloads, defaults, regions=(), mask=None, profiles=(), references=None, scale=None):
        name = label(name, 'Capture name')
        memo = note(memo, 'Capture note')
        capture = copy.deepcopy(capture)
        # Dimensions and format were validated by ingest; the store checks disk and commits one record.
        additional = sum(map(len, payloads.values()))
        if mask is not None:
            additional += len(mask[1])
        references = references or {}
        additional += sum(len(data) for _, data, _ in references.values())
        if additional > PROJECT_QUOTA:
            raise InputError('Capture exceeds the project source-storage budget.')
        with self._lock:
            if sum(1 for _ in self._dirs()) >= MAX_PROJECTS:
                raise ConflictError(f'The workspace already has {MAX_PROJECTS} captures.')
            self._check_disk(additional)
            project_id = new_id()
            while self._project_dir(project_id).exists():
                project_id = new_id()
            temporary = self.projects / f'.stage-{project_id}'
            final = self._project_dir(project_id)
            stamp = now()
            record = {
                'schema': SCHEMA,
                'id': project_id,
                'name': name,
                'note': memo,
                'created_at': stamp,
                'updated_at': stamp,
                'revision': 1,
                'capture': capture,
                'defaults': copy.deepcopy(defaults),
                'regions': list(regions),
                'profiles': list(profiles),
                'sessions': [],
                'mask': None,
                'references': {},
                'scale': scale,
            }
            try:
                blobs = temporary / 'source'
                blobs.mkdir(parents=True)
                for field, item in capture['sources'].items():
                    capture['sources'][field] = self._write_blob(blobs, item, payloads[field])
                if mask is not None:
                    record['mask'] = self._write_blob(blobs, *mask)
                for key, (item, data, box) in references.items():
                    record['references'][key] = {'file': self._write_blob(blobs, item, data), 'box': box}
                _atomic_bytes(temporary / _RECORD, json_bytes(record))
                temporary.rename(final)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return record

    def source_bytes(self, project_id, item):
        filename = item.get('blob')
        if not isinstance(filename, str) or not _BLOB_NAME.fullmatch(filename):
            raise ConflictError('Invalid source-image storage reference.')
        if filename != self._blob_name(item):
            raise ConflictError('Source-image storage reference does not match its identity.')
        path = self._project_dir(project_id) / 'source' / filename
        try:
            return path.read_bytes()
        except FileNotFoundError as error:
            raise ConflictError('An image referenced by the capture is missing from storage.') from error

    def _store_record(self, project_id, record):
        record['revision'] += 1
        record['updated_at'] = now()
        _atomic_bytes(self._project_dir(project_id) / _RECORD, json_bytes(record))
        return record

    def update(self, project_id, editor, *, expected_revision=None):
        with self._lock:
            record = self._record_unlocked(project_id)
            self._require_revision(record, expected_revision)
            editor(record)
            return self._store_record(project_id, record)

    @staticmethod
    def _require_revision(record, expected_revision):
        if expected_revision is not None and record['revision'] != expected_revision:
            raise ConflictError('The capture changed in another tab. Refresh before saving this edit.')

    def replace_image(self, project_id, item, payload, editor, *, expected_revision=None):
        with self._lock:
            record = self._record_unlocked(project_id)
            self._require_revision(record, expected_revision)
            directory = self._project_dir(project_id)
            blob = directory / 'source' / self._blob_name(item)
            exists = blob.is_file()
            if not exists:
                additional = len(payload)
                if self._project_bytes(directory) + additional > PROJECT_QUOTA:
                    raise ConflictError('This capture has reached its source-image storage budget.')
                self._check_disk(additional)
            try:
                image_item = self._write_blob(blob.parent, item, payload)
                editor(record, image_item)
                return self._store_record(project_id, record)
            except Exception:
                if not exists:
                    with contextlib.suppress(OSError):
                        blob.unlink(missing_ok=True)
                raise

    def delete(self, project_id, *, expected_revision=None):
        with self._lock:
            record = self._record_unlocked(project_id)
            self._require_revision(record, expected_revision)
            directory = self._project_dir(project_id)
            removed = self.projects / f'.removed-{project_id}-{new_id()}'
            directory.rename(removed)
            shutil.rmtree(removed)
