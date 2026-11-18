from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass

from . import calibration, demo, ingest, quality, registration, stokes
from .codec import raw_fraction, read_upload
from .domain import ANGLES, Box, InputError, integer, label, list_value, note, object_value
from .regions import Geometry, get_region, region_item
from .settings import Processing, default_settings, parse_settings
from .store import ProjectStore, fingerprint, source_identity

PIXEL_CACHE_BUDGET = 1_200_000


@dataclass(frozen=True)
class Inspection:
    decoded: ingest.DecodedCapture
    prepared: dict[int, calibration.CalibratedPlane]
    references: calibration.ReferenceMaps
    aligned: dict[int, registration.AlignedPlane]
    common: Box
    field: stokes.PolarField
    quality: quality.QualityField
    processing: Processing

    def channel(self, layer, azimuth=0):
        if layer.startswith('raw-'):
            plane = self.decoded.planes[int(layer.split('-')[-1])]
            return raw_fraction(plane), plane.valid
        if layer.startswith('aligned-'):
            plane = self.aligned[int(layer.split('-')[-1])]
            return plane.data.values, plane.valid
        return self.field.channel(layer, azimuth=azimuth)


class Workbench:
    def __init__(self, store: ProjectStore):
        self.store = store
        self._cache = OrderedDict()
        self._cache_lock = threading.RLock()
        self._inflight = weakref.WeakValueDictionary()

    @staticmethod
    def _memo(data):
        return note(data.get('note', ''), 'Capture note')

    def create_demo(self, payload):
        data = object_value(payload, 'Demo capture')
        key = data.get('demo', 'coated-panel')
        spec = demo.scene_spec(key)
        name = label(data.get('name', spec.name), 'Capture name')
        if self.store.list():
            current = {item['name'] for item in self.store.list()}
            proposed, occur = name, 2
            while proposed in current:
                suffix = f' ({occur})'
                proposed = name[:100 - len(suffix)] + suffix
                occur += 1
            name = proposed
        prepared = demo.demo_payload(key)
        prepared.metadata['demo'] = key
        prepared.metadata['synthetic'] = True
        regions = [region_item({'name': n, 'geometry': box}, spec.width, spec.height, index=i)
                   for i, (n, box) in enumerate(demo.sample_regions(spec))]
        return self.store.create(name=name, memo=f'Synthetic demonstration. {spec.description}', capture=prepared.metadata,
                                 payloads=prepared.payloads, defaults=default_settings(), mask=prepared.mask, regions=regions)

    def import_capture(self, form, files):
        name = label(form.get('name'), 'Capture name')
        memo = self._memo(form)
        source = ingest.files_from_form(form, files)
        return self.store.create(name=name, memo=memo, capture=source.metadata, payloads=source.payloads,
                                 defaults=default_settings(), mask=source.mask)

    def rename(self, project_id, payload):
        data = object_value(payload, 'Capture details')
        def update(record):
            record['name'] = label(data.get('name', record['name']), 'Capture name')
            record['note'] = note(data.get('note', record['note']), 'Capture note')
        return self.store.update(project_id, update, expected_revision=revision_of(data))

    def replace_mask(self, project_id, form, files):
        record = self.store.get(project_id)
        capture = record['capture']
        proposed = read_upload(files.get('file'), 'Mask')
        item, bytes_ = ingest.prepare_mask(proposed, (capture['height'], capture['width']))
        def assign(current, image_item):
            current['mask'] = image_item
        return self.store.replace_image(project_id, item, bytes_, assign, expected_revision=revision_of(form))

    def remove_mask(self, project_id, raw):
        def clear(record):
            record['mask'] = None
        return self.store.update(project_id, clear, expected_revision=revision_of(raw))

    def install_reference(self, project_id, key, file, box=None, expected_revision=None):
        record = self.store.get(project_id)
        shape = record['capture']['height'], record['capture']['width']
        item, payload = ingest.prepare_reference(file, shape, key)
        entry = {'box': None}
        if box is not None:
            if key != 'flat':
                raise InputError('The detector normalization box applies to the flat reference only.')
            entry['box'] = Box.parse(box, shape[1], shape[0]).as_dict()
        def assign(current, image):
            current['references'][key] = dict(entry, file=image)
        return self.store.replace_image(project_id, item, payload, assign, expected_revision=expected_revision)

    def use_demo_reference(self, project_id, key, raw):
        capture = self.store.get(project_id)['capture']
        if not capture.get('demo'):
            raise InputError('Example references apply to their matching synthetic capture only.')
        file = demo.demo_reference(capture['demo'], key)
        return self.install_reference(project_id, key, file, raw.get('box'), revision_of(raw))

    def remove_reference(self, project_id, key, raw):
        if key not in ingest.REFERENCE_FIELDS:
            raise InputError('The reference must be dark or flat.')
        def clear(record):
            record['defaults']['processing']['calibration'][f'use_{key}'] = False
            record['references'].pop(key, None)
        return self.store.update(project_id, clear, expected_revision=revision_of(raw))

    def set_reference_box(self, project_id, raw):
        data = object_value(raw, 'Flat normalization box')
        def edit(record):
            ref = record['references'].get('flat')
            if not ref:
                raise InputError('Upload a flat reference first.')
            dims = record['capture']
            box = Box.parse(data['box'], dims['width'], dims['height']) if data.get('box') is not None else None
            if box and box.area < 16:
                raise InputError('The flat normalization box must cover at least 16 detector pixels.')
            ref['box'] = box.as_dict() if box else None
        return self.store.update(project_id, edit, expected_revision=revision_of(data))

    def save_defaults(self, project_id, payload):
        raw = object_value(payload, 'Default view settings')
        def update(record):
            processing, view = parse_settings(raw.get('state', {}), record)
            # Construct the current field once so invalid reference combinations fail at the save boundary.
            self.inspect(record, processing)
            record['defaults'] = {'processing': processing.as_dict(), 'view': view.as_dict()}
        return self.store.update(project_id, update, expected_revision=revision_of(raw))

    def pending(self, record, state):
        options = state if state is not None else {}
        processing, view = parse_settings(options, record)
        return self.inspect(record, processing), view

    def _get_cached(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def inspect(self, record, processing):
        key = (record['id'], fingerprint(source_identity(record)), fingerprint(processing.as_dict()))
        with self._cache_lock:
            cached = self._get_cached(key)
            if cached is not None:
                return cached
            lock = self._inflight.setdefault(key, threading.Lock())
        with lock:
            with self._cache_lock:
                cached = self._get_cached(key)
                if cached is not None:
                    return cached
            inspection = self._evaluate(record, processing)
            pixels = inspection.field.s0.size
            if pixels <= PIXEL_CACHE_BUDGET:
                with self._cache_lock:
                    self._cache[key] = inspection
                    total = sum(case.field.s0.size for case in self._cache.values())
                    while total > PIXEL_CACHE_BUDGET or len(self._cache) > 6:
                        _, removed = self._cache.popitem(last=False)
                        total -= removed.field.s0.size
            return inspection

    def _evaluate(self, record, processing):
        decoded = ingest.decode(self.store, record, processing.limits.mask_threshold)
        refs = calibration.reference_maps(decoded, record, processing.calibration)
        corrected = {
            angle: calibration.prepare_channel(decoded.planes[angle], processing.calibration.tuning[angle], refs, processing.limits.saturation)
            for angle in ANGLES
        }
        aligned, common = registration.align_planes(corrected, processing.registration)
        field = stokes.reconstruct(aligned, common, decoded.mask, processing.limits, processing.zero_angle)
        report = quality.audit(aligned, common, decoded.mask, field, processing.limits)
        return Inspection(decoded, corrected, refs, aligned, common, field, report, processing)

    def registration_suggestions(self, record, raw):
        results, _ = self.pending(record, raw.get('state'))
        if raw.get('regions') is not None:
            windows = [self._registration_window(record, region_id, results.decoded) for region_id in _region_pair(raw)]
            return registration.suggest_registration_pair(results.prepared, windows, raw)
        return registration.suggest_registration(results.prepared, raw, results.decoded.width, results.decoded.height)

    @staticmethod
    def _registration_window(record, region_id, decoded):
        item = get_region(record, region_id)
        box = Geometry.parse(item['geometry'], decoded.width, decoded.height).box
        if min(box.width, box.height) < 16:
            raise InputError(f"Saved region '{item['name']}' is smaller than the 16 × 16 registration window minimum.")
        return item['name'], box

    def diagnostics(self, record, state):
        results, view = self.pending(record, state)
        readings = {}
        for angle in ANGLES:
            readings[str(angle)] = calibration.channel_evidence(results.decoded.planes[angle], results.prepared[angle])
        return {
            'input': ingest.describe_inputs(record),
            'common_footprint': results.common.as_dict(),
            'channel_footprints': results.quality.footprints,
            'quality': quality.summary(results.quality),
            'reference_maps': results.references.summary(),
            'analyzer_channels': readings,
            'processing': results.processing.as_dict(),
            'view': view.as_dict(),
        }

    def evict_capture(self, project_id):
        with self._cache_lock:
            keys = [key for key in self._cache if key[0] == project_id]
            for key in keys:
                self._cache.pop(key)


def _region_pair(raw):
    ids = list_value(raw.get('regions'), 'Registration regions', 2)
    if len(ids) != 2 or ids[0] == ids[1]:
        raise InputError('Choose two different saved regions to combine shift evidence.')
    return ids


def revision_of(data):
    rev = data.get('revision')
    return None if rev is None or rev == '' else integer(rev, 'Capture revision', 1, 2**31 - 1)
