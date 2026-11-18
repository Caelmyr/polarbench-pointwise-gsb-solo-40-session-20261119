from __future__ import annotations

import json

from flask import Blueprint, request

from . import capture_zip, ingest, profiles, regions, sessions
from .codec import read_upload
from .domain import InputError
from .http_support import app_bench, request_object, return_project, send_bytes, send_download, send_image, store
from .render import raster
from .settings import processing_schema
from .workbench import revision_of

api = Blueprint('captures', __name__, url_prefix='/api')


@api.get('/health')
def health():
    return {'status': 'ok', 'service': 'polarbench'}


@api.get('/configuration')
def get_config():
    from . import demo, quality
    return dict(processing_schema(), demos=demo.scenes(), mosaic_patterns=ingest.MOSAIC_PRESETS,
                quality_legend=quality.legends(), region_colors=regions.REGION_COLORS)


@api.get('/projects')
def list_projects():
    return {'projects': store().list()}


@api.get('/projects/<project_id>')
def get_project(project_id):
    return return_project(store().get(project_id))


@api.post('/projects/demo')
def create_demo():
    return return_project(app_bench().create_demo(request_object('Demo capture', optional=True)), 201)


@api.post('/projects')
def import_project():
    return return_project(app_bench().import_capture(request.form, request.files), 201)


@api.post('/mosaic-calibration')
def calibrate_mosaic_order():
    from .mosaic_solve import solve_from_request
    return solve_from_request(request.form, request.files)


@api.patch('/projects/<project_id>')
def edit_project(project_id):
    return return_project(app_bench().rename(project_id, request_object('Capture details')))


@api.delete('/projects/<project_id>')
def delete_project(project_id):
    bench = app_bench()
    bench.store.delete(project_id, expected_revision=revision_of(request_object(optional=True)))
    bench.evict_capture(project_id)
    return '', 204


@api.get('/projects/<project_id>/sources/<field>')
def download_source(project_id, field):
    payload, item = ingest.source_download(store(), store().get(project_id), field)
    mime = 'image/tiff' if item['format'] == 'TIFF' else 'image/png'
    return send_bytes(payload, mime=mime, name=item['name'])


@api.get('/projects/<project_id>/sources')
def list_sources(project_id):
    return ingest.describe_inputs(store().get(project_id))


@api.get('/projects/<project_id>/source-package')
def pack_source(project_id):
    return send_download(capture_zip.source_package(store(), store().get(project_id)))


@api.post('/projects/source-package')
def unpack_source():
    packet = capture_zip.decode_archive(request.files.get('package'), request.form.get('name'))
    source = packet.capture
    updated = store().create(name=packet.name, memo=packet.note, capture=source.metadata, payloads=source.payloads,
                             defaults=packet.settings, regions=packet.regions, mask=source.mask, profiles=packet.profiles,
                             references=packet.references, scale=packet.scale)
    return return_project(updated, 201)


@api.put('/projects/<project_id>/mask')
def upload_mask(project_id):
    return return_project(app_bench().replace_mask(project_id, request.form, request.files))


@api.delete('/projects/<project_id>/mask')
def clear_mask(project_id):
    return return_project(app_bench().remove_mask(project_id, request_object(optional=True)))


@api.put('/projects/<project_id>/references/<kind>')
def upload_reference(project_id, kind):
    box = request.form.get('box')
    if box:
        try:
            box = json.loads(box)
        except json.JSONDecodeError as error:
            raise InputError('Flat normalization coordinates must be a JSON object.') from error
    filename = read_upload(request.files.get('file'), f'{kind} reference')
    changed = app_bench().install_reference(project_id, kind, filename, box, revision_of(request.form))
    return return_project(changed)


@api.post('/projects/<project_id>/references/<kind>/demo')
def install_sample_reference(project_id, kind):
    updated = app_bench().use_demo_reference(project_id, kind, request_object(optional=True))
    return return_project(updated)


@api.delete('/projects/<project_id>/references/<kind>')
def remove_reference(project_id, kind):
    return return_project(app_bench().remove_reference(project_id, kind, request_object(optional=True)))


@api.patch('/projects/<project_id>/references/flat')
def update_flat_box(project_id):
    return return_project(app_bench().set_reference_box(project_id, request_object()))


@api.put('/projects/<project_id>/default-view')
def update_default_settings(project_id):
    return return_project(app_bench().save_defaults(project_id, request_object()))


def _edit_object(project_id, function, *identifiers, optional=False):
    data = request_object(optional=optional)
    expected = revision_of(data)
    captured = None
    def update(record):
        nonlocal captured
        captured = function(record, *identifiers, data)
    current = store().update(project_id, update, expected_revision=expected)
    return current, captured


@api.post('/projects/<project_id>/regions')
def add_region(project_id):
    record, item = _edit_object(project_id, regions.add_region)
    return {'region': item, 'revision': record['revision']}, 201


@api.patch('/projects/<project_id>/regions/<region_id>')
def edit_region(project_id, region_id):
    record, item = _edit_object(project_id, regions.change_region, region_id)
    return {'region': item, 'revision': record['revision']}


@api.delete('/projects/<project_id>/regions/<region_id>')
def delete_region(project_id, region_id):
    record, _ = _edit_object(project_id, lambda item, ident, data: regions.remove_region(item, ident), region_id, optional=True)
    return {'revision': record['revision']}


@api.post('/projects/<project_id>/profiles')
def save_scan_line(project_id):
    record, item = _edit_object(project_id, profiles.add_profile)
    return {'profile': item, 'revision': record['revision']}, 201


@api.delete('/projects/<project_id>/profiles/<profile_id>')
def delete_scan_line(project_id, profile_id):
    record, _ = _edit_object(project_id, lambda item, ident, data: profiles.remove_profile(item, ident), profile_id, optional=True)
    return {'revision': record['revision']}


@api.get('/projects/<project_id>/views')
def list_views(project_id):
    return {'views': sessions.session_list(store().get(project_id))}


@api.post('/projects/<project_id>/views')
def save_view(project_id):
    data = request_object('Saved inspection view')
    bench = app_bench()
    saved = None
    def edit(record):
        nonlocal saved
        bench.pending(record, data.get('state'))
        saved = sessions.add_session(record, data)
    record = bench.store.update(project_id, edit, expected_revision=revision_of(data))
    current = sessions.session_overview(saved, saved['source_identity'])
    return {'view': current, 'revision': record['revision']}, 201


@api.patch('/projects/<project_id>/views/<view_id>')
def edit_view(project_id, view_id):
    record, item = _edit_object(project_id, sessions.rename_session, view_id)
    return {'revision': record['revision'], 'view': next(v for v in sessions.session_list(record) if v['id'] == item['id'])}


@api.delete('/projects/<project_id>/views/<view_id>')
def delete_view(project_id, view_id):
    record, _ = _edit_object(project_id, lambda rec, ident, data: sessions.remove_session(rec, ident), view_id, optional=True)
    return {'revision': record['revision']}


@api.post('/projects/<project_id>/views/<view_id>/apply')
def apply_view(project_id, view_id):
    record, _ = _edit_object(project_id, lambda rec, ident, data: sessions.apply_saved_view(rec, ident), view_id, optional=True)
    return return_project(record)


@api.get('/projects/<project_id>/views/<view_id>/preview')
def revisit_view(project_id, view_id):
    bench = app_bench()
    archived, settings = sessions.replay(bench.store.get(project_id), view_id)
    results, view = bench.pending(archived, settings)
    return send_image(raster(results, view).payload)


@api.get('/projects/<project_id>/views/<view_id>/evidence')
def saved_evidence(project_id, view_id):
    from .artifacts import review_zip
    bench = app_bench()
    archived, settings = sessions.replay(bench.store.get(project_id), view_id)
    results, view = bench.pending(archived, settings)
    return send_download(review_zip(results, archived, view, {}))
