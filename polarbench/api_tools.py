from __future__ import annotations

from flask import Blueprint

from . import crop_capture, inspection_sheet, markers, pair_audit, scale
from .domain import enum_value
from .http_support import loaded_case, request_object, return_project, send_csv, send_download, send_image, store
from .workbench import revision_of

api = Blueprint('tools', __name__, url_prefix='/api/projects/<project_id>')


@api.post('/crop-capture')
def crop_capture_preview(project_id):
    case = loaded_case(project_id, 'Cropped capture')
    mode = enum_value(case.options.get('mode', 'preview'), ('preview', 'create'), 'Crop operation')
    if mode == 'preview':
        return crop_capture.validate_crop(case.record, case.options)
    return return_project(crop_capture.make_crop(case.bench.store, case.record, case.options), 201)


@api.get('/scale')
def get_scale(project_id):
    return scale.spatial_settings(store().get(project_id))


@api.put('/scale')
def set_scale(project_id):
    request = request_object('Detector scale')
    updated = store().update(project_id, lambda record: scale.update_scale(record, request), expected_revision=revision_of(request))
    return return_project(updated)


@api.delete('/scale')
def clear_scale(project_id):
    request = request_object(optional=True)
    updated = store().update(project_id, scale.clear_scale, expected_revision=revision_of(request))
    return return_project(updated)


@api.post('/spatial-measurement')
def spatial_measurement(project_id):
    case = loaded_case(project_id, 'Spatial measurement')
    return scale.measure_extent(case.record, case.options)


@api.post('/worksheet')
def worksheet(project_id):
    case = loaded_case(project_id, 'Inspection worksheet')
    result, view = case.results()
    return send_download(inspection_sheet.make_sheet(result, case.record, view, case.options))


@api.post('/pair-evidence')
def pair_evidence(project_id):
    case = loaded_case(project_id, 'Opposite analyzer pairs')
    result, _ = case.results()
    product = pair_audit.pair_audit(result, case.options)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Pair evidence output')
    if output == 'png':
        return send_image(pair_audit.audit_image(product))
    evidence = pair_audit.pair_summary(product)
    if output == 'csv':
        return send_csv(pair_audit.pair_histogram_csv(evidence), 'pair-evidence.csv')
    return evidence


@api.post('/signal-markers')
def locate_signals(project_id):
    case = loaded_case(project_id, 'Signal markers')
    result, view = case.results()
    report = markers.locate_markers(result, case.record, case.options, view)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Marker output')
    if output == 'png':
        return send_image(markers.marker_image(result, view, report))
    if output == 'csv':
        return send_csv(markers.markers_csv(report), 'detector-signal-markers.csv')
    return report


@api.post('/mosaic-review')
def raw_mosaic_review(project_id):
    from .mosaic_review import describe_mosaic, mosaic_study, preview_mosaic
    case = loaded_case(project_id, 'Native mosaic layout')
    study = mosaic_study(case.bench.store, case.record, case.options)
    if case.options.get('output') == 'png':
        return send_image(preview_mosaic(study))
    return describe_mosaic(study)
