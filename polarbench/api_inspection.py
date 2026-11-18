from __future__ import annotations

from flask import Blueprint

from . import artifacts, comparison, detector_bands, field_grid, masking, profiles, quality, reference_review, registration_review, statistics, sweep
from .domain import Box, InputError, enum_value
from .http_support import loaded_case, return_project, send_csv, send_download, send_image
from .render import histogram_chart, legend, profile_chart, raster
from .workbench import revision_of

api = Blueprint('inspection', __name__, url_prefix='/api/projects/<project_id>')


@api.post('/preview')
def preview(project_id):
    case = loaded_case(project_id)
    result, view = case.results()
    output = raster(result, view)
    box = output.viewport
    return send_image(output.payload, **{
        'X-Detector-Viewport': f'{box.x},{box.y},{box.width},{box.height}',
        'X-Rendered-Map': view.layer,
    })


@api.post('/legend')
def get_legend(project_id):
    from .settings import parse_settings
    case = loaded_case(project_id)
    _, view = parse_settings(case.options.get('state', {}), case.record)
    return legend(view)


@api.post('/diagnostics')
def diagnostics(project_id):
    case = loaded_case(project_id)
    return case.bench.diagnostics(case.record, case.options.get('state'))


@api.post('/registration/suggest')
def suggest_registration(project_id):
    case = loaded_case(project_id, 'Registration options')
    return case.bench.registration_suggestions(case.record, case.options)


@api.post('/probe')
def probe(project_id):
    case = loaded_case(project_id, 'Pixel probe')
    results, view = case.results()
    return profiles.probe(results, case.options, view)


@api.post('/measure')
def measure(project_id):
    case = loaded_case(project_id)
    result, view = case.results()
    data = statistics.measure(result, case.record, case.options, view)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Distribution output')
    if output == 'png':
        return send_image(histogram_chart(data))
    if output == 'csv':
        return send_csv(statistics.histogram_csv(data), 'detector-histogram.csv')
    return data


@api.post('/measure/table')
def table(project_id):
    case = loaded_case(project_id)
    results, view = case.results()
    rows = statistics.region_table(results, case.record, view, case.options)
    if case.options.get('output') == 'csv':
        return send_csv(statistics.table_csv(rows), 'detector-regions.csv')
    return {'regions': rows}


@api.post('/scan-line')
def scan(project_id):
    case = loaded_case(project_id)
    results, view = case.results()
    report = profiles.scan(results, case.record, case.options, view)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Scan output')
    if output == 'png':
        return send_image(profile_chart(report))
    if output == 'csv':
        return send_csv(profiles.scan_csv(report), 'detector-scan-line.csv')
    return report


@api.post('/bands')
def bands(project_id):
    case = loaded_case(project_id)
    results, view = case.results()
    report = detector_bands.detect_bands(results, case.record, case.options, view)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Band output')
    if output == 'png':
        return send_image(detector_bands.band_chart(report))
    if output == 'csv':
        return send_csv(detector_bands.bands_csv(report), 'detector-bands.csv')
    return report


@api.post('/orientation-grid')
def spatial_axes(project_id):
    case = loaded_case(project_id)
    results, _ = case.results()
    report = field_grid.orientation_grid(results, case.options)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Orientation output')
    if output == 'png':
        return send_image(field_grid.orientation_grid_png(report, results.processing.zero_angle))
    if output == 'csv':
        return send_csv(field_grid.orientation_csv(report), 'detector-orientation-grid.csv')
    return report


@api.post('/quality')
def inspect_quality(project_id):
    case = loaded_case(project_id)
    results, _ = case.results()
    params = case.options.get('box')
    box = Box.parse(params, results.decoded.width, results.decoded.height) if params is not None else None
    return quality.summary(results.quality, box)


@api.post('/quality/patches')
def image_patches(project_id):
    case = loaded_case(project_id)
    results, _ = case.results()
    from .quality_images import issue_profile, patch_overlay
    from .regions import csv_text
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Quality patch output')
    patches = quality.issue_patches(results.quality, case.options)
    if output == 'png':
        return send_image(patch_overlay(results, patches))
    if output == 'csv':
        axis = enum_value(case.options.get('axis', 'rows'), ('rows', 'columns'), 'Component coordinates')
        return send_csv(csv_text(('patch', 'first', 'after_last', 'pixels', 'touches_window'), issue_profile(patches, axis)),
                        'detector-quality-patches.csv')
    return patches


@api.post('/mask/evidence')
def mask_overview(project_id):
    case = loaded_case(project_id)
    result, _ = case.results()
    report = masking.inspect_mask(result.decoded, case.options)
    if case.options.get('output') == 'png':
        return send_image(masking.mask_inspection_png(result.decoded, report))
    return report


@api.post('/mask/edit')
def edit_mask(project_id):
    case = loaded_case(project_id)
    result, _ = case.results()
    edit = masking.edit_mask(case.record, result.decoded, case.options)
    action = enum_value(case.options.get('action', 'report'), ('report', 'preview', 'apply'), 'Mask action')
    if action == 'preview':
        return send_image(masking.preview_edit(edit))
    if action == 'report':
        return dict(edit.report(), legend=masking.mask_edit_legend())
    item, payload = edit.as_input()
    revision = revision_of(case.options) or case.record['revision']
    def assign(record, image):
        record['mask'] = image
    changed = case.bench.store.replace_image(project_id, item, payload, assign, expected_revision=revision)
    return return_project(changed)


@api.post('/references/review')
def review_reference(project_id):
    case = loaded_case(project_id)
    results, _ = case.results()
    study = reference_review.reference_study(results.decoded, case.record, results.processing, case.options)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'csv'), 'Reference review output')
    if output == 'png':
        return send_image(reference_review.reference_map_png(study, case.options))
    report = reference_review.summarize_reference(study, case.options)
    if output == 'csv':
        return send_csv(reference_review.reference_csv(report), 'detector-reference.csv')
    return report


@api.post('/references/neutral-balance')
def tune_neutral(project_id):
    case = loaded_case(project_id)
    results, _ = case.results()
    return reference_review.suggest_neutral_balance(results, case.record, case.options)


@api.post('/compare')
def compare(project_id):
    case = loaded_case(project_id, 'Capture comparison')
    reference_id = case.options.get('reference_id')
    if reference_id == project_id:
        raise InputError('Choose a different detector capture as reference.')
    reference = case.bench.store.get(reference_id)
    processing, _ = case.results()
    reference_results, _ = case.bench.pending(reference, case.options.get('reference_state'))
    result = comparison.compare_fields(processing, reference_results, case.record, reference, case.options)
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'evidence'), 'Comparison output')
    if output == 'png':
        return send_image(comparison.comparison_png(result))
    if output == 'evidence':
        from .comparison_pack import comparison_evidence
        return send_download(comparison_evidence(result, case.record, reference, processing.processing,
                                                 reference_results.processing, case.options))
    return dict(result.summary(), legend=comparison.compare_legend(result))


@api.post('/download')
def download_inspection(project_id):
    case = loaded_case(project_id)
    results, view = case.results()
    output = enum_value(case.options.get('output', 'png'), ('png', 'data', 'evidence', 'contact-sheet'), 'Download type')
    if output == 'evidence':
        return send_download(artifacts.review_zip(results, case.record, view, case.options))
    if output == 'contact-sheet':
        return send_download(artifacts.contact_sheet(results, case.record, view, case.options))
    return send_download(artifacts.map_download(results, case.record['name'], view, output))


@api.post('/registration/review')
def review_registration(project_id):
    case = loaded_case(project_id)
    results, _ = case.results()
    review = registration_review.registration_view(results, case.options)
    if case.options.get('output') == 'png':
        return send_image(registration_review.evaluation_image(review))
    return registration_review.summarize_evaluation(results, review)


@api.post('/sweep')
def perform_sweep(project_id):
    case = loaded_case(project_id)
    results, view = case.results()
    output = enum_value(case.options.get('output', 'json'), ('json', 'png', 'curve', 'csv'), 'Sweep output')
    if output == 'png':
        return send_image(sweep.sweep_sheet(results, case.record, view, case.options))
    data = sweep.measure_sweep(results, case.record, case.options)
    if output == 'curve':
        return send_image(sweep.sweep_curve(data))
    if output == 'csv':
        return send_csv(sweep.sweep_csv(data), 'detector-angle-sweep.csv')
    return data
