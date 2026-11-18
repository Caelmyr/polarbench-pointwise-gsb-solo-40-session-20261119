from __future__ import annotations

import dataclasses
import io
import zipfile

import numpy as np

from .artifacts import Download, _zip_entry, safe_stem
from .comparison import compare_legend, comparison_png
from .codec import gray_tiff
from .domain import Box, integer, object_value
from .regions import csv_text
from .store import fingerprint, json_bytes, source_identity


def grid_comparison(compared, raw):
    data = object_value(raw, 'Comparison evidence')
    view = compared.config.box
    tile = integer(data.get('tile', 40), 'Comparison cell size', 8, 300)
    cells = []
    for row, y in enumerate(range(0, view.height, tile)):
        height = min(tile, view.height - y)
        for column, x in enumerate(range(0, view.width, tile)):
            width = min(tile, view.width - x)
            cell = Box(x, y, width, height)
            section = cell.slices
            left, right = compared.reference_valid[section], compared.current_valid[section]
            both = left & right
            values = compared.delta[section][both]
            cells.append({
                'column': column, 'row': row,
                'box': Box(x + view.x, y + view.y, width, height).as_dict(),
                'shared': int(values.size),
                'only_reference': int(np.count_nonzero(left & ~right)),
                'only_current': int(np.count_nonzero(~left & right)),
                'median_change': float(np.median(values)) if values.size else None,
                'threshold_count': int(np.count_nonzero(np.abs(values) >= compared.config.threshold)),
            })
    return cells


def _csv(rows):
    return csv_text(('column', 'row', 'detector_x', 'detector_y', 'width', 'height', 'shared', 'only_reference', 'only_current', 'median_change', 'threshold_count'), (
        (r['column'], r['row'], *[r['box'][k] for k in ('x', 'y', 'width', 'height')],
         r['shared'], r['only_reference'], r['only_current'], r['median_change'], r['threshold_count']) for r in rows
    ))


def comparison_evidence(compared, current, reference, processing, reference_processing, settings):
    support = compared.shared
    points = np.where(support, compared.delta, np.nan)
    manifest = {
        'format': 'polarbench-comparison-1',
        'current_source': fingerprint(source_identity(current)),
        'reference_source': fingerprint(source_identity(reference)),
        'current_processing': processing.as_dict(),
        'reference_processing': reference_processing.as_dict(),
        'summary': compared.summary(),
        'legend': compare_legend(compared),
        'difference_sign': 'current minus reference',
    }
    package = io.BytesIO()
    with zipfile.ZipFile(package, 'w', compresslevel=4) as output:
        _zip_entry(output, 'comparison.json', json_bytes(manifest))
        _zip_entry(output, 'comparison.png', comparison_png(compared), stored=True)
        diagnostic = dataclasses.replace(compared, config=dataclasses.replace(compared.config, map_style='support'))
        _zip_entry(output, 'support.png', comparison_png(diagnostic), stored=True)
        _zip_entry(output, 'difference-linear.tif', gray_tiff(points), stored=True)
        _zip_entry(output, 'detector-cells.csv', _csv(grid_comparison(compared, settings)))
    return Download(package.getvalue(), f'{safe_stem(current["name"])}-comparison.zip', 'application/zip')
