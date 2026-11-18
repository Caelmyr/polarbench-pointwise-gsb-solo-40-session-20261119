from __future__ import annotations

import dataclasses
import hashlib
import io
import re
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import gray_tiff, rgba_png
from .domain import Box, InputError, enum_value, list_value, object_value
from .quality import legends, summary
from .regions import csv_text
from .render import legend, raster
from .settings import LAYERS, SCALAR_LAYERS
from .statistics import region_table, table_csv
from .store import fingerprint, json_bytes, source_identity


@dataclasses.dataclass(frozen=True)
class Download:
    payload: bytes
    name: str
    content_type: str


def safe_stem(name):
    stem = re.sub(r'[^a-zA-Z0-9_-]+', '-', name).strip('-_').lower()
    return stem[:62] if stem else 'polarbench-capture'


def map_download(result, name, view, output):
    stem = f'{safe_stem(name)}-{view.layer}'
    if output == 'png':
        return Download(raster(result, view, full_size=True).payload, f'{stem}.png', 'image/png')
    if output == 'data':
        if view.layer == 'quality':
            raise InputError('Quality flags are categorical. Use a quality evidence package.')
        return Download(numeric_layer(result, view), f'{stem}-linear.tif', 'image/tiff')
    raise InputError('Unknown map download format.')


def numeric_layer(result, view):
    values, support = result.channel(view.layer, view.analyzer_angle)
    region = view.crop.slices if view.crop else np.s_[:, :]
    copied = np.where(support[region], values[region], np.nan).astype(np.float32)
    return gray_tiff(copied)


def support_png(result, view):
    if view.layer == 'quality':
        coverage = result.quality.valid
    else:
        _, coverage = result.channel(view.layer, view.analyzer_angle)
    if view.crop:
        coverage = coverage[view.crop.slices]
    return rgba_png(np.where(coverage, 255, 0).astype(np.uint8))


def _bounded_font(size):
    return ImageFont.load_default(size=size)


def _fit_card(payload, fit):
    with Image.open(io.BytesIO(payload)) as current:
        rgba = current.convert('RGBA')
        rgba.thumbnail(fit, Image.Resampling.LANCZOS)
    return rgba


def _limited_title(text, width, font):
    if font.getlength(text) <= width:
        return text
    while text and font.getlength(text + '…') > width:
        text = text[:-1]
    return text + '…'


def contact_sheet(result, record, view, raw):
    request = object_value(raw, 'Contact sheet')
    layer_ids = list_value(request.get('layers', ['raw-0', 'raw-45', 'raw-90', 'raw-135']), 'Contact sheet maps', 8)
    if not layer_ids or len(layer_ids) != len(set(map(str, layer_ids))):
        raise InputError('Select up to eight different contact sheet maps.')
    entries = [LAYERS[enum_value(layer, LAYERS, 'Contact sheet map')] for layer in layer_ids]
    columns = 2 if len(entries) > 1 else 1
    rows = (len(entries) + columns - 1) // columns
    width, height = 90 + columns * 484, 121 + rows * 384
    sheet = Image.new('RGBA', (width, height), (19, 28, 41, 255))
    draw = ImageDraw.Draw(sheet)
    heading_font, body_font = _bounded_font(24), _bounded_font(13)
    draw.text((45, 31), _limited_title(record['name'], width - 90, heading_font), fill=(244, 248, 253), font=heading_font)
    scope = view.crop or Box.entire(result.decoded.width, result.decoded.height)
    draw.text((45, 69), f'Detector [{scope.x}, {scope.y}] · {scope.width} × {scope.height} px · {len(entries)} maps',
              fill=(161, 185, 211), font=body_font)
    for index, spec in enumerate(entries):
        x = 45 + (index % columns) * 484
        y = 110 + (index // columns) * 384
        draw.rounded_rectangle((x, y, x + 465, y + 363), radius=12, fill=(12, 19, 31), outline=(58, 73, 96))
        draw.text((x + 14, y + 12), spec.label, font=_bounded_font(16), fill=(233, 241, 249))
        from .domain import DisplayRange
        working = dataclasses.replace(view, layer=spec.id, palette=spec.palette, display=DisplayRange(spec.minimum, spec.maximum),
                                      gamma=1, invalid_style='checker', grid=False, vectors=False, max_side=600)
        exported = raster(result, working)
        frame = _fit_card(exported.payload, (431, 303))
        xx, yy = x + 17 + (431 - frame.width) // 2, y + 43 + (303 - frame.height) // 2
        sheet.alpha_composite(frame, (xx, yy))
    return Download(rgba_png(np.asarray(sheet)), f'{safe_stem(record["name"])}-contact-sheet.png', 'image/png')


def _zip_entry(output, path, data, *, stored=False):
    entry = zipfile.ZipInfo(path)
    entry.external_attr = 0o644 << 16
    entry.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    output.writestr(entry, data)


def _audit_rows(quality):
    source = summary(quality)
    return (('total', source['pixels']), ('usable', source['usable']), ('oriented', source['orientation_usable']), *source['issues'].items())


def review_zip(results, record, view, raw):
    data = object_value(raw, 'Evidence report')
    layer = data.get('measurement_layer', view.layer if view.layer != 'quality' else 'intensity')
    enum_value(layer, SCALAR_LAYERS, 'Review measurement layer')
    rendered = raster(results, view, full_size=True)
    original = source_identity(record)
    title = safe_stem(record['name'])
    summary_rows = region_table(results, record, view, {'layer': layer})
    files = {
        'view.png': rendered.payload,
        'view-support.png': support_png(results, view),
        'regions.csv': table_csv(summary_rows),
        'quality.csv': csv_text(('metric', 'pixels'), _audit_rows(results.quality)),
    }
    if view.layer != 'quality':
        files['map-linear.tif'] = numeric_layer(results, view)
    image_types = ('.png', '.tif')
    manifest = {path: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for path, data in files.items()}
    report = {
        'format': 'polarbench-review-1',
        'capture': {'id': record['id'], 'name': record['name'], 'revision': record['revision'], 'note': record['note']},
        'source_fingerprint': fingerprint(original),
        'source_identity': original,
        'processing': results.processing.as_dict(),
        'view': view.as_dict(),
        'render': rendered.metadata(),
        'reference': results.references.summary(),
        'quality': summary(results.quality),
        'quality_legend': legends(),
        'view_legend': legend(view),
        'viewport_quality': summary(results.quality, rendered.viewport),
        'file_manifest': manifest,
        'linear_missing_value': 'NaN' if view.layer != 'quality' else None,
        'regions': record['regions'],
        'region_measurement_layer': layer,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode='w', compresslevel=4) as output:
        _zip_entry(output, 'review.json', json_bytes(report))
        for path, content in files.items():
            _zip_entry(output, path, content, stored=path.endswith(image_types))
    return Download(buffer.getvalue(), f'{title}-evidence.zip', 'application/zip')
