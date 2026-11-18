from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .artifacts import Download, safe_stem
from .codec import rgba_png
from .domain import InputError, enum_value, list_value, object_value
from .quality import summary
from .regions import Geometry, get_region
from .render import legend, raster
from .scale import draw_scale, scale_bar_for
from .settings import LAYERS
from .statistics import measure


def _font(px):
    return ImageFont.load_default(size=px)


def _trim(value, font, width):
    if font.getlength(value) <= width:
        return value
    while value and font.getlength(value + '…') > width:
        value = value[:-1]
    return value + '…'


def _annotated_selection(record, request):
    ids = request.get('regions')
    if ids is None:
        return record['regions'][:8]
    ids = list_value(ids, 'Worksheet region selection', 8)
    chosen = [get_region(record, item) for item in ids]
    if len({item['id'] for item in chosen}) != len(chosen):
        raise InputError('A worksheet may list each region at most once.')
    return chosen


def _chart_metadata(record, view, quality):
    total = max(1, quality['pixels'])
    return [
        f'{LAYERS[view.layer].label}  ·  {quality["usable"]:,} pixels usable ({100 * quality["usable"] / total:.1f}%)',
        f'Reference {record["revision"]}  ·  {record["capture"]["width"]} × {record["capture"]["height"]} detector pixels',
    ]


def _frame(width, height, theme):
    if theme == 'paper':
        color = (242, 242, 236, 255)
        fg = (31, 41, 51)
        muted = (83, 95, 104)
    else:
        color = (16, 25, 39, 255)
        fg = (235, 242, 248)
        muted = (155, 176, 201)
    image = Image.new('RGBA', (width, height), color)
    return image, ImageDraw.Draw(image), fg, muted


def _region_pixels(frame, geometry):
    viewport, offset, size = frame
    x, y = offset
    factors = (size[0] / viewport.width, size[1] / viewport.height)
    def local(dx, dy):
        return x + (dx - viewport.x) * factors[0], y + (dy - viewport.y) * factors[1]
    if geometry.shape == 'polygon':
        return [local(point.x, point.y) for point in geometry.points]
    box = geometry.box
    return (*local(box.x, box.y), *local(box.right - 1, box.bottom - 1))


def _glyph(canvas, geometry, item, index, frame):
    viewport, offset, size = frame
    if viewport.intersection(geometry.box) is None:
        return
    region_draw = ImageDraw.Draw(canvas)
    coordinates = _region_pixels(frame, geometry)
    color, pen = item['color'], 2
    if geometry.shape == 'polygon':
        region_draw.line((*coordinates, coordinates[0]), fill=color, width=pen)
    elif geometry.shape == 'ellipse':
        region_draw.ellipse(coordinates, outline=color, width=pen)
    else:
        region_draw.rectangle(coordinates, outline=color, width=pen)
    x_ratio = size[0] / viewport.width
    y_ratio = size[1] / viewport.height
    xx = max(offset[0] + 2, offset[0] + round((geometry.box.x - viewport.x) * x_ratio) + 3)
    yy = max(offset[1] + 2, offset[1] + round((geometry.box.y - viewport.y) * y_ratio) + 3)
    region_draw.rounded_rectangle((xx, yy, xx + 21, yy + 21), radius=5, fill=color)
    region_draw.text((xx + 10, yy + 10), str(index), fill=(15, 26, 39), anchor='mm', font=_font(13))


def _view_legend(draw, viewport, definition, y, muted):
    if definition['type'] == 'category':
        cell = viewport // 5
        for idx, series in enumerate(definition['entries']):
            x = 36 + idx % 5 * cell
            row = y + idx // 5 * 26
            draw.rectangle((x, row + 2, x + 11, row + 13), fill=tuple(series['color']))
            draw.text((x + 16, row), _trim(series['label'], _font(11), cell - 18), fill=muted, font=_font(11))
        return y + 64
    length = min(440, viewport)
    for i, payload in enumerate(definition['colors']):
        x0 = 36 + round(i / len(definition['colors']) * length)
        x1 = 36 + round((i + 1) / len(definition['colors']) * length)
        draw.rectangle((x0, y, x1, y + 10), fill=tuple(payload))
    draw.text((36, y + 20), f'{definition["values"][0]:.3g}', fill=muted, font=_font(12))
    draw.text((36 + length, y + 20), f'{definition["values"][-1]:.3g} {definition["unit"]}', fill=muted, anchor='ra', font=_font(12))
    return y + 51


def _one_line_measure(draw, record, view, results, region, index, top, text, support):
    measure_result = measure(results, record, {'region_id': region['id'], 'layer': view.layer if view.layer != 'quality' else 'intensity'}, view)
    draw.rounded_rectangle((36, top + 1, 57, top + 23), radius=5, fill=region['color'])
    draw.text((47, top + 11), str(index), fill=(18, 27, 39), anchor='mm', font=_font(13))
    name = _trim(region['name'], _font(15), 260)
    draw.text((68, top + 4), name, fill=text, font=_font(15))
    metrics = measure_result['linear']
    axial = measure_result['direction']
    count = measure_result['layer_usable']
    main = f'{count:,} of {measure_result["pixels"]:,} usable'
    if metrics:
        main += f'    Median {metrics["percentiles"]["p50"]:.4g}'
    if axial and axial['mean_axis'] is not None:
        main += f'    Axis {axial["mean_axis"]:.1f}°'
    draw.text((355, top + 5), main, fill=support, font=_font(13))
    remarks = region['note'].splitlines()
    if remarks:
        draw.text((68, top + 28), _trim(remarks[0], _font(12), 675), fill=support, font=_font(12))


def make_sheet(results, record, view, payload):
    data = object_value(payload, 'Inspection worksheet')
    theme = enum_value(data.get('theme', 'dark'), ('dark', 'paper'), 'Worksheet theme')
    selected_items = _annotated_selection(record, data)
    rendered = raster(results, view)
    viewport = rendered.viewport
    with Image.open(io.BytesIO(rendered.payload)) as source:
        original = source.convert('RGBA')
        original.thumbnail((790, 560), Image.Resampling.LANCZOS)
    height = 240 + original.height + 58 * len(selected_items)
    total_width = 862
    canvas, draw, primary, details = _frame(total_width, height, theme)
    draw.text((36, 29), _trim(record['name'], _font(26), 790), fill=primary, font=_font(26))
    metadata = _chart_metadata(record, view, summary(results.quality, viewport))
    for i, line in enumerate(metadata):
        draw.text((36, 67 + i * 21), line, fill=details, font=_font(13))
    start = 116
    placement = (36 + (790 - original.width) // 2, start)
    detector = Image.new('RGBA', original.size, (25, 35, 48, 255))
    detector.alpha_composite(original)
    for idx, item in enumerate(selected_items, 1):
        wid, hgt = results.decoded.width, results.decoded.height
        shape = Geometry.parse(item['geometry'], wid, hgt)
        _glyph(detector, shape, item, idx, (viewport, (0, 0), original.size))
    if original.width >= 100 and original.height >= 46:
        bar = scale_bar_for(record, viewport, original.width)
        draw_scale(ImageDraw.Draw(detector), (13, original.height - 14), bar)
    canvas.alpha_composite(detector, placement)
    state = legend(view)
    y = _view_legend(draw, 790, state, start + original.height + 15, details)
    for idx, item in enumerate(selected_items, 1):
        _one_line_measure(draw, record, view, results, item, idx, y, primary, details)
        y += 58
    return Download(rgba_png(np.asarray(canvas)), f'{safe_stem(record["name"])}-worksheet.png', 'image/png')
