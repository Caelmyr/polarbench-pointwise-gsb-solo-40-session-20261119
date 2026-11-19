from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .domain import Box, InputError, enum_value, identifier, list_value, number, object_value
from .regions import csv_text
from .settings import LAYERS

TREND_LAYERS = ('degree', 'intensity', 'polarized', 'q', 'u', 'pair-error', 'pair-difference')
MAX_TREND_CAPTURES = 12
SEGMENT_COLORS = (
    (119, 225, 201), (241, 187, 118), (135, 172, 244), (232, 150, 192),
    (198, 168, 240), (216, 210, 120), (150, 214, 172),
)


@dataclass(frozen=True)
class TrendRequest:
    layer: str
    box: Box
    minimum_shared: float

    @classmethod
    def parse(cls, raw, width, height):
        data = object_value(raw, 'Process trend')
        return cls(
            layer=enum_value(data.get('layer', 'degree'), TREND_LAYERS, 'Trend signal'),
            box=Box.parse(data['box'], width, height) if data.get('box') is not None else Box.entire(width, height),
            minimum_shared=number(data.get('minimum_shared', 0.5), 'Minimum shared fraction', 0.05, 1),
        )


def trend_capture_ids(raw):
    data = object_value(raw, 'Process trend')
    listed = list_value(data.get('capture_ids'), 'Trend captures', MAX_TREND_CAPTURES)
    if len(listed) < 2:
        raise InputError('A process trend needs at least two captures.')
    ids = tuple(identifier(value, 'Capture') for value in listed)
    if len(set(ids)) != len(ids):
        raise InputError('Each capture can appear only once in a process trend.')
    return ids


def _resolve(bench, current, ids, state):
    """Load every step in process order, reusing the open capture and its live settings."""
    records, states = [], []
    for project_id in ids:
        if project_id == current['id']:
            records.append(current)
            states.append(state)
        else:
            records.append(bench.store.get(project_id))
            states.append(None)
    return records, states


def _matching_size(current, records):
    width, height = current['capture']['width'], current['capture']['height']
    for record in records:
        capture = record['capture']
        if (capture['width'], capture['height']) != (width, height):
            raise InputError('A process trend needs captures with equal detector dimensions. Align or crop the sources externally before importing.')
    return width, height


def _masks(bench, records, states, layer, box):
    cut = box.slices
    masks, measurable = [], []
    for record, state in zip(records, states):
        inspection, _ = bench.pending(record, state)
        _, support = inspection.channel(layer)
        mask = np.ascontiguousarray(support[cut], dtype=bool)
        masks.append(mask)
        measurable.append(int(mask.sum()))
    return masks, measurable


def shared_segments(masks, minimum_shared):
    """Return the maximal runs of consecutive captures that keep enough shared support.

    A run from step ``i`` to ``j`` qualifies while the intersection of every capture in it
    retains at least ``minimum_shared`` of the pixels measurable in any of those captures.
    The intersection only shrinks and the union only grows as the run extends, so each
    start position has one longest qualifying end.  Every start is examined and only runs
    contained in another run are discarded.  Neighbouring runs share the pixel batch they
    each use, so a later pair of steps is still reported when the whole sequence is not.
    """
    count = len(masks)
    intervals = []
    for start in range(count - 1):
        shared = masks[start].copy()
        combined = masks[start].copy()
        longest = None
        for end in range(start + 1, count):
            shared &= masks[end]
            combined |= masks[end]
            total = int(combined.sum())
            both = int(shared.sum())
            if both and total and both / total >= minimum_shared:
                longest = (start, end, shared.copy(), combined.copy())
            else:
                break
        if longest is not None:
            intervals.append(longest)
    kept = []
    for start, end, support, union in intervals:
        if any(other_start <= start and end <= other_end and (other_start, other_end) != (start, end)
               for other_start, other_end, _, _ in intervals):
            continue
        kept.append((start, end, support, union))
    return [{'start': start, 'end': end, 'support': support, 'union': union}
            for start, end, support, union in kept]


def _statistics(values, support):
    data = np.asarray(values[support], dtype=np.float64)
    data = data[np.isfinite(data)]
    if not data.size:
        return None
    lower, median, upper = np.percentile(data, (25, 50, 75))
    return {
        'samples': int(data.size),
        'median': float(median),
        'mean': float(data.mean()),
        'p25': float(lower),
        'p75': float(upper),
        'minimum': float(data.min()),
        'maximum': float(data.max()),
    }


def process_trend(bench, current, raw):
    records, states = _resolve(bench, current, trend_capture_ids(raw), raw.get('state'))
    width, height = _matching_size(current, records)
    request = TrendRequest.parse(raw, width, height)
    names = [record['name'] for record in records]
    masks, measurable = _masks(bench, records, states, request.layer, request.box)

    found = shared_segments(masks, request.minimum_shared)
    cut = request.box.slices
    values = {}
    for segment in found:
        for index in range(segment['start'], segment['end'] + 1):
            if index not in values:
                inspection, _ = bench.pending(records[index], states[index])
                values[index] = inspection.channel(request.layer)[0][cut]

    segments = []
    plotted = set()
    for number_, segment in enumerate(found):
        span = list(range(segment['start'], segment['end'] + 1))
        plotted.update(span)
        shared = int(segment['support'].sum())
        combined = int(segment['union'].sum())
        points = []
        for index in span:
            point = {'step': index + 1, 'id': records[index]['id'], 'name': names[index]}
            stats = _statistics(values[index], segment['support'])
            point.update(stats or {'samples': 0})
            points.append(point)
        segments.append({
            'segment': number_ + 1,
            'steps': [index + 1 for index in span],
            'captures': [
                {'step': index + 1, 'id': records[index]['id'], 'name': names[index]}
                for index in span
            ],
            'shared_pixels': shared,
            'union_pixels': combined,
            'shared_fraction': shared / combined if combined else 0.0,
            'box_pixels': request.box.area,
            'box_fraction': shared / request.box.area if request.box.area else 0.0,
            'points': points,
        })

    unplotted = [
        {
            'step': index + 1,
            'id': records[index]['id'],
            'name': names[index],
            'measurable_pixels': measurable[index],
            'note': ('No measurable pixels in this box.' if not measurable[index]
                     else f'No adjacent step keeps at least {request.minimum_shared:.0%} of the pixels measurable in the pair.'),
        }
        for index in range(len(records)) if index not in plotted
    ]
    return {
        'layer': request.layer,
        'label': LAYERS[request.layer].label,
        'unit': LAYERS[request.layer].unit,
        'box': request.box.as_dict(),
        'box_pixels': request.box.area,
        'minimum_shared': request.minimum_shared,
        'captures': [
            {'step': index + 1, 'id': record['id'], 'name': names[index], 'measurable_pixels': measurable[index]}
            for index, record in enumerate(records)
        ],
        'segments': segments,
        'unplotted': unplotted,
        'basis': 'Each segment reports every capture over one identical set of detector positions.',
    }


def _legend_rows(report):
    return max(1, len(report['segments'])) + 1


def _axis_ticks(segments, gap=0.7):
    """Place each segment's steps contiguously, separated by a visible break.

    A step that belongs to two neighbouring segments appears once per segment, because
    each segment compares it against a different shared pixel set.
    """
    ticks, positions = [], []
    cursor = 0.0
    for segment in segments:
        for point in segment['points']:
            ticks.append(point['step'])
            positions.append(cursor)
            cursor += 1.0
        cursor += gap - 1.0
    return ticks, positions


def trend_chart(report):
    width = 900
    height = 372 + 22 * _legend_rows(report)
    canvas = Image.new('RGBA', (width, height), (23, 32, 44, 255))
    canvas = canvas.crop((0, 0, width, min(height, 150))) if not report['segments'] else canvas
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=12)
    draw.text((28, 20), f'{report["label"]} by process step', fill=(237, 243, 248), font=title)
    box = report['box']
    draw.text(
        (28, 49),
        f'Box {box["x"]}, {box["y"]} - {box["width"]} x {box["height"]} px - '
        f'shared support >= {report["minimum_shared"]:.0%} of measurable pixels',
        fill=(158, 181, 204), font=small,
    )
    left, top, right, bottom = 86, 92, 862, 318
    segments = report['segments']
    if not segments:
        draw.text((28, 92),
                  'No pair of consecutive steps keeps enough shared measurable pixels in this box.',
                  fill=(226, 175, 124), font=small)
        draw.text((28, 116), 'Widen the box, relax the minimum shared fraction, or re-shoot the occluded step.',
                  fill=(158, 181, 204), font=small)
        return rgba_png(np.asarray(canvas))

    ticks, positions = _axis_ticks(segments)
    low_x, high_x = positions[0], positions[-1]

    def x_at(position):
        if high_x <= low_x:
            return (left + right) / 2
        return left + (right - left) * (position - low_x) / (high_x - low_x)

    values = [value for segment in segments for point in segment['points']
              for value in (point['p25'], point['median'], point['p75'])]
    low, high = min(values), max(values)
    span = high - low
    padding = span * 0.10 if span > 0 else max(abs(high) * 0.05, 0.01)
    low, high = low - padding, high + padding

    def y_at(value):
        return round(bottom - (bottom - top) * (value - low) / (high - low))

    for part in np.linspace(0, 1, 5):
        row = round(bottom - part * (bottom - top))
        draw.line((left, row, right, row), fill=(48, 64, 86), width=1)
        draw.text((left - 9, row), f'{low + part * (high - low):.4g}', anchor='rm', fill=(176, 194, 214), font=small)
    draw.line((left, top, left, bottom, right, bottom), fill=(110, 130, 152), width=1)
    for tick, position in zip(ticks, positions):
        x = round(x_at(position))
        draw.text((x, bottom + 8), str(tick), anchor='mt', fill=(176, 194, 214), font=small)
    draw.text(((left + right) / 2, bottom + 27), 'Process order', anchor='mt', fill=(158, 181, 204), font=small)
    skipped = report.get('unplotted', [])
    if skipped:
        note = ('Step ' + ', '.join(str(item['step']) for item in skipped)
                + (' has' if len(skipped) == 1 else ' have') + ' no shared batch in this window')
        draw.text((right, top - 16), note, anchor='rm', fill=(226, 175, 124), font=small)
    draw.text((left, top - 16), report['unit'], anchor='lm', fill=(158, 181, 204), font=small)
    colors = [SEGMENT_COLORS[index % len(SEGMENT_COLORS)] for index in range(len(segments))]
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    shade = ImageDraw.Draw(overlay)
    offset = 0
    bounds = []
    for segment, color in zip(segments, colors):
        points = segment['points']
        piece = positions[offset:offset + len(points)]
        bounds.append((offset, offset + len(points) - 1))
        offset += len(points)
        if len(points) < 2:
            continue
        floor = [(round(x_at(position)), y_at(point['p25'])) for position, point in zip(piece, points)]
        cap = [(round(x_at(position)), y_at(point['p75'])) for position, point in zip(piece, points)]
        shade.polygon([*floor, *reversed(cap)], fill=(*color, 44))
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)

    for (_, previous_end), (next_start, _) in zip(bounds, bounds[1:]):
        x = round((x_at(positions[previous_end]) + x_at(positions[next_start])) / 2)
        for row in range(top, bottom, 9):
            draw.line((x, row, x, min(row + 4, bottom)), fill=(96, 114, 137), width=1)

    offset = 0
    for segment, color in zip(segments, colors):
        points = segment['points']
        marks = [(round(x_at(position)), y_at(point['median']))
                 for point, position in zip(points, positions[offset:offset + len(points)])]
        offset += len(points)
        if len(marks) > 1:
            draw.line(marks, fill=color, width=3, joint='curve')
        for index, (point, (x, y)) in enumerate(zip(points, marks)):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(18, 26, 37), width=1)
            label = f'{point["median"]:.4g}'
            label_width = draw.textlength(label, font=small)
            at = x + 7 if x + 7 + label_width <= right else x - 7 - label_width
            draw.text((at, y - 9 if index % 2 == 0 else y + 2), label, fill=color, font=small)

    row = bottom + 48
    width_limit = right - left - 26
    for segment, color in zip(segments, colors):
        draw.rounded_rectangle((left, row + 3, left + 12, row + 13), radius=2, fill=color)
        first, last = segment['points'][0]['step'], segment['points'][-1]['step']
        span_text = f'{first}' if first == last else f'{first}-{last}'
        names = _shrink_names([f'{point["step"]} {point["name"]}' for point in segment['points']], 18)
        text = (f'Steps {span_text}: {names} - {segment["shared_pixels"]:,} px '
                f'({segment["shared_fraction"]:.0%} shared)')
        while len(text) > 8 and draw.textlength(text, font=small) > width_limit:
            text = text[:-4] + '...'
        draw.text((left + 20, row), text, fill=(205, 218, 232), font=small)
        row += 22
    draw.rounded_rectangle((left, row + 3, left + 12, row + 13), radius=2, fill=(96, 114, 137))
    draw.text((left + 20, row), '25-75% interval; dotted rules separate different shared pixel sets',
              fill=(158, 181, 204), font=small)
    return rgba_png(np.asarray(canvas))


def _shrink_names(names, limit):
    """Shorten a run of capture labels evenly so every step stays identifiable."""
    shortened = [name[:limit] for name in names]
    while sum(len(name) for name in shortened) > limit * 2 and any(len(name) > 6 for name in shortened):
        longest = max(range(len(shortened)), key=lambda index: len(shortened[index]))
        shortened[longest] = shortened[longest][:-2]
    return ' -> '.join(shortened)


def _segment_captures(segment):
    return ' -> '.join(f'{point["step"]} {point["name"]}' for point in segment['points'])


def trend_csv(report):
    columns = (
        'status', 'segment', 'step', 'capture', 'capture_id', 'shared_pixels', 'union_pixels',
        'shared_fraction', 'box_pixels', 'samples', 'median', 'mean', 'p25', 'p75',
        'segment_captures', 'note',
    )
    rows = []
    for segment in report['segments']:
        label = _segment_captures(segment)
        for point in segment['points']:
            rows.append((
                'plotted', segment['segment'], point['step'], point['name'], point['id'],
                segment['shared_pixels'], segment['union_pixels'], segment['shared_fraction'],
                segment['box_pixels'], point.get('samples'), point.get('median'), point.get('mean'),
                point.get('p25'), point.get('p75'), label, '',
            ))
    for item in report['unplotted']:
        rows.append((
            'not plotted', '', item['step'], item['name'], item['id'],
            0, 0, 0, report['box_pixels'], 0, '', '', '', '', '', item['note'],
        ))
    return csv_text(columns, rows)
