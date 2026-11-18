from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .domain import Box, InputError, integer, number, object_value
from .regions import Geometry, csv_text, select_region
from .scale import SpatialScale
from .stokes import axial_mean, polarity_difference

ANOMALY_COLOR = (246, 159, 64)
REFERENCE_COLOR = (105, 219, 175)
BRIDGE_COLOR = (215, 200, 113)
WINDOW_LIMIT = 40


@dataclass(frozen=True)
class LayupSpec:
    deviation: float
    split: float
    bridge_gap: int
    min_area: int

    @classmethod
    def parse(cls, raw):
        data = object_value(raw, 'Layup direction review')
        deviation = number(data.get('deviation', 20.0), 'Direction deviation threshold', 2.0, 90.0)
        split = number(data.get('split', 25.0), 'Patch split tolerance', 2.0, 90.0)
        bridge_gap = integer(data.get('bridge_gap', 2), 'Weak pixel bridge length', 0, 2)
        min_area = integer(data.get('min_area', 12), 'Minimum measured patch size', 1, 10_000_000)
        return cls(deviation, split, bridge_gap, min_area)


def _reference_axis(field, geometry):
    """Circular mean layup direction over the oriented pixels of the witness box."""
    cut = geometry.box.slices
    inside = geometry.local_mask()
    oriented = inside & field.oriented[cut]
    count = int(oriented.sum())
    if count < 4:
        raise InputError('The reference box needs at least four pixels with a usable layup direction.')
    q, u = (float(layer[cut][oriented].mean(dtype=np.float64)) for layer in (field.q, field.u))
    axis = axial_mean(q, u)
    if axis is None:
        raise InputError('Directions inside the reference box are balanced with no dominant layup direction.')
    return axis, count


def review_layup(results, record, raw):
    data = object_value(raw, 'Layup direction review')
    if not data.get('reference_box'):
        raise InputError('Draw a box over a normally laid section to define the reference direction.')
    height, width = results.field.shape
    reference = Box.parse(data['reference_box'], width, height)
    geometry, scope_name = select_region(record, data, width, height)
    cfg = LayupSpec.parse(data)
    field = results.field

    reference_geometry = Geometry('box', reference)
    baseline, reference_pixels = _reference_axis(field, reference_geometry)

    cut = geometry.box.slices
    inside = geometry.local_mask()
    oriented = inside & field.oriented[cut]
    # weak-axis = measured but orientation too weak. Everything else inside the window that is
    # not oriented (missing samples, masked strips, clipping, darkness, outside overlap) is a hard barrier.
    weak_axis = inside & results.quality.reason('weak-axis')[cut]
    bridgeable = weak_axis & ~oriented

    local_direction = field.direction[cut]
    local_signed = polarity_difference(local_direction, np.float32(baseline))
    local_deviation = np.abs(local_signed)
    anomaly_local = oriented & (local_deviation >= cfg.deviation)

    local_count = int(anomaly_local.sum())
    local_index = np.full(anomaly_local.shape, -1, dtype=np.int32)
    local_index[anomaly_local] = np.arange(local_count, dtype=np.int32)
    node_flat = np.flatnonzero(anomaly_local.ravel())
    flat_direction = local_direction.ravel()
    flat_signed = local_signed.ravel()

    edge_a, edge_b, bridge_weak = [], [], []
    # 4-neighbour links between measured anomaly pixels, kept only where the layup axes agree.
    for dy, dx in ((0, 1), (1, 0)):
        a = local_index[:local_index.shape[0] - dy, :local_index.shape[1] - dx]
        b = local_index[dy:, dx:]
        first, second = np.nonzero((a >= 0) & (b >= 0))
        if first.size:
            node_a, node_b = a[first, second], b[first, second]
            agree = np.abs(polarity_difference(flat_direction[node_flat[node_a]],
                                               flat_direction[node_flat[node_b]])) <= cfg.split
            edge_a.append(node_a[agree])
            edge_b.append(node_b[agree])

    # Bridges over 1 or 2 consecutive weak pixels, straight along a row or column only.
    h, w = anomaly_local.shape
    flat_anomaly = anomaly_local.ravel()
    flat_weak = bridgeable.ravel()
    for gap in range(1, cfg.bridge_gap + 1):
        for step in (1, w):  # horizontal then vertical
            starts = np.arange(0, flat_anomaly.size - (gap + 1) * step, dtype=np.int32)
            if step == 1:
                starts = starts[(starts // w) == ((starts + (gap + 1)) // w)]
            ends = starts + (gap + 1) * step
            middle = np.stack([starts + (k + 1) * step for k in range(gap)], axis=1)
            valid = flat_anomaly[starts] & flat_anomaly[ends] & flat_weak[middle].all(axis=1)
            if not valid.any():
                continue
            starts, ends, middle = starts[valid], ends[valid], middle[valid]
            agree = np.abs(polarity_difference(flat_direction[starts], flat_direction[ends])) <= cfg.split
            same_side = flat_signed[starts] * flat_signed[ends] > 0
            chosen = agree & same_side
            if not chosen.any():
                continue
            edge_a.append(local_index.ravel()[starts[chosen]])
            edge_b.append(local_index.ravel()[ends[chosen]])
            bridge_weak.append(middle[chosen].ravel())

    if local_count == 0:
        component_count, labels = 0, np.empty(0, dtype=np.int32)
        component_map = np.full(anomaly_local.shape, -1, dtype=np.int32)
    else:
        if edge_a:
            left, right = np.concatenate(edge_a), np.concatenate(edge_b)
            graph = sparse.csr_matrix((np.ones(left.size, dtype=np.int8), (left, right)),
                                      shape=(local_count, local_count))
        else:
            graph = sparse.csr_matrix((local_count, local_count), dtype=np.int8)
        component_count, labels = connected_components(graph, directed=False, return_labels=True)
        component_map = np.full(anomaly_local.shape, -1, dtype=np.int32)
        component_map[anomaly_local] = labels

    # Single sort groups every component's cells contiguously; detailed statistics are then
    # O(1) boundary slices instead of one full-array scan per component.
    order = np.argsort(labels, kind='stable') if component_count else np.empty(0, dtype=np.int64)
    sorted_flat = node_flat[order] if component_count else np.empty(0, dtype=np.int64)
    sorted_labels = labels[order] if component_count else np.empty(0, dtype=np.int32)
    group_start = np.zeros(component_count, dtype=np.int64)
    if component_count:
        group_start[np.unique(sorted_labels)] = np.flatnonzero(
            np.r_[True, np.diff(sorted_labels) != 0]
        )
    component_sizes = np.bincount(labels, minlength=component_count) if component_count else np.empty(0, dtype=np.int64)

    qualified_labels = np.flatnonzero(component_sizes >= cfg.min_area) if component_count else np.empty(0, dtype=np.int32)
    qualified_labels = qualified_labels[np.argsort(component_sizes[qualified_labels])[::-1]]
    shown_labels = qualified_labels[:WINDOW_LIMIT]

    bridge_members, bridge_mask = _bridge_members(bridge_weak, component_map, h, w, shown_labels)
    patches = _patch_rows(
        shown_labels, sorted_flat, group_start, component_sizes,
        local_direction, local_signed, local_deviation, geometry.box, record, bridge_members,
    )
    shown_component = np.isin(component_map, shown_labels) if shown_labels.size else np.zeros_like(anomaly_local)

    result = {
        'scope': scope_name,
        'window': geometry.box.as_dict(),
        'reference_box': reference.as_dict(),
        'reference_direction': round(baseline, 4),
        'reference_pixels': reference_pixels,
        'deviation_threshold': cfg.deviation,
        'split_tolerance': cfg.split,
        'bridge_gap': cfg.bridge_gap,
        'minimum_size': cfg.min_area,
        'oriented_pixels': int(oriented.sum()),
        'weak_pixels': int(bridgeable.sum()),
        'anomaly_pixels': local_count,
        'all_components': component_count,
        'qualified_components': int(qualified_labels.size),
        'shown': int(shown_labels.size),
        'patches': patches,
    }
    # Renderer-only arrays, never serialised to JSON.
    result['_render'] = {
        'anomaly_mask': shown_component,
        'bridge_mask': bridge_mask if shown_labels.size else np.zeros_like(bridge_mask),
    }
    return result


def _bridge_members(bridge_weak, component_map, height, width, shown_labels):
    """Attribute weak bridge pixels to the shown components they touch.

    Fully vectorised 4-neighbour dilation. Weak pixels are bridges only, never measured area.
    Returns the per-shown-component attribution and a full-size boolean bridge mask.
    """
    bridge_mask = np.zeros((height, width), dtype=bool)
    members = {int(label): set() for label in shown_labels.tolist()}
    if not bridge_weak or shown_labels.size == 0:
        return members, bridge_mask
    weak_flat = np.unique(np.concatenate(bridge_weak))
    padded = np.pad(component_map, 1, mode='constant', constant_values=-1)
    ys, xs = np.divmod(weak_flat, width)
    ys, xs = ys + 1, xs + 1  # padded coordinates
    neighbour_labels = np.stack([
        padded[ys - 1, xs], padded[ys + 1, xs], padded[ys, xs - 1], padded[ys, xs + 1],
    ], axis=1)
    shown_touch = np.isin(neighbour_labels, shown_labels).any(axis=1)
    bridge_mask.ravel()[weak_flat] = shown_touch

    # Build distinct (weak pixel, component) pairs in one pass, then group by component.
    repeated_weak = np.repeat(weak_flat, 4)
    flat_labels = neighbour_labels.ravel()
    keep = np.isin(flat_labels, shown_labels)
    repeated_weak, flat_labels = repeated_weak[keep], flat_labels[keep]
    if repeated_weak.size:
        base = component_map.size + 1
        pair_key = (repeated_weak.astype(np.int64) * base + flat_labels.astype(np.int64))
        pair_key = np.unique(pair_key)
        pair_weak, pair_label = pair_key // base, pair_key % base
        order = np.argsort(pair_label, kind='stable')
        pair_weak, pair_label = pair_weak[order], pair_label[order]
        starts = np.flatnonzero(np.r_[True, np.diff(pair_label) != 0])
        ends = np.r_[starts[1:], pair_label.size]
        for start, end in zip(starts, ends):
            members[int(pair_label[start])] = set(pair_weak[start:end].tolist())
    return members, bridge_mask


def _patch_rows(shown_labels, sorted_flat, group_start, component_sizes,
                direction, signed, deviation, window, record, bridge_members):
    if shown_labels.size == 0:
        return []
    scale = SpatialScale.from_record(record.get('scale'))
    width = direction.shape[1]
    flat_direction, flat_signed, flat_deviation = direction.ravel(), signed.ravel(), deviation.ravel()
    group_end = np.r_[group_start[1:], sorted_flat.size]
    rows = []
    for rank, label in enumerate(shown_labels, start=1):
        start, end = int(group_start[label]), int(group_end[label])
        member_flat = sorted_flat[start:end]
        measured = int(member_flat.size)
        ys, xs = np.divmod(member_flat, width)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        box = Box(window.x + x0, window.y + y0, x1 - x0, y1 - y0)
        axes = flat_direction[member_flat]
        deltas = flat_signed[member_flat]
        abs_delta = flat_deviation[member_flat]
        rows.append({
            'patch': rank,
            'box': box.as_dict(),
            'centroid': {
                'x': round(float(window.x + xs.mean()), 3),
                'y': round(float(window.y + ys.mean()), 3),
            },
            'measured_pixels': measured,
            'bridge_weak_pixels': len(bridge_members.get(int(label), set())),
            'median_direction': round(float(np.median(axes)), 3),
            'mean_deviation': round(float(abs_delta.mean()), 3),
            'max_deviation': round(float(abs_delta.max()), 3),
            'deviation_side': 'clockwise' if float(np.median(deltas)) >= 0 else 'counterclockwise',
            'fills_box': round(measured / box.area, 4),
            'measured_area': round(scale.area(measured), 6),
            'area_unit': f'{scale.symbol}²',
            'touches_window': bool(box.x == window.x or box.y == window.y or
                                   box.right == window.right or box.bottom == window.bottom),
        })
    return rows


def layup_csv(report):
    return csv_text((
        'patch', 'x', 'y', 'width', 'height', 'centroid_x', 'centroid_y',
        'measured_pixels', 'bridge_weak_pixels', 'median_direction',
        'mean_deviation', 'max_deviation', 'deviation_side', 'measured_area', 'area_unit', 'touches_window',
    ), (
        (
            row['patch'], row['box']['x'], row['box']['y'], row['box']['width'], row['box']['height'],
            row['centroid']['x'], row['centroid']['y'], row['measured_pixels'], row['bridge_weak_pixels'],
            row['median_direction'], row['mean_deviation'], row['max_deviation'], row['deviation_side'],
            row['measured_area'], row['area_unit'], row['touches_window'],
        )
        for row in report['patches']
    ))
