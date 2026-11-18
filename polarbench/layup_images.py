from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .layup import ANOMALY_COLOR, BRIDGE_COLOR, REFERENCE_COLOR
from .render import map_colors


def layup_overlay(results, report):
    window = report['window']
    x0, y0, width, height = (window[key] for key in ('x', 'y', 'width', 'height'))
    selected = np.s_[y0:y0 + height, x0:x0 + width]
    direction, support = results.field.channel('direction')
    base = map_colors(direction[selected] % 180, 0, 180, 'axial').astype(np.float32)
    # Dim unsupported positions so missing samples and masked strips read as hard barriers.
    base[~support[selected]] *= 0.28

    render = report['_render']
    base[render['bridge_mask']] = np.array(BRIDGE_COLOR, dtype=np.float32)
    base[render['anomaly_mask']] = np.array(ANOMALY_COLOR, dtype=np.float32)

    image = Image.fromarray(np.rint(base).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=11)
    reference = report['reference_box']
    draw.rectangle(
        (reference['x'] - x0, reference['y'] - y0,
         reference['x'] + reference['width'] - 1 - x0, reference['y'] + reference['height'] - 1 - y0),
        outline=REFERENCE_COLOR, width=2,
    )
    draw.text((reference['x'] - x0 + 2, max(0, reference['y'] - y0 - 12)),
              f"ref {report['reference_direction']:.1f}°", font=font, fill=REFERENCE_COLOR,
              stroke_width=2, stroke_fill=(23, 31, 47))
    for index, row in enumerate(report['patches'], 1):
        box = row['box']
        left, top = box['x'] - x0, box['y'] - y0
        right, bottom = left + box['width'] - 1, top + box['height'] - 1
        draw.rectangle((left, top, right, bottom), outline=(251, 240, 212), width=1)
        draw.text((left + 2, top + 2), str(index), font=font, fill=(247, 248, 243),
                  stroke_width=2, stroke_fill=(23, 31, 47))
    return rgba_png(np.asarray(image))


def layup_legend():
    return {'type': 'category', 'entries': [
        {'key': 'reference', 'label': 'Reference box (normal layup)', 'color': list(REFERENCE_COLOR),
         'explanation': 'Circular mean direction of the usable pixels in the witness box.'},
        {'key': 'anomaly', 'label': 'Deviating measured pixels', 'color': list(ANOMALY_COLOR),
         'explanation': 'Measured layup pixels whose axial deviation meets the threshold.'},
        {'key': 'bridge', 'label': 'Weak-direction bridge (not counted)', 'color': list(BRIDGE_COLOR),
         'explanation': 'Orientation too weak to measure; used only to join one continuous anomaly.'},
    ]}
