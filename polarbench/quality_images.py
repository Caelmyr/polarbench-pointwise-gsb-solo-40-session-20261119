from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .codec import rgba_png
from .quality import BY_ISSUE


def patch_overlay(results, report):
    issue = report['issue']
    box = report['box']
    x0, y0, width, height = (box[key] for key in ('x', 'y', 'width', 'height'))
    selected = np.s_[y0:y0 + height, x0:x0 + width]
    mask = results.quality.reason(issue)[selected]
    grayscale_signal = results.field.s0[selected]
    top = max(float(np.percentile(grayscale_signal, 96)), 0.1)
    levels = np.clip(grayscale_signal / top, 0, 1)
    colors = np.stack((22 + levels * 80, 29 + levels * 78, 45 + levels * 73), axis=-1).astype(np.uint8)
    colors[mask] = BY_ISSUE[issue].color
    image = Image.fromarray(colors)
    draw = ImageDraw.Draw(image)
    small = ImageFont.load_default(size=11)
    for i, patch in enumerate(report['patches'], 1):
        bounds = patch['box']
        left, top = bounds['x'] - x0, bounds['y'] - y0
        right, bottom = left + bounds['width'] - 1, top + bounds['height'] - 1
        draw.rectangle((left, top, right, bottom), outline=(251, 240, 212), width=1)
        draw.text((left + 2, top + 3), str(i), font=small, fill=(247, 248, 243),
                  stroke_width=2, stroke_fill=(23, 31, 47))
    return rgba_png(np.asarray(image))


def issue_profile(report, axis='rows'):
    """Place component evidence in original coordinates for external annotation tools."""
    output = []
    for index, patch in enumerate(report['patches'], 1):
        bounds = patch['box']
        if axis == 'rows':
            start, end = bounds['y'], bounds['y'] + bounds['height']
        else:
            start, end = bounds['x'], bounds['x'] + bounds['width']
        output.append((index, start, end, patch['pixels'], patch['touches_window']))
    return output
