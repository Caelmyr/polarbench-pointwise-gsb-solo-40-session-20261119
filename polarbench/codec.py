from __future__ import annotations

import hashlib
import io
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .domain import MAX_UPLOAD, InputError, check_shape

_ALLOWED_FORMATS = {'PNG', 'TIFF'}
_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_FILE_STEM = re.compile(r'[^\w. -]+', re.UNICODE)


@dataclass(frozen=True)
class ImagePlane:
    values: np.ndarray
    valid: np.ndarray
    bit_depth: int
    format: str
    original_mode: str
    sha256: str

    @property
    def shape(self):
        return self.values.shape

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def metadata(self):
        return {
            'width': self.width,
            'height': self.height,
            'bits': self.bit_depth,
            'format': self.format,
            'mode': self.original_mode,
            'invalid_pixels': int(self.valid.size - np.count_nonzero(self.valid)),
            'sha256': self.sha256,
        }


def meaningful_filename(filename: str | None, fallback: str = 'capture.png') -> str:
    if not filename:
        return fallback
    cleaned = _FILE_STEM.sub('_', Path(filename.replace('\\', '/')).name).strip(' .')
    return cleaned[:120] or fallback


def read_upload(file, field: str) -> tuple[bytes, str]:
    if file is None:
        raise InputError(f'{field} image is required.')
    payload = file.stream.read(MAX_UPLOAD + 1)
    if not payload:
        raise InputError(f'{field} image is empty.')
    if len(payload) > MAX_UPLOAD:
        raise InputError(f'{field} exceeds the {MAX_UPLOAD // 2**20} MB image upload limit.')
    return payload, meaningful_filename(file.filename)


def _validate_png_header(payload):
    if not payload.startswith(_PNG_SIGNATURE) or len(payload) < 33 or payload[12:16] != b'IHDR':
        raise InputError('The PNG header is incomplete.')
    width, height, depth, color = struct.unpack('>IIBB', payload[16:26])
    check_shape(height, width)
    if color in (2, 4, 6) and depth > 8:
        raise InputError('Use a grayscale PNG for 16-bit analyzer samples. Color and alpha PNGs are 8-bit here.')


def _validate_tiff_tags(image):
    tags = image.tag_v2
    orientation = tags.get(274, 1)
    if orientation != 1:
        raise InputError('Normalize TIFF orientation before uploading so detector coordinates remain explicit.')
    sample_format = tags.get(339, (1,))
    if isinstance(sample_format, int):
        sample_format = (sample_format,)
    if any(kind != 1 for kind in sample_format):
        raise InputError('TIFF analyzer samples must be unsigned integers.')
    photometric = tags.get(262, 1)
    if photometric not in (1, 2):
        raise InputError('Use black-is-zero grayscale or RGB TIFF to keep signal levels unambiguous.')
    bits = tags.get(258, (8,))
    if isinstance(bits, int):
        bits = (bits,)
    if any(bit not in (8, 16) for bit in bits) or (len(bits) > 1 and any(bit != 8 for bit in bits)):
        raise InputError('Use 8-bit or 16-bit grayscale TIFF, or equal-channel 8-bit RGB TIFF.')
    if tags.get(338) and image.mode not in ('LA', 'RGBA'):
        raise InputError('TIFF extra samples must have an explicit alpha channel.')


def _grayscale_rgb(channels):
    rgb = channels[:, :, :3]
    if np.any(rgb[:, :, 0] != rgb[:, :, 1]) or np.any(rgb[:, :, 0] != rgb[:, :, 2]):
        raise InputError('Analyzer images require grayscale samples. The RGB channels differ.')
    return rgb[:, :, 0]


def _read_mode(image):
    mode = image.mode
    if mode in ('L', 'I;16', 'I;16L', 'I;16B', 'I;16N'):
        values = np.array(image)
        if image.format == 'PNG' and 'transparency' in image.info:
            valid = values != image.info['transparency']
        else:
            valid = np.ones(values.shape, dtype=bool)
        return values, valid, 8 if mode == 'L' else 16
    if mode == 'LA':
        data = np.array(image)
        return data[:, :, 0], data[:, :, 1] == 255, 8
    if mode == 'RGB' and 'transparency' not in image.info:
        values = _grayscale_rgb(np.array(image))
        return values, np.ones(values.shape, dtype=bool), 8
    if mode in ('RGB', 'RGBA', 'P'):
        if mode == 'P' and 'transparency' not in image.info:
            data = np.array(image.convert('RGB'))
            values = _grayscale_rgb(data)
            return values, np.ones(values.shape, dtype=bool), 8
        data = np.array(image.convert('RGBA'))
        return _grayscale_rgb(data), data[:, :, 3] == 255, 8
    if mode == 'I' and image.format == 'PNG':
        values = np.array(image)
        if np.min(values) < 0 or np.max(values) > 65535:
            raise InputError('PNG integer values fall outside the 16-bit detector range.')
        transparent = image.info.get('transparency')
        support = values != transparent if transparent is not None else np.ones(values.shape, dtype=bool)
        return values.astype(np.uint16), support, 16
    raise InputError(f'Image mode {mode} cannot represent supported unsigned gray/alpha detector samples.')


def read_image(payload: bytes) -> ImagePlane:
    if len(payload) > MAX_UPLOAD:
        raise InputError('The source image exceeds the upload limit.')
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in _ALLOWED_FORMATS:
                raise InputError('Upload a PNG or a single-page TIFF with linear analyzer samples.')
            if getattr(image, 'n_frames', 1) != 1:
                raise InputError('Split multipage or animated input into separate image files.')
            check_shape(image.height, image.width)
            if image.format == 'PNG':
                _validate_png_header(payload)
            else:
                _validate_tiff_tags(image)
            values, valid, bit_depth = _read_mode(image)
            return ImagePlane(values, valid, bit_depth, image.format, image.mode, hashlib.sha256(payload).hexdigest())
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, ValueError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError('The detector image could not be decoded.') from error


def read_mask(payload: bytes, shape: tuple[int, int], threshold: int = 1):
    plane = read_image(payload)
    if plane.shape != shape:
        raise InputError(f'Mask needs the detector canvas size {shape[1]} × {shape[0]}.')
    if threshold < 1:
        raise InputError('A zero mask sample always excludes a pixel.')
    level = np.ceil(threshold / 255 * (2**plane.bit_depth - 1))
    return np.asarray(plane.valid & (plane.values >= level)), plane


def slice_plane(plane: ImagePlane, row: int, col: int):
    return ImagePlane(
        plane.values[row::2, col::2].copy(),
        plane.valid[row::2, col::2].copy(),
        plane.bit_depth,
        plane.format,
        plane.original_mode,
        plane.sha256,
    )


def raw_fraction(plane: ImagePlane):
    return plane.values.astype(np.float32) / (2**plane.bit_depth - 1)


def rgba_png(values: np.ndarray, valid: np.ndarray | None = None) -> bytes:
    pixels = np.asarray(values, dtype=np.uint8)
    if valid is not None:
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[:, :, None], 3, axis=2)
        support = np.where(valid, np.uint8(255), np.uint8(0))
        pixels = np.dstack((pixels, support))
    target = io.BytesIO()
    Image.fromarray(pixels).save(target, 'PNG', compress_level=3)
    return target.getvalue()


def plane_png(levels: np.ndarray, bits: int = 16, valid: np.ndarray | None = None) -> bytes:
    code = np.rint(np.clip(levels, 0, 1) * ((1 << bits) - 1))
    if bits == 8:
        return rgba_png(code.astype(np.uint8), valid)
    if valid is not None:
        code = np.where(valid, code, 0)
    target = io.BytesIO()
    Image.fromarray(code.astype(np.uint16)).save(target, 'PNG', compress_level=3)
    return target.getvalue()


def gray_tiff(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.float32)).save(output, 'TIFF', compression='tiff_deflate')
    return output.getvalue()


def input_packet(image: ImagePlane, *, name: str, byte_count: int) -> dict:
    return dict(image.metadata(), name=name, bytes=byte_count)
