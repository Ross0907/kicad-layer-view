"""Soft camera bloom from visible LED radiance; no renderer or new dependency."""
from functools import lru_cache

import numpy as np


def _box_axis(image, radius, axis):
    if radius < 1:
        return image
    padding = [(0, 0)] * image.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="constant")
    integral = np.cumsum(padded, axis=axis, dtype=np.float64)
    zero = list(integral.shape)
    zero[axis] = 1
    integral = np.concatenate((np.zeros(zero), integral), axis=axis)
    near, far = [slice(None)] * image.ndim, [slice(None)] * image.ndim
    near[axis], far[axis] = slice(None, -(2 * radius + 1)), slice(2 * radius + 1, None)
    return ((integral[tuple(far)] - integral[tuple(near)]) / (2 * radius + 1)).astype(np.float32)


def _soften(image, radius):
    # Three normalized boxes approximate a Gaussian without ringing or bright
    # edge replication. Zero padding lets energy leave the edge of the frame.
    for _ in range(3):
        image = _box_axis(_box_axis(image, radius, 1), radius, 0)
    return image


@lru_cache(maxsize=8)
def _axis_weights(source, target):
    positions = np.clip((np.arange(target, dtype=np.float64) + .5) * source / target - .5,
                        0, source - 1)
    near = positions.astype(np.intp)
    return near, np.minimum(near + 1, source - 1), (positions - near).astype(np.float32)


def _resize(image, height, width):
    """Same bilinear halo, using float32 tiles instead of full-frame doubles."""
    y0, y1, wy = _axis_weights(image.shape[0], height)
    x0, x1, wx = _axis_weights(image.shape[1], width)
    wx = wx[None, :, None]
    rows = np.empty((image.shape[0], width, 3), dtype=np.float32)
    for start in range(0, len(rows), 32):
        source = image[start:start+32]
        rows[start:start+32] = source[:, x0] * (1 - wx) + source[:, x1] * wx
    result = np.empty((height, width, 3), dtype=np.float32)
    for start in range(0, height, 32):
        end = min(start + 32, height)
        weight = wy[start:end, None, None]
        result[start:end] = rows[y0[start:end]] * (1 - weight) + rows[y1[start:end]] * weight
    return result


def _brightpass(image, exposure):
    image = np.maximum(image, 0) * exposure
    peak = image.max(axis=2, keepdims=True)
    knee = np.clip(peak - .5, 0., 1.)
    excess = np.maximum(peak - 1., 0.) + .25 * knee * knee
    image *= excess / np.maximum(peak, 1e-8)
    return image


def led_bloom(emission, *, exposure_ev=0., radius=1.):
    """Return linear HDR halo from direct LED emission, not copper highlights.

    A compact multiscale halo keeps a tight core and weaker broad tail. Only
    this smooth optical effect uses a reduced working buffer; the camera film,
    geometry, antialiasing and sharp LED core retain their original resolution.
    """
    image = np.asarray(emission, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("LED emission must be a nonempty H×W×3 array")
    if not np.isfinite(exposure_ev) or not -4 <= exposure_ev <= 4:
        raise ValueError("Exposure must be between -4 and +4 EV")
    if not np.isfinite(radius) or not .25 <= radius <= 4:
        raise ValueError("Bloom radius must be between 0.25 and 4")
    height, width = image.shape[:2]
    factor = max(1, int(np.ceil(max(height, width) / 512)))
    exposure = np.float32(2. ** exposure_ev)
    if factor > 1:
        pad_y, pad_x = (-height) % factor, (-width) % factor
        reduced = np.zeros(((height + pad_y)//factor, (width + pad_x)//factor, 3), dtype=np.float32)
        # Bright-pass BEFORE averaging preserves tiny bright LEDs. Processing
        # aligned strips avoids allocating multiple full-size HDR temporaries.
        step = max(factor, (64 // factor) * factor)
        for start in range(0, height, step):
            source = image[start:start+step]
            if source.max() <= .5 / exposure:
                continue  # Exact zero bright-pass; no thresholded visible halo.
            tile = _brightpass(source, exposure)
            tile = np.pad(tile, ((0, (-len(tile)) % factor), (0, pad_x), (0, 0)))
            count = len(tile) // factor
            reduced[start//factor:start//factor+count] = tile.reshape(
                count, factor, (width + pad_x)//factor, factor, 3).mean(axis=(1, 3))
        image = reduced
    else:
        image = _brightpass(image, exposure)
    if not np.any(image):
        return np.zeros((height, width, 3), dtype=np.float32)
    unit = max(.5, min(height, width) / (900 * factor)) * radius
    scales = [(weight, max(1, round(size * unit))) for weight, size in ((.55, 2), (.30, 5), (.15, 12))]
    # Compact-support kernels cannot affect pixels beyond three box radii.
    # Restrict work to that exact support around luminous pixels, not the PCB.
    ys = np.flatnonzero(np.any(image, axis=(1, 2)))
    xs = np.flatnonzero(np.any(image, axis=(0, 2)))
    support = 3 * max(size for _, size in scales)
    top, bottom = max(0, ys[0] - support), min(len(image), ys[-1] + support + 1)
    left, right = max(0, xs[0] - support), min(image.shape[1], xs[-1] + support + 1)
    crop = image[top:bottom, left:right]
    halo = np.zeros_like(image)
    halo[top:bottom, left:right] = sum(weight * _soften(crop, size) for weight, size in scales)
    if factor > 1:
        # Restore padded dimensions before cropping, so edge sources don't move.
        halo = _resize(halo, height + pad_y, width + pad_x)[:height, :width]
    return np.ascontiguousarray(halo, dtype=np.float32)
