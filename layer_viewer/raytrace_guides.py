"""Deterministic denoiser guides which retain copper beneath translucent resist.

Mitsuba's stock albedo AOV stops at the first surface, so a solder-mask slab
gives the denoiser an almost uniform image even where copper tracks lie below.
These guides follow straight-through (null) transmission on the *existing*
scene. They are auxiliary color/normal data, not a replacement beauty render.
Generate them once per camera/size change and reuse for all progressive samples.

The BSDF already weights diffuse reflectance by its blend opacity; multiply
only by the accumulated transmission of preceding surfaces. References:
https://github.com/mitsuba-renderer/mitsuba3/blob/v3.9.1/src/bsdfs/blendbsdf.cpp
https://github.com/mitsuba-renderer/mitsuba3/blob/v3.9.1/src/integrators/aov.cpp
"""
from __future__ import annotations

import numpy as np


def _trace_guides(mi, scene, ray, max_hits):
    import drjit as dr

    def advance(depth, ray, throughput, albedo, normal, active):
        hit = scene.ray_intersect(ray, active=active)
        valid = active & hit.is_valid()
        if not dr.is_jit_v(mi.Float) and not valid:
            return depth + 1, ray, throughput, albedo, normal, False
        bsdf = hit.bsdf()
        color = bsdf.eval_diffuse_reflectance(hit, valid)
        transmission = dr.clip(bsdf.eval_null_transmission(hit, valid), 0., 1.)
        albedo += dr.select(valid, throughput * color, 0.)
        facing_normal = mi.Vector3f(dr.select(dr.dot(hit.sh_frame.n, ray.d) > 0,
                                             -hit.sh_frame.n, hit.sh_frame.n))
        coverage = dr.mean(throughput * (1. - transmission))
        normal += dr.select(valid, facing_normal * coverage, 0.)
        throughput *= transmission
        active = valid & (dr.max(throughput) > 1e-4)
        # spawn_ray() uses a scene-scale offset which can jump over micron-thin
        # copper. A few float32 ulps separate consecutive intersections without
        # bridging the physical resist/copper gap in the normalized PCB scene.
        step = 4.76837158203125e-7 * dr.maximum(1e-2, dr.max(dr.abs(hit.p)))
        ray = mi.Ray3f(dr.select(valid, hit.p + ray.d * step, ray.o), ray.d)
        return depth + 1, ray, throughput, albedo, normal, active

    _, _, _, albedo, normal, _ = dr.while_loop(
        state=(mi.UInt32(0), ray, mi.Color3f(1.), mi.Color3f(0.),
               mi.Vector3f(0.), mi.Bool(True)),
        cond=lambda depth, ray, throughput, albedo, normal, active: active & (depth < max_hits),
        body=advance,
        max_iterations=max_hits,
        label="PCB transparent denoiser guides",
    )
    length = dr.norm(normal)
    normal = normal / dr.maximum(length, 1e-8)
    return dr.clip(albedo, 0., 1.), normal


def generate_guides(mi, scene, sensor, width, height, *, max_hits=32):
    """Return H×W×3 albedo and world-normal tensors, with four rays per pixel.

    Uses the current Mitsuba variant without changing global renderer state.
    Four fixed quarter-pixel rays match the box-filtered beauty's coverage at
    thin tracks, pad edges and component silhouettes. A single center sample
    gives OptiX jagged, all-or-nothing detail guides which do not match beauty
    pixels containing both copper and resist. CUDA traces all pixels together
    and keeps the results on the GPU. The
    scalar branch is a small-scene reference; CPU rendering has no OptiX
    denoiser and should not request these guides for ordinary full-size frames.
    """
    import drjit as dr

    width, height = int(width), int(height)
    if width < 1 or height < 1 or not 1 <= int(max_hits) <= 256:
        raise ValueError("Guide dimensions must be positive and max_hits must be 1–256")
    if dr.is_jit_v(mi.Float):
        sample = dr.arange(mi.UInt32, width * height * 4)
        index = sample // 4
        position = mi.Point2f((mi.Float(index % width) + .25 + .5 * mi.Float(sample % 2)) / width,
                             (mi.Float(index // width) + .25 + .5 * mi.Float((sample // 2) % 2)) / height)
        ray, _ = sensor.sample_ray(0., .5, position, mi.Point2f(.5))
        albedo, normal = _trace_guides(mi, scene, ray, int(max_hits))
        tensors = tuple(dr.mean(mi.TensorXf(dr.ravel(value), shape=(height, width, 4, 3)), axis=2)
                        for value in (albedo, normal))
        dr.eval(*tensors)
        return tensors
    colors = np.empty((height, width, 3), dtype=np.float32)
    normals = np.empty_like(colors)
    for row in range(height):
        for column in range(width):
            color_sum, normal_sum = mi.Color3f(0.), mi.Vector3f(0.)
            for dx, dy in ((.25, .25), (.75, .25), (.25, .75), (.75, .75)):
                position = mi.Point2f((column + dx) / width, (row + dy) / height)
                ray, _ = sensor.sample_ray(0., .5, position, mi.Point2f(.5))
                color, normal = _trace_guides(mi, scene, ray, int(max_hits))
                color_sum += color
                normal_sum += normal
            colors[row, column] = color_sum / 4
            normals[row, column] = normal_sum / 4
    return mi.TensorXf(colors), mi.TensorXf(normals)
