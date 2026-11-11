"""Scene loading: one complete scene by default, or staged surfaces first."""


def load_scene_stages(snapshot, *, via_details=False, progress=None):
    """Yield (scene, cache_hit); a preview never reduces mesh or pixel quality.

    Cached surfaces used to wait for every buried layer's checksum and VTP
    decode. Load them first and retain their mesh objects during completion,
    avoiding both a second disk read and a second GPU upload of those surfaces.
    """
    from dataclasses import replace
    from .cache import load_cached_scene, load_previous_scene
    from .geometry import build_scene, scene_keys

    preview_keys = scene_keys(snapshot, surface_only=True, via_details=False)
    requested = scene_keys(snapshot, via_details=via_details)
    previous = None
    preview = load_cached_scene(snapshot, requested_keys=preview_keys)
    if preview is not None:
        if preview_keys == requested:
            yield preview, True
            return
        preview._preview = True
        yield preview, True
        if progress is not None:
            progress.check_cancel()
        scene = load_cached_scene(snapshot, requested_keys=requested, previous_scene=preview)
        if scene is not None:
            yield scene, True
            return
    else:
        previous = load_previous_scene(snapshot, requested_keys=preview_keys)
        preview = build_scene(snapshot, progress=progress, previous_scene=previous,
                              requested_keys=preview_keys)
        if preview_keys == requested:
            yield preview, False
            return
        preview._preview = True
        yield preview, False
        if progress is not None:
            progress.check_cancel()
    # Read buried layers only after presenting the surface. This also handles
    # an exact surface cache hit whose inner signatures are stale or missing.
    # Retain validated preview objects instead of checksumming/reading them again.
    previous = load_previous_scene(snapshot, requested_keys=requested,
                                   previous_scene=preview)
    base = preview
    if previous is not None:
        visible = {part.key for part in preview.parts}
        base = replace(preview, parts=list(preview.parts) + [part for part in previous.parts if part.key not in visible],
                       _geometry_signatures={**previous._geometry_signatures, **preview._geometry_signatures})
    yield build_scene(snapshot, progress=progress, previous_scene=base,
                      requested_keys=requested), False


def load_complete_scene(snapshot, *, via_details=False, components=True, progress=None):
    """Return ``(scene, cache_hit)`` with every layer and the 3D models at once.

    Component models load on their own thread while the board is read from the
    cache or meshed, so the first frame already shows the finished board. The
    models ride along as ``scene._components = (parts, warnings, signature)``.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .cache import load_cached_scene, load_previous_scene
    from .geometry import build_scene, scene_keys

    requested = scene_keys(snapshot, via_details=via_details)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pcb-models")
    models = None
    try:
        if components and any(footprint.models for footprint in snapshot.footprints):
            from .components import build_component_parts
            slot_count = max(1, 2 * len(snapshot.stackup.copper) - 1)
            models = pool.submit(build_component_parts, snapshot, slot_count, progress)
        scene = load_cached_scene(snapshot, requested_keys=requested)
        cached = scene is not None
        if scene is None:
            previous = load_previous_scene(snapshot, requested_keys=requested)
            scene = build_scene(snapshot, progress=progress, previous_scene=previous,
                                requested_keys=requested)
        if models is not None:
            from .components import component_signature
            try:
                parts, warnings = models.result()
            except Exception as exc:  # noqa: BLE001 - the board must still open without models
                if progress is not None and hasattr(progress, "check_cancel"):
                    progress.check_cancel()
                parts, warnings = [], [f"3D components: {exc}"]
            scene._components = (parts, warnings, component_signature(snapshot))
        return scene, cached
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
