"""Show exact outer surfaces before reading/building the remaining inner foils."""


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
