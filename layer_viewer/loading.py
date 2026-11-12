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


class _ModelProgress:
    """Progress for the model thread that stops it once the load is abandoned."""

    def __init__(self, progress, stop):
        self.progress, self.stop = progress, stop

    def check_cancel(self):
        if self.stop.is_set():
            raise RuntimeError("Loading cancelled")
        if self.progress is not None and hasattr(self.progress, "check_cancel"):
            self.progress.check_cancel()

    def info(self, message):
        if not self.stop.is_set() and self.progress is not None and hasattr(self.progress, "info"):
            self.progress.info(message)


def load_complete_scene(snapshot, *, via_details=False, components=True, progress=None,
                        known_component_signature=None):
    """Return ``(scene, cache_hit)`` with every layer and the 3D models at once.

    Component models load on their own thread while the board is read from the
    cache or meshed, so the first frame already shows the finished board. The
    models ride along as ``scene._components = (parts, warnings, signature)``.
    Models whose signature equals ``known_component_signature`` (already shown)
    are not rebuilt. Cancellation or a board error never waits for a model.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    from .cache import load_cached_scene, load_previous_scene
    from .geometry import build_scene, scene_keys

    requested = scene_keys(snapshot, via_details=via_details)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pcb-models")
    stop = threading.Event()
    models = signature = None
    finished = False
    try:
        if components and any(footprint.models for footprint in snapshot.footprints):
            from .components import build_component_parts, component_signature
            # Taken before reading any model, so a file changed during the load
            # no longer matches and is reloaded afterwards.
            signature = component_signature(snapshot)
            if signature != known_component_signature:
                slot_count = max(1, 2 * len(snapshot.stackup.copper) - 1)
                models = pool.submit(build_component_parts, snapshot, slot_count,
                                     _ModelProgress(progress, stop))
        scene = load_cached_scene(snapshot, requested_keys=requested)
        cached = scene is not None
        if scene is None:
            previous = load_previous_scene(snapshot, requested_keys=requested)
            scene = build_scene(snapshot, progress=progress, previous_scene=previous,
                                requested_keys=requested)
        if models is not None:
            while True:
                try:
                    parts, warnings = models.result(timeout=.1)
                    break
                except FuturesTimeout:
                    if progress is not None and hasattr(progress, "check_cancel"):
                        progress.check_cancel()
                except Exception as exc:  # noqa: BLE001 - the board must still open without models
                    if progress is not None and hasattr(progress, "check_cancel"):
                        progress.check_cancel()
                    parts, warnings = [], [f"3D components: {exc}"]
                    break
            scene._components = (parts, warnings, signature)
        finished = True
        return scene, cached
    finally:
        if not finished:
            stop.set()
        pool.shutdown(wait=finished, cancel_futures=True)
