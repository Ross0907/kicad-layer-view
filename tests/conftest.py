"""Shared test configuration."""
import functools

import pytest

# These run Mitsuba's CPU renderer, whose symbolic loops need Dr.Jit's LLVM
# backend (an LLVM shared library on the machine). GPU-only installs lack it.
_NEEDS_LLVM = {
    "test_ray_materials.py::test_mask_transmits_colored_trace_detail_and_opaque_setting_hides_it",
    "test_raytrace.py::test_scalar_path_tracer_writes_nonempty_image_and_camera_orientation",
    "test_raytrace.py::test_neutral_lighting_is_symmetric_on_both_board_faces",
    "test_raytrace_backend.py::test_failed_gpu_probe_uses_real_cpu_scene",
    "test_raytrace_guides.py::test_first_denoised_path_sample_preserves_actual_copper_tracks_under_mask",
    "test_render_dialog.py::test_running_ray_worker_cancels_and_releases_capture",
}


@functools.lru_cache(maxsize=1)
def _llvm_backend():
    try:
        import drjit
        return bool(drjit.has_backend(drjit.JitBackend.LLVM))
    except Exception:  # noqa: BLE001 - absent or broken optional renderer
        return False


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="Mitsuba CPU rendering needs Dr.Jit's LLVM backend (LLVM not installed)")
    for item in items:
        key = f"{item.path.name}::{item.name.split('[')[0]}"
        if key in _NEEDS_LLVM and not _llvm_backend():
            item.add_marker(skip)


@pytest.fixture
def kicad_style(monkeypatch):
    """Resolve materials exactly like KiCad's 3D viewer (not the realistic default)."""
    from layer_viewer import finishes
    monkeypatch.setattr(finishes, "STYLE", "kicad")


@pytest.fixture(autouse=True)
def _private_mesh_cache(tmp_path_factory, monkeypatch):
    """Never read or write the user's real scene cache: a stale cached view of
    the demo board there would otherwise be shown as a load preview."""
    monkeypatch.setenv("LAYER_VIEW_CACHE_DIR", str(tmp_path_factory.mktemp("mesh-cache")))
