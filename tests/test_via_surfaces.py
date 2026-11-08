"""Surface-finish contrast must never add coplanar cap overlays or new holes."""
import pytest
from shapely.geometry import Point

from layer_viewer.geometry import copper_geometry
from layer_viewer.model import demo_snapshot
from layer_viewer.via_surfaces import split_cap_surfaces, via_patch_color


def test_cap_patch_partitions_copper_exactly_and_only_at_capped_endpoints():
    snapshot = demo_snapshot()
    capped = snapshot.vias[2]
    for layer in snapshot.stackup.copper:
        original = copper_geometry(snapshot, layer.id)
        base, patch = split_cap_surfaces(snapshot, layer.id, original)
        assert base.intersection(patch).area == pytest.approx(0, abs=1e-12)
        assert base.union(patch).symmetric_difference(original).area == pytest.approx(0, abs=1e-10)
        assert patch.covers(Point(capped.position)) == (layer.id in (capped.layers[0], capped.layers[-1]))
        assert not patch.covers(Point(snapshot.vias[0].position))
        assert not patch.covers(Point(snapshot.vias[1].position))
        assert not patch.covers(Point(28, 18))  # Board cutout.


def test_uncapped_fill_gets_no_metallic_patch_and_dark_palettes_stay_distinct():
    snapshot = demo_snapshot()
    for via in snapshot.vias:
        via.capped = False
        via.filled = True
    original = copper_geometry(snapshot, snapshot.stackup.copper[0].id)
    base, patch = split_cap_surfaces(snapshot, snapshot.stackup.copper[0].id, original)
    assert patch.is_empty and base.equals(original)
    for color in ((.77, .50, .24), (.8, .1, .1), (.01, .02, .01)):
        contrast = via_patch_color(color)
        assert max(abs(a-b) for a, b in zip(color, contrast)) > .1
        assert all(0 <= value <= 1 for value in contrast)
