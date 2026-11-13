"""Exercise the real Qt CLI shutdown with an outstanding cache worker."""
import os
from pathlib import Path
import json
import struct
import subprocess
import sys

import pytest
from layer_viewer.helper_process import run_cancellable


@pytest.mark.skipif(sys.platform.startswith("linux") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    or os.environ.get("QT_QPA_PLATFORM") == "offscreen"), reason="Qt rendering needs a display")
def test_uncached_demo_render_exits_cleanly_while_cache_worker_finishes(tmp_path):
    output = tmp_path / "demo.png"
    cache_dir = tmp_path / "isolated-empty-cache"
    assert not cache_dir.exists()
    environment = os.environ.copy()
    environment["LAYER_VIEW_CACHE_DIR"] = str(cache_dir)
    environment["QT_API"] = "pyside6"
    if os.name == "nt":
        # Qt UI tests use offscreen for controls, but VTK needs a native Windows
        # drawable for this actual screenshot subprocess.
        environment["QT_QPA_PLATFORM"] = "windows"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    # Hold the real background cache operation until closeEvent requests
    # cancellation. A small demo otherwise caches before the screenshot timer
    # fires and would fail to exercise the live-QThread shutdown regression.
    harness = """
import sys
import time
from layer_viewer import cache
original_save = cache.save_cached_scene
def delayed_save(snapshot, scene, should_cancel=None):
    print('cache-worker-started', flush=True)
    assert should_cancel is not None
    deadline = time.monotonic() + 20
    while not should_cancel() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert should_cancel(), 'CLI did not request worker cancellation when closing'
    result = original_save(snapshot, scene, should_cancel=should_cancel)
    print('cache-worker-finished', flush=True)
    return result
cache.save_cached_scene = delayed_save
from layer_viewer.__main__ import main
raise SystemExit(main())
"""
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    completed = subprocess.run(
        [sys.executable, "-c", harness, "--demo", "--render", str(output)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=60, startupinfo=startupinfo,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "cache-worker-started" in completed.stdout
    assert "cache-worker-finished" in completed.stdout
    assert "QThread: Destroyed" not in completed.stderr
    raw = output.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", raw[16:24])
    assert width >= 200 and height >= 200
    assert len(raw) > 10_000
    assert not list(cache_dir.glob(".pending-*"))


@pytest.mark.skipif(os.name != 'nt', reason='Native Windows shutdown regression')
def test_close_during_png_export_exits_app_and_its_python_helper(tmp_path):
    marker = tmp_path / 'helper-started.pid'
    environment = os.environ.copy()
    environment['QT_API'] = 'pyside6'
    environment['QT_QPA_PLATFORM'] = 'windows'
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    harness = r'''
import json,sys,time,subprocess
from pathlib import Path
from qtpy.QtCore import Qt,QTimer
from qtpy.QtWidgets import QApplication
from layer_viewer.app import MainWindow
from layer_viewer.render_dialog import RenderDialog,RayWorker
from layer_viewer.helper_process import run_cancellable
from layer_viewer import raytrace
marker=Path(sys.argv[1])
def export(*args,progress,**kwargs):
    script='import os,time; from pathlib import Path; '+f'Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)'
    run_cancellable([sys.executable,'-c',script],timeout=60,check_cancel=progress.check_cancel,
                    creationflags=subprocess.CREATE_NO_WINDOW)
raytrace.render_capture=export
app=QApplication([])
window=MainWindow()
window.setAttribute(Qt.WA_DontShowOnScreen,True)
window.show()
dialog=window.render_dialog=RenderDialog(window)
dialog.setAttribute(Qt.WA_DontShowOnScreen,True)
dialog.show()
worker=dialog.worker=RayWorker(object(),marker.with_suffix('.png'),16,16,1,dialog)
cancelled=[]
worker.cancelled.connect(lambda: cancelled.append(True))
worker.start()
started=time.monotonic()
closed=[]
def close_when_running():
    if marker.is_file():
        closed.append(time.monotonic())
        window.close()
    elif time.monotonic()-started > 10:
        window.close()
    else:
        QTimer.singleShot(25,close_when_running)
QTimer.singleShot(25,close_when_running)
app.exec()
assert closed and cancelled and not worker.isRunning()
assert worker.capture is None and not window.isVisible() and not dialog.isVisible()
assert not marker.with_suffix('.png').exists()
print(json.dumps(dict(clean_exit=True,close_seconds=time.monotonic()-closed[0],helper_pid=int(marker.read_text()))))
'''
    result = run_cancellable([sys.executable, '-c', harness, str(marker)], timeout=20,
        env=environment, cwd=Path(__file__).resolve().parents[1],
        creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stdout + result.stderr
    details = json.loads(result.stdout.strip().splitlines()[-1])
    assert details['clean_exit'] and details['close_seconds'] < 3
    assert 'QThread: Destroyed' not in result.stderr


def test_pyvista_start_up_defers_pyplot_until_it_is_used():
    code = ("import sys; from layer_viewer.__main__ import defer_pyplot; defer_pyplot(); "
            "import pyvista; assert 'matplotlib.figure' not in sys.modules; "
            "import matplotlib.pyplot as plt; assert plt.rcParams['axes.prop_cycle']; "
            "assert 'matplotlib.figure' in sys.modules")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
