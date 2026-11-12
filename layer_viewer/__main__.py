"""CLI and KiCad plugin application bootstrap."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import tempfile
import time


def main(plugin=False):
    parser = argparse.ArgumentParser(description="Read-only exploded KiCad PCB viewer")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--demo", action="store_true")
    sources.add_argument("--snapshot", type=Path)
    sources.add_argument("--board", type=Path)
    parser.add_argument("--gap", type=float, default=0.0, help="Separation between physical layers (mm)")
    parser.add_argument("--render", type=Path, help="Render viewport PNG and exit")
    parser.add_argument("--window-screenshot", type=Path, help="Save full window when --render completes")
    args = parser.parse_args()
    os.environ.setdefault("QT_API", "pyside6")
    log_dir = Path(tempfile.gettempdir()) / "kicad-layer-viewer"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"viewer-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"
    if plugin:
        stream = log_path.open("a", buffering=1, encoding="utf-8")
        for fd in (1, 2):
            try:
                os.dup2(stream.fileno(), fd)
            except OSError:
                pass
        sys.stdout = sys.stderr = stream
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Layer View")
    app.setOrganizationName("KiCad Layer Viewer")
    live = not (args.demo or args.snapshot or args.board or args.render)
    from . import __version__
    logging.info("Layer View %s; live=%s; editor-bound=%s", __version__, live,
                 bool(os.environ.get("KICAD_API_TOKEN")))
    if live:
        from .instance import activate_existing, instance_name
        name = instance_name()
        if activate_existing(name):
            return 0
        # Read the board from KiCad while Qt/VTK load and the window is built.
        from . import prefetch
        prefetch.start()
    from .app import MainWindow
    window = MainWindow(gap=max(0, min(args.gap, 30)))
    if live:
        from .instance import listen
        window.instance_server = listen(window, name)
        if window.instance_server is None:
            window.close()
            return 0
    window.automation = bool(args.render)
    exit_code = [0]
    rendered = [False]

    def render():
        if rendered[0] or window._closing:
            return
        # A staged surface frame is useful interactively, but automation must
        # capture the completed board and any requested component models.
        if (window._board_preview or window._detail_worker is not None
                or (window.kind_boxes["component"].isChecked()
                    and window._component_worker is not None)):
            QTimer.singleShot(150, render)
            return
        rendered[0] = True
        try:
            args.render.parent.mkdir(parents=True, exist_ok=True)
            window.viewport.screenshot(args.render)
            if args.window_screenshot:
                args.window_screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(args.window_screenshot))
        except Exception:
            logging.exception("Render failed")
            exit_code[0] = 1
        # closeEvent waits for the geometry/cache worker to finish. Let Qt quit
        # when the window actually closes; app.quit() here can destroy a live
        # QThread while a newly built scene is still being cached.
        window.close()

    def fail(message):
        if args.render:
            print(message, file=sys.stderr)
            exit_code[0] = 1
            QTimer.singleShot(100, window.close)

    if args.render:
        window.scene_ready.connect(lambda: QTimer.singleShot(800, render))
        window.load_failed.connect(fail)
    window.show()
    source = "demo" if args.demo else "snapshot" if args.snapshot else "board" if args.board else "ipc"
    QTimer.singleShot(100, lambda: window.load(source, args.snapshot or args.board))
    result = app.exec()
    return exit_code[0] or result


if __name__ == "__main__":
    raise SystemExit(main())
