"""Optional geometry work, performed after the PCB can already be inspected."""
from qtpy.QtCore import QThread, Signal


class SceneTask(QThread):
    ready = Signal(object, object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, snapshot, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot

    def check_cancel(self):
        if self.isInterruptionRequested():
            raise InterruptedError("Scene task cancelled")

    def info(self, message):
        self.check_cancel()
        self.progress.emit(str(message))

    def run(self):
        try:
            result = self.build()
            self.check_cancel()
            self.ready.emit(self.snapshot, result)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class ComponentTask(SceneTask):
    def __init__(self, snapshot, signature, slot_count, parent=None):
        super().__init__(snapshot, parent)
        self.signature, self.slot_count = signature, slot_count

    def build(self):
        from .components import build_component_parts
        return build_component_parts(self.snapshot, self.slot_count, progress=self)


class ViaDetailTask(SceneTask):
    def __init__(self, snapshot, scene, parent=None):
        super().__init__(snapshot, parent)
        self.scene = scene

    def build(self):
        from .geometry import build_scene, scene_keys
        return build_scene(self.snapshot, progress=self, previous_scene=self.scene,
                           requested_keys=scene_keys(self.snapshot, via_details=True))
