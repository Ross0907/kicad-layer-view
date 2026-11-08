"""Two plugin launches share one live window, including simultaneous startup."""
import os
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QObject
from qtpy.QtGui import QAction
from qtpy.QtWidgets import QApplication

from layer_viewer.instance import activate_existing, listen


def test_second_listener_activates_existing_window_instead_of_loading_twice():
    application = QApplication.instance() or QApplication([])

    class Window(QObject):
        def __init__(self):
            super().__init__()
            self.follow_action = QAction(self)
            self.follow_action.setCheckable(True)
            self.activations = 0

        def isMinimized(self):
            return False

        def show(self):
            pass

        def raise_(self):
            pass

        def activateWindow(self):
            self.activations += 1

    original, raced = Window(), Window()
    name = "layer-view-test-" + uuid.uuid4().hex
    server = listen(original, name)
    try:
        assert server.isListening()
        assert listen(raced, name) is None
        application.processEvents()
        assert original.activations == 1
        assert original.follow_action.isChecked()
        assert raced.activations == 0
        assert activate_existing(name)
        application.processEvents()
        assert original.activations == 2
    finally:
        server.close()
        application.processEvents()
