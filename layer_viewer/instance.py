"""Reopen the existing live window instead of loading the board a second time."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from qtpy.QtCore import QLockFile
from qtpy.QtNetwork import QLocalServer, QLocalSocket


def instance_name():
    from kipy.kicad import _default_socket_path
    endpoint = _default_socket_path()
    # The same URL can be advertised by two Windows KiCad processes when one
    # failed to bind its listener. Never activate another editor's live viewer.
    identity = endpoint + "\0" + os.environ.get("KICAD_API_TOKEN", "")
    return "kicad-layer-view-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def activate_existing(name):
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(150):
        return False
    socket.write(b"activate\n")
    socket.flush()
    socket.waitForBytesWritten(150)
    socket.disconnectFromServer()
    return True


def listen(window, name):
    """Own the endpoint, or return None if another launch won the race."""
    lock_directory = Path(tempfile.gettempdir()) / "kicad-layer-viewer"
    lock_directory.mkdir(exist_ok=True)
    lock = QLockFile(str(lock_directory / (name + ".lock")))
    # A live viewer can remain open for days; only a dead process makes this
    # lock stale. Windows permits several named-pipe servers with the same name,
    # so QLocalServer.listen() by itself does not provide mutual exclusion.
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        if lock.error() != QLockFile.LockFailedError:
            raise RuntimeError(f"Could not lock the live viewer endpoint: {lock.error()}")
        activate_existing(name)
        return None
    server = QLocalServer(window)
    server.instance_lock = lock
    server.destroyed.connect(lock.unlock)
    server.setSocketOptions(QLocalServer.UserAccessOption)

    def connected():
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            socket.disconnected.connect(socket.deleteLater)
            socket.disconnectFromServer()
            window.showNormal() if window.isMinimized() else window.show()
            window.raise_()
            window.activateWindow()
            if not window.follow_action.isChecked():
                window.follow_action.setChecked(True)

    server.newConnection.connect(connected)
    if not server.listen(name):
        # On Unix an unclean exit can leave a socket name. Probe before cleanup
        # so a concurrent plugin launch never removes a live viewer endpoint.
        if activate_existing(name):
            lock.unlock()
            server.deleteLater()
            return None
        QLocalServer.removeServer(name)
        if not server.listen(name):
            lock.unlock()
            raise RuntimeError(f"Could not register the live viewer: {server.errorString()}")
    return server
