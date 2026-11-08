"""Cancellation must reap the actual helper process and close both pipes."""
import os
import subprocess
import sys
from pathlib import Path
import time

import pytest

from layer_viewer.helper_process import run_cancellable


@pytest.mark.parametrize('reason', ['cancel', 'timeout'])
def test_running_child_is_reaped_on_exit(monkeypatch, reason):
    original = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    checks = 0
    def cancel():
        nonlocal checks
        checks += 1
        if reason == 'cancel' and checks >= 3:
            assert children and children[0].poll() is None
            raise RuntimeError('closed by user')
    with pytest.raises(RuntimeError if reason == 'cancel' else subprocess.TimeoutExpired):
        run_cancellable([sys.executable, '-c', 'import time; time.sleep(60)'],
            timeout=.25, check_cancel=cancel,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed


def test_cancelled_probe_does_not_cache_a_driver_failure(monkeypatch):
    from layer_viewer import raytrace_backend
    raytrace_backend.clear_probe_cache()
    def cancel():
        raise RuntimeError('closing viewer')
    with pytest.raises(RuntimeError, match='closing viewer'):
        raytrace_backend.gpu_backend_available(check_cancel=cancel)
    monkeypatch.setattr(raytrace_backend, 'run_cancellable', lambda *args, **kwargs:
        subprocess.CompletedProcess(args[0], 0, '{"usable":true,"reason":"verified"}', ''))
    assert raytrace_backend.gpu_backend_available() == (True, 'verified')
    raytrace_backend.clear_probe_cache()


def _windows_process_api():
    import ctypes as c
    from ctypes import wintypes as w
    api = c.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes, api.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
    api.WaitForSingleObject.argtypes, api.WaitForSingleObject.restype = [w.HANDLE, w.DWORD], w.DWORD
    api.TerminateProcess.argtypes, api.TerminateProcess.restype = [w.HANDLE, w.UINT], w.BOOL
    api.CloseHandle.argtypes, api.CloseHandle.restype = [w.HANDLE], w.BOOL
    return api


def _tree_script(tmp_path):
    grandchild = "import os,time; from pathlib import Path; " + (
        f"Path({str(tmp_path / 'grandchild.pid')!r}).write_text(str(os.getpid())); time.sleep(60)")
    return ("import os,sys,subprocess,time; from pathlib import Path; "
        f"Path({str(tmp_path / 'helper.pid')!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); time.sleep(60)")


def _open_tree_handles(api, tmp_path):
    paths = [tmp_path / name for name in ('helper.pid', 'grandchild.pid')]
    if not all(path.is_file() and path.stat().st_size for path in paths):
        return []
    handles = [api.OpenProcess(0x100001, False, int(path.read_text())) for path in paths]
    assert all(handles), 'Helper processes must actually be alive before the close'
    return handles


def _check_and_release_tree(api, handles):
    try:
        assert len(handles) == 2
        assert all(api.WaitForSingleObject(handle, 3000) == 0 for handle in handles), 'Orphan helper remains'
    finally:
        for handle in handles:
            if api.WaitForSingleObject(handle, 0) != 0:
                api.TerminateProcess(handle, 1)
                api.WaitForSingleObject(handle, 3000)
            api.CloseHandle(handle)


@pytest.mark.skipif(os.name != 'nt', reason='Windows redirector and job lifecycle')
@pytest.mark.parametrize('reason', ['cancel', 'timeout'])
def test_real_python_redirector_and_descendant_are_reaped(tmp_path, reason):
    api, handles = _windows_process_api(), []
    def check():
        if not handles:
            handles.extend(_open_tree_handles(api, tmp_path))
        if handles and reason == 'cancel':
            raise RuntimeError('close the viewer')
    try:
        with pytest.raises(RuntimeError if reason == 'cancel' else subprocess.TimeoutExpired):
            run_cancellable([sys.executable, '-c', _tree_script(tmp_path)], timeout=1.5,
                check_cancel=check, creationflags=subprocess.CREATE_NO_WINDOW)
    finally:
        _check_and_release_tree(api, handles)


@pytest.mark.skipif(os.name != 'nt', reason='Windows job owner termination')
def test_abrupt_owner_exit_reaps_helper_tree(tmp_path):
    api, handles = _windows_process_api(), []
    package = Path(__file__).resolve().parents[1]
    script = (f"import sys; sys.path.insert(0,{str(package)!r}); "
        "from layer_viewer.helper_process import run_cancellable; "
        f"run_cancellable({[sys.executable, '-c', _tree_script(tmp_path)]!r}, timeout=60, creationflags=0x08000000)")
    # Use the real executable for the test owner so terminating this PID really
    # simulates loss of the viewer, not loss of another venv redirector.
    owner = subprocess.Popen([sys._base_executable, '-c', script],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 5
        while not handles and owner.poll() is None and time.monotonic() < deadline:
            handles.extend(_open_tree_handles(api, tmp_path))
            time.sleep(.02)
        assert handles, owner.stderr.read() if owner.poll() is not None else 'Helpers did not start'
        owner.terminate()
        owner.wait(timeout=3)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=3)
        owner.stderr.close()
        _check_and_release_tree(api, handles)
