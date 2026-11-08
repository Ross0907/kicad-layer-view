"""Own only a helper's process tree, including the Windows Python redirector.

Start the helper suspended, assign it to a kill-on-close job, then resume its
primary thread. No child can escape ownership before Python starts executing.
Windows closes the non-inherited job handle even if the viewer is terminated.
https://learn.microsoft.com/windows/win32/procthread/job-objects
"""
import ctypes as c
from ctypes import wintypes as w


class _BasicLimits(c.Structure):
    _fields_ = [('process_time', c.c_int64), ('job_time', c.c_int64),
                ('flags', w.DWORD), ('min_working_set', c.c_size_t),
                ('max_working_set', c.c_size_t), ('active_processes', w.DWORD),
                ('affinity', c.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]


class _ExtendedLimits(c.Structure):
    _fields_ = [('basic', _BasicLimits), ('io_counters', c.c_uint64 * 6),
                ('process_memory', c.c_size_t), ('job_memory', c.c_size_t),
                ('peak_process_memory', c.c_size_t), ('peak_job_memory', c.c_size_t)]


class _ThreadEntry(c.Structure):
    _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('thread_id', w.DWORD),
                ('process_id', w.DWORD), ('base_priority', w.LONG),
                ('delta_priority', w.LONG), ('flags', w.DWORD)]


class WindowsJob:
    def __init__(self):
        self.api = c.WinDLL('kernel32', use_last_error=True)
        definitions = {
            'CreateJobObjectW': ([w.LPVOID, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, c.c_int, w.LPVOID, w.DWORD], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'CreateToolhelp32Snapshot': ([w.DWORD, w.DWORD], w.HANDLE),
            'Thread32First': ([w.HANDLE, c.POINTER(_ThreadEntry)], w.BOOL),
            'Thread32Next': ([w.HANDLE, c.POINTER(_ThreadEntry)], w.BOOL),
            'OpenThread': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'ResumeThread': ([w.HANDLE], w.DWORD),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in definitions.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        # No security attributes: the helper cannot inherit the owning handle.
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.close()
            raise error

    def start(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise c.WinError(c.get_last_error())
        snapshot = self.api.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        try:
            entry = _ThreadEntry()
            entry.size = c.sizeof(entry)
            found = self.api.Thread32First(snapshot, c.byref(entry))
            while found:
                if entry.process_id == process.pid:
                    thread = self.api.OpenThread(2, False, entry.thread_id)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise c.WinError(c.get_last_error())
                    try:
                        if self.api.ResumeThread(thread) == 0xffffffff:
                            raise c.WinError(c.get_last_error())
                    finally:
                        self.api.CloseHandle(thread)
                    return
                found = self.api.Thread32Next(snapshot, c.byref(entry))
            raise RuntimeError('The helper primary thread could not be resumed')
        finally:
            self.api.CloseHandle(snapshot)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
