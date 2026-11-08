"""Run a viewer-owned helper and always reap it on cancellation or timeout."""
from __future__ import annotations

import subprocess
import time
import os


def run_cancellable(command, *, timeout, check_cancel=None, **options):
    if check_cancel is not None:
        check_cancel()
    job = None
    if os.name == 'nt':
        from .windows_job import WindowsJob
        job = WindowsJob()
        options['creationflags'] = options.get('creationflags', 0) | 4  # CREATE_SUSPENDED
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, **options) as process:
            deadline = time.monotonic() + timeout
            try:
                if job is not None:
                    job.start(process)
                while True:
                    if check_cancel is not None:
                        check_cancel()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        stdout, stderr = process.communicate(timeout=min(.1, remaining))
                        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                # Windows venv python.exe redirects to another process. Reap
                # the entire owned tree, not just the Popen launcher PID.
                if job is not None:
                    job.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
    finally:
        if job is not None:
            job.close()
