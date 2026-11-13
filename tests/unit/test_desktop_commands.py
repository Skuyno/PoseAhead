import os
import queue
import subprocess
import sys
import threading

import pytest

from poseahead.desktop.commands import listen_for_stop


@pytest.fixture
def command_pipe():
    read_fd, write_fd = os.pipe()
    stop, finished = threading.Event(), threading.Event()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    errors = []

    def read():
        try:
            listen_for_stop(stream, stop, finished)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        yield write_fd, stop, finished, thread
    finally:
        finished.set()
        thread.join(timeout=1)
        os.close(write_fd)
        stream.close()
        assert not thread.is_alive()
        assert not errors


@pytest.mark.parametrize("command", [b"stop\n", b" stop\r\n", b"unknown\nstop\n"])
def test_stop_command_is_read_from_a_real_pipe(command_pipe, command):
    fd, stop, _, thread = command_pipe
    os.write(fd, command)
    assert stop.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_partial_stop_does_not_block_or_trigger_early(command_pipe):
    fd, stop, finished, _ = command_pipe
    os.write(fd, b"st")
    assert not stop.wait(.1)
    os.write(fd, b"op\r")
    assert not stop.wait(.1)
    os.write(fd, b"\n")
    assert stop.wait(1)
    assert not finished.is_set()


def test_oversized_or_unknown_commands_do_not_hide_following_stop(command_pipe):
    fd, stop, _, _ = command_pipe
    os.write(fd, b"x" * 1024 + b"stop\nSTOP\n")
    assert not stop.wait(.1)
    os.write(fd, b"stop\n")
    assert stop.wait(1)


def test_idle_command_reader_can_be_finished_without_closing_stdin(command_pipe):
    _, stop, finished, thread = command_pipe
    finished.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not stop.is_set()


def test_closed_command_pipe_exits_without_requesting_stop():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with os.fdopen(read_fd, "rb", buffering=0) as stream:
        stop = threading.Event()
        thread = threading.Thread(
            target=listen_for_stop, args=(stream, stop, threading.Event()), daemon=True
        )
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert not stop.is_set()


def test_native_import_with_idle_command_pipe_does_not_deadlock():
    # Keep stdin OPEN while native DLLs initialize. communicate(input=...) would
    # close it and mask the Windows deadlock that occurred in the actual UI.
    code = """
import sys, threading, time
from poseahead.desktop.commands import listen_for_stop
stop, finished = threading.Event(), threading.Event()
thread = threading.Thread(target=listen_for_stop, args=(sys.stdin, stop, finished))
thread.start()
time.sleep(.1)
import numpy
print('native-ready', flush=True)
stopped = stop.wait(10)
finished.set()
thread.join(timeout=1)
print('stopped' if stopped else 'missing-stop', flush=True)
sys.exit(0 if stopped and not thread.is_alive() else 1)
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    lines = queue.Queue()

    def output():
        for line in process.stdout:
            lines.put(line.decode("utf-8", errors="replace").strip())

    reader = threading.Thread(target=output, daemon=True)
    reader.start()
    try:
        assert lines.get(timeout=10) == "native-ready"
        process.stdin.write(b"st")
        process.stdin.flush()
        process.stdin.write(b"op\n")
        process.stdin.flush()
        assert process.wait(timeout=10) == 0
        assert lines.get(timeout=1) == "stopped"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        reader.join(timeout=1)
        process.stdin.close()
        process.stdout.close()
