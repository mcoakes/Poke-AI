#!/usr/bin/env python3
"""Compile the pinned socket functions in a small POSIX regression harness.

Exercises real OS descriptor reuse and EOF; it does not compile the Qt frontend.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def function(source, name):
    match = re.search(r"^[^\n]*\b" + re.escape(name) + r"\([^\n]*\) \{\n.*?^\}",
                      source, re.M | re.S)
    if not match:
        raise ValueError(f"cannot find pinned function {name}")
    return match.group(0)


def run(source_dir):
    source = (source_dir / "src/script/socket.c").read_text()
    close = function(source, "_mScriptSocketClose")
    recv = function(source, "_mScriptSocketRecv")
    compiler = shutil.which("cc")
    if not compiler:
        raise RuntimeError("a C compiler is required (build-essential)")
    # Minimal script-value scaffolding; the two function bodies are read directly
    # from the selected mGBA tree, and the socket operations use the actual OS.
    harness = r'''
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef int Socket;
#define INVALID_SOCKET (-1)
#define SOCKET_FAILED(s) ((s) < 0)
#define mSCRIPT_SOCKERR_OK 0
static int SocketClose(Socket s) { return close(s) >= 0; }
static int SocketError(void) { return errno; }
static ssize_t SocketRecv(Socket s, void* buffer, size_t size) { return read(s, buffer, size); }
struct mScriptSocket { Socket socket; int32_t error; };
struct mScriptString { char* buffer; size_t size; };
struct mScriptValue { union { struct mScriptString* string; } value; };
static struct mScriptValue* mScriptStringCreateEmpty(uint32_t n) {
    struct mScriptValue* v = calloc(1, sizeof(*v));
    v->value.string = calloc(1, sizeof(*v->value.string));
    v->value.string->buffer = calloc(1, n);
    return v;
}
static void _mScriptSocketSetError(struct mScriptSocket* s, int32_t error) { s->error = error; }
'''+close+'\n'+recv+r'''
int main(void) {
    int original[2], replacement[2], eof[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, original)) return 2;
    int reused = fcntl(original[0], F_DUPFD, 64);
    if (reused < 0) return 2;
    close(original[0]); close(original[1]);
    struct mScriptSocket old = { .socket = reused };
    _mScriptSocketClose(&old);  /* explicit Lua close */
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, replacement)) return 2;
    if (dup2(replacement[0], reused) != reused) return 2;
    close(replacement[0]);
    _mScriptSocketClose(&old);  /* later userdata destructor */
    int close_failed = fcntl(reused, F_GETFD) < 0;
    if (!close_failed) {
        char c = 0;
        if (write(reused, "x", 1) != 1 || read(replacement[1], &c, 1) != 1 || c != 'x') return 2;
        close(reused);
    }
    close(replacement[1]);
    printf("descriptor reuse after repeated close: %s\n", close_failed ? "FAIL" : "PASS");

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, eof)) return 2;
    struct mScriptSocket reader = { .socket = eof[0] };
    close(eof[1]);
    errno = EAGAIN; /* Successful read(2) returning EOF does not reset errno. */
    struct mScriptValue* data = _mScriptSocketRecv(&reader, 8);
    int eof_failed = data->value.string->size != 0 || reader.error != 0;
    free(data->value.string->buffer); free(data->value.string); free(data);
    close(eof[0]);
    printf("EOF with stale errno: %s\n", eof_failed ? "FAIL" : "PASS");

    if (socketpair(AF_UNIX, SOCK_STREAM, 0, eof)) return 2;
    if (fcntl(eof[0], F_SETFL, O_NONBLOCK) < 0) return 2;
    reader.socket = eof[0];
    data = _mScriptSocketRecv(&reader, 8);
    int again_failed = reader.error != EAGAIN && reader.error != EWOULDBLOCK;
    free(data->value.string->buffer); free(data->value.string); free(data);
    close(eof[0]); close(eof[1]);
    printf("nonblocking no-data remains retryable: %s\n", again_failed ? "FAIL" : "PASS");
    return close_failed || eof_failed || again_failed;
}
'''
    with tempfile.TemporaryDirectory(prefix="mgba-socket-test-") as directory:
        cfile, executable = Path(directory) / "test.c", Path(directory) / "test"
        cfile.write_text(harness)
        subprocess.run([compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", str(cfile),
                        "-o", str(executable)], check=True)
        return subprocess.run([str(executable)], check=False).returncode


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "third_party/mgba")
    args = parser.parse_args()
    return run(args.source)


if __name__ == "__main__":
    raise SystemExit(main())
