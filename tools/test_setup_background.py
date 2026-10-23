"""setup --background (start_in_background): the server runs in a session of its own and setup gives the terminal
back at once; a server that stops in its first second is reported with the end of its output.  And a start, setup
or update is refused while this folder's server runs.

    python tools/test_setup_background.py
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from strata_control import procs  # noqa: E402

# a stand-in server: loads for a long while, or refuses its config at once
SERVER = """import sys, time
print("loading ...", flush=True)
if "--fail" in sys.argv:
    print("--config: no such file", flush=True)
    sys.exit(2)
time.sleep(60)
"""


@unittest.skipIf(setup.WIN, "sessions and signals: POSIX")
class Background(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.script = self.dir / "server.py"
        self.script.write_text(SERVER, encoding="utf-8")
        self.cfg = {"log": str(self.dir / "strata-q2_0.log"), "host": "0.0.0.0"}

    def start(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = setup.start_in_background([sys.executable, str(self.script), *extra], dict(os.environ), self.cfg, 8080)
        return code, out.getvalue()

    def server_pid(self, text):
        return int(text.split("(pid ")[1].split(")")[0])

    def test_gives_the_terminal_back_while_it_loads(self):
        t0 = time.monotonic()
        code, text = self.start()
        pid = self.server_pid(text)
        self.addCleanup(lambda: os.kill(pid, signal.SIGKILL) if alive(pid) else None)
        self.assertEqual(code, 0, text)
        self.assertLess(time.monotonic() - t0, 3)                           # not waiting for the model
        self.assertTrue(alive(pid))                                         # still loading after setup returned
        self.assertNotEqual(os.getsid(pid), os.getsid(0))                   # its own session: a closed terminal is no SIGHUP
        log = self.dir / "strata-q2_0-server.log"
        self.assertIn(f"tail -f {log}", text)
        self.assertIn("other devices: http://<this computer's IP>:8080/", text)
        self.assertIn("--stop", text)
        self.assertIn("loading ...", log.read_text())
        self.assertEqual(log.stat().st_mode & 0o077, 0)                     # owner only

    def test_a_server_that_stops_at_once_is_reported(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            setup.start_in_background([sys.executable, str(self.script), "--fail"], dict(os.environ), self.cfg, 8080)
        self.assertIn("no such file", out.getvalue())


class SecondStart(unittest.TestCase):
    def test_refused_while_this_folder_serves(self):
        looked = []
        running = procs.Ident(4242, "l:1", ("python3", "serve/server.py"))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)), \
                mock.patch.object(setup, "CONTROL", False), \
                mock.patch.object(procs, "servers_of", lambda script: looked.append(script) or [running]), \
                contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            setup.refuse_if_running()
        self.assertEqual(looked, [str(Path(d) / "serve" / "server.py")])        # the folder setup runs in now
        self.assertIn("already running from this folder (pid 4242)", out.getvalue())
        self.assertIn("--stop", out.getvalue())

    def test_nothing_running_goes_on(self):
        with mock.patch.object(setup, "CONTROL", False), mock.patch.object(procs, "servers_of", lambda script: []):
            self.assertIsNone(setup.refuse_if_running())


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with contextlib.suppress(OSError):                                     # a zombie of this test is not alive
        if Path(f"/proc/{pid}/stat").read_text().split(") ")[1].startswith("Z"):
            return False
    return True


if __name__ == "__main__":
    unittest.main()
