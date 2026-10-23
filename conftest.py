"""pytest: setup.py's data-folder migration (#198) saves the data folder it settled on in the user's settings file
(%APPDATA%\Strata\settings.json, ~/.config/strata/settings.json) and can MOVE model files from a folder it finds there.
A test that reaches it must never touch the real file, so the whole session gets its own config home.

The session's temporary files go to a folder of its own as well (TMPDIR, so also those of the programs the tests start),
removed with the config home when the session ends: a test that leaves its files behind leaves nothing.  Not on a Mac,
whose TMPDIR is already long and a Unix socket path has a 104-byte limit there."""
import atexit
import os
import shutil
import sys
import tempfile

if sys.platform != "darwin":
    _tmp = tempfile.mkdtemp(prefix="strata-test-")
    atexit.register(shutil.rmtree, _tmp, True)
    for _name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[_name] = _tmp
    tempfile.tempdir = _tmp
_home = tempfile.mkdtemp(prefix="strata-pytest-config-")
atexit.register(shutil.rmtree, _home, True)
os.environ["APPDATA"] = _home
os.environ["XDG_CONFIG_HOME"] = _home
