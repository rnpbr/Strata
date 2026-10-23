#!/bin/sh
# Strata for Linux: the first run installs everything and starts the model; later runs just start it.
# Needs only an NVIDIA driver (or, for an AMD Radeon card, the kernel's amdgpu driver: see docs/AMD_HIP.md).
# Python: on macOS and 64-bit Linux a pinned standalone CPython (the Terminal archive's) goes into .python, so every PC
# runs the same, current interpreter with no sudo and nothing installed system-wide; elsewhere a system Python 3.11+.
# NixOS (or STRATA_UV=1) with uv installed: uv makes the project venv; every other Linux uses the standard venv.
cd "$(dirname "$0")" || exit 1
# A bundled Python runtime (the python/ folder of the terminal archive) is used when it is there: nothing to find,
# install or download. update.sh runs this file, so it uses it too.
if [ -x python/bin/python3 ]; then
  PYTHONNOUSERSITE=1; export PYTHONNOUSERSITE
  unset PYTHONHOME PYTHONPATH
  exec python/bin/python3 setup.py "$@"
fi
# python-build-standalone 20261021, CPython 3.14.9: the files packaging/package_terminal.py pins
PBS_VERSION=3.14.9
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PBS_ARCH=aarch64-apple-darwin; PBS_SUM=226d696642a8a53d03b4d5b255b1c319b1575a6d44bd723a953c9f507755e36a ;;
  Linux-aarch64) PBS_ARCH=aarch64-unknown-linux-gnu; PBS_SUM=847895ba727440b012207bb0192282d7e182eb991ae5c7bb5c35c6a12a99e732 ;;
  Linux-x86_64) PBS_ARCH=x86_64-unknown-linux-gnu; PBS_SUM=dfa4c5a40c47f338f41658017be550c72247c391e6bfb1fa4da03ebc8451abdf ;;
  *) PBS_ARCH="" ;;
esac
PBS="https://github.com/astral-sh/python-build-standalone/releases/download/20261021/cpython-3.14.9%2B20261021-$PBS_ARCH-install_only.tar.gz"
# Python 3.11+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; }
# uv's own Python only on NixOS (its Python cannot load pip wheels' libraries), or when asked: STRATA_UV=1.
# Everywhere else the standard venv stays the default (#662: a silent toolchain change was not wanted).
UV=""
if [ "${STRATA_UV:-}" = "1" ] || grep -qi '^ID=nixos' /etc/os-release 2>/dev/null; then
  UV="$(command -v uv 2>/dev/null || true)"
fi
# The .venv is made again when it is half-made (no pip), too old (Pillow 13 needs 3.11) or not on the pinned interpreter
# (an earlier setup's): never while a Strata server of this user runs (serve/server.py, or -m serve.server as the
# Terminal starts it; any folder's, to be safe) - then on a later run, once it is stopped.
if [ -x .venv/bin/python ]; then
  STALE=""
  if ! .venv/bin/python -m pip --version >/dev/null 2>&1 ||
     ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    STALE=1
  elif [ -n "$PBS_ARCH" ] && [ -z "$UV" ] &&
       [ "$(sed -n -e 's/^version = //p' -e 's/^version_info = //p' .venv/pyvenv.cfg 2>/dev/null | head -1)" != "$PBS_VERSION" ]; then
    STALE=1
  fi
  if [ -n "$STALE" ] && ! pgrep -u "$(id -u)" -f 'serve[/.]server' >/dev/null 2>&1; then
    rm -rf .venv
  fi
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  if [ -n "$UV" ]; then
    # Prefer uv's own interpreter: unlike Nix's Python, it works naturally with pip wheels using dlopen.
    PY=3.14
  elif [ -n "$PBS_ARCH" ]; then
    if [ "$(cat .python/SHA256 2>/dev/null)" != "$PBS_SUM" ] || ! ok_py .python/python/bin/python3; then
      echo "Downloading the pinned Python $PBS_VERSION into .python (about 30 MB, once) ..."
      rm -rf .python && mkdir -p .python
      if curl -fL --retry 3 -o .python/python.tar.gz "$PBS"; then
        SUM="$( (sha256sum .python/python.tar.gz 2>/dev/null || shasum -a 256 .python/python.tar.gz) | cut -d" " -f1)"
        if [ "$SUM" != "$PBS_SUM" ]; then
          rm -rf .python; echo "the Python download does not match its pinned SHA-256"; exit 1
        fi
        tar xzf .python/python.tar.gz -C .python && rm -f .python/python.tar.gz && echo "$PBS_SUM" > .python/SHA256
      else
        rm -rf .python; echo "could not download Python: trying this system's own"
      fi
    fi
    ok_py .python/python/bin/python3 && PY=.python/python/bin/python3
  fi
  if [ -z "$PY" ]; then
    for c in python3 python; do
      if command -v $c >/dev/null 2>&1 && ok_py $c; then
        PY=$c; break
      fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "Please install Python 3.11 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv), then run"
    echo "./setup.sh again. On NixOS, add uv to environment.systemPackages."
    exit 1
  fi
  if [ -n "$UV" ]; then
    # Force an uv-managed interpreter: pip wheels may dlopen libraries outside the Nix store.
    "$UV" venv --managed-python --seed --python "$PY" .venv || { echo "Could not create the uv environment in .venv."; exit 1; }
  else
    # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
    $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
  fi
fi
exec .venv/bin/python setup.py "$@"
