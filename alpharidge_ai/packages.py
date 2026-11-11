"""Repairs an environment upgraded in place from bittensor 10.0.

The old `scalecodec` package and its replacement `cyscale` install the same module, and
async-substrate-interface refuses to load while both are present. Pip never removes the old
one on upgrade, so the neuron entry points call `repair_substrate_codec()` before importing
bittensor: it removes both, reinstalls `cyscale` and restarts the process once.
"""
import importlib.metadata as md
import os
import subprocess
import sys


def repair_substrate_codec() -> None:
    try:
        md.version("scalecodec")
        cyscale = md.version("cyscale")
    except md.PackageNotFoundError:
        return
    print("[SETUP] Replacing scalecodec with cyscale (one-time, after the bittensor upgrade)",
          flush=True)
    pip = [sys.executable, "-m", "pip"]
    try:
        subprocess.run(pip + ["uninstall", "-y", "scalecodec", "cyscale"], check=True)
        subprocess.run(pip + ["install", "--no-deps", "--force-reinstall",
                              f"cyscale=={cyscale}"], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"[SETUP] Could not replace scalecodec ({e}). Run: pip uninstall -y "
                         f"scalecodec cyscale && pip install cyscale=={cyscale}")
    os.execv(sys.executable, getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
