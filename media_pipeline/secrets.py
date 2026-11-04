"""API keys come from environment variables only.

On macOS you can keep them in the Keychain and have the CLI read them at
runtime instead: set e.g. ANTHROPIC_KEYCHAIN_SERVICE=<your keychain item name>
and the key is read with `security find-generic-password -w` into the
process environment. Nothing is ever written to disk or logged.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


def get_secret(env_name: str) -> Optional[str]:
    value = os.environ.get(env_name)
    if value:
        return value.strip()
    service = os.environ.get(env_name.replace("_API_KEY", "") + "_KEYCHAIN_SERVICE")
    if service and sys.platform == "darwin":
        try:
            out = subprocess.run(["security", "find-generic-password", "-s", service, "-w"],
                                 capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode == 0 and out.stdout.strip():
            os.environ[env_name] = out.stdout.strip()  # so the SDK finds it too
            return os.environ[env_name]
    return None
