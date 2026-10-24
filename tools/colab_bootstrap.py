"""Colab bootstrap without depending on the kernel's pip/ensurepip or Python version."""

import hashlib
import io
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

UV_VERSION = "0.12.23"
UV_WHEEL = (
    "https://files.pythonhosted.org/packages/e5/83/85a6c63c24905af4924fddb11a499b934913f59a134248367a1ef1a4716f/"
    "uv-0.12.23-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
UV_SHA256 = "565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff"


def ensure_uv(directory):
    found = shutil.which("uv")
    if found:
        version = subprocess.check_output([found, "--version"], text=True).split()[1]
        if version == UV_VERSION:
            return found
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("This bootstrap targets Colab Linux x86_64")
    target = Path(directory) / "uv"
    if target.exists():
        version = subprocess.check_output([str(target), "--version"], text=True).split()[1]
        if version == UV_VERSION:
            return str(target)
        raise RuntimeError("Unexpected bootstrap executable version")
    with urllib.request.urlopen(UV_WHEEL, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != UV_SHA256:
        raise RuntimeError("uv download checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        binary = archive.read(f"uv-{UV_VERSION}.data/scripts/uv")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    temporary.write_bytes(binary)
    temporary.chmod(0o755)
    os.replace(temporary, target)
    return str(target)


def bootstrap(project, extras, environment="/content/streamcut-env312"):
    project, environment = Path(project).resolve(), Path(environment).resolve()
    uv = ensure_uv(environment.parent / "streamcut-bootstrap")
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    subprocess.run([uv, "python", "install", "3.12"], check=True, env=env)
    args = [
        uv,
        "sync",
        "--locked",
        "--project",
        str(project),
        "--python",
        "3.12",
        "--no-default-groups",
    ]
    for extra in extras.split(","):
        args += ["--extra", extra]
    subprocess.run(args, check=True, env=env)
    return str(environment / "bin/python")
