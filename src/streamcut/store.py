"""Atomic records, task identities and private command diagnostics."""

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from .model import CutError

SCHEMA = 1


def code_identity():
    return fingerprint({p.name: digest(p) for p in Path(__file__).parent.glob("*.py")})


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(
            f, parse_constant=lambda _: (_ for _ in ()).throw(CutError("nonfinite JSON number"))
        )


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as f:
            os.chmod(temporary, 0o600)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def within(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise CutError("task path must be relative")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise CutError("task path escapes the project")
    return path


def command(args, log, timeout=7200, cwd=None):
    """Do not print command lines; media URLs may contain credentials."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as f:
        os.chmod(log, 0o600)
        try:
            result = subprocess.run(
                [str(a) for a in args],
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=f,
                timeout=timeout,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            raise CutError("command timed out; inspect the private stage log") from None
    if result.returncode:
        raise CutError("command failed; inspect the private stage log")


def task(root, stage, index, payload):
    body = {"schema": SCHEMA, "stage": stage, "index": index, "payload": payload}
    body["key"] = fingerprint(body)
    write(Path(root) / "tasks" / stage / f"{index:04d}.json", body)
    return body


def result_path(root, spec):
    return Path(root) / "results" / spec["stage"] / (spec["key"] + ".json")


def cached(root, spec):
    path = result_path(root, spec)
    if not path.exists():
        return None
    try:
        data = read(path)
        if data.get("key") != spec["key"] or data.get("schema") != SCHEMA:
            raise CutError("cached result identity mismatch")
        if data.get("error"):
            raise CutError("worker failed; restart with --resume after fixing the cause")
        if data.get("value_hash") != fingerprint(data.get("value")):
            raise CutError("cached result was modified")
        for relative, expected in data.get("files", {}).items():
            if digest(within(root, relative)) != expected:
                raise CutError("cached media was modified")
        return data["value"]
    except (OSError, ValueError, KeyError, TypeError):
        raise CutError("cached result is incomplete or corrupt") from None


def save_result(root, spec, value, files=(), elapsed=0):
    write(
        result_path(root, spec),
        {
            "schema": SCHEMA,
            "key": spec["key"],
            "value": value,
            "value_hash": fingerprint(value),
            "seconds": round(elapsed, 3),
            "files": {str(Path(p).relative_to(root)): digest(p) for p in files},
        },
    )


def wait(root, specs, timeout):
    deadline = time.monotonic() + timeout
    announced = 0
    verified = {}
    while True:
        for spec in specs:
            if spec["key"] not in verified:
                value = cached(root, spec)
                if value is not None:
                    verified[spec["key"]] = value
        results = [verified.get(s["key"]) for s in specs]
        done = sum(r is not None for r in results)
        if done == len(specs):
            return results
        if time.monotonic() > deadline:
            raise CutError("waiting for workers timed out; check worker notebooks and resume")
        if time.monotonic() - announced > 20:
            print(f"{specs[0]['stage']}: {done}/{len(specs)} 完了", flush=True)
            announced = time.monotonic()
        time.sleep(2)
