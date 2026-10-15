#!/usr/bin/env python3
"""PACKAGE_MANIFEST.json 생성·검증 (배포자용) — git 추적 파일별 path·bytes·mode·sha256.

  python scripts/package_manifest.py           # 생성(덮어쓰기)
  python scripts/package_manifest.py --check   # 검증 — 추적 파일 집합·바이트·sha256·mode 가 다르면 종료코드 1
manifest 자신은 목록에 넣지 않는다. 형식은 cicd 패키지 공통 형식(manifest_version 1.0.0)이다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "PACKAGE_MANIFEST.json"
MANIFEST_VERSION = "1.0.0"
PACKAGE_NAME = "langchain-model-preset-gen"
INVENTORY_EXCLUDES = [MANIFEST, ".git/", "__pycache__/", ".pytest_cache/"]
JSON_INDENT = 2


def tracked_files(root: Path = ROOT) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.splitlines() if p != MANIFEST)


def entry(root: Path, rel: str) -> dict:
    path = root / rel
    data = path.read_bytes()
    return {"path": rel, "bytes": len(data), "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
            "sha256": hashlib.sha256(data).hexdigest()}


def build(root: Path = ROOT) -> dict:
    files = [entry(root, rel) for rel in tracked_files(root)]
    return {"manifest_version": MANIFEST_VERSION, "package": PACKAGE_NAME, "inventory_excludes": INVENTORY_EXCLUDES,
            "file_count": len(files), "files": files}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PACKAGE_MANIFEST.json 생성·검증")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    want = build()
    path = ROOT / MANIFEST
    if not a.check:
        path.write_text(json.dumps(want, ensure_ascii=False, indent=JSON_INDENT) + "\n", encoding="utf-8")
        print(f"MANIFEST written ({want['file_count']} files)")
        return 0
    have = json.loads(path.read_text(encoding="utf-8"))
    if have != want:
        have_files = {f["path"]: f for f in have["files"]}
        want_files = {f["path"]: f for f in want["files"]}
        diff = sorted(p for p in have_files.keys() | want_files.keys() if have_files.get(p) != want_files.get(p))
        print(f"MANIFEST MISMATCH: {diff}")
        return 1
    print(f"MANIFEST OK ({want['file_count']} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
