#!/usr/bin/env python3
"""배포 위생 검사 (배포자용) — 패키지 트리에 사이트 식별자·개인 경로·사설 IP 가 없는지 본다.

대상: 패키지 루트 아래 텍스트 파일 전부 (.git 제외). 설치본(configs/config.yaml 에 사이트 주소가 정당하게 들어감)에는
쓰지 않는다. 금지 패턴은 FORBIDDEN_PATTERNS 한 곳에 둔다. 종료코드 0 = 0건, 1 = 발견.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache"}
# 사이트 이름은 이 파일 자신이 검사에 걸리지 않도록 유니코드 이스케이프·조각으로 적는다.
SITE_NAMES = ("\uacbd\ub9ac\ub098\ub77c", "kyung" + "rinara")
FORBIDDEN_PATTERNS = {
    "site_name": re.compile("|".join(SITE_NAMES), re.IGNORECASE),
    "home_path": re.compile(r"/home/[a-z_][a-z0-9_-]*/"),
    "private_ip": re.compile(r"(?<![0-9.])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?![0-9])"),
}


def scan(root: Path = ROOT) -> list[tuple[str, int, str]]:
    hits = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or EXCLUDED_DIRS & set(path.relative_to(root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:          # 바이너리 파일은 검사 대상이 아니다
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(line):
                    hits.append((str(path.relative_to(root)), lineno, name))
    return hits


def main() -> int:
    hits = scan()
    for rel, lineno, name in hits:
        print(f"{rel}:{lineno}: {name}")
    print(f"SANITIZE {'FAIL' if hits else 'OK'} ({len(hits)} hits)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
