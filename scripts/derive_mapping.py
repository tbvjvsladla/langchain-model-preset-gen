#!/usr/bin/env python3
"""CLI 진입점 — 로직은 langchain_model_preset_gen.derive_mapping 에 있다. 사용법: python scripts/derive_mapping.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_model_preset_gen.derive_mapping import main  # noqa: E402

sys.exit(main())
