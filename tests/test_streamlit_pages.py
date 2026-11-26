"""Headless Smoke Tests for all 10 Streamlit Pages via Streamlit AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
ALL_PAGES = [APP_DIR / "Home.py"] + sorted((APP_DIR / "pages").glob("*.py"))


@pytest.mark.parametrize("page_path", ALL_PAGES, ids=[p.name for p in ALL_PAGES])
def test_streamlit_page_renders_without_exception(page_path: Path) -> None:
    at = AppTest.from_file(str(page_path), default_timeout=25)
    at.run()
    assert not at.exception, f"Streamlit exception in {page_path.name}: {at.exception}"
