"""Tests for the ``ocr-extract`` command-line interface."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ocr_extractor.cli import main

from .conftest import SAMPLE_TEXT

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def test_plain_text_output(text_image_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(text_image_path)]) == 0
    assert capsys.readouterr().out.strip() == SAMPLE_TEXT


def test_json_output(text_image_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(text_image_path), "--json", "--no-preprocess"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == SAMPLE_TEXT
    assert payload["preprocessing"] == []
    assert len(payload["words"]) == 3


def test_multiple_files_and_output_file(text_image_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    assert main([str(text_image_path), str(text_image_path), "-o", str(out)]) == 0
    content = out.read_text()
    assert content.count(SAMPLE_TEXT) == 2
    assert "==> " in content


def test_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope.png")]) == 2
    assert "file not found" in capsys.readouterr().err


def test_bad_language(text_image_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(text_image_path), "--lang", "xx_bad"]) == 1
    assert "error" in capsys.readouterr().err


def test_list_langs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-langs"]) == 0
    assert "eng" in capsys.readouterr().out.split()


def test_verbose_logs_pipeline(text_image_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert main([str(text_image_path), "--verbose"]) == 0
    assert any("Preprocessing applied" in record.getMessage() for record in caplog.records)


def test_requires_an_image(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    assert "at least one image" in capsys.readouterr().err
