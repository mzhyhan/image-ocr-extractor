"""Command-line interface for image-ocr-extractor.

Installed as the ``ocr-extract`` console script (also runnable as
``python -m ocr_extractor``).

Examples:
    Extract plain text::

        $ ocr-extract receipt.jpg

    Word-level JSON in Ukrainian and English, without preprocessing::

        $ ocr-extract scan.png --json --lang eng+ukr --no-preprocess

    Process several files and save the output::

        $ ocr-extract page1.png page2.png -o pages.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ocr_extractor import __version__
from ocr_extractor.ocr import OCRError, OCRResult, available_languages, extract_text
from ocr_extractor.preprocessing import PreprocessConfig

__all__ = ["build_parser", "main"]

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_OCR_ERROR = 1
EXIT_USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ``ocr-extract``.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="ocr-extract",
        description="Extract text from images using OpenCV preprocessing + Tesseract OCR.",
    )
    parser.add_argument("images", nargs="*", type=Path, help="image file(s) to process")
    parser.add_argument("-l", "--lang", default="eng", help="Tesseract language(s), e.g. eng+ukr")
    parser.add_argument("--json", action="store_true", help="output JSON with word-level details")
    parser.add_argument("--no-preprocess", action="store_true", help="skip image preprocessing")
    parser.add_argument(
        "--threshold",
        choices=["otsu", "adaptive", "none"],
        default="otsu",
        help="binarisation method (adaptive helps with uneven lighting)",
    )
    parser.add_argument("--no-deskew", action="store_true", help="disable rotation correction")
    parser.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode")
    parser.add_argument("-o", "--output", type=Path, help="write output to a file")
    parser.add_argument("--list-langs", action="store_true", help="list installed languages")
    parser.add_argument("-v", "--verbose", action="store_true", help="log pipeline details")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _format_output(results: list[tuple[Path, OCRResult]], as_json: bool) -> str:
    """Render OCR results as plain text or JSON.

    Args:
        results: ``(path, result)`` pairs in input order.
        as_json: Emit JSON (a single object for one file, a list for several).

    Returns:
        The formatted output string.
    """
    if as_json:
        payload = [{"file": str(path), **result.to_dict()} for path, result in results]
        return json.dumps(payload[0] if len(payload) == 1 else payload, indent=2)
    if len(results) == 1:
        return results[0][1].text
    return "\n\n".join(f"==> {path} <==\n{result.text}" for path, result in results)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ocr-extract`` command.

    Args:
        argv: Command-line arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` if OCR failed, ``2`` for usage errors
        such as a missing input file.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("ocr_extractor").setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    try:
        if args.list_langs:
            print("\n".join(available_languages()))
            return EXIT_OK
        if not args.images:
            parser.error("at least one image is required")

        config = PreprocessConfig(threshold=args.threshold, deskew=not args.no_deskew)
        results: list[tuple[Path, OCRResult]] = []
        for path in args.images:
            if not path.is_file():
                print(f"error: file not found: {path}", file=sys.stderr)
                return EXIT_USAGE_ERROR
            logger.info("Processing %s", path)
            result = extract_text(
                path,
                lang=args.lang,
                preprocess_image=not args.no_preprocess,
                config=config,
                psm=args.psm,
                with_words=args.json,
            )
            results.append((path, result))
    except OCRError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_OCR_ERROR

    output = _format_output(results, as_json=args.json)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        logger.info("Wrote output to %s", args.output)
    else:
        print(output)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
