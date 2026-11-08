"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from .io import load_image, load_video
from .report import compare_images, compare_videos
from .temporal import DEFAULT_CUTOFF_HZ
from .validation import controls_table


def _resolve_fps(requested: float | None, test_fps: float | None, reference_fps: float | None) -> float:
    if requested is not None:
        return requested
    known = {fps for fps in (test_fps, reference_fps) if fps is not None}
    if not known:
        raise ValueError("frame rate unknown; pass --fps")
    if len(known) > 1:
        raise ValueError(f"videos have different frame rates {sorted(known)}; pass --fps to choose one")
    return known.pop()


def _compare(args: argparse.Namespace) -> int:
    test, test_fps = load_video(args.test, args.max_frames)
    reference, reference_fps = load_video(args.reference, args.max_frames)
    fps = _resolve_fps(args.fps, test_fps, reference_fps)
    report = compare_videos(test, reference, fps, cutoff_hz=args.cutoff_hz)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_text())
    return 0


def _compare_images(args: argparse.Namespace) -> int:
    report = compare_images(load_image(args.test), load_image(args.reference))
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_text())
    return 0


def _validate(args: argparse.Namespace) -> int:
    print(controls_table(seed=args.seed))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fast-fourier-transform-quality", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compare = commands.add_parser("compare", help="compare a test video with a reference")
    compare.add_argument("test", help="video file or .npy array")
    compare.add_argument("reference", help="video file or .npy array of the same frame size")
    compare.add_argument("--fps", type=float, help="frame rate; required for .npy input")
    compare.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ)
    compare.add_argument("--max-frames", type=int, help="read only the first N frames")
    compare.add_argument("--json", action="store_true", help="print JSON instead of text")
    compare.set_defaults(handler=_compare)

    images = commands.add_parser("compare-images", help="compare a test image with a reference")
    images.add_argument("test", help="image file or .npy array")
    images.add_argument("reference", help="image file or .npy array of the same size")
    images.add_argument("--json", action="store_true", help="print JSON instead of text")
    images.set_defaults(handler=_compare_images)

    validate = commands.add_parser("validate", help="run the synthetic controls")
    validate.add_argument("--seed", type=int, default=0)
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
