"""Compatibility entry point forwarding to the decomposition CLI."""

from mofbb.decompose.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
