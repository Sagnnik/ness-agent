#!/usr/bin/env python3
"""Refresh and inspect Ness Agent's runtime OpenRouter model catalog."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from ness_cli.paths import cache_dir_from_env
from ness_cli.providers.openrouter.catalog import OpenRouterCatalog


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh OpenRouter model metadata. Normal model listing uses cached "
            "or offline data and does not refresh automatically."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch OpenRouter and models.dev metadata and replace the cache",
    )
    args = parser.parse_args(argv)
    if not args.refresh:
        parser.error("--refresh is required")

    catalog = OpenRouterCatalog.from_cache_dir(cache_dir_from_env())
    result = asyncio.run(catalog.refresh(force=True))
    if not result.refreshed:
        detail = result.error or "no updated catalog was written"
        print(f"error: catalog refresh failed: {detail}", file=sys.stderr)
        return 1

    print(f"cached {result.models} models at {catalog.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
