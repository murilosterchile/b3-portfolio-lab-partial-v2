from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from portfolio_core.data.cvm_registry import materialize_cvm_registry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical ticker/CVM mappings from official CVM CAD + FCA open data"
    )
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("start-year must be <= end-year")
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    result = materialize_cvm_registry(
        data_dir=data_dir,
        years=range(args.start_year, args.end_year + 1),
        force=args.force,
    )
    print(result)


if __name__ == "__main__":
    main()
