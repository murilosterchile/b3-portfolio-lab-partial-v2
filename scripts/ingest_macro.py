from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from portfolio_core.data.bcb import materialize_default_macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()
    outputs = materialize_default_macro(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        base_url=os.getenv("BCB_SGS_BASE_URL", "https://api.bcb.gov.br/dados/serie/bcdata.sgs"),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
