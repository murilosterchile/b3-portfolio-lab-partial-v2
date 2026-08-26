from __future__ import annotations

import argparse
import os
from pathlib import Path

from portfolio_core.data.cvm import download_cvm_document, extract_cvm_statements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--document", choices=["ITR", "DFP"], default="ITR")
    args = parser.parse_args()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    base = os.getenv("CVM_BASE_URL", "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC")
    archive = download_cvm_document(
        document=args.document, year=args.year, base_url=base, data_dir=data_dir
    )
    outputs = extract_cvm_statements(
        archive, data_dir=data_dir, document=args.document, year=args.year
    )
    print(f"materialized={len(outputs)}")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
