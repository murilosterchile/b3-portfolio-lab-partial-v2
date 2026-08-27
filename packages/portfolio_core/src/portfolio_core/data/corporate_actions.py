from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np
import polars as pl

from portfolio_core.data_quality import DataQualityError, validate_prices

from .common import sha256_file, utc_now_iso

B3_SUPPLEMENT_URL = (
    "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/"
    "CompanyCall/GetListedSupplementCompany"
)

_CASH_LABELS = {
    "DIVIDENDO": "cash_dividend",
    "JRS CAP PROPRIO": "jcp",
    "JUROS CAP PROPRIO": "jcp",
    "RENDIMENTO": "cash_dividend",
}
_STOCK_LABELS = {
    "DESDOBRAMENTO": "stock_split",
    "GRUPAMENTO": "reverse_split",
    "BONIFICACAO": "bonus_shares",
    "BONIFICAÇÃO": "bonus_shares",
}


@dataclass(frozen=True)
class CorporateActionSyncSummary:
    requested_roots: int
    successful_roots: int
    failed_roots: int
    failed_root_names: tuple[str, ...]
    action_rows: int
    raw_dir: str
    parquet_path: str
    cache_hits: int = 0
    downloaded: int = 0
    empty_responses: int = 0
    retries: int = 0
    timeouts: int = 0
    bytes_received: int = 0
    elapsed_seconds: float = 0.0
    parse_seconds: float = 0.0
    request_p50_seconds: float = 0.0
    request_p95_seconds: float = 0.0
    request_max_seconds: float = 0.0


@dataclass(frozen=True)
class _FetchResult:
    root: str
    url: str
    payload: Any
    status: str
    http_status: int
    retries: int
    timeouts: int
    duration_seconds: float
    bytes_received: int


def load_failed_corporate_action_roots(data_dir: Path) -> tuple[str, ...]:
    """Load failed issuer roots from an existing corporate-action sync manifest."""
    actions_dir = data_dir / "silver" / "b3_corporate_actions"
    if not (actions_dir / "part-000.parquet").exists():
        raise FileNotFoundError("Existing B3 corporate-action Parquet was not found")
    manifest_path = actions_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Existing B3 corporate-action sync manifest was not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest.get("failed_root_names")
    if names is None:
        names = [str(item).partition(":")[0] for item in manifest.get("failures", [])]
    return tuple(sorted({str(name).strip().upper()[:4] for name in names if str(name).strip()}))


def _encode_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _normalise_label(value: object) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            pass
    return None


def _parse_decimal(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    text = str(value).strip()
    if not text:
        return None
    # B3 endpoints commonly return pt-BR formatted decimals.
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        result = float(text)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def _unwrap_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    return payload if isinstance(payload, dict) else None


def _payload_is_valid(payload: Any) -> bool:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return False
    return isinstance(payload, dict) or (
        isinstance(payload, list)
        and (not payload or all(isinstance(item, dict) for item in payload))
    )


def fetch_company_actions(
    issuer_root: str,
    *,
    client: httpx.Client | None = None,
    timeout_seconds: float | None = 30.0,
    connect_timeout_seconds: float | None = None,
    read_timeout_seconds: float | None = None,
    retries: int = 3,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Fetch the B3 listed-company supplemental payload for one issuer root.

    The endpoint is public but not a formal stable API. The caller persists the raw payload so a
    research run remains reproducible if the upstream response changes later.
    """
    root = issuer_root.strip().upper()[:4]
    if len(root) != 4 or not root.isalnum():
        raise ValueError(f"Invalid B3 issuer root: {issuer_root!r}")
    encoded = _encode_payload({"issuingCompany": root, "language": "pt-br"})
    url = f"{B3_SUPPLEMENT_URL}/{encoded}"
    owned_client = client is None
    connect_timeout = connect_timeout_seconds or timeout_seconds or 10.0
    read_timeout = read_timeout_seconds or timeout_seconds or 30.0
    http = client or httpx.Client(
        timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    try:
        last_error: Exception | None = None
        attempts = max(1, retries + 1)
        retry_count = 0
        timeout_count = 0
        started = time.monotonic()
        for attempt in range(attempts):
            try:
                response = http.get(url)
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                if response.status_code >= 400:
                    response.raise_for_status()
                payload = response.json()
                if not _payload_is_valid(payload):
                    raise RuntimeError("B3 corporate-action endpoint returned an unexpected payload")
                return {
                    "url": url,
                    "payload": payload,
                    "http_status": response.status_code,
                    "retry_count": retry_count,
                    "timeout_count": timeout_count,
                    "duration_seconds": time.monotonic() - started,
                    "bytes_received": len(response.content),
                }
            except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                last_error = exc
                if isinstance(exc, httpx.TimeoutException):
                    timeout_count += 1
                transient = isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))
                response_error = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
                if response_error is not None:
                    transient = response_error.status_code in {429, 500, 502, 503, 504}
                if not transient or attempt + 1 >= attempts:
                    break
                retry_count += 1
                retry_after = response_error.headers.get("Retry-After") if response_error else None
                try:
                    delay = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    delay = 0.0
                if delay <= 0:
                    delay = min(8.0, 1.0 * (2**attempt)) + random.uniform(0.0, 0.25)
                sleep(delay)
        raise RuntimeError(f"Failed to fetch B3 corporate actions for {root}: {last_error}")
    finally:
        if owned_client:
            http.close()


def parse_company_actions(issuer_root: str, payload: Any) -> pl.DataFrame:
    """Normalize cash dividends/JCP and share-count actions from a B3 payload."""
    root = issuer_root.strip().upper()[:4]
    item = _unwrap_payload(payload)
    if item is None:
        return pl.DataFrame(schema=_action_schema())
    rows: list[dict[str, object]] = []

    for raw in item.get("cashDividends") or []:
        label = _normalise_label(raw.get("label") or raw.get("corporateAction"))
        kind = _CASH_LABELS.get(label)
        amount = _parse_decimal(raw.get("rate") or raw.get("valueCash"))
        last_prior = _parse_date(raw.get("lastDatePrior") or raw.get("lastDatePriorEx"))
        isin = str(raw.get("assetIssued") or raw.get("isinCode") or "").strip().upper()
        if kind is None or amount is None or amount <= 0 or last_prior is None or not isin:
            continue
        rows.append(
            {
                "issuer_root": root,
                "isin": isin,
                "action_type": kind,
                "label": label,
                "last_date_prior": last_prior,
                "approved_on": _parse_date(raw.get("approvedOn")),
                "payment_date": _parse_date(raw.get("paymentDate")),
                "cash_amount": amount,
                "share_factor": None,
                "source": "B3 listedCompaniesProxy",
            }
        )

    for raw in item.get("stockDividends") or []:
        label = _normalise_label(raw.get("label"))
        kind = _STOCK_LABELS.get(label)
        factor = _parse_decimal(raw.get("factor"))
        last_prior = _parse_date(raw.get("lastDatePrior"))
        isin = str(raw.get("isinCode") or raw.get("assetIssued") or "").strip().upper()
        if kind is None or factor is None or factor <= 0 or last_prior is None or not isin:
            continue
        rows.append(
            {
                "issuer_root": root,
                "isin": isin,
                "action_type": kind,
                "label": label,
                "last_date_prior": last_prior,
                "approved_on": _parse_date(raw.get("approvedOn")),
                "payment_date": None,
                "cash_amount": None,
                "share_factor": factor,
                "source": "B3 listedCompaniesProxy",
            }
        )

    if not rows:
        return pl.DataFrame(schema=_action_schema())
    return (
        pl.DataFrame(rows, schema=_action_schema())
        .unique(subset=["isin", "action_type", "last_date_prior", "cash_amount", "share_factor"])
        .sort(["isin", "last_date_prior", "action_type"])
    )


def _action_schema() -> dict[str, pl.DataType]:
    return {
        "issuer_root": pl.Utf8,
        "isin": pl.Utf8,
        "action_type": pl.Utf8,
        "label": pl.Utf8,
        "last_date_prior": pl.Date,
        "approved_on": pl.Date,
        "payment_date": pl.Date,
        "cash_amount": pl.Float64,
        "share_factor": pl.Float64,
        "source": pl.Utf8,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    with partial.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def _cache_paths(raw_dir: Path, root: str) -> tuple[Path, Path]:
    issuer_dir = raw_dir / f"issuer={root}"
    return issuer_dir / "response.json", issuer_dir / "metadata.json"


def _read_cached_payload(raw_dir: Path, root: str) -> tuple[Any, bool]:
    response_path, metadata_path = _cache_paths(raw_dir, root)
    if response_path.exists() and metadata_path.exists():
        response_bytes = response_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported corporate-action cache schema")
        if hashlib.sha256(response_bytes).hexdigest() != metadata.get("sha256"):
            raise ValueError("corporate-action cache checksum mismatch")
        payload = json.loads(response_bytes)
        if not _payload_is_valid(payload):
            raise ValueError("invalid corporate-action cache payload")
        return payload, False

    # Read the partial-v2 legacy envelope and migrate it without another HTTP request.
    legacy_path = raw_dir / f"{root}.json"
    if not legacy_path.exists():
        raise FileNotFoundError(root)
    envelope = json.loads(legacy_path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if not _payload_is_valid(payload):
        raise ValueError("invalid legacy corporate-action cache payload")
    _persist_cache(
        raw_dir,
        root,
        payload,
        source_url=str(envelope.get("source_url", B3_SUPPLEMENT_URL)),
        http_status=200,
        status="success" if parse_company_actions(root, payload).height else "empty",
        retry_count=0,
    )
    return payload, True


def _persist_cache(
    raw_dir: Path,
    root: str,
    payload: Any,
    *,
    source_url: str,
    http_status: int,
    status: str,
    retry_count: int,
) -> int:
    response_path, metadata_path = _cache_paths(raw_dir, root)
    response_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    metadata = {
        "issuer_identifier": root,
        "identifier_type": "B3 issuingCompany",
        "fetched_at": utc_now_iso(),
        "source": source_url,
        "http_status": http_status,
        "sha256": hashlib.sha256(response_bytes).hexdigest(),
        "schema_version": 1,
        "status": status,
        "retry_count": retry_count,
        "bytes": len(response_bytes),
    }
    _atomic_write(response_path, response_bytes)
    _atomic_write(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return len(response_bytes)


def _persist_failure(raw_dir: Path, root: str, error: Exception) -> None:
    _, metadata_path = _cache_paths(raw_dir, root)
    failure_path = metadata_path.with_name("last_failure.json")
    _atomic_write(
        failure_path,
        json.dumps(
            {
                "issuer_identifier": root,
                "identifier_type": "B3 issuingCompany",
                "fetched_at": utc_now_iso(),
                "source": B3_SUPPLEMENT_URL,
                "schema_version": 1,
                "status": "failure",
                "error": str(error),
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
    )


def sync_corporate_actions(
    issuer_roots: Iterable[str],
    *,
    data_dir: Path,
    force: bool = False,
    concurrency: int | None = None,
    connect_timeout_seconds: float | None = None,
    read_timeout_seconds: float | None = None,
    retries: int | None = None,
    progress_every: int = 25,
) -> CorporateActionSyncSummary:
    """Fetch issuer payloads concurrently, then deterministically normalize them.

    A successful/empty response is atomically cached by the worker before it is reported complete,
    so an interrupted run resumes from the remaining issuers. Parsing and consolidation stay on the
    main thread, which avoids shared mutable Polars objects and makes output independent of response
    order.
    """
    roots = sorted({root.strip().upper()[:4] for root in issuer_roots if str(root).strip()})
    concurrency = concurrency or int(os.getenv("B3_HTTP_CONCURRENCY", "6"))
    connect_timeout = connect_timeout_seconds or float(os.getenv("B3_HTTP_TIMEOUT_CONNECT", "10"))
    read_timeout = read_timeout_seconds or float(os.getenv("B3_HTTP_TIMEOUT_READ", "30"))
    retry_limit = retries if retries is not None else int(os.getenv("B3_HTTP_RETRIES", "3"))
    if not 1 <= concurrency <= 32:
        raise ValueError("B3 HTTP concurrency must be between 1 and 32")
    if connect_timeout <= 0 or read_timeout <= 0 or not 0 <= retry_limit <= 10:
        raise ValueError("invalid B3 HTTP timeout/retry configuration")
    raw_dir = data_dir / "raw" / "b3_corporate_actions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    payloads: dict[str, Any] = {}
    failed: list[str] = []
    failed_names: list[str] = []
    metrics = {
        "cache_hits": 0,
        "downloaded": 0,
        "empty": 0,
        "retries": 0,
        "timeouts": 0,
        "bytes": 0,
    }
    request_durations: list[float] = []
    pending: list[str] = []
    for root in roots:
        if force:
            pending.append(root)
            continue
        try:
            payload, _migrated = _read_cached_payload(raw_dir, root)
            payloads[root] = payload
            metrics["cache_hits"] += 1
        except FileNotFoundError:
            pending.append(root)
        except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
            # Corrupt entries are never trusted; fetch a clean replacement.
            print(f"[B3 actions] invalid cache issuer={root}: {exc}; downloading again")
            pending.append(root)

    print(f"[B3 actions] issuers discovered: {len(roots)}")
    print(
        f"[B3 actions] workers={concurrency} cache_hits={metrics['cache_hits']} "
        f"pending={len(pending)}"
    )
    semaphore = threading.BoundedSemaphore(concurrency)
    completed = metrics["cache_hits"]

    def fetch_one(root: str, client: httpx.Client) -> _FetchResult:
        with semaphore:
            result = fetch_company_actions(
                root,
                client=client,
                connect_timeout_seconds=connect_timeout,
                read_timeout_seconds=read_timeout,
                retries=retry_limit,
            )
        payload = result["payload"]
        status = "empty" if parse_company_actions(root, payload).is_empty() else "success"
        _persist_cache(
            raw_dir,
            root,
            payload,
            source_url=result["url"],
            http_status=result["http_status"],
            status=status,
            retry_count=result["retry_count"],
        )
        return _FetchResult(
            root=root,
            url=result["url"],
            payload=payload,
            status=status,
            http_status=result["http_status"],
            retries=result["retry_count"],
            timeouts=result["timeout_count"],
            duration_seconds=result["duration_seconds"],
            bytes_received=result["bytes_received"],
        )

    with httpx.Client(
        timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    ) as client:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="b3-actions") as executor:
            futures = {executor.submit(fetch_one, root, client): root for root in pending}
            for future in as_completed(futures):
                root = futures[future]
                completed += 1
                try:
                    result = future.result()
                    payloads[root] = result.payload
                    metrics["downloaded"] += 1
                    metrics["empty"] += int(result.status == "empty")
                    metrics["retries"] += result.retries
                    metrics["timeouts"] += result.timeouts
                    metrics["bytes"] += result.bytes_received
                    request_durations.append(result.duration_seconds)
                except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
                    failed.append(f"{root}: {exc}")
                    failed_names.append(root)
                    _persist_failure(raw_dir, root, exc)
                if completed % max(1, progress_every) == 0 or completed == len(roots):
                    elapsed = max(0.001, time.monotonic() - started)
                    rate = completed / elapsed
                    eta = int((len(roots) - completed) / rate) if rate else 0
                    print(
                        f"[{completed}/{len(roots)}] ok={len(payloads)} "
                        f"empty={metrics['empty']} failed={len(failed_names)} "
                        f"cached={metrics['cache_hits']} rate={rate:.1f}/s "
                        f"ETA={eta // 60:02d}:{eta % 60:02d}"
                    )

    # Parsing happens after all I/O and in issuer order for deterministic output.
    parse_started = time.monotonic()
    frames: list[pl.DataFrame] = []
    for root in roots:
        payload = payloads.get(root)
        if payload is None:
            continue
        try:
            frame = parse_company_actions(root, payload)
            if not frame.is_empty():
                frames.append(frame)
        except (RuntimeError, ValueError) as exc:
            failed.append(f"{root}: {exc}")
            failed_names.append(root)
    parse_seconds = time.monotonic() - parse_started

    target = data_dir / "silver" / "b3_corporate_actions" / "part-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    actions = (
        pl.concat(frames, how="vertical_relaxed")
        if frames
        else pl.DataFrame(schema=_action_schema())
    )
    actions = actions.unique(
        subset=["isin", "action_type", "last_date_prior", "cash_amount", "share_factor"]
    ).sort(["issuer_root", "isin", "last_date_prior", "action_type"])
    partial_target = target.with_suffix(".parquet.part")
    actions.write_parquet(partial_target, compression="zstd")
    partial_target.replace(target)
    elapsed_seconds = time.monotonic() - started
    durations = sorted(request_durations)
    p50 = statistics.median(durations) if durations else 0.0
    p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))] if durations else 0.0
    unique_failed_names = sorted(set(failed_names))
    manifest = {
        "source": "B3 listedCompaniesProxy/GetListedSupplementCompany",
        "fetched_at": utc_now_iso(),
        "requested_roots": len(roots),
        "successful_roots": len(payloads),
        "failed_roots": len(unique_failed_names),
        "failed_root_names": unique_failed_names,
        "failures": failed,
        "rows": actions.height,
        "cache_hits": metrics["cache_hits"],
        "downloaded": metrics["downloaded"],
        "empty": metrics["empty"],
        "retries": metrics["retries"],
        "timeouts": metrics["timeouts"],
        "bytes_received": metrics["bytes"],
        "elapsed_seconds": elapsed_seconds,
        "parse_seconds": parse_seconds,
        "request_p50_seconds": p50,
        "request_p95_seconds": p95,
        "request_max_seconds": max(durations, default=0.0),
        "complete": not unique_failed_names,
        "parquet_sha256": sha256_file(target),
    }
    _atomic_write(
        target.parent / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    _atomic_write(
        raw_dir / "watermark.json",
        json.dumps(
            {
                "last_corporate_actions_sync": utc_now_iso(),
                "complete": not unique_failed_names,
                "issuers": len(roots),
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
    )
    print("B3 CORPORATE ACTIONS SUMMARY")
    print(
        f"issuers_total={len(roots)} cache_hits={metrics['cache_hits']} "
        f"downloaded={metrics['downloaded']} empty={metrics['empty']} "
        f"failed={len(unique_failed_names)} retries={metrics['retries']} "
        f"timeouts={metrics['timeouts']} elapsed={elapsed_seconds:.1f}s "
        f"request_p50={p50:.2f}s request_p95={p95:.2f}s "
        f"request_max={max(durations, default=0.0):.2f}s parse={parse_seconds:.2f}s"
    )
    return CorporateActionSyncSummary(
        requested_roots=len(roots),
        successful_roots=len(payloads),
        failed_roots=len(unique_failed_names),
        failed_root_names=tuple(unique_failed_names),
        action_rows=actions.height,
        raw_dir=str(raw_dir),
        parquet_path=str(target),
        cache_hits=metrics["cache_hits"],
        downloaded=metrics["downloaded"],
        empty_responses=metrics["empty"],
        retries=metrics["retries"],
        timeouts=metrics["timeouts"],
        bytes_received=metrics["bytes"],
        elapsed_seconds=elapsed_seconds,
        parse_seconds=parse_seconds,
        request_p50_seconds=p50,
        request_p95_seconds=p95,
        request_max_seconds=max(durations, default=0.0),
    )


def _stock_price_multiplier(action_type: str, reported_factor: float) -> float:
    if reported_factor <= 0 or not np.isfinite(reported_factor):
        raise DataQualityError(f"invalid corporate-action share factor: {reported_factor}")
    if action_type == "stock_split":
        # B3 publishes splits as the percentage of new shares issued.
        multiplier = 1.0 / (1.0 + reported_factor / 100.0)
    elif action_type == "reverse_split":
        multiplier = 1.0 / reported_factor
    elif action_type == "bonus_shares":
        # B3 publishes bonuses as percentages in the supplemental endpoint.
        multiplier = 1.0 / (1.0 + reported_factor / 100.0)
    else:
        return 1.0
    if not 1e-7 <= multiplier <= 1e7:
        raise DataQualityError(
            f"corporate-action adjustment multiplier outside safety bounds: {multiplier}"
        )
    return multiplier


def adjust_prices_for_total_return(
    prices: pl.DataFrame,
    actions: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build an analytically adjusted close from official B3 actions.

    Raw COTAHIST is never changed. Invalid raw price observations are quarantined, and the returned
    analytical table contains only finite positive closes. Share-count actions are backward-adjusted
    first; cash dividends/JCP then apply a Yahoo-style backward total-return factor using the last
    cum-dividend close. Unknown/non-normalized actions are intentionally not applied.
    """
    required = {"trade_date", "ticker", "isin", "close"}
    if missing := required - set(prices.columns):
        raise ValueError(f"prices missing columns required for adjustment: {sorted(missing)}")
    invalid = prices.filter(
        pl.col("close").is_null()
        | ~pl.col("close").is_finite()
        | (pl.col("close") <= 0)
    )
    clean = prices.join(
        invalid.select("ticker", "trade_date").with_columns(pl.lit(True).alias("_invalid")),
        on=["ticker", "trade_date"],
        how="left",
    ).filter(pl.col("_invalid").is_null()).drop("_invalid")
    if clean.is_empty():
        raise DataQualityError("no valid prices remain after quarantine")
    validate_prices(clean, context="corporate-action clean prices")

    if actions.is_empty():
        adjusted = clean.with_columns(
            pl.col("close").alias("adjusted_close"),
            pl.lit(1.0).alias("split_adjustment_factor"),
            pl.lit(1.0).alias("cash_adjustment_factor"),
        )
        return adjusted, invalid

    action_required = {"isin", "action_type", "last_date_prior", "cash_amount", "share_factor"}
    if missing := action_required - set(actions.columns):
        raise ValueError(f"actions missing required columns: {sorted(missing)}")

    action_groups = {key[0] if isinstance(key, tuple) else key: frame for key, frame in actions.partition_by("isin", as_dict=True).items()}
    output_groups: list[pl.DataFrame] = []
    for price_group in clean.sort(["isin", "trade_date"]).partition_by("isin", maintain_order=True):
        isin_value = price_group["isin"][0]
        dates = np.asarray(price_group["trade_date"].to_list(), dtype="datetime64[D]")
        close = price_group["close"].to_numpy().astype(float)
        split_factor = np.ones(len(close), dtype=float)
        cash_factor = np.ones(len(close), dtype=float)
        group_actions = action_groups.get(isin_value)
        if group_actions is not None and not group_actions.is_empty():
            rows = group_actions.sort("last_date_prior").iter_rows(named=True)
            stock_rows: list[dict[str, object]] = []
            cash_rows: list[dict[str, object]] = []
            for row in rows:
                if row["action_type"] in {"stock_split", "reverse_split", "bonus_shares"}:
                    stock_rows.append(row)
                elif row["action_type"] in {"cash_dividend", "jcp"}:
                    cash_rows.append(row)
            for row in stock_rows:
                factor = row.get("share_factor")
                if factor is None:
                    continue
                last_prior = np.datetime64(row["last_date_prior"], "D")
                prior_dates = dates <= last_prior
                if not prior_dates.any() or prior_dates.all():
                    continue
                split_factor[prior_dates] *= _stock_price_multiplier(
                    str(row["action_type"]), float(factor)
                )
            for row in cash_rows:
                amount = row.get("cash_amount")
                if amount is None or float(amount) <= 0:
                    continue
                last_prior = np.datetime64(row["last_date_prior"], "D")
                pos = int(np.searchsorted(dates, last_prior, side="right")) - 1
                if pos < 0:
                    continue
                # Dividend yield must use the contemporaneous raw cum-dividend price.
                # Later split adjustments affect both historical price and per-share cash amount
                # units; using an already split-adjusted price with the original cash amount would
                # distort the yield.
                cum_price = float(close[pos])
                value = float(amount)
                if not np.isfinite(cum_price) or cum_price <= 0 or value >= cum_price:
                    # An event larger than the cum price is almost certainly a mapping/unit problem.
                    raise DataQualityError(
                        f"invalid cash-action yield for {isin_value} on {row['last_date_prior']}: "
                        f"amount={value} close={cum_price}"
                    )
                cash_factor[dates <= last_prior] *= (cum_price - value) / cum_price
        adjusted_close = close * split_factor * cash_factor
        if not np.all(np.isfinite(adjusted_close)) or np.any(adjusted_close <= 0):
            raise DataQualityError(f"non-positive/non-finite adjusted prices for ISIN {isin_value}")
        output_groups.append(
            price_group.with_columns(
                pl.Series("split_adjustment_factor", split_factor),
                pl.Series("cash_adjustment_factor", cash_factor),
                pl.Series("adjusted_close", adjusted_close),
            )
        )

    adjusted = pl.concat(output_groups, how="vertical_relaxed").sort(["trade_date", "ticker"])
    return adjusted, invalid


def materialize_adjusted_price_history(
    *,
    data_dir: Path,
    excluded_issuer_roots: Iterable[str] = (),
) -> dict[str, object]:
    """Read COTAHIST + B3 actions and write adjusted yearly partitions.

    Issuer roots whose corporate-action source could not be refreshed are excluded from the
    adjusted research universe instead of silently falling back to raw, non-total-return prices.
    The excluded observations are materialized as a separate coverage quarantine.
    """
    parts = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    actions_path = data_dir / "silver" / "b3_corporate_actions" / "part-000.parquet"
    if not parts:
        raise FileNotFoundError("No COTAHIST silver partitions were found")
    if not actions_path.exists():
        raise FileNotFoundError(
            "Corporate actions are missing; run scripts/ingest_corporate_actions.py first"
        )
    prices = pl.concat([pl.read_parquet(part) for part in parts], how="diagonal_relaxed")
    if "traded_value_brl" in prices.columns and "volume" in prices.columns:
        prices = prices.with_columns(
            pl.coalesce("traded_value_brl", "volume").alias("traded_value_brl")
        )
    elif "traded_value_brl" not in prices.columns and "volume" in prices.columns:
        prices = prices.with_columns(pl.col("volume").alias("traded_value_brl"))
    excluded = sorted({str(root).strip().upper()[:4] for root in excluded_issuer_roots if str(root).strip()})
    coverage_quarantine = prices.head(0).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("coverage_reason")
    )
    if excluded:
        coverage_quarantine = prices.filter(
            pl.col("ticker").str.slice(0, 4).is_in(excluded)
        ).with_columns(pl.lit("corporate_action_source_unavailable").alias("coverage_reason"))
        prices = prices.filter(~pl.col("ticker").str.slice(0, 4).is_in(excluded))
    if prices.is_empty():
        raise DataQualityError("no prices remain after corporate-action coverage quarantine")
    actions = pl.read_parquet(actions_path)
    adjusted, quarantined = adjust_prices_for_total_return(prices, actions)

    output_root = data_dir / "silver" / "b3_prices_adjusted"
    output_root.mkdir(parents=True, exist_ok=True)
    years = sorted(set(adjusted["trade_date"].dt.year().to_list()))
    for year in years:
        path = output_root / f"year={year}" / "part-000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        adjusted.filter(pl.col("trade_date").dt.year() == year).write_parquet(
            path, compression="zstd"
        )
    quarantine_path = data_dir / "silver" / "b3_price_quarantine" / "invalid_prices.parquet"
    coverage_path = (
        data_dir / "silver" / "b3_price_quarantine" / "corporate_action_coverage.parquet"
    )
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_parquet(quarantine_path, compression="zstd")
    coverage_quarantine.write_parquet(coverage_path, compression="zstd")
    return {
        "rows": adjusted.height,
        "years": years,
        "corporate_actions": actions.height,
        "quarantined_rows": quarantined.height,
        "coverage_quarantined_rows": coverage_quarantine.height,
        "excluded_issuer_roots": excluded,
        "output_root": str(output_root),
        "quarantine": str(quarantine_path),
        "coverage_quarantine": str(coverage_path),
    }
