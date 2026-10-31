"""Download pinned public data and build strictly complete UTC 4-hour candles.

The upstream archive is a third-party mirror of Binance spot one-minute klines,
not an exchange-certified dataset. Source commits and SHA256 hashes are recorded.
No price, volume, or missing candle is forward-filled. A 4-hour candle is emitted
only when all 240 distinct, valid one-minute candles are present.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
MINUTE_COLUMNS = OHLCV + ["quote_volume", "trade_count", "taker_buy_volume"]
BAR = pd.Timedelta(hours=4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sources(root: Path, source_config: Path, symbols: list[str]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    sources = json.loads(source_config.read_text())
    result = {}
    for symbol in symbols:
        source = sources[symbol]
        if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
            raise ValueError("Sources must be pinned to a full git commit")
        path = root / f"{symbol}.tar.gz"
        url = f"https://codeload.github.com/{source['repository']}/tar.gz/{source['commit']}"
        if not path.exists():
            print(f"Downloading {symbol}: {url}", flush=True)
            temporary = path.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "crypto-regime-research"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
            temporary.replace(path)
        result[symbol] = {**source, "url": url, "archive_sha256": sha256_file(path)}
    return result


def clean_minutes(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate a month; reject conflicting duplicates rather than choosing one."""
    required = {"timestamp", *MINUTE_COLUMNS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    frame = raw.copy()
    timestamp = pd.to_numeric(frame["timestamp"], errors="raise")
    # Binance switched public CSV archives from milliseconds to microseconds.
    divisor = pd.Series(np.where(timestamp.abs() >= 10**14, 1_000_000,
                                np.where(timestamp.abs() >= 10**11, 1_000, 1)), index=frame.index)
    seconds = timestamp // divisor
    frame["_aligned_timestamp"] = timestamp % divisor == 0
    frame.index = pd.to_datetime(seconds, unit="s", utc=True)
    frame.index.name = "timestamp"
    frame = frame[MINUTE_COLUMNS + ["_aligned_timestamp"]].copy()
    frame[MINUTE_COLUMNS] = frame[MINUTE_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    rows_received = len(frame)
    if not frame.index.is_unique:
        duplicate_rows = frame.index.duplicated(keep=False)
        if frame.loc[duplicate_rows].duplicated(keep=False).sum() != duplicate_rows.sum():
            raise ValueError("Conflicting duplicate one-minute records")
        frame = frame.loc[~frame.index.duplicated(keep="first")]
    frame = frame.sort_index()
    valid = frame.pop("_aligned_timestamp").to_numpy(dtype=bool) & (frame.index.asi8 % 60_000_000_000 == 0)
    valid &= np.isfinite(frame.to_numpy()).all(axis=1)
    valid &= (frame[["open", "high", "low", "close"]] > 0).all(axis=1).to_numpy()
    valid &= (frame["high"] >= frame[["open", "close", "low"]].max(axis=1)).to_numpy()
    valid &= (frame["low"] <= frame[["open", "close", "high"]].min(axis=1)).to_numpy()
    valid &= (frame[["volume", "quote_volume", "trade_count", "taker_buy_volume"]] >= 0).all(axis=1).to_numpy()
    valid &= (frame["taker_buy_volume"] <= frame["volume"] * (1 + 1e-8)).to_numpy()
    valid &= (frame["trade_count"] % 1 == 0).to_numpy()
    audit = {"rows_received": rows_received, "identical_duplicates_removed": rows_received - len(frame),
             "invalid_rows_removed": int((~valid).sum())}
    return frame.loc[valid], audit


def aggregate_complete_4h(minutes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """OHLC/volume aggregation, with a completeness mask on the source minutes."""
    if minutes.empty:
        return pd.DataFrame(columns=MINUTE_COLUMNS + ["minute_count"]), {}
    if not minutes.index.is_unique or not minutes.index.is_monotonic_increasing:
        raise ValueError("Minutes must be unique and sorted")
    count = minutes["close"].resample("4h", label="left", closed="left").count()
    groups = minutes.resample("4h", label="left", closed="left")
    bars = groups.agg({"open": "first", "high": "max", "low": "min", "close": "last",
                       "volume": "sum", "quote_volume": "sum", "trade_count": "sum",
                       "taker_buy_volume": "sum"})
    bars["minute_count"] = count
    expected_start = minutes.index.min().floor("4h")
    expected_end = minutes.index.max().floor("4h")
    expected = pd.date_range(expected_start, expected_end, freq="4h", tz="UTC")
    observed = minutes.index.floor("4h")
    complete = bars["minute_count"].eq(240)
    complete &= observed.value_counts().reindex(bars.index, fill_value=0).eq(240)
    # Unique minute-aligned observations plus count=240 guarantees exact coverage.
    result = bars.loc[complete].reindex(expected[complete.reindex(expected, fill_value=False)])
    result.index.name = "timestamp"
    audit = {"candidate_4h_bars": len(expected), "complete_4h_bars": len(result),
             "incomplete_4h_bars_rejected": len(expected) - len(result),
             "valid_minutes_used": int(result["minute_count"].sum())}
    return result, audit


def build_symbol(archive: Path, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Stream selected monthly CSVs from a pinned archive without unpacking it all."""
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    pieces, files = [], []
    totals = {"rows_received": 0, "identical_duplicates_removed": 0, "invalid_rows_removed": 0,
              "candidate_4h_bars": 0, "complete_4h_bars": 0, "incomplete_4h_bars_rejected": 0,
              "valid_minutes_used": 0}
    pattern = re.compile(rf"/1m/(\d{{4}})/(\d{{2}})/{symbol}_1m_\1-\2\.csv$")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            match = pattern.search(member.name)
            if not member.isfile() or not match:
                continue
            month = pd.Timestamp(year=int(match[1]), month=int(match[2]), day=1, tz="UTC")
            if month < start_ts.replace(day=1) or month >= end_ts:
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read {member.name}")
            content = stream.read()
            minutes, audit = clean_minutes(pd.read_csv(io.BytesIO(content)))
            minutes = minutes.loc[(minutes.index >= start_ts) & (minutes.index < end_ts)]
            bars, bar_audit = aggregate_complete_4h(minutes)
            pieces.append(bars)
            for key, value in {**audit, **bar_audit}.items():
                totals[key] += value
            files.append({"path": member.name, "sha256": hashlib.sha256(content).hexdigest(),
                          "source_bytes": len(content), **audit, **bar_audit})
    files.sort(key=lambda row: row["path"])
    expected_months = pd.date_range(start_ts.floor("D").replace(day=1), end_ts - pd.Timedelta("1min"), freq="MS")
    if len(files) != len(expected_months):
        raise ValueError(f"{symbol}: expected {len(expected_months)} monthly files, found {len(files)}")
    if not pieces:
        raise ValueError(f"No source files for {symbol}")
    bars = pd.concat(pieces).sort_index()
    if not bars.index.is_unique:
        raise ValueError("Overlapping monthly candles")
    bars.insert(0, "symbol", symbol)
    totals["missing_4h_grid_slots"] = len(pd.date_range(bars.index.min(), bars.index.max(), freq="4h")) - len(bars)
    totals["first_candle"] = bars.index.min().isoformat()
    totals["last_candle"] = bars.index.max().isoformat()
    return bars, {"symbol": symbol, **totals, "files": files}


def prepare_data(data_dir: Path, archive_dir: Path, source_config: Path,
                 symbols: list[str], start: str, end: str) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = download_sources(archive_dir, source_config, symbols)
    audits = []
    for symbol in symbols:
        bars, audit = build_symbol(archive_dir / f"{symbol}.tar.gz", symbol, start, end)
        bars.to_parquet(data_dir / f"{symbol}_4h.parquet")
        bars.to_csv(data_dir / f"{symbol}_4h.csv")
        audit["processed_sha256"] = sha256_file(data_dir / f"{symbol}_4h.parquet")
        audit["source"] = sources[symbol]
        audits.append(audit)
        print(f"{symbol}: {len(bars):,} complete 4H bars; rejected {audit['incomplete_4h_bars_rejected']:,} incomplete bars", flush=True)
    manifest = {"schema_version": 1, "timeframe": "4h", "timestamp_convention": "UTC candle open; candle known at timestamp + 4h",
                "source_type": "Third-party GitHub mirror of Binance spot 1m klines; not exchange-certified",
                "requested_start": start, "requested_end_exclusive": end,
                "aggregation": "240 distinct valid minutes; open=first, high=max, low=min, close=last; volumes/counts=sum",
                "missing_data_policy": "Reject incomplete bars; never fill prices or volumes",
                "symbols": audits}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
