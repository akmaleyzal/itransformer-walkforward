#!/usr/bin/env python3
"""
spot_klines_btc.py
==================

Pengunduh klines PASAR SPOT Binance via REST API publik (tanpa API key).

Ini adalah Stage 1.

Dirancang untuk opsi A: variat berbasis fitur dari SATU aset. Karena itu
skrip ini mempertahankan SELURUH 11 kolom bermakna dari respons klines,
bukan hanya OHLCV. Tiga kolom di bawah ini tidak bisa diturunkan dari OHLC
dan menjadi tulang punggung famili variat F3/F4/F5:

    quote_volume     -> VWAP intrabar = quote_volume / volume
    trades           -> ukuran order rata-rata = quote_volume / trades
    taker_buy_base   -> ketidakseimbangan aliran order (bertanda, bukan magnitudo)

Skrip ini MURNI PENGUNDUH. Tidak ada rekayasa fitur, tidak ada pengisian
nilai hilang, tidak ada penskalaan. Celah data dilaporkan apa adanya, tidak
ditambal -- keputusan menambal adalah keputusan pemodelan, bukan pengunduhan.

Pemakaian
---------
    python spot_klines_btc.py --self-test          # tanpa jaringan
    python spot_klines_btc.py                      # default: BTCUSDT 1h 2018-01..2026-08
    python spot_klines_btc.py --symbol BTCUSDT --interval 1h \
        --start 2018-01-01 --end 2026-08-01 --outdir ./data/raw
    python spot_klines_btc.py --rebuild-only       # rakit ulang dari JSONL, tanpa jaringan

Argumen --end bersifat EKSKLUSIF. Default 2026-08-01 berarti bar terakhir
memiliki open_time 2026-07-31 23:00 UTC.

EKSKLUSIVITAS ITU DITEGAKKAN, BUKAN DIASUMSIKAN. Binance mengembalikan
bar ber-open_time == endTime, jadi tabel yang dirakit dari JSONL berisi satu
bar di luar jendela. Bar itu tidak menggeser deteksi celah -- find_gaps sudah
memakai inclusive="left" -- tapi ia menaikkan bars_actual satu, sehingga
missing_bars terhitung 121 bukan 122 dan cakupan 2026 keluar 100.02%, angka
yang mustahil. clip_to_window() membuangnya sebelum laporan maupun parquet
ditulis.

Keluaran (di --outdir, default ./data/raw yang IMMUTABLE)
---------------------------------------------------------
    BTCUSDT_1h_raw.jsonl        respons API mentah, append-only, untuk resume
    BTCUSDT_1h.parquet          tabel bersih terurut, indeks UTC
    BTCUSDT_1h.csv              opsional, via --csv
    BTCUSDT_1h_report.json      laporan cakupan dan integritas
    BTCUSDT_1h_gaps.csv         daftar celah, kalau ada

Dependensi: requests, pandas, pyarrow (untuk parquet).

CATATAN POLARS. Data plane (segmentasi, validasi, fitur di src/) memakai
polars, bukan pandas. Stage 1 dikecualikan karena berkas ini tidak menghitung
satu pun jendela bergulir dan tidak menurunkan fitur. Jangan jadikan berkas ini
preseden untuk memakai pandas di hilir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

# Urutan failover. data-api.binance.vision adalah mirror khusus data publik;
# sering lolos ketika api.binance.com kena pembatasan regional.
ENDPOINTS = (
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api-gcp.binance.com",
)

KLINES_PATH = "/api/v3/klines"
MAX_LIMIT = 1000  # batas keras /api/v3/klines

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

PANDAS_FREQ = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D",
}

# Skema respons /api/v3/klines. Kolom ke-12 ("ignore") selalu 0 dan dibuang.
RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

FLOAT_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote",
]
INT_COLUMNS = ["trades"]

# Kolom yang wajib utuh supaya famili variat F1-F5 opsi A bisa dibangun.
REQUIRED_FOR_OPTION_A = [
    "open", "high", "low", "close",      # F1 lintasan harga, F2 estimator volatilitas
    "volume", "quote_volume", "trades",  # F3 intensitas, F5 lokasi harga intrabar
    "taker_buy_base",                    # F4 aliran order
]

# Ambang bobot per menit. Batas resmi Binance saat ini 6000/menit per IP;
# kita rem di 80% supaya tidak pernah menyentuh 429.
WEIGHT_CEILING = 4800


# --------------------------------------------------------------------------
# Utilitas waktu
# --------------------------------------------------------------------------

def to_utc_ms(value: str | int | datetime) -> int:
    """Konversi tanggal apa pun ke epoch milidetik UTC.

    Naive datetime DIANGGAP UTC, tidak pernah waktu lokal. Ini disengaja:
    melokalkan ke zona waktu mesin adalah bug senyap klasik yang menggeser
    seluruh dataset beberapa jam tanpa pesan error.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Lapisan jaringan
# --------------------------------------------------------------------------

class KlineFetcher:
    """Klien klines sadar-bobot dengan backoff dan failover endpoint."""

    def __init__(self, timeout: int = 20, verbose: bool = True):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "binance-spot-klines/1.0"})
        self.timeout = timeout
        self.verbose = verbose
        self.endpoint_idx = 0
        self.used_weight = 0
        self.request_count = 0

    # -- internal ----------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def _rotate(self) -> None:
        self.endpoint_idx = (self.endpoint_idx + 1) % len(ENDPOINTS)
        self._log(f"  -> ganti endpoint: {ENDPOINTS[self.endpoint_idx]}")

    def _throttle(self) -> None:
        """Rem preventif kalau bobot terpakai mendekati plafon."""
        if self.used_weight >= WEIGHT_CEILING:
            self._log(f"  bobot {self.used_weight} >= {WEIGHT_CEILING}, jeda 60 dtk")
            time.sleep(60)
            self.used_weight = 0

    # -- publik ------------------------------------------------------------

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = MAX_LIMIT,
        max_retries: int = 6,
    ) -> list[list]:
        """Ambil satu batch. Mengembalikan list-of-list mentah dari API."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }

        for attempt in range(max_retries):
            self._throttle()
            url = ENDPOINTS[self.endpoint_idx] + KLINES_PATH
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                wait = min(2 ** attempt, 30)
                self._log(f"  jaringan gagal ({exc.__class__.__name__}), tunggu {wait}s")
                time.sleep(wait)
                if attempt >= 2:
                    self._rotate()
                continue

            self.request_count += 1
            hdr = resp.headers.get("x-mbx-used-weight-1m")
            if hdr and hdr.isdigit():
                self.used_weight = int(hdr)

            if resp.status_code == 200:
                return resp.json()

            # 429 = melewati rate limit; 418 = IP diblokir sementara.
            if resp.status_code in (429, 418):
                retry_after = int(resp.headers.get("Retry-After", 60))
                self._log(f"  HTTP {resp.status_code}, hormati Retry-After={retry_after}s")
                time.sleep(retry_after + 1)
                self.used_weight = 0
                continue

            if resp.status_code == 451:
                self._log("  HTTP 451 (blokir regional) -> rotasi endpoint")
                self._rotate()
                continue

            if 500 <= resp.status_code < 600:
                wait = min(2 ** attempt, 30)
                self._log(f"  HTTP {resp.status_code}, tunggu {wait}s")
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"HTTP {resp.status_code} dari {url}: {resp.text[:300]}"
            )

        raise RuntimeError(f"Gagal setelah {max_retries} percobaan pada startTime={start_ms}")


# --------------------------------------------------------------------------
# Pagination + persistensi mentah
# --------------------------------------------------------------------------

def resume_cursor(raw_path: Path, interval_ms: int) -> int | None:
    """Baca open_time terbesar dari JSONL yang sudah ada, kembalikan kursor lanjut.

    Baris JSONL yang rusak (mis. tulis terpotong saat proses dibunuh) dilewati
    diam-diam; append-only berarti hanya baris terakhir yang bisa rusak.
    """
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return None
    max_open = None
    with raw_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                batch = json.loads(line)
            except json.JSONDecodeError:
                continue
            if batch:
                last = batch[-1][0]
                if max_open is None or last > max_open:
                    max_open = last
    return None if max_open is None else max_open + interval_ms


def fetch_range(
    fetcher: KlineFetcher,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    raw_path: Path,
) -> int:
    """Paginasi maju dari start_ms sampai end_ms, append tiap batch ke JSONL.

    Mengembalikan jumlah bar baru yang diunduh.
    """
    step = INTERVAL_MS[interval]
    cursor = resume_cursor(raw_path, step)
    if cursor is None:
        cursor = start_ms
    else:
        print(f"Melanjutkan dari {ms_to_iso(cursor)}", file=sys.stderr)

    total_expected = max(0, (end_ms - cursor) // step)
    fetched = 0
    t0 = time.time()

    with raw_path.open("a") as fh:
        while cursor < end_ms:
            batch = fetcher.get_klines(symbol, interval, cursor, end_ms)

            # Batch kosong berarti benar-benar tidak ada bar pada [cursor, end_ms].
            # Binance melewati celah, jadi kosong BUKAN berarti "sedang di dalam
            # celah" -- kalau ada bar setelah celah, dia akan dikembalikan.
            if not batch:
                break

            fh.write(json.dumps(batch, separators=(",", ":")) + "\n")
            fh.flush()

            fetched += len(batch)
            last_open = batch[-1][0]
            next_cursor = last_open + step

            # Penjaga anti-loop-tak-hingga.
            if next_cursor <= cursor:
                raise RuntimeError(
                    f"Kursor tidak maju: {cursor} -> {next_cursor}. "
                    f"Batch terakhir open_time={last_open}"
                )
            cursor = next_cursor

            if total_expected:
                pct = min(100.0, 100.0 * fetched / total_expected)
                print(
                    f"\r  {fetched:>7,} bar  ({pct:5.1f}%)  "
                    f"hingga {ms_to_iso(last_open)[:16]}  "
                    f"req={fetcher.request_count} w={fetcher.used_weight}",
                    end="", file=sys.stderr, flush=True,
                )

    print(f"\nSelesai unduh: {fetched:,} bar dalam {time.time() - t0:.1f} dtk "
          f"({fetcher.request_count} request)", file=sys.stderr)
    return fetched


# --------------------------------------------------------------------------
# Perakitan tabel
# --------------------------------------------------------------------------

def assemble(raw_path: Path) -> pd.DataFrame:
    """JSONL mentah -> DataFrame bersih, indeks DatetimeIndex UTC."""
    rows: list[list] = []
    with raw_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.extend(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        raise RuntimeError(f"Tidak ada baris terbaca dari {raw_path}")

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df = df.drop(columns=["ignore"])

    # API mengembalikan angka sebagai STRING. Tanpa koersi eksplisit kolom-kolom
    # ini menjadi objek dan semua aritmetika hilir diam-diam salah.
    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Batch yang tumpang tindih setelah resume bisa menghasilkan duplikat.
    df = df.drop_duplicates(subset="open_time", keep="last")
    df = df.sort_values("open_time").set_index("open_time")
    df.index.name = "open_time"
    return df


def clip_to_window(df: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Potong ke [start, end) -- setengah terbuka, persis seperti --end EKSKLUSIF.

    Binance mengembalikan bar ber-open_time == endTime, jadi tabel yang dirakit
    dari JSONL berisi satu bar di luar jendela. Membiarkannya lolos tidak
    merusak deteksi celah (find_gaps memakai inclusive="left"), tapi merusak
    setiap hitungan yang diturunkan dari len(df):

        bars_actual  75.095 bukan 75.094
        missing_bars    121 bukan 122      <- tidak cocok dengan gaps.csv
        cakupan 2026 100.02%               <- mustahil

    Perbandingan dilakukan pada epoch integer, bukan string tanggal (setiap
    timestamp berbasis epoch dan dibandingkan sebagai integer).
    """
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    n_before = len(df)
    df = df[(df.index >= start) & (df.index < end)]
    dropped = n_before - len(df)
    if dropped:
        print(f"Dibuang {dropped} bar di luar [{start.isoformat()}, "
              f"{end.isoformat()}) -- jendela --end bersifat eksklusif.",
              file=sys.stderr)
    return df


# --------------------------------------------------------------------------
# Laporan integritas
# --------------------------------------------------------------------------

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """sha256 heksadesimal dari sebuah berkas, dibaca per potongan."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_gaps(index: pd.DatetimeIndex, start: pd.Timestamp,
              end: pd.Timestamp, freq: str) -> pd.DataFrame:
    """Kembalikan tabel celah (blok bar hilang yang berurutan)."""
    full = pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC")
    missing = full.difference(index)
    if len(missing) == 0:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_bars"])

    delta = pd.Timedelta(freq)
    s = pd.Series(missing, index=missing)
    block = (s.diff() != delta).cumsum()
    grouped = s.groupby(block).agg(["min", "max", "count"])
    return pd.DataFrame({
        "gap_start": grouped["min"].values,
        "gap_end": grouped["max"].values,
        "missing_bars": grouped["count"].values,
    })


def integrity_report(df: pd.DataFrame, start_ms: int, end_ms: int,
                     interval: str, symbol: str) -> tuple[dict, pd.DataFrame]:
    freq = PANDAS_FREQ[interval]
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC")

    expected = len(pd.date_range(start, end, freq=freq, inclusive="left", tz="UTC"))
    gaps = find_gaps(df.index, start, end, freq)

    # Pelanggaran sanity OHLC. Tidak pernah terjadi pada data Binance yang
    # sehat; kalau muncul, ada yang salah di perakitan, bukan di pasar.
    hi = df[["open", "close", "high", "low"]].max(axis=1)
    lo = df[["open", "close", "high", "low"]].min(axis=1)
    ohlc_bad = int(((df["high"] < hi - 1e-9) | (df["low"] > lo + 1e-9)).sum())

    # taker_buy_base adalah SUBSET dari volume; tidak boleh melebihinya.
    taker_bad = int((df["taker_buy_base"] > df["volume"] * (1 + 1e-9)).sum())

    per_year = (
        df.groupby(df.index.year).size().rename("bars").to_frame()
    )
    per_year["expected"] = [
        len(pd.date_range(
            max(start, pd.Timestamp(f"{y}-01-01", tz="UTC")),
            min(end, pd.Timestamp(f"{y + 1}-01-01", tz="UTC")),
            freq=freq, inclusive="left", tz="UTC"))
        for y in per_year.index
    ]
    per_year["coverage_pct"] = (100 * per_year["bars"] / per_year["expected"]).round(3)

    report = {
        "symbol": symbol,
        "interval": interval,
        "requested_start_utc": start.isoformat(),
        "requested_end_utc_exclusive": end.isoformat(),
        "actual_first_bar_utc": df.index[0].isoformat(),
        "actual_last_bar_utc": df.index[-1].isoformat(),
        "bars_expected": expected,
        "bars_actual": int(len(df)),
        "coverage_pct": round(100 * len(df) / expected, 4) if expected else None,
        "missing_bars": int(expected - len(df)),
        "gap_blocks": int(len(gaps)),
        "largest_gap_bars": int(gaps["missing_bars"].max()) if len(gaps) else 0,
        "duplicate_timestamps": 0,  # sudah dibuang saat perakitan
        "monotonic_index": bool(df.index.is_monotonic_increasing),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "option_a_columns_complete": {
            c: int(df[c].isna().sum()) == 0 for c in REQUIRED_FOR_OPTION_A
        },
        "ohlc_violations": ohlc_bad,
        "taker_buy_exceeds_volume": taker_bad,
        "zero_volume_bars": int((df["volume"] == 0).sum()),
        "zero_trade_bars": int((df["trades"] == 0).sum()),
        "per_year_coverage": per_year.to_dict(orient="index"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report, gaps


# --------------------------------------------------------------------------
# Self-test (tanpa jaringan)
# --------------------------------------------------------------------------

def _synth_batch(start_ms: int, n: int, step: int, price: float = 40000.0) -> list[list]:
    """Bangun batch mirip-API. Semua angka STRING, persis seperti Binance."""
    out = []
    for i in range(n):
        ot = start_ms + i * step
        o = price + i
        out.append([
            ot, f"{o:.2f}", f"{o + 20:.2f}", f"{o - 20:.2f}", f"{o + 5:.2f}",
            f"{100 + i:.8f}", ot + step - 1, f"{(100 + i) * o:.8f}", 500 + i,
            f"{(100 + i) * 0.55:.8f}", f"{(100 + i) * 0.55 * o:.8f}", "0",
        ])
    return out


def self_test() -> int:
    import tempfile
    step = INTERVAL_MS["1h"]
    start_ms = to_utc_ms("2018-01-01")
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "OK  " if cond else "GAGAL"
        print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("Self-test (tanpa jaringan)\n")

    # 1. Konversi waktu memperlakukan naive sebagai UTC
    check("naive datetime dianggap UTC",
          to_utc_ms("2018-01-01") == 1514764800000,
          f"dapat {to_utc_ms('2018-01-01')}")
    check("input bersuffix Z konsisten",
          to_utc_ms("2018-01-01T00:00:00Z") == to_utc_ms("2018-01-01"))

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "t.jsonl"

        # Batch 1: 100 bar rapat. Batch 2: mulai 5 bar setelah akhir batch 1
        # -> celah 5 bar yang disengaja. Batch 3: tumpang tindih dengan batch 2
        # -> duplikat yang disengaja.
        # b1 menutupi bar 0..99. b2 mulai di bar 105 -> celah disengaja 5 bar
        # (bar 100..104). b3 mulai di bar 150, tumpang tindih 5 bar terakhir b2
        # (bar 150..154) -> duplikat disengaja. b3 juga ditulis dua kali.
        # Bar unik: 100 + 50 + 25 = 175, dari rentang penuh 180.
        b1 = _synth_batch(start_ms, 100, step)
        b2 = _synth_batch(start_ms + 105 * step, 50, step)
        b3 = _synth_batch(start_ms + 150 * step, 30, step)
        with raw.open("w") as fh:
            for b in (b1, b2, b3, b3):
                fh.write(json.dumps(b) + "\n")

        df = assemble(raw)

        check("dtype float terkoersi dari string",
              df["close"].dtype == "float64", str(df["close"].dtype))
        check("dtype trades integer",
              df["trades"].dtype == "int64", str(df["trades"].dtype))
        check("kolom 'ignore' dibuang", "ignore" not in df.columns)
        check("indeks bertimezone UTC", str(df.index.tz) == "UTC", str(df.index.tz))
        check("indeks monoton naik", df.index.is_monotonic_increasing)
        check("duplikat terbuang", len(df) == 175, f"len={len(df)}")

        end_ms = start_ms + 180 * step
        rep, gaps = integrity_report(df, start_ms, end_ms, "1h", "TEST")

        check("celah terdeteksi tepat 1 blok",
              rep["gap_blocks"] == 1, f"dapat {rep['gap_blocks']}")
        check("ukuran celah = 5 bar",
              rep["largest_gap_bars"] == 5, f"dapat {rep['largest_gap_bars']}")
        check("bar hilang = 5", rep["missing_bars"] == 5, f"dapat {rep['missing_bars']}")

        # -- regresi: bar tepat di batas eksklusif ---------------------
        # Binance mengembalikan bar ber-open_time == endTime. Tanpa
        # clip_to_window bar itu ikut terhitung, dan SETIAP hitungan turunan
        # bergeser satu ke arah yang menyanjung data: missing_bars turun,
        # cakupan naik. Di data asli inilah yang membuat cakupan 2026 keluar
        # 100.02% dan missing_bars 121 bukan 122.
        raw_edge = Path(tmp) / "edge.jsonl"
        with raw_edge.open("w") as fh:
            for b in (b1, b2, b3):
                fh.write(json.dumps(b) + "\n")
            fh.write(json.dumps(_synth_batch(end_ms, 1, step)) + "\n")

        df_unclipped = assemble(raw_edge)
        df_clipped = clip_to_window(df_unclipped, start_ms, end_ms)

        check("bar batas ada sebelum dipotong",
              len(df_unclipped) == 176, f"len={len(df_unclipped)}")
        check("clip_to_window membuang tepat bar batas",
              len(df_clipped) == 175, f"len={len(df_clipped)}")
        check("bar terakhir yang bertahan ada di dalam jendela",
              df_clipped.index[-1] < pd.Timestamp(end_ms, unit="ms", tz="UTC"))

        rep_unclipped, _ = integrity_report(
            df_unclipped, start_ms, end_ms, "1h", "TEST")
        rep_clipped, _ = integrity_report(
            df_clipped, start_ms, end_ms, "1h", "TEST")

        # Bentuk defek yang sebenarnya: deteksi celah TIDAK berubah -- find_gaps
        # sudah memakai inclusive="left" -- jadi gaps.csv dan missing_bars saling
        # bertentangan sampai bar itu dibuang. Itulah yang membuatnya lolos ulasan.
        check("tanpa clip, missing_bars bertentangan dengan celah nyata",
              rep_unclipped["missing_bars"] == 4 and rep_unclipped["gap_blocks"] == 1,
              f"dapat missing={rep_unclipped['missing_bars']} "
              f"blok={rep_unclipped['gap_blocks']}")
        check("dengan clip, missing_bars cocok dengan celah nyata",
              rep_clipped["missing_bars"] == 5 and rep_clipped["gap_blocks"] == 1,
              f"dapat missing={rep_clipped['missing_bars']} "
              f"blok={rep_clipped['gap_blocks']}")
        check("clip idempoten",
              len(clip_to_window(df_clipped, start_ms, end_ms)) == 175)
        check("tidak ada pelanggaran OHLC", rep["ohlc_violations"] == 0)
        check("taker_buy <= volume", rep["taker_buy_exceeds_volume"] == 0)
        check("kolom opsi A lengkap",
              all(rep["option_a_columns_complete"].values()))

        # Kursor resume harus menunjuk tepat satu langkah setelah bar terakhir
        cur = resume_cursor(raw, step)
        check("kursor resume benar",
              cur == start_ms + 180 * step, f"dapat {cur}")

        # Deteksi pelanggaran OHLC: rusak satu baris secara sengaja
        bad = df.copy()
        bad.iloc[10, bad.columns.get_loc("high")] = bad["low"].iloc[10] - 100
        rep_bad, _ = integrity_report(bad, start_ms, end_ms, "1h", "TEST")
        check("pelanggaran OHLC terdeteksi", rep_bad["ohlc_violations"] == 1,
              f"dapat {rep_bad['ohlc_violations']}")

        # Baris JSONL rusak harus dilewati, bukan menggagalkan seluruh muat
        with raw.open("a") as fh:
            fh.write("{rusak tidak lengkap\n")
        check("baris JSONL rusak dilewati", len(assemble(raw)) == 175)

    print()
    if failures:
        print(f"{len(failures)} tes GAGAL: {', '.join(failures)}")
        return 1
    print("Semua tes lolos.")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Unduh klines spot Binance dengan 11 kolom penuh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h", choices=sorted(INTERVAL_MS))
    p.add_argument("--start", default="2018-01-01", help="inklusif, UTC")
    p.add_argument("--end", default="2026-08-01", help="EKSKLUSIF, UTC")
    p.add_argument("--outdir", default="./data/raw")
    p.add_argument("--csv", action="store_true", help="tulis CSV selain parquet")
    p.add_argument("--fresh", action="store_true", help="abaikan JSONL lama, unduh ulang")
    p.add_argument("--rebuild-only", action="store_true",
                   help="rakit ulang parquet/laporan dari JSONL yang sudah ada, "
                        "tanpa menyentuh jaringan")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--self-test", action="store_true", help="uji logika tanpa jaringan")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    start_ms = to_utc_ms(args.start)
    end_ms = to_utc_ms(args.end)
    if start_ms >= end_ms:
        p.error("--start harus sebelum --end")

    now_ms = int(time.time() * 1000)
    if end_ms > now_ms:
        print(f"Peringatan: --end ({args.end}) di masa depan. "
              f"Dipotong ke sekarang.", file=sys.stderr)
        end_ms = now_ms

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.symbol}_{args.interval}"
    raw_path = outdir / f"{stem}_raw.jsonl"

    if args.fresh and raw_path.exists():
        raw_path.unlink()

    print(f"{args.symbol} {args.interval}  "
          f"{ms_to_iso(start_ms)[:10]} -> {ms_to_iso(end_ms)[:10]} (eksklusif)",
          file=sys.stderr)

    if args.rebuild_only:
        if not raw_path.exists():
            p.error(f"--rebuild-only butuh {raw_path}, yang tidak ada")
        print("Mode rakit-ulang: JSONL yang ada dipakai apa adanya, "
              "jaringan tidak disentuh.", file=sys.stderr)
    else:
        fetcher = KlineFetcher(verbose=not args.quiet)
        fetch_range(fetcher, args.symbol, args.interval, start_ms, end_ms, raw_path)

    # Potong SEBELUM laporan maupun parquet: keduanya harus melihat jendela
    # yang sama, dan jendela itu setengah terbuka.
    df = clip_to_window(assemble(raw_path), start_ms, end_ms)
    report, gaps = integrity_report(df, start_ms, end_ms, args.interval, args.symbol)

    pq_path = outdir / f"{stem}.parquet"
    try:
        df.to_parquet(pq_path)
        print(f"Tulis {pq_path}", file=sys.stderr)
    except Exception as exc:
        print(f"Parquet gagal ({exc}); pasang pyarrow. Jatuh ke CSV.", file=sys.stderr)
        args.csv = True

    if args.csv:
        csv_path = outdir / f"{stem}.csv"
        df.to_csv(csv_path)
        print(f"Tulis {csv_path}", file=sys.stderr)

    if len(gaps):
        gaps.to_csv(outdir / f"{stem}_gaps.csv", index=False)

    # sha256 setiap artefak masuk ke laporan. Angka yang
    # diproduksi di bawah hash artefak masukan berbeda tidak sebanding dan
    # tidak boleh berbagi tabel -- tanpa hash yang tercatat, tidak ada run
    # yang bisa membuktikan vintage mana yang dikonsumsinya.
    report["artifact_sha256"] = {
        path.name: sha256_of(path)
        for path in sorted(outdir.glob(f"{stem}*"))
        if path.suffix != ".json"
    }
    (outdir / f"{stem}_report.json").write_text(json.dumps(report, indent=2))

    # -- ringkasan ke stdout ----------------------------------------------
    print()
    print(f"  bar diharapkan : {report['bars_expected']:,}")
    print(f"  bar diperoleh  : {report['bars_actual']:,}")
    print(f"  cakupan        : {report['coverage_pct']}%")
    print(f"  bar hilang     : {report['missing_bars']:,} "
          f"dalam {report['gap_blocks']} blok "
          f"(terbesar {report['largest_gap_bars']} bar)")
    print(f"  rentang aktual : {report['actual_first_bar_utc'][:16]} .. "
          f"{report['actual_last_bar_utc'][:16]}")
    print(f"  pelanggaran OHLC / taker : "
          f"{report['ohlc_violations']} / {report['taker_buy_exceeds_volume']}")
    print()
    print("  cakupan per tahun:")
    for year, row in report["per_year_coverage"].items():
        print(f"    {year}  {row['bars']:>6,} / {row['expected']:>6,}  "
              f"{row['coverage_pct']:>7.3f}%")
    print()
    incomplete = [c for c, ok in report["option_a_columns_complete"].items() if not ok]
    if incomplete:
        print(f"  PERINGATAN: kolom opsi A tidak lengkap: {incomplete}")
    else:
        print("  Semua kolom yang dibutuhkan famili variat F1-F5 utuh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())