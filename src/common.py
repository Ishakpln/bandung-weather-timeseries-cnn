"""Shared I/O, validation rules, and provenance for the Bandung study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "output" / "manifests"
TIMEZONE = "Asia/Jakarta"
START, END = "2022-01-01 00:00:00", "2025-12-31 23:00:00"
REQUESTED_COORDINATES = (-6.9175, 107.6191)
# A transparent project sanity check, not an API or municipal boundary rule.
LOCATION_TOLERANCE_KM = 10.0
API_DOC = "https://open-meteo.com/en/docs/historical-weather-api"
CONTINUOUS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
              "precipitation", "pressure_msl", "cloud_cover", "wind_speed_10m"]
WEATHER = CONTINUOUS + ["weather_code"]
UNITS = dict(zip(WEATHER, ["°C", "%", "°C", "mm", "hPa", "%", "km/h", "wmo code"]))
LABELS = {"temperature_2m": "Suhu (°C)", "relative_humidity_2m": "Kelembapan (%)",
          "dew_point_2m": "Titik embun (°C)", "precipitation": "Presipitasi (mm)",
          "pressure_msl": "Tekanan muka laut (hPa)", "cloud_cover": "Tutupan awan (%)",
          "wind_speed_10m": "Kecepatan angin (km/jam)"}
WMO_CODES = {0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65,
             66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99}
STAGES = {
    1: "stage1_validation.py", 2: "stage2_eda.py", 3: "stage3_preprocessing.py",
    4: "stage4_visualization.py", 5: "stage5_patterns.py",
}


def prepare_dirs():
    for path in (TABLES, FIGURES, REPORTS, PROCESSED, MANIFESTS):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    def encode(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (Path, pd.Timestamp)):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                     default=encode, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def markdown_table(frame, decimals=3):
    """Small dependency-free Markdown table (no optional tabulate dependency)."""
    def fmt(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, (float, np.floating)):
            return f"{x:.{decimals}f}"
        return str(x).replace("|", "\\|").replace("\n", " ")
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def save_table(name, frame, index=False):
    path = TABLES / f"{name}.csv"
    frame.to_csv(path, index=index, encoding="utf-8", float_format="%.10g")
    return path


def expected_index():
    return pd.date_range(START, END, freq="h", tz=TIMEZONE)


def raw_argument(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw", type=Path, help="CSV mentah; default: satu CSV di data/raw")
    return parser


def resolve_raw(raw=None):
    if raw is None:
        candidates = sorted((ROOT / "data" / "raw").glob("*.csv"))
        if len(candidates) != 1:
            raise ValueError("Harus ada tepat satu CSV data/raw, atau tentukan --raw.")
        raw = candidates[0]
    raw = Path(raw).resolve()
    if not raw.is_relative_to((ROOT / "data" / "raw").resolve()):
        raise ValueError("CSV mentah harus berada di folder data/raw proyek ini.")
    if not raw.is_file():
        raise FileNotFoundError(raw)
    return raw


def load_raw(raw):
    """Read Open-Meteo CSV metadata separately from its hourly table."""
    with Path(raw).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header = next((i for i, row in enumerate(rows) if row and row[0] == "time"), None)
    if header is None or header < 2:
        raise ValueError("Header time atau metadata Open-Meteo tidak ditemukan.")
    if len(rows[0]) != len(rows[1]) or len(set(rows[0])) != len(rows[0]):
        raise ValueError("Metadata CSV tidak konsisten.")
    metadata = dict(zip(rows[0], rows[1]))
    for key in ("latitude", "longitude", "elevation", "utc_offset_seconds"):
        metadata[key] = float(metadata[key])
    frame = pd.read_csv(raw, skiprows=header, encoding="utf-8-sig")
    original_columns = frame.columns.tolist()
    rename = {name: name.split(" (")[0] for name in original_columns}
    if len(set(rename.values())) != len(rename):
        raise ValueError("Nama variabel menjadi ambigu setelah normalisasi.")
    return frame.rename(columns=rename), metadata, original_columns, header


def local_times(series):
    parsed = pd.to_datetime(series, format="%Y-%m-%dT%H:%M", errors="coerce")
    return parsed.dt.tz_localize(TIMEZONE)


def distance_km(lat, lon):
    lat1, lon1, lat2, lon2 = map(math.radians, (*REQUESTED_COORDINATES, lat, lon))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(a)))


def quality_masks(frame):
    masks = {
        "humidity_outside_0_100": ~frame.relative_humidity_2m.between(0, 100),
        "cloud_outside_0_100": ~frame.cloud_cover.between(0, 100),
        "negative_precipitation": frame.precipitation < 0,
        "negative_wind": frame.wind_speed_10m < 0,
        "nonpositive_pressure": frame.pressure_msl <= 0,
        "temperature_below_absolute_zero": frame.temperature_2m < -273.15,
        "dewpoint_below_absolute_zero": frame.dew_point_2m < -273.15,
        "dewpoint_above_temperature_tolerance_0_3C": frame.dew_point_2m > frame.temperature_2m + 0.3,
        "unknown_or_fractional_weather_code": ~frame.weather_code.isin(WMO_CODES),
        "infinite_numeric_value": pd.Series(np.isinf(frame[WEATHER]).any(axis=1), index=frame.index),
    }
    return masks


def source_signature(stage):
    paths = [Path(__file__), ROOT / "src" / STAGES[stage]]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def require_previous(stage, raw):
    for previous in range(1, stage):
        path = MANIFESTS / f"stage{previous}.json"
        if not path.exists():
            raise RuntimeError(f"Tahap {previous} harus berhasil sebelum tahap {stage}.")
        manifest = read_json(path)
        if manifest.get("status") != "passed" or manifest["raw_sha256"] != sha256(raw):
            raise RuntimeError(f"Tahap {previous} gagal atau dataset berubah; jalankan ulang dari tahap 1.")
        if manifest["code_sha256"] != source_signature(previous):
            raise RuntimeError(f"Kode tahap {previous} berubah; jalankan ulang tahap tersebut.")
        for item in manifest["artifacts"]:
            artifact = ROOT / item["path"]
            if not artifact.exists() or sha256(artifact) != item["sha256"]:
                raise RuntimeError(f"Artefak tahap {previous} hilang/berubah: {artifact}")


def begin_stage(stage, raw):
    prepare_dirs()
    require_previous(stage, raw)
    # Invalidate previous success before attempting a rerun.
    (MANIFESTS / f"stage{stage}.json").unlink(missing_ok=True)
    return sha256(raw)


def finish_stage(stage, raw, original_hash, artifacts, **details):
    if sha256(raw) != original_hash:
        raise RuntimeError("Hash data mentah berubah selama eksekusi.")
    manifest = {
        "stage": stage, "status": "passed",
        "completed_at_jakarta": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "raw_file": str(raw.relative_to(ROOT)), "raw_sha256": original_hash,
        "code_sha256": source_signature(stage),
        "artifacts": [{"path": str(Path(p).relative_to(ROOT)), "sha256": sha256(p)} for p in artifacts],
        **details,
    }
    json_write(MANIFESTS / f"stage{stage}.json", manifest)
    print(f"TAHAP {stage} LULUS: {len(artifacts)} artefak; SHA-256 raw tetap {original_hash[:12]}...", flush=True)


def load_clean():
    frame = pd.read_csv(PROCESSED / "bandung_weather_hourly_clean.csv")
    frame["time"] = pd.to_datetime(frame.time, utc=True).dt.tz_convert(TIMEZONE)
    return frame


def versions():
    names = ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "tzdata", "pillow"]
    return {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in names}}
