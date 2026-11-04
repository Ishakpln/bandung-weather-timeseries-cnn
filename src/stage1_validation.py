"""Tahap 1: audit metadata, schema, location, period, and hourly grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (API_DOC, END, LOCATION_TOLERANCE_KM, REPORTS, REQUESTED_COORDINATES,
                    START, TABLES, TIMEZONE, UNITS, WEATHER, begin_stage, distance_km,
                    expected_index, finish_stage, json_write, load_raw, local_times,
                    markdown_table, raw_argument, resolve_raw, save_table, versions)


def run(raw=None):
    raw = resolve_raw(raw)
    original_hash = begin_stage(1, raw)
    frame, metadata, originals, header = load_raw(raw)
    tests = []

    def check(name, passed, observed):
        tests.append({"check": name, "passed": bool(passed), "observed": str(observed)})

    check("schema_exact", frame.columns.tolist() == ["time"] + WEATHER, frame.columns.tolist())
    if not tests[-1]["passed"]:
        save_table("validation_checks", pd.DataFrame(tests))
        raise ValueError("Schema tidak sesuai; lihat output/tables/validation_checks.csv")
    for name, original in zip(frame.columns[1:], originals[1:]):
        check(f"unit_{name}", original == f"{name} ({UNITS[name]})", original)
    times = local_times(frame.time)
    wanted = expected_index()
    observed = pd.DatetimeIndex(times.dropna())
    missing, extra = wanted.difference(observed), observed.difference(wanted)
    distance = distance_km(metadata["latitude"], metadata["longitude"])
    check("rows", len(frame) == len(wanted), f"{len(frame)} / {len(wanted)}")
    check("columns", frame.shape[1] == 9, frame.shape[1])
    check("parseable_timestamp", times.notna().all(), int(times.isna().sum()))
    check("period_start", times.min() == wanted[0], times.min())
    check("period_end", times.max() == wanted[-1], times.max())
    check("missing_hour", len(missing) == 0, len(missing))
    check("extra_timestamp", len(extra) == 0, len(extra))
    check("unique_timestamp", not times.duplicated().any(), int(times.duplicated().sum()))
    check("timezone", metadata["timezone"] == TIMEZONE, metadata["timezone"])
    check("utc_offset_seconds", metadata["utc_offset_seconds"] == 25200, metadata["utc_offset_seconds"])
    check("location_sanity_check", np.isfinite(distance) and distance <= LOCATION_TOLERANCE_KM,
          f"{distance:.6f} km <= {LOCATION_TOLERANCE_KM} km")
    for name in WEATHER:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        check(f"numeric_{name}", not (frame[name].notna() & numeric.isna()).any(), str(frame[name].dtype))
    checks = pd.DataFrame(tests)
    checks_path = save_table("validation_checks", checks)
    save_table("missing_timestamps", pd.DataFrame({"time": missing}))
    save_table("extra_timestamps", pd.DataFrame({"time": extra}))
    schema = pd.DataFrame({"original_name": originals, "canonical_name": frame.columns,
                           "raw_dtype": frame.dtypes.astype(str).values,
                           "unit": ["local ISO8601"] + [UNITS[n] for n in WEATHER]})
    schema_path = save_table("data_dictionary_raw", schema)
    metadata_path = TABLES / "source_metadata.json"
    json_write(metadata_path, {"raw_path": str(raw), "raw_sha256": original_hash,
                              "metadata": metadata, "table_header_zero_based_line": header,
                              "requested_latitude": REQUESTED_COORDINATES[0],
                              "requested_longitude": REQUESTED_COORDINATES[1],
                              "distance_km": distance, "expected_rows": len(wanted),
                              "start_local": str(times.min()), "end_local": str(times.max()),
                              "intervals_seconds": {str(k.total_seconds()): int(v)
                                                    for k, v in times.diff().value_counts().items()},
                              "already_sorted": times.is_monotonic_increasing})
    environment_path = TABLES / "environment.json"
    json_write(environment_path, versions())
    success = bool(checks.passed.all())
    report = REPORTS / "01_data_validation.md"
    report.write_text(f"""# Tahap 1 — Pengumpulan dan validasi data

Status: **{'LULUS' if success else 'GAGAL'}**. Dataset yang sudah tersedia digunakan tanpa mengunduh ulang.
File: `{raw.relative_to(raw.parents[2])}`. SHA-256: `{original_hash}`.
CSV memiliki metadata pada dua baris awal, pemisah kosong, dan header tabel pada baris {header + 1}.
Metadata bukan observasi; data hourly berukuran **{len(frame):,} baris × {frame.shape[1]} kolom**.

Periode lokal: {times.min()} sampai {times.max()}; ekspektasi {len(wanted):,} jam
(2022: 8.760; 2023: 8.760; 2024: 8.784; 2025: 8.760).
Timestamp hilang: {len(missing)}; tambahan: {len(extra)}; duplikat: {times.duplicated().sum()}.
Urutan awal menaik: {times.is_monotonic_increasing}. Interval sebelum pengurutan:
{dict((str(k), int(v)) for k, v in times.diff().value_counts().items())}.

## Lokasi dan asal data

Titik penelitian: {REQUESTED_COORDINATES}; metadata grid: ({metadata['latitude']}, {metadata['longitude']});
jarak haversine **{distance:.3f} km**, elevasi metadata {metadata['elevation']:.1f} m.
Menurut [dokumentasi Open-Meteo]({API_DOC}), koordinat respons adalah pusat grid yang dapat
berjarak beberapa kilometer dari titik permintaan. Pemeriksaan kewajaran proyek menggunakan ambang
{LOCATION_TOLERANCE_KM:.0f} km, bukan batas administratif atau jaminan ketepatan spasial.
Data konsisten dengan sekitar titik Bandung yang diminta; kecocokan persis dengan stasiun atau
representativitas seluruh wilayah kota tidak dibuktikan oleh file ini.

Zona waktu metadata `{metadata['timezone']}` dengan offset {metadata['utc_offset_seconds']:.0f} detik.
Waktu pada CSV dibaca sebagai WIB, bukan UTC. Pengguna menyatakan sumber Historical Weather API;
CSV tidak menyimpan URL permintaan, tanggal pengunduhan, pilihan model, atau versi model.
Karena itu asal model spesifik dan proses pengambilan tidak dapat diautentikasi dari file saja.
Analisis diperlakukan sebagai data grid historis, bukan pengukuran stasiun langsung.
Arsipkan URL asli dan tanggal unduh jika tersedia untuk pelaporan akademik berikutnya.

## Nama, satuan, dan tipe mentah

{markdown_table(schema)}

`weather_code` adalah kategori berkode angka; rata-rata kode tidak ditafsirkan sebagai besaran cuaca.
Header dibaca sebagai UTF-8; simbol derajat pada CSV dipertahankan dalam kamus data.

## Seluruh pemeriksaan

{markdown_table(checks)}

Hanya hasil LULUS mengizinkan tahap 2. Hash file mentah dan setiap artefak dicatat dalam manifest.
""", encoding="utf-8")
    if not success:
        raise ValueError("Validasi gagal. Tahap berikutnya dihentikan; baca reports/01_data_validation.md")
    artifacts = [checks_path, schema_path, metadata_path, environment_path, report,
                 TABLES / "missing_timestamps.csv", TABLES / "extra_timestamps.csv"]
    finish_stage(1, raw, original_hash, artifacts, rows=len(frame), columns=frame.shape[1])
    return frame


if __name__ == "__main__":
    run(raw_argument(__doc__).parse_args().raw)
