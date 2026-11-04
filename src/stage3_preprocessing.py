"""Tahap 3: preserve measured/modelled values and derive explicit time features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CONTINUOUS, PROCESSED, REPORTS, TABLES, TIMEZONE, UNITS, WEATHER,
                    begin_stage, expected_index, finish_stage, json_write, load_raw,
                    local_times, markdown_table, quality_masks, raw_argument,
                    resolve_raw, save_table, sha256)


def run(raw=None):
    raw = resolve_raw(raw)
    original_hash = begin_stage(3, raw)
    frame, source_metadata, _, _ = load_raw(raw)
    if frame.isna().any().any():
        raise ValueError("Ditemukan missing baru. Tentukan kebijakan per variabel; jangan mengisi hujan/kode dengan tebakan.")
    if any(mask.any() for mask in quality_masks(frame).values()):
        raise ValueError("Pelanggaran kualitas perlu ditangani dan didokumentasikan sebelum melanjutkan.")
    if frame.time.duplicated().any():
        raise ValueError("Duplikasi timestamp membutuhkan audit; tidak boleh dihapus tanpa keputusan.")
    frame["time"] = local_times(frame.time)
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    if not pd.DatetimeIndex(frame.time).equals(expected_index()):
        raise ValueError("Grid hourly tidak lengkap; tidak melakukan pembuatan data pengganti.")
    for name in CONTINUOUS:
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype("float64")
    frame["weather_code"] = frame.weather_code.astype("int16")
    clock = frame.time.dt
    calendar = {"hour": clock.hour, "day": clock.day, "day_of_week": clock.dayofweek,
                "day_of_year": clock.dayofyear, "month": clock.month, "year": clock.year,
                "is_weekend": (clock.dayofweek >= 5).astype("int8")}
    for name, values in calendar.items():
        frame[name] = values.astype("int16")
    frame["date"] = clock.strftime("%Y-%m-%d")
    phases = {"hour": frame.hour / 24, "month": (frame.month - 1) / 12,
              "day_of_year": (frame.day_of_year - 1) / np.where(clock.is_leap_year, 366, 365)}
    for name, phase in phases.items():
        frame[f"{name}_sin"] = np.sin(2 * np.pi * phase)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * phase)

    path = PROCESSED / "bandung_weather_hourly_clean.csv"
    export = frame.copy()
    export["time"] = export.time.map(lambda value: value.isoformat())
    export.to_csv(path, index=False, encoding="utf-8")
    reloaded = pd.read_csv(path)
    before, _, _, _ = load_raw(raw)
    before = before.assign(time=local_times(before.time)).sort_values("time")
    preserved = bool(np.array_equal(before[WEATHER].to_numpy(), reloaded[WEATHER].to_numpy()))
    if not preserved:
        raise AssertionError("Nilai cuaca berubah saat round-trip ekspor CSV.")

    definitions = {
        "time": ("timestamp", "ISO8601 +07:00", "Waktu lokal WIB; hourly unik"),
        "hour": ("derived_calendar", "0..23", "Jam lokal"),
        "day": ("derived_calendar", "1..31", "Tanggal dalam bulan"),
        "day_of_week": ("derived_calendar", "0..6", "Senin=0; Minggu=6"),
        "day_of_year": ("derived_calendar", "1..365/366", "Urutan hari; tahun kabisat dipertahankan"),
        "month": ("derived_calendar", "1..12", "Bulan kalender"),
        "year": ("derived_calendar", "2022..2025", "Tahun; bukan fitur wajib CNN"),
        "is_weekend": ("derived_calendar", "0/1", "Sabtu/Minggu=1; bukan asumsi pengaruh cuaca"),
        "date": ("derived_calendar", "YYYY-MM-DD", "Label tanggal lokal untuk agregasi"),
    }
    dictionary = []
    for name in frame.columns:
        if name in WEATHER:
            role = "categorical_weather" if name == "weather_code" else "continuous_weather"
            unit, description = UNITS[name], "Nilai dari CSV mentah; tidak diubah"
        elif name.endswith(("_sin", "_cos")):
            role, unit, description = "derived_cyclical", "[-1,1]", "Transformasi sinus/kosinus kalender deterministik"
        else:
            role, unit, description = definitions[name]
        dictionary.append({"column": name, "dtype_in_memory": str(frame[name].dtype),
                           "role": role, "unit_or_domain": unit, "description": description})
    schema = pd.DataFrame(dictionary)
    schema_path = save_table("data_dictionary_processed", schema)
    metadata_path = PROCESSED / "dataset_metadata.json"
    json_write(metadata_path, {
        "raw_sha256": original_hash, "processed_sha256": sha256(path),
        "rows": len(frame), "columns": len(frame.columns), "timezone": TIMEZONE,
        "start": frame.time.iloc[0], "end": frame.time.iloc[-1], "frequency": "1h",
        "weather_values_preserved": preserved, "imputed_cells": 0, "dropped_rows": 0,
        "added_observations": 0, "scaling_applied": False, "source_metadata": source_metadata,
        "calendar_features": list(calendar) + ["date"],
        "cyclical_features": [f"{n}_{s}" for n in phases for s in ("sin", "cos")],
    })
    audit = pd.DataFrame([
        {"operation": "Raw cells missing", "count": int(before.isna().sum().sum())},
        {"operation": "Rows dropped", "count": 0},
        {"operation": "Observations generated", "count": 0},
        {"operation": "Cells imputed", "count": 0},
        {"operation": "Weather cells changed", "count": 0},
        {"operation": "Derived columns", "count": len(frame.columns) - 9},
    ])
    audit_path = save_table("preprocessing_audit", audit)
    report = REPORTS / "03_preprocessing.md"
    report.write_text(f"""# Tahap 3 — Preprocessing

Dataset bersih: `data/processed/bandung_weather_hourly_clean.csv`.
Hasil: **{len(frame):,} baris × {len(frame.columns)} kolom**, terdiri atas 9 kolom asal
(nama disederhanakan) dan {len(frame.columns) - 9} fitur kalender/siklik.
Dataset mentah terpisah di `data/raw/` dan hash-nya tidak berubah.

## Keputusan dan audit

{markdown_table(audit, 0)}

1. Metadata CSV dibaca terpisah; satuan disimpan dalam kamus data.
2. Nama variabel distandarkan tanpa mengubah besaran atau satuan.
3. Timestamp dilokalisasi ke Asia/Jakarta dan diekspor sebagai ISO8601 dengan offset `+07:00`.
4. Data diurutkan berdasarkan waktu; duplikasi dan ketidaklengkapan grid memicu penghentian.
5. Variabel kontinu bertipe float64; `weather_code` tetap kategori berkode integer.
6. Ditambahkan kalender serta encoding siklik jam, bulan, dan hari dalam tahun.
7. Dilakukan baca ulang CSV dan perbandingan seluruh nilai cuaca: identik = **{preserved}**.

## Missing values dan nilai ekstrem

Tidak ada missing values sehingga **tidak ada imputasi**. Tidak dilakukan interpolasi,
forward/backward fill, pengisian hujan dengan nol, penghalusan, clipping, atau penghapusan outlier.
Kandidat ekstrem pada tahap 2 dipertahankan karena belum ada bukti kesalahan data.
Script sengaja berhenti jika dataset pengganti mempunyai missing atau pelanggaran fisik;
kebijakan baru harus ditentukan berdasarkan variabel dan panjang gap, lalu didokumentasikan.
Untuk analisis retrospektif, interpolasi gap pendek variabel kontinu dapat ditinjau bila benar-benar dibutuhkan;
hujan dan kode kategorikal tidak layak diinterpolasi linear. Untuk CNN, imputasi harus kausal dan
diestimasi pada data latih saja. Aturan hipotetis ini **tidak diterapkan** pada dataset sekarang.

## Fitur waktu

Hour: sin/cos(2π × hour/24). Month: sin/cos(2π × (month−1)/12).
Day-of-year: sin/cos(2π × (day_of_year−1)/D), D=365 atau 366 sesuai tahun.
Kalender merupakan turunan timestamp asli, bukan observasi cuaca sintetis.
Kolom tahun, hari, akhir pekan, dan tanggal tersedia untuk analisis; relevansinya sebagai input CNN
harus dievaluasi pada tahap berikutnya. CSV tidak menyimpan dtype pandas: gunakan
`pd.to_datetime(df['time'], utc=True).dt.tz_convert('Asia/Jakarta')` saat memuat ulang.

## Kamus data hasil

{markdown_table(schema)}

## Kesiapan analisis dan tahap CNN selanjutnya

Grid hourly berurutan dan lengkap, satuan konsisten, seluruh nilai cuaca terjaga.
Dataset ini **belum dinormalisasi dan belum dibentuk menjadi window CNN**.
Statistik EDA seluruh periode hanya untuk pelaporan; tidak boleh digunakan untuk fitting scaler,
seleksi fitur berbasis target, atau penentuan threshold model dari validation/test.
Rekomendasi input, target, split waktu, dan leakage dibahas di `06_cnn_readiness.md` setelah tahap 5.
""", encoding="utf-8")
    finish_stage(3, raw, original_hash, [path, metadata_path, schema_path, audit_path, report],
                 rows=len(frame), columns=len(frame.columns), weather_values_preserved=preserved)
    return frame


if __name__ == "__main__":
    run(raw_argument(__doc__).parse_args().raw)
