"""Tahap 2: descriptive statistics and quality diagnostics, without changing data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CONTINUOUS, REPORTS, TABLES, WEATHER, begin_stage,
                    finish_stage, json_write, load_raw, local_times, markdown_table,
                    quality_masks, raw_argument, resolve_raw, save_table)


def longest_constant_run(series):
    groups = series.ne(series.shift()).cumsum()
    return int(series.groupby(groups).size().max())


def run(raw=None):
    raw = resolve_raw(raw)
    original_hash = begin_stage(2, raw)
    frame, _, _, _ = load_raw(raw)
    times = local_times(frame.time)
    artifacts = []

    def table(name, data):
        artifacts.append(save_table(name, data))

    description = frame[CONTINUOUS].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    description["skewness"] = frame[CONTINUOUS].skew()
    description["excess_kurtosis"] = frame[CONTINUOUS].kurt()
    description["unique_values"] = frame[CONTINUOUS].nunique()
    description["zero_count"] = frame[CONTINUOUS].eq(0).sum()
    description.index.name = "variable"
    table("descriptive_statistics", description.reset_index())
    missing = pd.DataFrame({"variable": frame.columns, "missing_count": frame.isna().sum().values,
                            "missing_pct": frame.isna().mean().values * 100})
    table("missing_values", missing)
    masks = quality_masks(frame)
    quality = pd.DataFrame([{"check": name, "violations": int(mask.sum())} for name, mask in masks.items()])
    table("quality_checks", quality)
    flags = []
    for name, mask in masks.items():
        flagged = frame.loc[mask].copy()
        flagged.insert(0, "reason", name)
        flags.append(flagged)
    table("quality_flagged_rows", pd.concat(flags, ignore_index=True))
    codes = frame.weather_code.value_counts().sort_index().rename_axis("weather_code").reset_index(name="count")
    codes["percentage"] = codes["count"] / len(frame) * 100
    table("weather_code_frequency", codes)
    coverage = pd.DataFrame({"time": times, "year": times.dt.year, "month": times.dt.month})
    coverage = coverage.groupby(["year", "month"]).agg(hours=("time", "size"),
                                                         start=("time", "min"), end=("time", "max")).reset_index()
    coverage["expected_hours"] = pd.to_datetime(coverage[["year", "month"]].assign(day=1)).dt.days_in_month * 24
    coverage["complete"] = coverage.hours.eq(coverage.expected_hours)
    table("monthly_coverage", coverage)
    repeated = pd.DataFrame([{"variable": n, "longest_unchanged_run_hours": longest_constant_run(frame[n]),
                              "mode": frame[n].mode().iloc[0],
                              "mode_pct": frame[n].value_counts(normalize=True).iloc[0] * 100}
                             for n in WEATHER])
    table("constant_runs", repeated)

    # Rainfall is zero-inflated: wet-hour fences are diagnostic only.
    outliers = []
    for name in CONTINUOUS:
        values = frame.loc[frame[name] > 0, name] if name == "precipitation" else frame[name]
        q1, q3 = values.quantile([.25, .75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mask = (values < low) | (values > high)
        outliers.append({"variable": name, "basis": "positive_hours_only" if name == "precipitation" else "all_hours",
                         "basis_n": len(values), "q1": q1, "q3": q3, "lower_fence": low,
                         "upper_fence": high, "flagged_count": int(mask.sum()),
                         "flagged_pct_of_basis": mask.mean() * 100})
    fences = pd.DataFrame(outliers)
    table("iqr_diagnostics", fences)
    pearson, spearman = frame[CONTINUOUS].corr(), frame[CONTINUOUS].corr(method="spearman")
    table("correlation_pearson", pearson.rename_axis("variable").reset_index())
    table("correlation_spearman", spearman.rename_axis("variable").reset_index())
    positive = frame.precipitation[frame.precipitation > 0]
    summary = {"rows": len(frame), "columns": len(frame.columns),
               "missing_cells": int(frame.isna().sum().sum()), "exact_duplicate_rows": int(frame.duplicated().sum()),
               "duplicate_timestamps": int(times.duplicated().sum()),
               "all_months_complete": bool(coverage.complete.all()),
               "physical_quality_violations": int(quality.violations.sum()),
               "zero_rain_hours": int(frame.precipitation.eq(0).sum()), "wet_hours_gt0": len(positive),
               "wet_hours_ge0_1": int(frame.precipitation.ge(.1).sum()),
               "wet_hours_ge1": int(frame.precipitation.ge(1).sum()),
               "zero_rain_pct": float(frame.precipitation.eq(0).mean() * 100),
               "wet_hour_median_mm": float(positive.median())}
    summary_path = TABLES / "eda_summary.json"
    json_write(summary_path, summary)
    artifacts.append(summary_path)
    compact = description[["count", "mean", "std", "min", "50%", "max", "skewness"]].reset_index()
    report = REPORTS / "02_eda.md"
    report.write_text(f"""# Tahap 2 — Exploratory Data Analysis

Analisis menggunakan seluruh {len(frame):,} observasi hourly, bukan sampel acak.
Statistik dihitung sebelum preprocessing; nilai cuaca tidak diubah.
Standar deviasi menggunakan sampel (`ddof=1`); kurtosis dalam tabel lengkap adalah excess kurtosis.

## Statistik deskriptif

{markdown_table(compact)}

Tabel lengkap, termasuk persentil 1/5/25/50/75/95/99, jumlah nilai unik, dan nol:
`output/tables/descriptive_statistics.csv`. Distribusi divisualisasikan pada tahap 4.
Kode cuaca dianalisis sebagai kategori dan dikecualikan dari statistik kontinu serta korelasi.

## Missing values, duplikasi, dan waktu

Jumlah sel kosong: **{summary['missing_cells']}**. Baris identik duplikat: **{summary['exact_duplicate_rows']}**.
Timestamp duplikat: **{summary['duplicate_timestamps']}**. Semua 48 bulan lengkap: **{summary['all_months_complete']}**.
Tidak ada jam yang ditambahkan atau dihapus pada tahap ini.
Perincian tersedia pada `missing_values.csv` dan `monthly_coverage.csv`.
Tahun kabisat 2024 memiliki 8.784 jam, bukan 8.760 jam.

## Pemeriksaan kualitas fisik dan konsistensi

{markdown_table(quality)}

Aturan memeriksa rentang persen [0,100], nilai negatif pada hujan/angin, tekanan nonpositif,
suhu di bawah nol absolut, angka tak hingga, kode kategori tak dikenal, serta titik embun di atas suhu.
Toleransi titik embun 0,3 °C adalah pilihan pemeriksaan pembulatan proyek, bukan batas kualitas universal.
Lolos aturan ini menunjukkan konsistensi internal, bukan pembuktian akurasi terhadap pengukuran lapangan.
Tidak digunakan batas suhu regional yang dibuat-buat untuk menghapus nilai ekstrem.

## Distribusi hujan dan kategori cuaca

Hujan nol terjadi pada **{summary['zero_rain_hours']:,} jam ({summary['zero_rain_pct']:.2f}%)**.
Definisi operasional jam basah: presipitasi > 0 mm; jumlah **{len(positive):,} jam**.
Median pada jam basah {positive.median():.2f} mm. Sensitivitas ambang:
≥0,1 mm = {summary['wet_hours_ge0_1']:,} jam; ≥1 mm = {summary['wet_hours_ge1']:,} jam.
Ambang ini mendeskripsikan file, bukan klasifikasi intensitas resmi.
Nol merupakan hasil data yang sah dan tidak dianggap missing.

{markdown_table(codes)}

## Nilai ekstrem dan pengulangan

{markdown_table(fences)}

Fence Tukey Q1−1,5×IQR sampai Q3+1,5×IQR hanya menandai kandidat pemeriksaan.
Untuk presipitasi, fence dihitung pada jam positif agar nol dominan tidak mendefinisikan hujan sebagai kesalahan.
Tidak ada outlier yang otomatis dibuang, diganti, atau di-winsorize.
Variabel berbatas seperti awan dapat menghasilkan flag distribusi tanpa kesalahan fisik.

{markdown_table(repeated)}

Pengulangan nilai dapat muncul akibat pembulatan, batas 100%, periode kering, atau keluaran model.
Panjang run sendiri tidak membuktikan sensor macet; ini data grid.

## Hubungan awal

Pearson suhu–kelembapan = **{pearson.loc['temperature_2m', 'relative_humidity_2m']:.4f}**;
Spearman = **{spearman.loc['temperature_2m', 'relative_humidity_2m']:.4f}**.
Korelasi memakai pasangan pada timestamp sama. Presipitasi tidak harus linear terhadap variabel lain;
karena itu Spearman dan analisis kondisional juga disediakan. Tidak dihitung p-value berbasis asumsi
observasi hourly independen. Autokorelasi dan pola waktu bersama membatasi interpretasi.
Korelasi tidak membuktikan hubungan sebab-akibat.

## Keputusan sebelum tahap 3

Missing values: {summary['missing_cells']}; pelanggaran fisik: {summary['physical_quality_violations']}.
Preprocessing akan mempertahankan nilai cuaca yang valid, menstandarkan nama/timestamp,
mengurutkan waktu, dan menambah fitur kalender. Jika masalah kualitas muncul pada dataset pengganti,
script preprocessing berhenti agar kebijakan penanganan ditinjau dan didokumentasikan terlebih dahulu.
""", encoding="utf-8")
    artifacts.append(report)
    finish_stage(2, raw, original_hash, artifacts, **summary)
    return summary


if __name__ == "__main__":
    run(raw_argument(__doc__).parse_args().raw)
