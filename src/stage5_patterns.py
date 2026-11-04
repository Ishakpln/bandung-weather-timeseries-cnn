"""Tahap 5: numeric evidence, pattern interpretation, and CNN readiness documentation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CONTINUOUS, REPORTS, TABLES, begin_stage, finish_stage, json_write,
                    load_clean, markdown_table, raw_argument, resolve_raw, save_table)
from stage4_visualization import MONTHS


def conditional_rain(frame, variable, bins):
    groups = pd.cut(frame[variable], bins=bins, include_lowest=True)
    return frame.groupby(groups, observed=True).agg(
        n_hours=("precipitation", "size"), wet_hours=("precipitation", lambda x: x.gt(0).sum()),
        wet_pct=("precipitation", lambda x: x.gt(0).mean() * 100),
        mean_rain_mm=("precipitation", "mean"),
        mean_rain_positive_mm=("precipitation", lambda x: x[x > 0].mean()),
    ).reset_index().rename(columns={variable: "bin"})


def run(raw=None):
    raw = resolve_raw(raw)
    original_hash = begin_stage(5, raw)
    frame = load_clean()
    monthly = pd.read_csv(TABLES / "monthly_summary.csv")
    daily = pd.read_csv(TABLES / "daily_summary.csv", parse_dates=["time"])
    annual = pd.read_csv(TABLES / "annual_summary.csv").set_index("year")
    month = pd.read_csv(TABLES / "monthly_pattern.csv").set_index("month")
    hour = pd.read_csv(TABLES / "hourly_pattern.csv").set_index("hour")
    anomalies = pd.read_csv(TABLES / "temperature_anomaly_scores.csv", parse_dates=["time"])
    acf = pd.read_csv(TABLES / "autocorrelation.csv").set_index("lag_hours")
    pearson = frame[CONTINUOUS].corr()
    spearman = frame[CONTINUOUS].corr(method="spearman")
    residuals = frame[CONTINUOUS] - frame.groupby(["month", "hour"])[CONTINUOUS].transform("mean")
    residual_corr = residuals.corr()
    cloud_bins = conditional_rain(frame, "cloud_cover", [0, 25, 50, 75, 90, 100])
    humidity_bins = conditional_rain(frame, "relative_humidity_2m", [0, 40, 60, 70, 80, 90, 100])
    wet = frame.precipitation > 0
    dry_wet = frame.assign(state=np.where(wet, "wet_gt0", "dry_eq0")).groupby("state").agg(
        n_hours=("precipitation", "size"), temperature_mean=("temperature_2m", "mean"),
        humidity_mean=("relative_humidity_2m", "mean"), cloud_mean=("cloud_cover", "mean"),
        precipitation_mean=("precipitation", "mean")).reset_index()
    annual["temperature_change_vs_previous_C"] = annual.temperature_mean.diff()
    annual["rain_change_vs_previous_pct"] = annual.precipitation_sum.pct_change() * 100
    anomaly_year = anomalies.assign(year=anomalies.time.dt.year).groupby("year").agg(
        n_hours=("temperature_2m", "size"), candidates=("candidate", "sum"))
    anomaly_year["candidate_pct"] = anomaly_year.candidates / anomaly_year.n_hours * 100
    candidates = anomalies.loc[anomalies.candidate].copy()
    candidates["absolute_modified_z"] = candidates.modified_z.abs()
    ranked = candidates.sort_values("absolute_modified_z", ascending=False).head(20)
    top_rain_hours = frame.nlargest(10, "precipitation")[["time", "precipitation", "relative_humidity_2m", "cloud_cover"]]
    top_rain_days = daily.nlargest(10, "precipitation_sum")[["time", "precipitation_sum", "wet_hours"]]
    # A line fitted to daily means is a descriptive summary, not a climate trend test.
    elapsed_years = (daily.time - daily.time.iloc[0]).dt.total_seconds() / (365.2425 * 86400)
    slope = float(np.polyfit(elapsed_years, daily.temperature_mean, deg=1)[0])
    residual_daily = pd.Series(residuals.temperature_2m.to_numpy(), index=frame.time).resample("D").mean()
    residual_slope = float(np.polyfit(elapsed_years, residual_daily, deg=1)[0])
    max_month, min_month = int(month.temperature_mean.idxmax()), int(month.temperature_mean.idxmin())
    wet_month, dry_month = int(month.mean_monthly_rain_mm.idxmax()), int(month.mean_monthly_rain_mm.idxmin())
    max_hour, min_hour = int(hour.temperature_mean.idxmax()), int(hour.temperature_mean.idxmin())
    wet_hour, dry_hour = int(hour.wet_pct.idxmax()), int(hour.wet_pct.idxmin())
    absolute_min, absolute_max = frame.temperature_2m.min(), frame.temperature_2m.max()
    min_rows, max_rows = frame.loc[frame.temperature_2m.eq(absolute_min)], frame.loc[frame.temperature_2m.eq(absolute_max)]
    saturation = frame.cloud_cover.eq(100)
    saturated_dry = (saturation & ~wet).sum()
    sensitivity = pd.DataFrame([{"threshold_mm": threshold,
                                 "wet_hours_ge_threshold": int(frame.precipitation.ge(threshold).sum()),
                                 "percentage": frame.precipitation.ge(threshold).mean() * 100}
                                for threshold in [.1, 1., 5., 10.]])
    rain_total = float(frame.precipitation.sum())
    evidence = {
        "temperature_mean_C": float(frame.temperature_2m.mean()),
        "humidity_mean_pct": float(frame.relative_humidity_2m.mean()),
        "rain_total_timestamp_grouped_mm": rain_total,
        "wet_hours_gt0": int(wet.sum()), "wet_pct": float(wet.mean() * 100),
        "daily_temperature_linear_slope_C_per_year_descriptive": slope,
        "month_hour_residual_daily_slope_C_per_year_descriptive": residual_slope,
        "temp_humidity_pearson": float(pearson.loc['temperature_2m', 'relative_humidity_2m']),
        "temp_humidity_residual_pearson": float(residual_corr.loc['temperature_2m', 'relative_humidity_2m']),
        "hottest_month": max_month, "coolest_month": min_month,
        "warmest_hour_WIB": max_hour, "coolest_hour_WIB": min_hour,
        "anomaly_candidates": len(candidates),
        "zero_MAD_hours": int(anomalies.month_hour_mad_C.eq(0).sum()),
        "raw_sha256": original_hash,
    }
    artifacts = []
    for name, data in [("conditional_rain_cloud", cloud_bins), ("conditional_rain_humidity", humidity_bins),
                       ("dry_wet_comparison", dry_wet), ("annual_changes", annual.reset_index()),
                       ("anomaly_candidates_by_year", anomaly_year.reset_index()),
                       ("top_temperature_anomalies", ranked), ("top_rain_hours", top_rain_hours),
                       ("top_rain_days", top_rain_days), ("rain_threshold_sensitivity", sensitivity),
                       ("correlation_month_hour_residual_pearson", residual_corr.rename_axis("variable").reset_index())]:
        artifacts.append(save_table(name, data))
    evidence_path = TABLES / "pattern_evidence.json"
    json_write(evidence_path, evidence)
    artifacts.append(evidence_path)
    pair_rows = []
    for x, y in [("temperature_2m", "relative_humidity_2m"), ("cloud_cover", "precipitation"),
                 ("relative_humidity_2m", "precipitation")]:
        pair_rows.append({"pair": f"{x} / {y}", "Pearson": pearson.loc[x, y],
                          "Spearman": spearman.loc[x, y],
                          "Pearson_month_hour_residual": residual_corr.loc[x, y], "n": len(frame)})
    pair_table = pd.DataFrame(pair_rows)
    artifacts.append(save_table("key_associations", pair_table))

    report = REPORTS / "05_pattern_analysis_insight.md"
    report.write_text(f"""# Tahap 5 — Analisis pola dan insight

Seluruh angka dihitung dari {len(frame):,} observasi hourly yang lolos tahap 1–4.
`output/tables/pattern_evidence.json` menyimpan angka utama; grafik bernomor berada di `output/figures/`.
Penelitian ini deskriptif dan retrospektif. Tidak ada model, prediksi, atau eksperimen sebab-akibat.

## Karakteristik umum

Rata-rata suhu **{frame.temperature_2m.mean():.3f} °C** (SD {frame.temperature_2m.std():.3f} °C),
kelembapan **{frame.relative_humidity_2m.mean():.3f}%**, dan tutupan awan **{frame.cloud_cover.mean():.3f}%**.
Suhu minimum **{absolute_min:.1f} °C** tercatat {len(min_rows)} kali; kemunculan pertama {min_rows.time.iloc[0]}.
Maksimum **{absolute_max:.1f} °C** tercatat {len(max_rows)} kali; pertama {max_rows.time.iloc[0]}.
Presipitasi total dari label timestamp yang tersedia **{rain_total:.1f} mm** selama empat tahun;
{wet.sum():,} jam ({wet.mean() * 100:.2f}%) bernilai >0 mm, dan {(~wet).sum():,} jam bernilai nol.
Dasar: `descriptive_statistics.csv`, grafik 01–04 dan 11. Jumlah hujan bukan observasi stasiun tervalidasi.

## Pola harian

Rata-rata suhu tertinggi terjadi pada **{max_hour:02d}:00 WIB ({hour.loc[max_hour, 'temperature_mean']:.3f} °C)**,
terendah pada **{min_hour:02d}:00 WIB ({hour.loc[min_hour, 'temperature_mean']:.3f} °C)**.
Selisih puncak–minimum profil rata-rata **{hour.loc[max_hour, 'temperature_mean'] - hour.loc[min_hour, 'temperature_mean']:.3f} °C**.
Ini amplitudo profil rata-rata per jam, bukan rata-rata rentang harian; rata-rata rentang harian individual
adalah **{daily.diurnal_range.mean():.3f} °C** (`daily_summary.csv`). Grafik 06 memperlihatkan profil tiap tahun
dan IQR observasi, sehingga variasi antarhari tetap terlihat.

Frekuensi jam basah tertinggi berlabel **{wet_hour:02d}:00 WIB ({hour.loc[wet_hour, 'wet_pct']:.2f}%)**,
terendah **{dry_hour:02d}:00 WIB ({hour.loc[dry_hour, 'wet_pct']:.2f}%)**.
Setiap jam lokal memiliki {int(hour.n_hours.min()):,}–{int(hour.n_hours.max()):,} observasi.
Presipitasi berlabel {wet_hour:02d}:00 mewakili akumulasi jam sebelumnya; jangan menafsirkan label sebagai
waktu mulai hujan atau sebagai durasi kejadian hujan terpisah. Dasar: `hourly_pattern.csv`, grafik 12–13.
Pola yang berulang ini merupakan dasar untuk mempertimbangkan fitur jam siklik dan lookback harian pada CNN,
belum merupakan bukti bahwa pilihan tersebut meningkatkan kinerja prediksi.

## Pola bulanan dan variasi antartahun

Suhu gabungan tertinggi pada **{MONTHS[max_month-1]} ({month.loc[max_month, 'temperature_mean']:.3f} °C)**,
terendah pada **{MONTHS[min_month-1]} ({month.loc[min_month, 'temperature_mean']:.3f} °C)**;
selisih **{month.loc[max_month, 'temperature_mean'] - month.loc[min_month, 'temperature_mean']:.3f} °C**.
Rata-rata total presipitasi bulanan terbesar pada **{MONTHS[wet_month-1]}
({month.loc[wet_month, 'mean_monthly_rain_mm']:.2f} mm/bulan)**, terkecil pada **{MONTHS[dry_month-1]}
({month.loc[dry_month, 'mean_monthly_rain_mm']:.2f} mm/bulan)**.
Setelah mempertimbangkan panjang bulan, rata-rata mm/hari pada kedua bulan tersebut masing-masing
{month.loc[wet_month, 'mean_rain_mm_per_day']:.3f} dan {month.loc[dry_month, 'mean_rain_mm_per_day']:.3f}.
Ini ringkasan empat tahun, bukan normal klimatologis jangka panjang.
Dasar: `monthly_pattern.csv`, `monthly_summary.csv`, grafik 05, 12, dan 13.

{markdown_table(month[['n_hours', 'temperature_mean', 'humidity_mean', 'mean_monthly_rain_mm', 'wet_pct']].reset_index())}

## Tren deskriptif dan fluktuasi

{markdown_table(annual.reset_index())}

Rata-rata suhu 2025 dikurangi 2022 = **{annual.loc[2025, 'temperature_mean'] - annual.loc[2022, 'temperature_mean']:.3f} °C**.
Garis lurus pada 1.461 rata-rata harian memberi kemiringan **{slope:.4f} °C/tahun**;
setelah mengurangi rerata bulan–jam secara retrospektif, kemiringan rata-rata residual harian
**{residual_slope:.4f} °C/tahun**. Keduanya hanya ringkasan jendela empat tahun dan bukan estimasi
pemanasan jangka panjang. Tidak ada uji signifikansi, atribusi iklim, atau proyeksi dari kemiringan ini.
Presipitasi tahunan terendah pada **{annual.precipitation_sum.idxmin()} ({annual.precipitation_sum.min():.1f} mm)**
dan tertinggi **{annual.precipitation_sum.idxmax()} ({annual.precipitation_sum.max():.1f} mm)**;
fluktuasi ini tidak menunjukkan pola kenaikan hujan yang selalu monoton.
Tabel `rain_mm_per_day` menyesuaikan perbedaan durasi 2024 yang kabisat. Dasar: grafik 01 dan 03,
`annual_changes.csv`. Tidak diklaim penyebab seperti ENSO tanpa data pembanding dan analisis tambahan.

## Hubungan antarvariabel

{markdown_table(pair_table, 4)}

Grafik 07–10 memperlihatkan asosiasi simultan. Korelasi suhu–kelembapan berubah dari
{pearson.loc['temperature_2m', 'relative_humidity_2m']:.4f} menjadi
{residual_corr.loc['temperature_2m', 'relative_humidity_2m']:.4f} sesudah setiap variabel dikurangi
rerata pada kelompok bulan–jam yang sama. Perbandingan ini menunjukkan bahwa struktur waktu bersama
memengaruhi besarnya asosiasi. Residualisasi ini bukan kontrol seluruh faktor perancu atau analisis kausal,
dan dihitung dari seluruh data hanya untuk EDA.

Kelembapan–presipitasi memiliki Pearson gabungan **{pearson.loc['relative_humidity_2m', 'precipitation']:.4f}**
dan Spearman **{spearman.loc['relative_humidity_2m', 'precipitation']:.4f}**, sedangkan Pearson residual
bulan–jam = **{residual_corr.loc['relative_humidity_2m', 'precipitation']:.4f}**. Asosiasi agregat yang kecil
tidak cukup untuk menyimpulkan bahwa hubungan pada semua kondisi waktu juga kecil.
Perbedaan ini perlu diperiksa bersama distribusi nol dan pola jam; tidak menunjukkan pengaruh kausal.

### Presipitasi menurut tutupan awan

{markdown_table(cloud_bins)}

### Presipitasi menurut kelembapan

{markdown_table(humidity_bins)}

Batas bin ditetapkan eksplisit dalam script; jumlah observasi tiap bin harus diperhatikan.
Tutupan awan tepat 100% terdapat pada **{saturation.sum():,} jam**; di antaranya
**{saturated_dry:,} jam ({saturated_dry / saturation.sum() * 100:.2f}%)** memiliki hujan nol.
Dengan demikian kondisi awan penuh tidak identik dengan adanya hujan pada file ini.
Rata-rata hujan semua jam berbeda dari intensitas bersyarat hanya jam basah.

{markdown_table(dry_wet)}

Perbedaan kelompok basah/kering bersifat deskriptif dan dapat bercampur dengan pola jam, bulan,
serta sifat data model. Banyak nol, batas 100%, hubungan nonlinear, dan perbedaan dukungan waktu
(hujan interval vs variabel instan) membatasi tafsir korelasi. Jangan menyatakan awan atau kelembapan
menyebabkan presipitasi berdasarkan tabel ini.

## Autokorelasi dan dependensi waktu

Suhu: ACF lag 1 = **{acf.loc[1, 'temperature_raw']:.4f}**, lag 24 = **{acf.loc[24, 'temperature_raw']:.4f}**,
lag 168 = **{acf.loc[168, 'temperature_raw']:.4f}**. Setelah pengurangan rerata bulan–jam,
ACF residual suhu lag 1 = **{acf.loc[1, 'temperature_month_hour_residual']:.4f}** dan
lag 24 = **{acf.loc[24, 'temperature_month_hour_residual']:.4f}**.
Presipitasi: ACF lag 1 = **{acf.loc[1, 'precipitation']:.4f}**, lag 24 = **{acf.loc[24, 'precipitation']:.4f}**.
Dasar: `autocorrelation.csv`, grafik 14. Observasi hourly tidak dapat diperlakukan sebagai sampel IID;
random split berisiko memberi evaluasi terlalu optimistis karena kedekatan waktu.

## Kejadian bernilai besar dan kandidat anomali

Presipitasi hourly terbesar **{top_rain_hours.iloc[0].precipitation:.1f} mm** pada **{top_rain_hours.iloc[0].time}**.
Total per label tanggal tertinggi **{top_rain_days.iloc[0].precipitation_sum:.1f} mm** pada
**{top_rain_days.iloc[0].time}**. Ini nilai maksimum dataset, bukan klaim rekor Bandung.
Tabel sepuluh nilai terbesar: `top_rain_hours.csv` dan `top_rain_days.csv`.

{markdown_table(sensitivity)}

Terdapat **{len(candidates)} kandidat anomali suhu ({len(candidates) / len(frame) * 100:.3f}%)** dengan
|modified z| > 3,5 terhadap median/MAD bulan–jam; MAD nol terjadi pada {evidence['zero_MAD_hours']} jam.
Dasar: grafik 15, `temperature_anomaly_candidates.csv`, dan tabel berikut.

{markdown_table(anomaly_year.reset_index())}

Kandidat teratas (daftar hingga 20 tersedia di `top_temperature_anomalies.csv`):

{markdown_table(ranked[['time', 'temperature_2m', 'month_hour_median_C', 'month_hour_mad_C', 'modified_z']].head(10), 3)}

Kandidat dapat menunjukkan variasi meteorologis yang sah, kejadian tidak biasa, atau artefak model;
tanpa data stasiun/kejadian pembanding penyebabnya tidak dapat dipastikan. Semuanya dipertahankan.
MAD dihitung secara retrospektif dari semua tahun dan **tidak boleh digunakan sebagai transformasi CNN
sebelum split**. Tidak ada label anomali yang diklaim sebagai ground truth.

## Batas kesimpulan dan tindak lanjut yang disiapkan

Pola jam/bulan, hubungan suhu–kelembapan, serta sebaran presipitasi didukung grafik dan tabel di atas.
Temuan berlaku pada grid, periode, zona waktu, dan definisi variabel file ini. Empat tahun tidak cukup
untuk generalisasi klimatologis; validitas eksternal terhadap stasiun atau seluruh Kota Bandung belum diuji.
Pilihan model sumber tidak tercantum. Analisis ini berhenti pada tahap 5.
Dokumentasi CNN selanjutnya mencantumkan kandidat fitur/target serta split waktu tanpa membangun
window, melatih arsitektur, menjalankan prediksi, atau memilih model.
""", encoding="utf-8")
    artifacts.append(report)

    # Planning artifacts only: no split arrays, windows, fitted transforms, or model.
    split_rows = []
    for name, first, last in [("train", 2022, 2023), ("validation", 2024, 2024), ("test", 2025, 2025)]:
        part = frame.loc[frame.year.between(first, last)]
        split_rows.append({"split": name, "start_local": part.time.iloc[0], "end_local": part.time.iloc[-1],
                           "rows": len(part), "share_pct": len(part) / len(frame) * 100,
                           "illustration_L48_h1_windows": max(0, len(part) - 48 - 1 + 1),
                           "illustration_L48_h24_windows": max(0, len(part) - 48 - 24 + 1)})
    splits = pd.DataFrame(split_rows)
    artifacts.append(save_table("cnn_recommended_split_plan", splits))
    cnn = REPORTS / "06_cnn_readiness.md"
    cnn.write_text(f"""# Kesiapan dataset untuk CNN — dokumentasi, tanpa implementasi model

Status: tahap 1–5 telah selesai; data bersih tersedia, tetapi belum dinormalisasi, di-window,
dibagi menjadi tensor, dijadikan target, atau dipakai melatih model. Tidak ada arsitektur atau prediksi.
Tujuan peramalan dan horizon masih perlu ditetapkan pada penelitian berikutnya.

## Struktur dan kontrak data

- File: `data/processed/bandung_weather_hourly_clean.csv`; **35.064 baris, 23 kolom**.
- Satu baris = satu timestamp hourly pada satu grid sekitar Bandung, Asia/Jakarta, UTC+7.
- Rentang: 2022-01-01 00:00 sampai 2025-12-31 23:00 WIB; termasuk 29 Februari 2024.
- Nilai meteorologis asli tidak berubah; missing/duplikat/gap = 0 pada dataset saat ini.
- Metadata, satuan, dan dtype: `data/processed/dataset_metadata.json`, `data_dictionary_processed.csv`.
- Hash mentah: `{original_hash}`. Versi library dan manifest tersedia di `output/`.

## Kandidat variabel input

| Kelompok | Kandidat | Catatan pemakaian selanjutnya |
| --- | --- | --- |
| Cuaca kontinu | temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, pressure_msl, cloud_cover, wind_speed_10m | Gunakan riwayat yang telah tersedia pada waktu penerbitan prediksi; lakukan ablation di validation |
| Kalender siklik | hour_sin/cos, month_sin/cos, day_of_year_sin/cos | Sudah tersedia; nilai kalender masa depan diketahui, namun bentuk input diputuskan nanti |
| Kalender tambahan | hour, day, day_of_week, month, year, is_weekend | Untuk analisis tersedia; bukan otomatis seluruhnya dimasukkan ke model |
| Kategori opsional | weather_code | Gunakan one-hot/encoding kategori yang ditetapkan dari train; jangan mengasumsikan jarak numerik antarkode |
| Identitas waktu | time, date | Indeks, pengurutan, split, dan audit; jangan masukkan string timestamp sebagai fitur numerik mentah |

Suhu dan kelembapan berkorelasi (Pearson {pearson.loc['temperature_2m', 'relative_humidity_2m']:.4f});
hal ini menunjukkan potensi redundansi, bukan alasan otomatis membuang salah satunya.
Fitur suhu lag dapat menjadi input untuk target suhu masa depan selama timestamp input ≤ t.
Kode cuaca dibentuk dari variabel cuaca menurut sumber API; memasukkan kode pada waktu target
untuk memprediksi hujan pada waktu sama berpotensi kebocoran informasi. Gunakan hanya kode historis
yang diketahui pada waktu inferensi, atau keluarkan fitur ini dari baseline berikutnya.

## Kandidat target dan waktu informasi

Pilihan utama yang masuk akal untuk dirumuskan nanti: `temperature_2m` pada t+1 jam atau t+24 jam.
Alternatif: kelembapan masa depan; presipitasi masa depan sebagai regresi; atau indikator jam basah
masa depan sebagai klasifikasi. Definisi >0 mm pada EDA adalah operasional; tetapkan ambang target
berdasarkan tujuan penelitian sebelum mengevaluasi test.
Target alternatif belum dibuat dan tidak dipilih berdasarkan performa test.

Untuk target suhu t+h: input riwayat t−L+1, ..., t, dan label pada t+h, h≥1.
Gunakan variabel di timestamp ≤ t yang benar-benar sudah tersedia saat penerbitan prediksi.
Hujan timestamp t adalah akumulasi jam sebelumnya; perhatikan akhir interval dan latensi sumber.
Data grid historis dapat direvisi dan belum tentu tersedia pada waktu kejadian.
Keberhasilan evaluasi offline pada arsip ini kelak tidak otomatis membuktikan performa operasional.

## Rekomendasi split kronologis

{markdown_table(splits)}

Train: dua tahun awal; validation dan test masing-masing satu tahun penuh agar seluruh bulan terwakili.
Persentase mendekati 50/25/25%; batas kalender lebih utama daripada rasio baris persis.
Tabel hanya **rencana**, bukan split yang dipakai melatih model.
2025 disimpan sebagai holdout akhir dan tidak dipakai untuk tuning, pemilihan fitur, atau scaler.
EDA seluruh periode sudah dilakukan sesuai ruang lingkup penelitian; karena itu 2025 belum menjadi
holdout yang sepenuhnya tidak pernah dilihat. Bekukan protokol sebelum pemodelan dan laporkan
paparan EDA ini; bila keputusan model mengikuti pola test, butuhkan periode eksternal yang belum dilihat.
Walk-forward pada rentang sebelum test dapat direncanakan untuk memeriksa stabilitas antartahun.

## Rekomendasi window dan preprocessing berikutnya

1. Tentukan target, horizon, dan ketersediaan input terlebih dahulu. Kandidat lookback untuk dibandingkan
   hanya pada validation: 24, 48, 72, atau 168 jam. ACF suhu lag 24 = {acf.loc[24, 'temperature_raw']:.4f}
   dan lag 168 = {acf.loc[168, 'temperature_raw']:.4f}; nilai ini mendukung pemeriksaan kandidat, bukan pemilihan final.
2. Split berdasarkan waktu **sebelum** membuat window bertumpang tindih atau fitting transformasi.
   Rekomendasi konservatif: input dan seluruh label suatu window tetap berada pada split yang sama,
   sehingga tidak ada raw timestamp yang dipakai dua split. Jangan melakukan random split window.
3. Untuk ilustrasi single-target horizon h dan panjang input L, jumlah window valid mandiri
   = max(0, N−L−h+1), stride 1. Tabel di atas menghitung kapasitas L=48 untuk h=1/24,
   **tanpa membangun window**. Format yang direncanakan: X=(samples, L, jumlah_fitur),
   y=(samples, 1) untuk satu nilai target pada t+h. Target harus tercatat dengan timestamp tersendiri.
4. Pendekatan alternatif dapat membawa konteks historis train ke awal validation dan validation ke test
   apabila target tetap berada di split tujuan dan konteks sudah tersedia saat inferensi. Jika dipilih,
   dokumentasikan overlap konteks dan gunakan evaluasi temporal yang sesuai; tabel di sini memakai
   pendekatan mandiri tanpa konteks lintas split.
5. Fit scaler numerik pada train saja, lalu transform validation/test. StandardScaler atau MinMaxScaler
   merupakan kandidat, bukan transformasi yang telah diterapkan. Nilai di luar rentang train tidak
   otomatis di-clip. Scaler target terpisah jika dibutuhkan; simpan parameter train dan inverse transform.
6. Presipitasi memiliki {(~wet).mean() * 100:.2f}% nol dan ekor kanan; log1p atau pendekatan target
   basah/kering dapat dievaluasi pada validation. Tidak mengubah data nol atau menciptakan hujan.
   Jika klasifikasi, pertimbangkan ketidakseimbangan kelas hanya berdasarkan distribusi train.
7. Fitur siklik sudah memiliki rentang tetap. Hindari penggandaan kalender tanpa alasan; encoding WMO
   harus menangani kategori yang tidak ada pada train. Tidak menggunakan angka WMO sebagai skala kontinu.
8. Tidak perlu imputasi saat ini. Bila ada gap pada data baru, audit penyebab dan gunakan kebijakan
   per variabel yang kausal; jangan backward fill, interpolasi lintas split, atau statistik dari masa depan.
   Validasi kontiguitas window dan catat mask missing bila kebijakan berikutnya memerlukannya.
9. Jangan otomatis menghapus kandidat ekstrem. Statistik MAD/IQR/rerata bulan–jam dari EDA seluruh
   periode hanya diagnostik; jika dijadikan transformasi/threshold model, fit ulang pada train saja.
10. Simpan seed bila kelak memakai pengacakan, batas split, daftar fitur, unit, urutan kolom,
    timestamp input/label, parameter scaler, versi data, dan hash artefak untuk reproduksibilitas.

## Potensi leakage yang harus diaudit

- Scaler, imputasi, threshold outlier, residualisasi, atau seleksi fitur menggunakan seluruh periode.
- Target atau variabel cuaca masa depan masuk ke input; kesalahan shift t versus t+h.
- Random split setelah window dibuat; overlap input/label serta label horizon melintasi batas split.
- Rolling centered, backward fill, atau interpolasi memakai observasi sesudah waktu prediksi.
- Kode cuaca pada waktu target, atau variabel turunan target, dipakai menjelaskan target yang sama.
- Pemilihan horizon/fitur berdasarkan grafik test 2025 yang sudah dibaca pada EDA ini.
- Arsip historis yang direvisi diasumsikan sama dengan data yang tersedia secara real time.

Tahap berikutnya memerlukan protokol pemodelan terpisah. Pekerjaan saat ini selesai di tahap 5.
""", encoding="utf-8")
    artifacts.append(cnn)
    finish_stage(5, raw, original_hash, artifacts, analysis_complete=True,
                 cnn_training_performed=False, predictions_performed=False)
    return evidence


if __name__ == "__main__":
    run(raw_argument(__doc__).parse_args().raw)
