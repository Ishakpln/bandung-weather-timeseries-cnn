# Analisis time-series cuaca Bandung — tahap 1–5

Analisis akademik dataset hourly Open-Meteo periode 1 Januari 2022–31 Desember 2025,
zona waktu Asia/Jakarta. Pekerjaan berhenti pada analisis pola dan dokumentasi kesiapan CNN.
Tidak ada training, arsitektur, atau prediksi CNN.

Dataset yang tersedia: `data/raw/BandungRawWeather.csv`, berisi metadata lokasi dan tabel
35.064 baris × 9 kolom. File mentah tidak diubah atau diunduh ulang.
Hasil preprocessing: 35.064 baris × 23 kolom; nilai cuaca tetap, ditambah 14 fitur waktu.

## Menjalankan ulang

Lingkungan hasil saat ini: Python 3.13.9. Untuk mengulang lingkungan yang digunakan,
pakai Python 3.13.9 dan versi pada `requirements.txt`.
Jalankan dari direktori proyek:

```powershell
python -m pip install -r requirements.txt
python src/run_all.py
python src/verify_results.py
```

Paket yang diperlukan sudah tersedia saat artefak ini dibuat; instalasi ulang hanya diperlukan
di environment baru. Pipeline menggunakan data lokal dan tidak mengakses jaringan.
Setiap tahap diperiksa sebelum tahap berikutnya dimulai; kegagalan menghentikan pipeline.
Versi paket tersimpan pada `output/tables/environment.json`.

Untuk eksekusi satu per satu, jalankan **secara berurutan**:

```powershell
python src/stage1_validation.py
python src/stage2_eda.py
python src/stage3_preprocessing.py
python src/stage4_visualization.py
python src/stage5_patterns.py
python src/verify_results.py
```

Default input adalah satu CSV dalam `data/raw/`. Jika ada beberapa CSV, tentukan
`--raw data/raw/BandungRawWeather.csv` pada setiap perintah, atau pada `run_all.py`.
Kode menggunakan path relatif terhadap lokasi script, sehingga output tetap di proyek ini.
Rerun mengganti artefak turunan pada folder hasil; file raw selalu dibaca saja.
Jika kode tahap, data sumber, atau artefak prasyarat berubah, jalankan ulang dari tahap yang
terpengaruh; perubahan `common.py` memerlukan rerun dari tahap 1.

## Artefak dan laporan

| Hasil | Lokasi |
| --- | --- |
| Script tiap tahap | [src/](src/) |
| Validasi data/lokasi/periode | [reports/01_data_validation.md](reports/01_data_validation.md) |
| Laporan EDA | [reports/02_eda.md](reports/02_eda.md) |
| Laporan preprocessing | [reports/03_preprocessing.md](reports/03_preprocessing.md) |
| Dataset bersih | [data/processed/bandung_weather_hourly_clean.csv](data/processed/bandung_weather_hourly_clean.csv) |
| Metadata dataset | [data/processed/dataset_metadata.json](data/processed/dataset_metadata.json) |
| 15 grafik PNG 300 dpi | [output/figures/](output/figures/) |
| Daftar grafik dan caption | [reports/04_visualization_guide.md](reports/04_visualization_guide.md) |
| Laporan pola dan insight | [reports/05_pattern_analysis_insight.md](reports/05_pattern_analysis_insight.md) |
| Kesiapan CNN, fitur/target/split/leakage | [reports/06_cnn_readiness.md](reports/06_cnn_readiness.md) |
| Statistik, agregasi, korelasi, kandidat anomali | [output/tables/](output/tables/) |
| Bukti keberhasilan tiap tahap dan SHA-256 | [output/manifests/](output/manifests/) |
| Hasil pemeriksaan integritas akhir | [output/tables/verification_results.json](output/tables/verification_results.json) |

`verify_results.py` memeriksa hash dan kelengkapan artefak, identitas nilai cuaca sebelum/sesudah,
grid waktu dan tahun kabisat, konsistensi total agregasi, korelasi yang dihitung ulang melalui
NumPy/SciPy, fitur kalender, rencana split, gambar PNG/dpi, serta penolakan saat prasyarat tidak sah.

## Keputusan metodologis

Tidak ditemukan missing, duplikat, atau jam hilang; tidak dilakukan imputasi atau penghapusan outlier.
Nol presipitasi dipertahankan. Kode cuaca adalah kategori dan dikecualikan dari korelasi kontinu.
Data mentah memuat koordinat grid, yang berbeda sedikit dari koordinat permintaan Bandung;
jarak dihitung dan batas verifikasi spasial dijelaskan di laporan validasi.
Model sumber, URL asli pengambilan, serta tanggal unduh tidak tersedia dalam CSV.

Definisi grid, timezone, satuan, dan interval presipitasi merujuk
[dokumentasi resmi Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
(diakses 4 November 2026). Tidak ada pemanggilan endpoint unduh pada pipeline.
Jumlah hujan mengikuti label timestamp; definisi jam sebelumnya dan keterbatasan batas kalender
dijelaskan pada panduan visualisasi. Korelasi dan slope empat tahun bersifat deskriptif.

Rencana CNN memakai train 2022–2023, validation 2024, dan test 2025, tanpa fitting transformasi saat ini.
Karena EDA sudah mencakup 2025, dokumentasi menjelaskan batas holdout yang sudah dilihat serta
perlunya membekukan protokol sebelum modeling. Scaler, imputasi, atau threshold model berikutnya
harus dipelajari dari train; seluruh hasil EDA bukan parameter preprocessing CNN.

## Tahap 6–9: CNN 1D, evaluasi, insight, dan paper

CNN pipeline:

```bash
python src/stage6_9_cnn_pipeline.py
```

Pipeline membandingkan lookback `[24, 48, 72]` jam dengan horizon `[1, 6, 24]` jam,
fit StandardScaler pada train (2022–2023) saja, memakai validation 2024 untuk pemilihan,
dan test 2025 untuk evaluasi. Model memakai PyTorch CPU, seed 42, training window stride 3
untuk efisiensi dan evaluasi stride 1. Hasil disimpan pada `output/cnn/`.

Paper penelitian: `paper/Paper_CNN_TimeSeries_Bandung.docx`.
Laporan insight/QA: `reports/07_cnn_experiments_insight.md`.

**Catatan interpretasi:** 2025 sudah ikut EDA tahap sebelumnya, sehingga test kronologis ini
bukan holdout yang sepenuhnya blind. Baseline persistence disertakan. CNN mengungguli baseline
pada horizon 6 jam tetapi belum pada horizon 1 atau 24 jam; jangan klaim keunggulan universal.
