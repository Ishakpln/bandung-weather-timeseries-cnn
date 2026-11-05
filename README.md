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

## CNN representation-learning experiment (new)

Requires PyTorch and scikit-learn in addition to the existing `requirements.txt`:

```bash
python -m pip install torch scikit-learn
python src/stage6_cnn_experiments.py --epochs 3
```

The script reads the existing cleaned CSV, preserves chronological splits (2022–23 / 2024 / 2025), fits scaling on train only, and trains separate 1D multivariate and 2D temperature recurrence-similarity convolutional autoencoders. Outputs: `output/cnn/` (models, figures, embeddings, scaler, metrics), and `reports/07_cnn_paper.md` (research draft). The default three-epoch CPU run is a **smoke-test baseline**, not a tuned or conclusive experiment. The 2D matrix is continuous recurrence similarity, **not a binary RP**. MSE between 1D and 2D target spaces must not be compared directly.


## Fokus penelitian revisi: interpretasi visual, bukan penemuan pola baru

Penelitian menelaah **apa yang dapat dilihat** dari visualisasi time-series asli, peta aktivasi CNN 1D, citra kemiripan rekuren, peta aktivasi CNN 2D, rekonstruksi, dan proyeksi embedding. Forecasting dan discovery meteorologi baru bukan tujuan. MSE masing-masing autoencoder merupakan diagnostik rekonstruksi, **bukan skor perbandingan langsung**.

Setelah menjalankan eksperimen baseline (`python src/stage6_cnn_experiments.py --epochs 3`), buat enam grafik interpretasi tambahan dari checkpoint tersimpan dengan:

```bash
python src/stage7_visual_interpretation.py
```

Grafik tersimpan di `output/cnn/`; keterangan dan keterbatasannya berada di `visual_interpretation_manifest.json`. Paper revisi: `reports/07_cnn_paper.pdf` dan `.md`. Grafik interpretasi menggunakan contoh window test pertama, tidak merepresentasikan keseluruhan tahun.
