# Kesiapan dataset untuk CNN — dokumentasi, tanpa implementasi model

Status: tahap 1–5 telah selesai; data bersih tersedia, tetapi belum dinormalisasi, di-window,
dibagi menjadi tensor, dijadikan target, atau dipakai melatih model. Tidak ada arsitektur atau prediksi.
Tujuan peramalan dan horizon masih perlu ditetapkan pada penelitian berikutnya.

## Struktur dan kontrak data

- File: `data/processed/bandung_weather_hourly_clean.csv`; **35.064 baris, 23 kolom**.
- Satu baris = satu timestamp hourly pada satu grid sekitar Bandung, Asia/Jakarta, UTC+7.
- Rentang: 2022-01-01 00:00 sampai 2025-12-31 23:00 WIB; termasuk 29 Februari 2024.
- Nilai meteorologis asli tidak berubah; missing/duplikat/gap = 0 pada dataset saat ini.
- Metadata, satuan, dan dtype: `data/processed/dataset_metadata.json`, `data_dictionary_processed.csv`.
- Hash mentah: `8e3b4eda200da0b49fb416ec5bd985e8ac2c413dd24ab59fe140c649046ee469`. Versi library dan manifest tersedia di `output/`.

## Kandidat variabel input

| Kelompok | Kandidat | Catatan pemakaian selanjutnya |
| --- | --- | --- |
| Cuaca kontinu | temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, pressure_msl, cloud_cover, wind_speed_10m | Gunakan riwayat yang telah tersedia pada waktu penerbitan prediksi; lakukan ablation di validation |
| Kalender siklik | hour_sin/cos, month_sin/cos, day_of_year_sin/cos | Sudah tersedia; nilai kalender masa depan diketahui, namun bentuk input diputuskan nanti |
| Kalender tambahan | hour, day, day_of_week, month, year, is_weekend | Untuk analisis tersedia; bukan otomatis seluruhnya dimasukkan ke model |
| Kategori opsional | weather_code | Gunakan one-hot/encoding kategori yang ditetapkan dari train; jangan mengasumsikan jarak numerik antarkode |
| Identitas waktu | time, date | Indeks, pengurutan, split, dan audit; jangan masukkan string timestamp sebagai fitur numerik mentah |

Suhu dan kelembapan berkorelasi (Pearson -0.8578);
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

| split | start_local | end_local | rows | share_pct | illustration_L48_h1_windows | illustration_L48_h24_windows |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2022-01-01 00:00:00+07:00 | 2023-12-31 23:00:00+07:00 | 17520 | 49.966 | 17472 | 17449 |
| validation | 2024-01-01 00:00:00+07:00 | 2024-12-31 23:00:00+07:00 | 8784 | 25.051 | 8736 | 8713 |
| test | 2025-01-01 00:00:00+07:00 | 2025-12-31 23:00:00+07:00 | 8760 | 24.983 | 8712 | 8689 |

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
   hanya pada validation: 24, 48, 72, atau 168 jam. ACF suhu lag 24 = 0.9145
   dan lag 168 = 0.8680; nilai ini mendukung pemeriksaan kandidat, bukan pemilihan final.
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
6. Presipitasi memiliki 73.25% nol dan ekor kanan; log1p atau pendekatan target
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
