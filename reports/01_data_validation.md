# Tahap 1 — Pengumpulan dan validasi data

Status: **LULUS**. Dataset yang sudah tersedia digunakan tanpa mengunduh ulang.
File: `data\raw\BandungRawWeather.csv`. SHA-256: `8e3b4eda200da0b49fb416ec5bd985e8ac2c413dd24ab59fe140c649046ee469`.
CSV memiliki metadata pada dua baris awal, pemisah kosong, dan header tabel pada baris 4.
Metadata bukan observasi; data hourly berukuran **35,064 baris × 9 kolom**.

Periode lokal: 2022-01-01 00:00:00+07:00 sampai 2025-12-31 23:00:00+07:00; ekspektasi 35,064 jam
(2022: 8.760; 2023: 8.760; 2024: 8.784; 2025: 8.760).
Timestamp hilang: 0; tambahan: 0; duplikat: 0.
Urutan awal menaik: True. Interval sebelum pengurutan:
{'0 days 01:00:00': 35063}.

## Lokasi dan asal data

Titik penelitian: (-6.9175, 107.6191); metadata grid: (-6.9244285, 107.60539);
jarak haversine **1.698 km**, elevasi metadata 703.0 m.
Menurut [dokumentasi Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), koordinat respons adalah pusat grid yang dapat
berjarak beberapa kilometer dari titik permintaan. Pemeriksaan kewajaran proyek menggunakan ambang
10 km, bukan batas administratif atau jaminan ketepatan spasial.
Data konsisten dengan sekitar titik Bandung yang diminta; kecocokan persis dengan stasiun atau
representativitas seluruh wilayah kota tidak dibuktikan oleh file ini.

Zona waktu metadata `Asia/Jakarta` dengan offset 25200 detik.
Waktu pada CSV dibaca sebagai WIB, bukan UTC. Pengguna menyatakan sumber Historical Weather API;
CSV tidak menyimpan URL permintaan, tanggal pengunduhan, pilihan model, atau versi model.
Karena itu asal model spesifik dan proses pengambilan tidak dapat diautentikasi dari file saja.
Analisis diperlakukan sebagai data grid historis, bukan pengukuran stasiun langsung.
Arsipkan URL asli dan tanggal unduh jika tersedia untuk pelaporan akademik berikutnya.

## Nama, satuan, dan tipe mentah

| original_name | canonical_name | raw_dtype | unit |
| --- | --- | --- | --- |
| time | time | object | local ISO8601 |
| temperature_2m (°C) | temperature_2m | float64 | °C |
| relative_humidity_2m (%) | relative_humidity_2m | int64 | % |
| dew_point_2m (°C) | dew_point_2m | float64 | °C |
| precipitation (mm) | precipitation | float64 | mm |
| pressure_msl (hPa) | pressure_msl | float64 | hPa |
| cloud_cover (%) | cloud_cover | int64 | % |
| wind_speed_10m (km/h) | wind_speed_10m | float64 | km/h |
| weather_code (wmo code) | weather_code | int64 | wmo code |

`weather_code` adalah kategori berkode angka; rata-rata kode tidak ditafsirkan sebagai besaran cuaca.
Header dibaca sebagai UTF-8; simbol derajat pada CSV dipertahankan dalam kamus data.

## Seluruh pemeriksaan

| check | passed | observed |
| --- | --- | --- |
| schema_exact | True | ['time', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'pressure_msl', 'cloud_cover', 'wind_speed_10m', 'weather_code'] |
| unit_temperature_2m | True | temperature_2m (°C) |
| unit_relative_humidity_2m | True | relative_humidity_2m (%) |
| unit_dew_point_2m | True | dew_point_2m (°C) |
| unit_precipitation | True | precipitation (mm) |
| unit_pressure_msl | True | pressure_msl (hPa) |
| unit_cloud_cover | True | cloud_cover (%) |
| unit_wind_speed_10m | True | wind_speed_10m (km/h) |
| unit_weather_code | True | weather_code (wmo code) |
| rows | True | 35064 / 35064 |
| columns | True | 9 |
| parseable_timestamp | True | 0 |
| period_start | True | 2022-01-01 00:00:00+07:00 |
| period_end | True | 2025-12-31 23:00:00+07:00 |
| missing_hour | True | 0 |
| extra_timestamp | True | 0 |
| unique_timestamp | True | 0 |
| timezone | True | Asia/Jakarta |
| utc_offset_seconds | True | 25200.0 |
| location_sanity_check | True | 1.698189 km <= 10.0 km |
| numeric_temperature_2m | True | float64 |
| numeric_relative_humidity_2m | True | int64 |
| numeric_dew_point_2m | True | float64 |
| numeric_precipitation | True | float64 |
| numeric_pressure_msl | True | float64 |
| numeric_cloud_cover | True | int64 |
| numeric_wind_speed_10m | True | float64 |
| numeric_weather_code | True | int64 |

Hanya hasil LULUS mengizinkan tahap 2. Hash file mentah dan setiap artefak dicatat dalam manifest.
