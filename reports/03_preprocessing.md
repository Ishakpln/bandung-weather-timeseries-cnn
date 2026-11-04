# Tahap 3 — Preprocessing

Dataset bersih: `data/processed/bandung_weather_hourly_clean.csv`.
Hasil: **35,064 baris × 23 kolom**, terdiri atas 9 kolom asal
(nama disederhanakan) dan 14 fitur kalender/siklik.
Dataset mentah terpisah di `data/raw/` dan hash-nya tidak berubah.

## Keputusan dan audit

| operation | count |
| --- | --- |
| Raw cells missing | 0 |
| Rows dropped | 0 |
| Observations generated | 0 |
| Cells imputed | 0 |
| Weather cells changed | 0 |
| Derived columns | 14 |

1. Metadata CSV dibaca terpisah; satuan disimpan dalam kamus data.
2. Nama variabel distandarkan tanpa mengubah besaran atau satuan.
3. Timestamp dilokalisasi ke Asia/Jakarta dan diekspor sebagai ISO8601 dengan offset `+07:00`.
4. Data diurutkan berdasarkan waktu; duplikasi dan ketidaklengkapan grid memicu penghentian.
5. Variabel kontinu bertipe float64; `weather_code` tetap kategori berkode integer.
6. Ditambahkan kalender serta encoding siklik jam, bulan, dan hari dalam tahun.
7. Dilakukan baca ulang CSV dan perbandingan seluruh nilai cuaca: identik = **True**.

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

| column | dtype_in_memory | role | unit_or_domain | description |
| --- | --- | --- | --- | --- |
| time | datetime64[ns, Asia/Jakarta] | timestamp | ISO8601 +07:00 | Waktu lokal WIB; hourly unik |
| temperature_2m | float64 | continuous_weather | °C | Nilai dari CSV mentah; tidak diubah |
| relative_humidity_2m | float64 | continuous_weather | % | Nilai dari CSV mentah; tidak diubah |
| dew_point_2m | float64 | continuous_weather | °C | Nilai dari CSV mentah; tidak diubah |
| precipitation | float64 | continuous_weather | mm | Nilai dari CSV mentah; tidak diubah |
| pressure_msl | float64 | continuous_weather | hPa | Nilai dari CSV mentah; tidak diubah |
| cloud_cover | float64 | continuous_weather | % | Nilai dari CSV mentah; tidak diubah |
| wind_speed_10m | float64 | continuous_weather | km/h | Nilai dari CSV mentah; tidak diubah |
| weather_code | int16 | categorical_weather | wmo code | Nilai dari CSV mentah; tidak diubah |
| hour | int16 | derived_calendar | 0..23 | Jam lokal |
| day | int16 | derived_calendar | 1..31 | Tanggal dalam bulan |
| day_of_week | int16 | derived_calendar | 0..6 | Senin=0; Minggu=6 |
| day_of_year | int16 | derived_calendar | 1..365/366 | Urutan hari; tahun kabisat dipertahankan |
| month | int16 | derived_calendar | 1..12 | Bulan kalender |
| year | int16 | derived_calendar | 2022..2025 | Tahun; bukan fitur wajib CNN |
| is_weekend | int16 | derived_calendar | 0/1 | Sabtu/Minggu=1; bukan asumsi pengaruh cuaca |
| date | object | derived_calendar | YYYY-MM-DD | Label tanggal lokal untuk agregasi |
| hour_sin | float64 | derived_cyclical | [-1,1] | Transformasi sinus/kosinus kalender deterministik |
| hour_cos | float64 | derived_cyclical | [-1,1] | Transformasi sinus/kosinus kalender deterministik |
| month_sin | float64 | derived_cyclical | [-1,1] | Transformasi sinus/kosinus kalender deterministik |
| month_cos | float64 | derived_cyclical | [-1,1] | Transformasi sinus/kosinus kalender deterministik |
| day_of_year_sin | float64 | derived_cyclical | [-1,1] | Transformasi sinus/kosinus kalender deterministik |
| day_of_year_cos | float64 | derived_cyclical | [-1,1] | Transformasi sinus/kosinus kalender deterministik |

## Kesiapan analisis dan tahap CNN selanjutnya

Grid hourly berurutan dan lengkap, satuan konsisten, seluruh nilai cuaca terjaga.
Dataset ini **belum dinormalisasi dan belum dibentuk menjadi window CNN**.
Statistik EDA seluruh periode hanya untuk pelaporan; tidak boleh digunakan untuk fitting scaler,
seleksi fitur berbasis target, atau penentuan threshold model dari validation/test.
Rekomendasi input, target, split waktu, dan leakage dibahas di `06_cnn_readiness.md` setelah tahap 5.
