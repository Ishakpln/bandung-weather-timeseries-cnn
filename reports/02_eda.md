# Tahap 2 — Exploratory Data Analysis

Analisis menggunakan seluruh 35,064 observasi hourly, bukan sampel acak.
Statistik dihitung sebelum preprocessing; nilai cuaca tidak diubah.
Standar deviasi menggunakan sampel (`ddof=1`); kurtosis dalam tabel lengkap adalah excess kurtosis.

## Statistik deskriptif

| variable | count | mean | std | min | 50% | max | skewness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| temperature_2m | 35064.000 | 23.025 | 3.136 | 13.900 | 22.500 | 33.500 | 0.409 |
| relative_humidity_2m | 35064.000 | 79.465 | 14.171 | 18.000 | 84.000 | 100.000 | -1.099 |
| dew_point_2m | 35064.000 | 18.945 | 1.863 | 4.100 | 19.300 | 23.300 | -1.639 |
| precipitation | 35064.000 | 0.231 | 1.007 | 0.000 | 0.000 | 27.800 | 9.111 |
| pressure_msl | 35064.000 | 1011.665 | 1.951 | 1004.700 | 1011.700 | 1018.500 | -0.126 |
| cloud_cover | 35064.000 | 80.132 | 31.340 | 0.000 | 99.000 | 100.000 | -1.382 |
| wind_speed_10m | 35064.000 | 6.050 | 3.403 | 0.000 | 5.500 | 24.900 | 0.954 |

Tabel lengkap, termasuk persentil 1/5/25/50/75/95/99, jumlah nilai unik, dan nol:
`output/tables/descriptive_statistics.csv`. Distribusi divisualisasikan pada tahap 4.
Kode cuaca dianalisis sebagai kategori dan dikecualikan dari statistik kontinu serta korelasi.

## Missing values, duplikasi, dan waktu

Jumlah sel kosong: **0**. Baris identik duplikat: **0**.
Timestamp duplikat: **0**. Semua 48 bulan lengkap: **True**.
Tidak ada jam yang ditambahkan atau dihapus pada tahap ini.
Perincian tersedia pada `missing_values.csv` dan `monthly_coverage.csv`.
Tahun kabisat 2024 memiliki 8.784 jam, bukan 8.760 jam.

## Pemeriksaan kualitas fisik dan konsistensi

| check | violations |
| --- | --- |
| humidity_outside_0_100 | 0 |
| cloud_outside_0_100 | 0 |
| negative_precipitation | 0 |
| negative_wind | 0 |
| nonpositive_pressure | 0 |
| temperature_below_absolute_zero | 0 |
| dewpoint_below_absolute_zero | 0 |
| dewpoint_above_temperature_tolerance_0_3C | 0 |
| unknown_or_fractional_weather_code | 0 |
| infinite_numeric_value | 0 |

Aturan memeriksa rentang persen [0,100], nilai negatif pada hujan/angin, tekanan nonpositif,
suhu di bawah nol absolut, angka tak hingga, kode kategori tak dikenal, serta titik embun di atas suhu.
Toleransi titik embun 0,3 °C adalah pilihan pemeriksaan pembulatan proyek, bukan batas kualitas universal.
Lolos aturan ini menunjukkan konsistensi internal, bukan pembuktian akurasi terhadap pengukuran lapangan.
Tidak digunakan batas suhu regional yang dibuat-buat untuk menghapus nilai ekstrem.

## Distribusi hujan dan kategori cuaca

Hujan nol terjadi pada **25,686 jam (73.25%)**.
Definisi operasional jam basah: presipitasi > 0 mm; jumlah **9,378 jam**.
Median pada jam basah 0.20 mm. Sensitivitas ambang:
≥0,1 mm = 9,378 jam; ≥1 mm = 1,901 jam.
Ambang ini mendeskripsikan file, bukan klasifikasi intensitas resmi.
Nol merupakan hasil data yang sah dan tidak dianggap missing.

| weather_code | count | percentage |
| --- | --- | --- |
| 0 | 3193 | 9.106 |
| 1 | 3054 | 8.710 |
| 2 | 2462 | 7.021 |
| 3 | 16967 | 48.389 |
| 45 | 10 | 0.029 |
| 51 | 6176 | 17.614 |
| 53 | 1301 | 3.710 |
| 55 | 365 | 1.041 |
| 61 | 658 | 1.877 |
| 63 | 603 | 1.720 |
| 65 | 148 | 0.422 |
| 80 | 63 | 0.180 |
| 81 | 32 | 0.091 |
| 82 | 4 | 0.011 |
| 95 | 28 | 0.080 |

## Nilai ekstrem dan pengulangan

| variable | basis | basis_n | q1 | q3 | lower_fence | upper_fence | flagged_count | flagged_pct_of_basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temperature_2m | all_hours | 35064 | 20.800 | 25.200 | 14.200 | 31.800 | 109 | 0.311 |
| relative_humidity_2m | all_hours | 35064 | 72.000 | 91.000 | 43.500 | 119.500 | 838 | 2.390 |
| dew_point_2m | all_hours | 35064 | 18.100 | 20.200 | 14.950 | 23.350 | 1373 | 3.916 |
| precipitation | positive_hours_only | 9378 | 0.100 | 0.700 | -0.800 | 1.600 | 1284 | 13.692 |
| pressure_msl | all_hours | 35064 | 1010.300 | 1013.100 | 1006.100 | 1017.300 | 70 | 0.200 |
| cloud_cover | all_hours | 35064 | 69.000 | 100.000 | 22.500 | 146.500 | 3665 | 10.452 |
| wind_speed_10m | all_hours | 35064 | 3.600 | 7.900 | -2.850 | 14.350 | 852 | 2.430 |

Fence Tukey Q1−1,5×IQR sampai Q3+1,5×IQR hanya menandai kandidat pemeriksaan.
Untuk presipitasi, fence dihitung pada jam positif agar nol dominan tidak mendefinisikan hujan sebagai kesalahan.
Tidak ada outlier yang otomatis dibuang, diganti, atau di-winsorize.
Variabel berbatas seperti awan dapat menghasilkan flag distribusi tanpa kesalahan fisik.

| variable | longest_unchanged_run_hours | mode | mode_pct |
| --- | --- | --- | --- |
| temperature_2m | 5 | 21.500 | 1.660 |
| relative_humidity_2m | 7 | 93.000 | 4.831 |
| dew_point_2m | 6 | 19.900 | 3.009 |
| precipitation | 265 | 0.000 | 73.255 |
| pressure_msl | 4 | 1012.200 | 2.048 |
| cloud_cover | 136 | 100.000 | 48.708 |
| wind_speed_10m | 5 | 5.400 | 2.076 |
| weather_code | 69 | 3.000 | 48.389 |

Pengulangan nilai dapat muncul akibat pembulatan, batas 100%, periode kering, atau keluaran model.
Panjang run sendiri tidak membuktikan sensor macet; ini data grid.

## Hubungan awal

Pearson suhu–kelembapan = **-0.8578**;
Spearman = **-0.8368**.
Korelasi memakai pasangan pada timestamp sama. Presipitasi tidak harus linear terhadap variabel lain;
karena itu Spearman dan analisis kondisional juga disediakan. Tidak dihitung p-value berbasis asumsi
observasi hourly independen. Autokorelasi dan pola waktu bersama membatasi interpretasi.
Korelasi tidak membuktikan hubungan sebab-akibat.

## Keputusan sebelum tahap 3

Missing values: 0; pelanggaran fisik: 0.
Preprocessing akan mempertahankan nilai cuaca yang valid, menstandarkan nama/timestamp,
mengurutkan waktu, dan menambah fitur kalender. Jika masalah kualitas muncul pada dataset pengganti,
script preprocessing berhenti agar kebijakan penanganan ditinjau dan didokumentasikan terlebih dahulu.
