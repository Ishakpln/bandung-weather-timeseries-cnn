# Tahap 4 — Panduan visualisasi

Seluruh 15 grafik PNG telah dirender pada **300 dpi**, dengan ukuran fisik dan
label sumbu untuk penggunaan dalam paper. Seluruh observasi digunakan; tidak ada subsampling.
File dan caption: `output/figures/figure_manifest.json`. Tabel sumber tersedia di `output/tables/`.

| filename | width_px | height_px | dpi | caption |
| --- | --- | --- | --- | --- |
| 01_temperature_timeseries.png | 3260 | 1411 | 300 | Time-series suhu; data hourly dan rata-rata per tanggal lokal. |
| 02_humidity_timeseries.png | 3261 | 1411 | 300 | Time-series kelembapan; data hourly dan rata-rata per tanggal lokal. |
| 03_precipitation_analysis.png | 3262 | 2750 | 300 | Agregasi presipitasi: tanggal, bulan, tahun. Pengelompokan mengikuti timestamp yang tersedia. |
| 04_precipitation_distribution.png | 2962 | 1411 | 300 | Nol ditampilkan terpisah; histogram positif memakai sumbu jumlah logaritmik. |
| 05_monthly_temperature_pattern.png | 2961 | 2446 | 300 | Boxplot rata-rata harian: kotak Q1–Q3, garis median, whisker 1,5 IQR; panel bawah empat tahun terpisah. |
| 06_hourly_temperature_humidity.png | 2960 | 2294 | 300 | Suhu dan kelembapan per jam; pita adalah rentang kuartil observasi, bukan confidence interval. |
| 07_temperature_vs_humidity.png | 2145 | 1685 | 300 | Suhu vs kelembapan; hexbin seluruh data, termasuk nol hujan; korelasi simultan bersifat deskriptif. |
| 08_cloud_cover_vs_precipitation.png | 2140 | 1685 | 300 | Tutupan awan vs presipitasi; hexbin seluruh data, termasuk nol hujan; korelasi simultan bersifat deskriptif. |
| 09_humidity_vs_precipitation.png | 2140 | 1685 | 300 | Kelembapan vs presipitasi; hexbin seluruh data, termasuk nol hujan; korelasi simultan bersifat deskriptif. |
| 10_correlation_heatmaps.png | 3833 | 1788 | 300 | Korelasi tujuh variabel kontinu; kode WMO dan fitur kalender dikecualikan. |
| 11_variable_distributions.png | 4162 | 2141 | 300 | Histogram seluruh variabel kontinu dan frekuensi kategori cuaca; hujan memakai sumbu jumlah log. |
| 12_month_hour_patterns.png | 2742 | 2294 | 300 | Heatmap bulan–jam dari seluruh observasi; persentase menggunakan jumlah jam pada sel masing-masing. |
| 13_monthly_hourly_rain_patterns.png | 3261 | 1411 | 300 | Presipitasi bulanan dirata-ratakan atas empat total tahunan; jam basah memakai ambang >0 mm. |
| 14_autocorrelation.png | 2961 | 1411 | 300 | Autokorelasi lag 1–168 jam; residual bulan–jam dihitung retrospektif hanya untuk EDA. |
| 15_temperature_anomaly_candidates.png | 3260 | 1471 | 300 | Median/MAD per bulan–jam; kandidat bukan bukti kesalahan data atau kejadian ekstrem terverifikasi. |

## Cara membaca dan batasan

Time-series menampilkan variasi hourly dan rata-rata harian tanpa smoothing yang mengubah data sumber.
Pita pada pola jam adalah IQR observasi, bukan interval keyakinan. Boxplot pola bulan menggunakan
**rata-rata harian**, sedangkan garis pola bulanan merata-ratakan observasi hourly.
Rata-rata gabungan per bulan/jam dibobot menurut jumlah jam; Februari 2024 mencakup hari kabisat.
Rata-rata total hujan per bulan kalender dihitung dari empat total bulanan, bukan jumlah empat tahun.
Perbandingan intensitas antarbulan juga tersedia sebagai mm/hari agar durasi bulan tidak diabaikan.

Hujan nol dipisahkan dari histogram jam positif. Hexbin menunjukkan kepadatan semua pasangan,
termasuk nol hujan; warna berskala log membantu menghindari tumpang tindih titik.
Pearson dan Spearman hanya untuk variabel cuaca kontinu pada timestamp sama.

Menurut [definisi variabel Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api),
presipitasi adalah jumlah pada jam sebelumnya. Agregasi di sini mengikuti **label timestamp asli**:
nilai 00:00 berasal dari interval sebelumnya. Jadi jumlah per tanggal/bulan/tahun merupakan
kelompok label yang tersedia; batas fisik kalender tepat memerlukan rekaman jam berikutnya
yang tidak ada pada akhir dataset. Tidak dilakukan pergeseran atau penambahan jam.

Autokorelasi tidak memakai pita signifikansi IID karena sampel hourly bergantung secara temporal.
Residual terhadap rerata bulan–jam memakai seluruh periode dan hanya untuk deskripsi retrospektif.
Anomali suhu memakai modified z = 0,67448975 × (suhu−median bulan/jam) / MAD bulan/jam,
dengan ambang absolut >3,5. MAD nol akan menghasilkan skor NA, bukan pembagian nol.
Semua kandidat tetap tersimpan; ambang ditetapkan sebagai diagnostik dan bukan label supervisi CNN.
