# Tahap 5 — Analisis pola dan insight

Seluruh angka dihitung dari 35,064 observasi hourly yang lolos tahap 1–4.
`output/tables/pattern_evidence.json` menyimpan angka utama; grafik bernomor berada di `output/figures/`.
Penelitian ini deskriptif dan retrospektif. Tidak ada model, prediksi, atau eksperimen sebab-akibat.

## Karakteristik umum

Rata-rata suhu **23.025 °C** (SD 3.136 °C),
kelembapan **79.465%**, dan tutupan awan **80.132%**.
Suhu minimum **13.9 °C** tercatat 1 kali; kemunculan pertama 2024-07-13 06:00:00+07:00.
Maksimum **33.5 °C** tercatat 1 kali; pertama 2023-10-28 13:00:00+07:00.
Presipitasi total dari label timestamp yang tersedia **8115.7 mm** selama empat tahun;
9,378 jam (26.75%) bernilai >0 mm, dan 25,686 jam bernilai nol.
Dasar: `descriptive_statistics.csv`, grafik 01–04 dan 11. Jumlah hujan bukan observasi stasiun tervalidasi.

## Pola harian

Rata-rata suhu tertinggi terjadi pada **12:00 WIB (27.475 °C)**,
terendah pada **06:00 WIB (19.261 °C)**.
Selisih puncak–minimum profil rata-rata **8.213 °C**.
Ini amplitudo profil rata-rata per jam, bukan rata-rata rentang harian; rata-rata rentang harian individual
adalah **9.087 °C** (`daily_summary.csv`). Grafik 06 memperlihatkan profil tiap tahun
dan IQR observasi, sehingga variasi antarhari tetap terlihat.

Frekuensi jam basah tertinggi berlabel **15:00 WIB (64.68%)**,
terendah **07:00 WIB (5.89%)**.
Setiap jam lokal memiliki 1,461–1,461 observasi.
Presipitasi berlabel 15:00 mewakili akumulasi jam sebelumnya; jangan menafsirkan label sebagai
waktu mulai hujan atau sebagai durasi kejadian hujan terpisah. Dasar: `hourly_pattern.csv`, grafik 12–13.
Pola yang berulang ini merupakan dasar untuk mempertimbangkan fitur jam siklik dan lookback harian pada CNN,
belum merupakan bukti bahwa pilihan tersebut meningkatkan kinerja prediksi.

## Pola bulanan dan variasi antartahun

Suhu gabungan tertinggi pada **Okt (23.848 °C)**,
terendah pada **Jul (22.360 °C)**;
selisih **1.488 °C**.
Rata-rata total presipitasi bulanan terbesar pada **Nov
(287.95 mm/bulan)**, terkecil pada **Agu
(63.85 mm/bulan)**.
Setelah mempertimbangkan panjang bulan, rata-rata mm/hari pada kedua bulan tersebut masing-masing
9.598 dan 2.060.
Ini ringkasan empat tahun, bukan normal klimatologis jangka panjang.
Dasar: `monthly_pattern.csv`, `monthly_summary.csv`, grafik 05, 12, dan 13.

| month | n_hours | temperature_mean | humidity_mean | mean_monthly_rain_mm | wet_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | 2976 | 22.758 | 82.388 | 182.375 | 36.694 |
| 2 | 2712 | 22.784 | 82.684 | 188.950 | 38.680 |
| 3 | 2976 | 22.934 | 82.749 | 224.625 | 34.341 |
| 4 | 2880 | 23.037 | 83.095 | 197.000 | 29.306 |
| 5 | 2976 | 23.211 | 82.554 | 216.950 | 25.571 |
| 6 | 2880 | 22.670 | 81.554 | 88.400 | 19.062 |
| 7 | 2976 | 22.360 | 78.511 | 76.450 | 14.651 |
| 8 | 2976 | 22.952 | 73.212 | 63.850 | 13.340 |
| 9 | 2880 | 23.206 | 72.912 | 128.275 | 17.604 |
| 10 | 2976 | 23.848 | 73.308 | 162.150 | 23.051 |
| 11 | 2880 | 23.319 | 80.173 | 287.950 | 34.514 |
| 12 | 2976 | 23.207 | 80.722 | 211.950 | 34.980 |

## Tren deskriptif dan fluktuasi

| year | n_hours | temperature_mean | temperature_std | humidity_mean | precipitation_sum | wet_hours | wind_mean | cloud_mean | wet_pct | rain_mm_per_day | temperature_change_vs_previous_C | rain_change_vs_previous_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 8760 | 22.584 | 2.852 | 82.152 | 2131.700 | 2542 | 6.220 | 86.800 | 29.018 | 5.840 | NA | NA |
| 2023 | 8760 | 23.050 | 3.545 | 77.205 | 1422.600 | 1975 | 6.674 | 74.031 | 22.546 | 3.898 | 0.466 | -33.265 |
| 2024 | 8784 | 23.221 | 3.237 | 79.256 | 2297.800 | 2249 | 6.160 | 77.321 | 25.603 | 6.278 | 0.171 | 61.521 |
| 2025 | 8760 | 23.245 | 2.806 | 79.247 | 2263.600 | 2612 | 5.145 | 82.382 | 29.817 | 6.202 | 0.023 | -1.488 |

Rata-rata suhu 2025 dikurangi 2022 = **0.661 °C**.
Garis lurus pada 1.461 rata-rata harian memberi kemiringan **0.2439 °C/tahun**;
setelah mengurangi rerata bulan–jam secara retrospektif, kemiringan rata-rata residual harian
**0.2025 °C/tahun**. Keduanya hanya ringkasan jendela empat tahun dan bukan estimasi
pemanasan jangka panjang. Tidak ada uji signifikansi, atribusi iklim, atau proyeksi dari kemiringan ini.
Presipitasi tahunan terendah pada **2023 (1422.6 mm)**
dan tertinggi **2024 (2297.8 mm)**;
fluktuasi ini tidak menunjukkan pola kenaikan hujan yang selalu monoton.
Tabel `rain_mm_per_day` menyesuaikan perbedaan durasi 2024 yang kabisat. Dasar: grafik 01 dan 03,
`annual_changes.csv`. Tidak diklaim penyebab seperti ENSO tanpa data pembanding dan analisis tambahan.

## Hubungan antarvariabel

| pair | Pearson | Spearman | Pearson_month_hour_residual | n |
| --- | --- | --- | --- | --- |
| temperature_2m / relative_humidity_2m | -0.8578 | -0.8368 | -0.6717 | 35064 |
| cloud_cover / precipitation | 0.1201 | 0.2538 | 0.0929 | 35064 |
| relative_humidity_2m / precipitation | 0.0339 | -0.0402 | 0.2814 | 35064 |

Grafik 07–10 memperlihatkan asosiasi simultan. Korelasi suhu–kelembapan berubah dari
-0.8578 menjadi
-0.6717 sesudah setiap variabel dikurangi
rerata pada kelompok bulan–jam yang sama. Perbandingan ini menunjukkan bahwa struktur waktu bersama
memengaruhi besarnya asosiasi. Residualisasi ini bukan kontrol seluruh faktor perancu atau analisis kausal,
dan dihitung dari seluruh data hanya untuk EDA.

Kelembapan–presipitasi memiliki Pearson gabungan **0.0339**
dan Spearman **-0.0402**, sedangkan Pearson residual
bulan–jam = **0.2814**. Asosiasi agregat yang kecil
tidak cukup untuk menyimpulkan bahwa hubungan pada semua kondisi waktu juga kecil.
Perbedaan ini perlu diperiksa bersama distribusi nol dan pola jam; tidak menunjukkan pengaruh kausal.

### Presipitasi menurut tutupan awan

| bin | n_hours | wet_hours | wet_pct | mean_rain_mm | mean_rain_positive_mm |
| --- | --- | --- | --- | --- | --- |
| (-0.001, 25.0] | 4050 | 143 | 3.531 | 0.009 | 0.248 |
| (25.0, 50.0] | 2924 | 495 | 16.929 | 0.039 | 0.231 |
| (50.0, 75.0] | 2488 | 531 | 21.342 | 0.080 | 0.376 |
| (75.0, 90.0] | 2562 | 549 | 21.429 | 0.184 | 0.858 |
| (90.0, 100.0] | 23040 | 7660 | 33.247 | 0.317 | 0.952 |

### Presipitasi menurut kelembapan

| bin | n_hours | wet_hours | wet_pct | mean_rain_mm | mean_rain_positive_mm |
| --- | --- | --- | --- | --- | --- |
| (-0.001, 40.0] | 582 | 45 | 7.732 | 0.019 | 0.251 |
| (40.0, 60.0] | 3482 | 773 | 22.200 | 0.062 | 0.280 |
| (60.0, 70.0] | 4150 | 1341 | 32.313 | 0.150 | 0.464 |
| (70.0, 80.0] | 6513 | 2271 | 34.869 | 0.446 | 1.280 |
| (80.0, 90.0] | 11564 | 3017 | 26.090 | 0.234 | 0.895 |
| (90.0, 100.0] | 8773 | 1931 | 22.011 | 0.189 | 0.857 |

Batas bin ditetapkan eksplisit dalam script; jumlah observasi tiap bin harus diperhatikan.
Tutupan awan tepat 100% terdapat pada **17,079 jam**; di antaranya
**10,824 jam (63.38%)** memiliki hujan nol.
Dengan demikian kondisi awan penuh tidak identik dengan adanya hujan pada file ini.
Rata-rata hujan semua jam berbeda dari intensitas bersyarat hanya jam basah.

| state | n_hours | temperature_mean | humidity_mean | cloud_mean | precipitation_mean |
| --- | --- | --- | --- | --- | --- |
| dry_eq0 | 25686 | 22.634 | 79.562 | 75.801 | 0.000 |
| wet_gt0 | 9378 | 24.096 | 79.198 | 91.993 | 0.865 |

Perbedaan kelompok basah/kering bersifat deskriptif dan dapat bercampur dengan pola jam, bulan,
serta sifat data model. Banyak nol, batas 100%, hubungan nonlinear, dan perbedaan dukungan waktu
(hujan interval vs variabel instan) membatasi tafsir korelasi. Jangan menyatakan awan atau kelembapan
menyebabkan presipitasi berdasarkan tabel ini.

## Autokorelasi dan dependensi waktu

Suhu: ACF lag 1 = **0.9347**, lag 24 = **0.9145**,
lag 168 = **0.8680**. Setelah pengurangan rerata bulan–jam,
ACF residual suhu lag 1 = **0.8590** dan
lag 24 = **0.4891**.
Presipitasi: ACF lag 1 = **0.3877**, lag 24 = **0.1917**.
Dasar: `autocorrelation.csv`, grafik 14. Observasi hourly tidak dapat diperlakukan sebagai sampel IID;
random split berisiko memberi evaluasi terlalu optimistis karena kedekatan waktu.

## Kejadian bernilai besar dan kandidat anomali

Presipitasi hourly terbesar **27.8 mm** pada **2024-01-07 12:00:00+07:00**.
Total per label tanggal tertinggi **67.1 mm** pada
**2023-05-04 00:00:00+07:00**. Ini nilai maksimum dataset, bukan klaim rekor Bandung.
Tabel sepuluh nilai terbesar: `top_rain_hours.csv` dan `top_rain_days.csv`.

| threshold_mm | wet_hours_ge_threshold | percentage |
| --- | --- | --- |
| 0.100 | 9378 | 26.745 |
| 1.000 | 1901 | 5.422 |
| 5.000 | 349 | 0.995 |
| 10.000 | 78 | 0.222 |

Terdapat **129 kandidat anomali suhu (0.368%)** dengan
|modified z| > 3,5 terhadap median/MAD bulan–jam; MAD nol terjadi pada 0 jam.
Dasar: grafik 15, `temperature_anomaly_candidates.csv`, dan tabel berikut.

| year | n_hours | candidates | candidate_pct |
| --- | --- | --- | --- |
| 2022 | 8760 | 36 | 0.411 |
| 2023 | 8760 | 25 | 0.285 |
| 2024 | 8784 | 27 | 0.307 |
| 2025 | 8760 | 41 | 0.468 |

Kandidat teratas (daftar hingga 20 tersedia di `top_temperature_anomalies.csv`):

| time | temperature_2m | month_hour_median_C | month_hour_mad_C | modified_z |
| --- | --- | --- | --- | --- |
| 2025-08-14 10:00:00+07:00 | 21.400 | 26.100 | 0.500 | -6.340 |
| 2022-12-27 09:00:00+07:00 | 20.600 | 25.200 | 0.500 | -6.205 |
| 2023-03-02 10:00:00+07:00 | 21.100 | 26.150 | 0.550 | -6.193 |
| 2023-03-02 11:00:00+07:00 | 21.200 | 27.150 | 0.650 | -6.174 |
| 2022-05-26 12:00:00+07:00 | 22.600 | 27.900 | 0.600 | -5.958 |
| 2025-04-12 11:00:00+07:00 | 23.400 | 27.350 | 0.450 | -5.921 |
| 2025-08-14 11:00:00+07:00 | 23.400 | 27.600 | 0.500 | -5.666 |
| 2022-12-27 08:00:00+07:00 | 20.400 | 23.750 | 0.400 | -5.649 |
| 2022-04-21 11:00:00+07:00 | 23.600 | 27.350 | 0.450 | -5.621 |
| 2022-12-27 10:00:00+07:00 | 21.000 | 26.400 | 0.650 | -5.603 |

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
