# Visualisasi dan analisis pola cuaca Bandung menggunakan CNN

Penelitian mengeksplorasi pola pada window cuaca melalui convolutional autoencoder, visualisasi
embedding, pengelompokan, serta pemeriksaan karakteristik meteorologis setiap kelompok.
Tujuan aktif adalah **representasi dan pencarian pola**, tanpa forecasting.

Data lokal: Open-Meteo, 2022–2025, hourly, 35.064 observasi. Lokasi berada di sekitar titik Bandung;
`Asia/Jakarta` menunjukkan zona waktu WIB (UTC+7). Koordinat respons adalah titik grid sekitar
1,698 km dari titik permintaan. CSV mentah tetap utuh dan tidak diunduh ulang.

## Deliverable

- **[Laporan IEEE PDF](reports/13524022_13524063_Visualisasi_CNN_IEEE_Format.pdf)**
- [Sumber Markdown laporan](reports/13524022_13524063_Visualisasi_CNN_IEEE_Format.md)
- [Laporan pola dan validasi CNN](reports/08_pattern_analysis_validation.md)
- [Dataset bersih: 35.064 × 23 kolom](data/processed/bandung_weather_hourly_clean.csv)
- [15 visualisasi EDA](output/figures/figure_manifest.json)
- [18 visualisasi CNN/pola, semuanya PNG 300 dpi](output/cnn/visual_interpretation_manifest.json)
- [Hasil eksperimen per seed](output/cnn/results.json), [hasil pola](output/cnn/pattern_results.json),
  dan [pemeriksaan CNN/laporan](output/cnn/verification_results.json)

`reports/07_cnn_paper.md` dan `.pdf` tetap tersedia sebagai salinan laporan utama dengan nama lama.
Laporan 01–06 merupakan catatan tahap persiapan; skenario forecasting dalam laporan kesiapan awal
tidak menjadi tujuan eksperimen akhir. Versi PDF sebelum revisi disimpan di `reports/archive/`;
eksperimen tiga epoch beserta gambar lamanya berada di `output/cnn/legacy_smoke_test/`.

## Temuan yang didukung hasil saat ini

- Model 1D terpilih menurut validation mempunyai MSE rekonstruksi test **0,05781**;
  profil jam train pada window test yang sama menghasilkan **0,67258**. Ini rekonstruksi, bukan forecasting.
- Dua kelompok test memiliki rentang suhu rata-rata **8,95 vs 11,57 °C**, kelembapan **79,95 vs 73,70%**,
  dan proporsi jam basah **31,33 vs 16,87%**. Kelompok merupakan ringkasan algoritmis, bukan kelas cuaca alami.
- **163/182 window (89,56%)** mendapat label sama pada tiga encoder setelah penyelarasan berdasarkan train.
  ARI antar-encoder rata-rata **0,624**; batas kelompok masih sensitif terhadap seed.
- Pada embedding tetap, pergantian seed K-Means menghasilkan ARI **1,000**. Ketidakstabilan yang teramati
  terutama berkaitan dengan encoder pada protokol ini. CNN belum terbukti unggul universal atas PCA/statistik.

Semua angka, contoh representatif, dan batas interpretasinya dijelaskan pada laporan utama.
Periode test 2025 sudah dilihat dalam EDA; hasil tetap eksploratif dan bukan validasi eksternal.

## Lingkungan dan reproduksi

Lingkungan yang dipakai: Windows, Python 3.13.9, PyTorch 2.10.0+cpu, scikit-learn 1.9.1.
Versi paket dipatok dalam `requirements.txt`. Pembuat PDF menggunakan font Times New Roman
di `C:/Windows/Fonts` dan menghasilkan A4 dua kolom dengan font tertanam.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/run_all.py --with-cnn
```

Perintah terakhir menjalankan tahap 1–5, pemeriksaan EDA, pelatihan CNN, analisis pola,
pembangunan laporan, serta pemeriksaan checkpoint/artefak/PDF secara berurutan.
Kegagalan menghentikan pipeline. Jika environment sudah siap, cukup jalankan perintah terakhir.
Pipeline analisis tidak memerlukan jaringan; instalasi dependency dilakukan terpisah.

Untuk menjalankan bagian eksperimen dan laporan saja setelah tahap 1–5 valid:

```powershell
.\.venv\Scripts\python.exe src/stage6_cnn_experiments.py --epochs 150 --patience 20 --seeds 42 43 44
.\.venv\Scripts\python.exe src/stage7_visual_interpretation.py
.\.venv\Scripts\python.exe src/build_ieee_report.py
.\.venv\Scripts\python.exe src/verify_cnn_results.py
```

Untuk pemeriksaan tanpa melatih ulang, jalankan `verify_cnn_results.py`. Untuk membangun ulang
PDF dari artefak yang sudah valid, jalankan `build_ieee_report.py` lalu pemeriksaannya.
PDF dan Markdown dihasilkan dari isi yang sama; perubahan naskah yang akan digenerasikan ulang
dilakukan pada `build_content()` di `src/build_ieee_report.py`.
Opsi `--reuse-checkpoints` pada tahap 6 hanya memulihkan evaluasi setelah interupsi;
checkpoint diperiksa berdasarkan versi arsitektur, seed, dan hash raw.
Perubahan kode/data/artefak prasyarat harus diikuti rerun bagian yang terpengaruh.

## Protokol penelitian

1. **Audit dan EDA:** metadata dipisahkan dari observasi; frekuensi, periode, satuan, missing,
   duplikasi, kualitas fisik, statistik, distribusi, korelasi, dan pola waktu diperiksa.
2. **Preprocessing:** nilai cuaca dipertahankan dan fitur waktu ditambahkan. Dataset mentah di
   `data/raw/`, dataset bersih di `data/processed/`; tidak ada imputasi yang diperlukan.
3. **Pembelajaran representasi:** train 2022–2023, validation 2024, test 2025. Scaler difit pada train.
   Window nonoverlap 48 jam menghasilkan 365/183/182 sampel. Sisa 24 jam terakhir test tetap
   berada dalam dataset/EDA tetapi tidak membentuk window CNN lengkap.
4. **CNN:** 1D multivariat menjadi analisis utama, dengan bottleneck 8×12. CNN 2D pada citra
   kemiripan suhu menjadi eksperimen pendamping dengan target ruang berbeda. Tiga seed dilatih
   maksimum 150 epoch, patience 20; checkpoint dan seed utama dipilih menggunakan validation.
   Keluaran rekonstruksi kelembapan/awan dibatasi 0–100% serta hujan/angin tidak negatif,
   menggunakan batas domain yang ditransformasikan dengan statistik train. Input tidak di-clipping.
5. **Pola:** K-Means pada embedding penuh; k=2–6 dipilih pada validation dengan batas jumlah
   anggota minimal. PCA tampilan difit pada train. Pembanding mencakup encoder sebelum training,
   PCA input tanpa whitening, dan 28 statistik sederhana; parameter semuanya fit pada train.
6. **Validasi/interpretasi:** profil fisik, kurva median/IQR, contoh aktual terdekat centroid,
   frekuensi bulanan, ARI lintas encoder, dan ARI inisialisasi K-Means diperiksa. Label antar-seed
   diselaraskan pada train untuk menghitung kesepakatan tiap window. Semua kasus tetap dievaluasi.
7. **Pelaporan:** angka dan gambar disalin dari artefak numerik, dengan sumber Markdown,
   pembangun PDF, manifest hash, dan preview semua halaman. Tidak ada klaim sebab-akibat.

## Struktur hasil

| Lokasi | Isi |
| --- | --- |
| `src/stage1_validation.py`–`stage5_patterns.py` | Audit, EDA, preprocessing, visualisasi, pola data asli |
| `src/stage6_cnn_experiments.py` | Model, training tiga seed, checkpoint, tensor, metrik per variabel |
| `src/stage7_visual_interpretation.py` | Pengelompokan, pembanding, stabilitas, profil, dan grafik |
| `src/build_ieee_report.py` | Sumber laporan dan PDF IEEE dua kolom |
| `src/verify_results.py` | Pemeriksaan **tahap 1–5 saja** |
| `src/verify_cnn_results.py` | Pemeriksaan CNN, pola, PDF, dan render semua halaman |
| `output/cnn/window_metadata.csv` | Pemetaan tensor ke waktu/baris cuaca asli |
| `output/cnn/cluster_weather_profiles.csv` | Statistik kelompok train/validation/test |
| `output/cnn/cluster_stability.csv` | ARI encoder dan inisialisasi K-Means |
| `output/cnn/encoder_seed_window_agreement.csv` | Label seluruh seed dan konsensus per window |
| `output/cnn/*manifest.json` | Provenance, hash artefak, daftar gambar/caption |
| `output/report_preview/` | Render halaman PDF untuk pemeriksaan tata letak |

Definisi data merujuk [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api).
Rujukan akademik tentang autoencoder, silhouette, ARI, recurrence plot, dan PCA dicantumkan
dalam laporan. Ketepatan spasial, sumber model meteorologi, dependensi waktu, dan keterbatasan
generalisasi tetap dinyatakan.
