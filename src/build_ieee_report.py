"""Build the submission PDF and editable Markdown from verified experiment outputs.

The layout uses A4, two columns, embedded Times New Roman, numbered references,
and 300-dpi source figures. This is an IEEE-style course report, not an assertion
of acceptance or automated compliance with any particular IEEE conference.
"""
from __future__ import annotations
import html
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from common import ROOT, REPORTS, json_write, read_json, sha256
from stage7_visual_interpretation import OUT, check_manifest

STEM = "13524022_13524063_Visualisasi_CNN_IEEE_Format"
TITLE = "Visualisasi Pola Time-Series Cuaca Bandung Menggunakan Convolutional Autoencoder"
AUTHORS = "Ishak Palentino Napitupulu (13524022) dan Marcel Luther Sitorus (13524063)"
AFFILIATION = "Sistem Cerdas TF4018, Institut Teknologi Bandung, 2026"
REPO_URL = "https://github.com/Ishakpln/bandung-weather-timeseries-cnn"


def build_content():
    result = read_json(OUT/"results.json")
    pattern = read_json(OUT/"pattern_results.json")
    source = read_json(ROOT/"output/tables/source_metadata.json")
    eda = read_json(ROOT/"output/tables/pattern_evidence.json")
    stats = pd.read_csv(OUT/"cluster_weather_profiles.csv")
    feature = pd.read_csv(OUT/"cnn1d_per_feature_reconstruction.csv")
    stability = pd.read_csv(OUT/"cluster_stability.csv")
    associations = pd.read_csv(OUT/"latent_weather_associations.csv")
    months = pd.read_csv(OUT/"test_cluster_month_percentages.csv").set_index("month")
    profiles = pattern["test_cluster_profiles"]
    first, last = profiles[0], profiles[-1]
    mse1, mse2 = result["cnn1d"]["mse"]["test"], result["cnn2d"]["mse"]["test"]
    benchmarks = {r["method"]: r["test_mse"] for r in pattern["reconstruction_benchmarks"]}
    reduction = (1-mse1/benchmarks["train_hour_profile"])*100
    agreement = pattern["three_seed_agreement"]
    ari1 = pattern["encoder_seed_ARI"]["cnn1d"]["mean"]
    primary = pattern["primary"]
    short = {"temperature_2m": "Suhu", "relative_humidity_2m": "Kelembapan",
             "dew_point_2m": "Titik embun", "precipitation": "Presipitasi",
             "pressure_msl": "Tekanan MSL", "cloud_cover": "Tutupan awan", "wind_speed_10m": "Angin"}
    units = ["°C", "%", "°C", "mm", "hPa", "%", "km/jam"]
    blocks, figure_count, table_count = [], 0, 0
    def para(text):
        blocks.append({"type": "paragraph", "text": text})
    def heading(text, sub=False):
        blocks.append({"type": "subheading" if sub else "heading", "text": text})
    def table(title, columns, rows):
        nonlocal table_count
        table_count += 1
        blocks.append({"type": "table", "caption": f"TABEL {table_count}. {title}", "columns": columns, "rows": rows})
    def figure(name, caption):
        nonlocal figure_count
        figure_count += 1
        blocks.append({"type": "figure", "path": str(OUT/name), "caption": f"Gambar {figure_count}. {caption}"})

    heading("ABSTRAK")
    para(f"Penelitian ini memvisualisasikan dan mengeksplorasi pola temporal cuaca Bandung melalui representasi convolutional autoencoder (CAE). "
         f"Dataset mencakup 35.064 observasi hourly tahun 2022–2025. CAE 1D mempelajari tujuh variabel dalam window 48 jam; "
         f"CAE 2D menjadi eksperimen pendamping pada citra kemiripan suhu. Tiga seed dilatih dengan maksimum 150 epoch dan seleksi validation-only. "
         f"Embedding 1D dikelompokkan memakai K-Means dengan k yang dipilih pada validation. Model terpilih mencapai MSE rekonstruksi test "
         f"{mse1:.5f}, dibandingkan {benchmarks['train_hour_profile']:.5f} untuk profil jam train. Dua kelompok test memiliki rentang suhu rata-rata "
         f"{first['temperature_range']:.2f} dan {last['temperature_range']:.2f} °C serta frekuensi jam basah {first['wet_pct']:.2f}% dan {last['wet_pct']:.2f}%. "
         f"Kesepakatan seluruh seed mencakup {agreement['n_windows']}/{agreement['total_test_windows']} window ({agreement['percentage']:.2f}%), "
         f"dengan ARI antar-encoder rata-rata {ari1:.3f}. Hasil mendukung interpretasi gradien kondisi yang lebih lembap/berawan dan lebih besar "
         f"variasi suhunya, tetapi batas kelompok masih sensitif terhadap seed. Kontribusi utama adalah hubungan terukur antara representasi CNN, "
         f"profil cuaca asli, dan ketidakpastian kelompok. Penelitian bersifat eksploratif tanpa forecasting atau klaim kelas cuaca alami.")
    para("Kata kunci—cuaca Bandung; time-series; convolutional autoencoder; visualisasi; pengelompokan; stabilitas representasi.")
    heading("I. PENDAHULUAN")
    para("Grafik time-series memperlihatkan perubahan nilai cuaca dan siklus harian, tetapi pemeriksaan banyak potongan waktu multivariat "
         "secara bersamaan sulit dilakukan hanya dengan kurva panjang. Representasi yang merangkum bentuk temporal dapat membantu menelusuri "
         "kemiripan antarperiode. Tantangan akademiknya adalah menghubungkan representasi tersebut kembali ke karakteristik cuaca yang terukur, "
         "serta membedakan pola yang didukung data dari bentuk visual yang berasal dari transformasi atau inisialisasi model.")
    para("Penelitian menjawab tiga pertanyaan: (1) seberapa baik CAE mempertahankan variasi temporal pada rekonstruksi; "
         "(2) kelompok window seperti apa yang dihasilkan representasi CNN dan bagaimana profil cuaca aslinya; serta "
         "(3) seberapa konsisten kelompok tersebut pada beberapa seed dan dibanding representasi sederhana. Tujuannya adalah visualisasi "
         "dan eksplorasi pola, sehingga tidak dilakukan peramalan suhu masa depan. Rekonstruksi masukan pada waktu yang sama digunakan "
         "sebagai diagnostik kualitas representasi, bukan sebagai skor forecasting.")
    para("Kontribusi penelitian meliputi pipeline kronologis yang dapat direproduksi, visualisasi contoh nyata per kelompok, "
         "profil kuantitatif variabel cuaca, pembanding representasi, serta pemisahan ketidakstabilan encoder dari ketidakstabilan K-Means. "
         "Kelompok tidak diasumsikan sebagai musim atau kategori meteorologi sebelum karakteristik datanya diperiksa.")
    heading("II. LANDASAN METODE")
    para("Autoencoder mempelajari kode berdimensi lebih rendah dengan meminimalkan galat rekonstruksi input [2]. "
         "Konvolusi 1D mengolah lingkungan lokal sepanjang sumbu waktu, sedangkan embedding encoder dapat dipakai untuk pencarian "
         "kemiripan atau pengelompokan. Galat rendah tidak dengan sendirinya membuktikan bahwa embedding menghasilkan kelas meteorologi "
         "yang terpisah. Contoh penerapan convolutional autoencoder untuk deret waktu juga tersedia pada dokumentasi Keras [6].")
    para("PCA dipakai untuk proyeksi linear dua dimensi dan pelaporan explained variance [7]. Pengelompokan tetap dilakukan pada embedding penuh. "
         "Silhouette menilai kekompakan dan pemisahan berdasarkan geometri ruang yang dievaluasi [3]. ARI membandingkan dua partisi "
         "dengan koreksi terhadap kesepakatan kebetulan [4]; pada penelitian ini partisinya berasal dari model berbeda, bukan label ground truth.")
    para("Eksperimen pendamping menggunakan kemiripan pasangan waktu yang berkaitan dengan gagasan recurrence plot [5]. "
         "Matriks yang digunakan bersifat kontinu, S(i,j)=exp(−|u_i−u_j|/0,25), dengan u suhu hasil min–max pada satu window. "
         "Diagonal bernilai satu dan simetri berasal dari rumus tersebut. Matriks ini tidak diperlakukan sebagai recurrence plot biner "
         "atau sebagai pola yang ditemukan oleh CNN.")
    heading("III. DATA DAN METODOLOGI")
    heading("A. Sumber, lokasi, dan kualitas data", True)
    para(f"CSV mengikuti format Open-Meteo Historical Weather API [1], dengan periode 1 Januari 2022 pukul 00.00 sampai "
         f"31 Desember 2025 pukul 23.00 WIB. Metadata mencatat grid ({source['metadata']['latitude']}, {source['metadata']['longitude']}), "
         f"berjarak {source['distance_km']:.3f} km dari titik permintaan (−6,9175; 107,6191). Asia/Jakarta adalah zona waktu WIB (UTC+7), "
         "sedangkan lokasi penelitian berada di sekitar titik Bandung tersebut. Data merepresentasikan grid dan tidak diasumsikan sebagai "
         "pengukuran stasiun tepat di koordinat permintaan. Model meteorologi spesifik, URL permintaan, dan tanggal unduh tidak tercantum dalam CSV.")
    para("Tabel hourly memiliki 35.064 baris dan sembilan kolom: waktu serta delapan variabel cuaca. Tidak ditemukan missing, duplikasi timestamp, "
         "atau jam hilang pada audit. Tahun kabisat 2024 mencakup 8.784 jam. Nilai cuaca mentah dipertahankan, tanpa imputasi, clipping, "
         "penghapusan outlier, atau observasi sintetis. Dataset bersih berisi 23 kolom setelah penambahan 14 fitur kalender/siklik. "
         "Input CNN dibatasi pada suhu, kelembapan, titik embun, presipitasi, tekanan muka laut, tutupan awan, dan kecepatan angin; "
         "kode cuaca bersifat kategori dan tidak dimasukkan sebagai besaran kontinu. Kalender digunakan sebagai anotasi analisis.")
    para("Presipitasi adalah jumlah pada jam sebelumnya menurut definisi sumber [1]. Pengelompokan tanggal/bulan mengikuti label timestamp "
         "asli; pembacaan batas kalender perlu memperhatikan interval tersebut. Tekanan yang digunakan adalah pressure_msl, bukan tekanan "
         "permukaan pada elevasi Bandung. Validasi internal tidak membuktikan akurasi terhadap stasiun pengamatan.")
    heading("B. Split dan window", True)
    table("Pembagian waktu dan cakupan window 48 jam", ["Split", "Jam", "Window", "Sisa jam"],
          [["Train 2022–23", "17.520", "365", "0"], ["Val 2024", "8.784", "183", "0"], ["Test 2025", "8.760", "182", "24"]])
    para("StandardScaler difit pada jam train 2022–2023 saja. Validation dan test ditransformasikan memakai parameter train. "
         "Window panjang 48 jam dan stride 48 jam dibentuk mandiri pada setiap split, dimulai pukul 00.00 WIB. Tidak ada overlap "
         "antarwindow maupun lintas split. Sebanyak 24 jam terakhir tahun 2025 tidak membentuk window lengkap dan tidak masuk evaluasi CNN; "
         "jam tersebut tetap ada dalam dataset bersih dan EDA. Bulan window ditentukan dari waktu awal; sebagian window melintasi batas bulan.")
    heading("C. Autoencoder dan protokol pelatihan", True)
    para("Encoder 1D: Conv(7→32,k=5,s=1,p=2), Conv(32→16,k=4,s=2,p=1), dan Conv(16→8,k=4,s=2,p=1), "
         "masing-masing diikuti LeakyReLU(0,1). Input berukuran 7×48 dikompresi menjadi 8×12 atau 96 nilai. Decoder menggunakan "
         "ConvTranspose(8→16,k=4,s=2,p=1), ConvTranspose(16→32,k=4,s=2,p=1), dan Conv(32→7,k=5,p=2). "
         "Keluaran mengembalikan ukuran input, dengan kendala domain: kelembapan/awan 0–100%, serta hujan/angin tidak negatif. "
         "Batas fisik ditransformasikan memakai statistik train dan diterapkan selama training serta inference. "
         "Kendala ini tidak mengubah atau memotong observasi input. Posisi temporal dipertahankan tanpa adaptive pooling kasar pada arsitektur lama.")
    para("Encoder 2D memakai kanal 1→8→16→8→8, dengan tiga pengurangan resolusi stride 2, menghasilkan latent 8×4×4. "
         "Decoder ConvTranspose mengembalikan resolusi 32×32 dan diakhiri Sigmoid agar sesuai target kemiripan 0–1. "
         "Input 2D menggunakan 32 sampel aktual dari suhu window 48 jam, tanpa interpolasi observasi. Normalisasi per-window "
         "menghilangkan level suhu absolut. Karena jumlah variabel dan ruang target berbeda, skor 1D dan 2D tidak dirangking langsung.")
    para("Kedua arsitektur dilatih terpisah untuk seed 42, 43, dan 44, menggunakan Adam, learning rate 0,001, weight decay 10⁻⁵, "
         "batch 32, clipping norma gradien 5, maksimum 150 epoch, dan early stopping patience 20 dengan min_delta 10⁻⁵. "
         "Checkpoint tiap seed dan seed utama dipilih berdasarkan MSE validation. MSE validation ditimbang menurut jumlah elemen, "
         "sehingga batch terakhir tidak mendapat bobot berlebihan. CPU dua thread dan algoritme deterministik diaktifkan. "
         "Perubahan arsitektur dan durasi pelatihan dilakukan bersama; peningkatan dari eksperimen lama tidak diatribusikan pada satu faktor saja.")
    run_rows = []
    for tag in ("cnn1d", "cnn2d"):
        for run in result[tag]["runs"]:
            run_rows.append(["1D" if tag == "cnn1d" else "2D", str(run["seed"]), str(run["best_epoch"]), f"{run['metrics']['test']['mse']:.5f}"])
    table("Checkpoint terbaik menurut validation", ["Model", "Seed", "Epoch", "MSE test"], run_rows)
    figure("cnn1d_loss.png", "Loss rekonstruksi tiga seed CNN 1D. Pemilihan checkpoint memakai validation; penurunan pada akhir batas epoch tidak menjamin konvergensi global.")
    heading("D. Pencarian pola dan pembanding", True)
    para("Embedding 1D distandarkan berdasarkan embedding train lalu dikelompokkan dengan K-Means (n_init=20, seed 42). "
         "Kandidat k=2–6 dipilih memakai silhouette validation, dengan syarat tiap kelompok memiliki minimal 10 window train dan "
         "tiga window validation. Test tidak digunakan memilih k. Penomoran C1, C2, dan seterusnya diurutkan berdasarkan rerata "
         "rentang suhu kelompok train. Penamaan ini bersifat deskriptif, bukan label musim atau ground truth.")
    para("Pembanding representasi meliputi encoder 1D sebelum pelatihan, PCA input terstandardisasi dengan 96 komponen tanpa whitening, "
         "serta 28 statistik window: mean, simpangan baku, minimum, dan maksimum untuk tujuh variabel. Seluruh transformasi dan centroid "
         "difit pada train; k setiap representasi dipilih pada validation dengan aturan yang sama. Silhouette pada ruang representasi "
         "berbeda tidak merupakan ukuran universal keunggulan. Sebagai pemeriksaan tambahan, label tiap metode juga dinilai pada ruang "
         "input terstandardisasi yang sama. Tidak dilakukan uji signifikansi IID terhadap window yang masih bergantung secara temporal.")
    heading("E. Validasi pola dan interpretasi", True)
    para("Kestabilan encoder dinilai dengan ARI pada label test dari tiga encoder yang dilatih independen, memakai k tetap hasil "
         "pemilihan validation arsitektur tersebut. Centroid setiap encoder tetap difit pada train. Stabilitas inisialisasi K-Means "
         "diukur terpisah pada satu embedding tetap dengan lima seed. Untuk menghitung kesepakatan per window, label antar-encoder "
         "diselaraskan ke model utama menggunakan Hungarian matching pada label train; penyelarasan tidak memakai label test.")
    para("Profil kelompok dihitung dari cuaca asli dalam satuan fisik. Contoh representatif adalah window test aktual terdekat dengan "
         "centroid train pada setiap kelompok. Median dan IQR kurva digunakan sebagai ringkasan observasi, bukan data sintetis atau "
         "confidence interval. Window tanpa variasi pada suatu variabel dikecualikan hanya dari korelasi temporal variabel tersebut, "
         "karena korelasinya tidak terdefinisi; observasinya tetap disertakan pada MSE/MAE.")
    heading("IV. HASIL DAN PEMBAHASAN")
    heading("A. Pola cuaca pada data asli", True)
    para(f"EDA menghasilkan rata-rata suhu {eda['temperature_mean_C']:.3f} °C dan kelembapan {eda['humidity_mean_pct']:.3f}%. "
         "Profil rata-rata suhu mencapai maksimum sekitar pukul 12.00 WIB (27,475 °C) dan minimum sekitar 06.00 WIB (19,261 °C). "
         "Sebanyak 73,25% jam memiliki presipitasi nol. Korelasi Pearson suhu–kelembapan adalah −0,858. "
         "Temuan ini menunjukkan pola harian dan hubungan simultan yang sudah terlihat sebelum CNN. Pola tersebut tidak diklaim "
         "sebagai penemuan model; asosiasi tidak membuktikan sebab-akibat.")
    heading("B. Informasi temporal yang dipertahankan", True)
    para(f"Seed terpilih berdasarkan validation adalah {result['cnn1d']['selected_seed']} untuk 1D dan {result['cnn2d']['selected_seed']} untuk 2D. "
         f"MSE test 1D sebesar {mse1:.5f}, sedangkan rerata dan SD antar-seed adalah "
         f"{result['cnn1d']['test_mse_mean_across_seeds']:.5f} ± {result['cnn1d']['test_mse_std_across_seeds']:.5f}. "
         f"Dibanding profil jam train pada window test yang sama, penurunan MSE rekonstruksi adalah {reduction:.2f}%. "
         "Pembanding tidak mendapat input window aktual, sehingga hasil ini menilai retensi informasi oleh autoencoder, "
         "bukan bukti keunggulan forecasting atau representasi dibanding semua metode kompresi.")
    table("Pembanding rekonstruksi dalam ruang target masing-masing", ["Ruang", "Metode", "MSE test"],
          [["1D", "Rerata train", f"{benchmarks['train_mean']:.5f}"],
           ["1D", "Profil jam train", f"{benchmarks['train_hour_profile']:.5f}"],
           ["1D", "CNN 1D", f"{mse1:.5f}"], ["2D", "Citra rerata train", f"{benchmarks['train_mean_image']:.5f}"],
           ["2D", "CNN 2D", f"{mse2:.5f}"]])
    table("Galat rekonstruksi fisik dan korelasi temporal 1D", ["Variabel", "Unit", "RMSE", "r window"],
          [[short[row.feature], unit, f"{row.rmse_original_units:.3f}", f"{row.mean_window_temporal_correlation:.3f}"]
           for row, unit in zip(feature.itertuples(), units)])
    correlation_counts = "; ".join(f"{short[row.feature]} {int(row.nonconstant_windows_for_correlation)}"
                                   for row in feature.itertuples())
    domain_violations = sum(row["violations_tolerance_1e_4"] for row in pattern["reconstruction_domain_checks"])
    para("RMSE pada tabel adalah galat rekonstruksi pada waktu yang sama, bukan ketidakpastian prediksi masa depan. "
         "Korelasi temporal merupakan rerata korelasi per window yang tidak konstan. Presipitasi dan tutupan awan mempunyai "
         "nilai berulang yang dapat membuat korelasi tidak terdefinisi. Jumlah window yang masuk korelasi masing-masing: "
         f"{correlation_counts}. Korelasi ini bersyarat pada input dan output yang tidak konstan; "
         "MSE/MAE tetap memakai seluruh 182 window. Hujan nol dominan dan puncak singkat perlu diperhatikan saat membaca rata-rata galat. "
         f"Pemeriksaan batas keluaran untuk kelembapan, awan, hujan, dan angin menemukan {domain_violations} pelanggaran "
         "pada toleransi numerik 10⁻⁴ dalam satuan fisik. Ini pemeriksaan domain per variabel, bukan jaminan konsistensi semua hubungan fisik antarvariabel.")
    figure("representative_windows.png", "Suhu asli dan rekonstruksi pada contoh aktual terdekat centroid tiap kelompok. Variasi harian kini terlihat terpulihkan; contoh tidak menggantikan evaluasi seluruh window.")
    heading("C. Kelompok pola dan ciri cuacanya", True)
    para(f"Pemilihan validation menghasilkan k={primary['k']}, dengan silhouette validation {primary['validation_silhouette']:.3f} "
         f"dan test {primary['test_silhouette']:.3f} pada ruang embedding terstandardisasi. Kelompok test tidak seimbang: "
         f"C1 berisi {first['n_windows']} dan C2 berisi {last['n_windows']} window. Proyeksi PCA memakai parameter train; "
         f"dua komponen pertama mencakup {pattern['pca']['cnn1d']['two_components_variance_pct']:.2f}% variasi embedding train. "
         "Kelompok didefinisikan di ruang penuh 96 dimensi, sehingga jarak pada gambar dua dimensi bukan keseluruhan jarak representasi.")
    figure("pattern_embedding.png", "Proyeksi test dengan PCA train dan warna kelompok dari embedding penuh. Titik berdekatan belum otomatis berarti kondisi cuaca identik.")
    table("Profil kelompok pada seluruh window test", ["C", "n", "Suhu °C", "Rentang °C", "RH %", "Basah %"],
          [[f"C{r['cluster']}", str(r['n_windows']), f"{r['temperature_mean']:.2f}", f"{r['temperature_range']:.2f}",
            f"{r['humidity_mean']:.2f}", f"{r['wet_pct']:.2f}"] for r in profiles])
    para(f"C1 memiliki kelembapan {first['humidity_mean']:.2f}%, tutupan awan {first['cloud_mean']:.2f}%, "
         f"dan rata-rata jumlah hujan per window 48 jam {first['rain_sum']:.2f} mm. C2 memiliki kelembapan {last['humidity_mean']:.2f}%, "
         f"awan {last['cloud_mean']:.2f}%, dan hujan {last['rain_sum']:.2f} mm. Rentang suhu C2 lebih besar "
         f"{last['temperature_range']-first['temperature_range']:.2f} °C, sedangkan frekuensi jam basahnya lebih rendah "
         f"{first['wet_pct']-last['wet_pct']:.2f} poin persentase. Profil ini mendukung istilah deskriptif "
         "'lebih lembap/berawan dengan rentang suhu lebih kecil' untuk C1 dan 'rentang suhu lebih besar dengan lebih sedikit jam basah' "
         "untuk C2. Istilah tersebut tidak sama dengan penetapan musim hujan/kemarau atau penjelasan mekanisme penyebab.")
    figure("cluster_profiles.png", "Median dan IQR suhu/kelembapan menurut posisi dalam window. Perbedaan kelompok dibaca bersama tabel fisik dan cakupan observasi.")
    para("Arah perbedaan kelompok juga muncul pada data train dan validation: kelompok berentang suhu lebih besar memiliki kelembapan "
         "dan frekuensi jam basah lebih rendah. Namun, komposisi dan intensitas hujan berubah antarperiode. "
         "Konsistensi arah deskriptif ini tidak membuktikan invariansi distribusi atau dua rezim cuaca yang benar-benar diskret.")
    assoc1 = associations.loc[(associations.representation == "cnn1d")&(associations.component == 1)].set_index("descriptor")
    para(f"Pada test, PC1 memiliki korelasi {assoc1.loc['temperature_range','test_pearson_r']:.3f} dengan rentang suhu "
         f"dan {assoc1.loc['humidity_mean','test_pearson_r']:.3f} dengan kelembapan rata-rata. "
         "Ini menghubungkan variasi latent dengan besaran cuaca yang dapat dibaca, tanpa menganggap PC1 sebagai sensor atau variabel fisik. "
         "Tanda sumbu PCA dapat terbalik tanpa mengubah struktur representasi; korelasi ini eksploratif dan tidak disertai p-value IID.")
    heading("D. Distribusi waktu dan pola yang berulang", True)
    para(f"Pada test 2025, kelompok C2 mencakup {months.loc[10,str(last['cluster'])]:.2f}% window berawal Oktober, "
         f"{months.loc[9,str(last['cluster'])]:.2f}% berawal September, dan {months.loc[7,str(last['cluster'])]:.2f}% berawal Juli. "
         "Distribusi ini merupakan pola frekuensi pada satu tahun evaluasi, bukan normal klimatologis atau bukti bahwa bulan menentukan kelompok. "
         "Jumlah window per bulan kecil dan window dapat melintasi batas bulan; persentase dibaca bersama jumlah observasinya.")
    figure("cluster_month_frequency.png", "Frekuensi kelompok per bulan awal window test, dinormalisasi dalam setiap bulan. Nol berarti tidak ada window yang ditetapkan ke kelompok itu pada sampel tersebut.")
    heading("E. Konsistensi pola dan sumber ketidakpastian", True)
    seed_values = stability.loc[(stability.representation == "cnn1d")&(stability.stability_type == "encoder_seed"), "test_ARI"]
    kmeans_values = stability.loc[stability.stability_type == "kmeans_initialization", "test_ARI"]
    para(f"ARI antar-encoder 1D rata-rata {seed_values.mean():.3f}, dengan rentang {seed_values.min():.3f}–{seed_values.max():.3f}. "
         f"Pada embedding utama tetap, ARI antar-inisialisasi K-Means rata-rata {kmeans_values.mean():.3f}. "
         "Artinya, pengulangan K-Means pada representasi yang sama jauh lebih konsisten daripada hasil dari encoder yang dilatih berbeda. "
         "Pada protokol ini ketidakpastian batas kelompok terutama berkaitan dengan variasi representasi encoder, "
         "bukan semata-mata seed K-Means. Rekonstruksi yang baik ternyata tidak menjamin partisi latent yang sama.")
    para(f"Setelah penyelarasan label berdasarkan train, {agreement['n_windows']} dari {agreement['total_test_windows']} window "
         f"({agreement['percentage']:.2f}%) mendapat label yang sama pada ketiga seed. Sebanyak "
         f"{agreement['total_test_windows']-agreement['n_windows']} window lain berubah label. "
         "Seluruh window tetap masuk hasil utama; analisis inti konsensus hanya diagnostik tambahan dan tidak dipakai untuk membuang "
         "kasus sulit atau menaikkan skor utama. Persentase ini adalah kesepakatan label, bukan akurasi; dominasi C1 membuatnya "
         "perlu dibaca bersama ARI yang mengoreksi kesepakatan kebetulan. Dua kelompok merupakan ringkasan gradien kondisi dengan batas yang belum stabil penuh.")
    table("Profil inti konsensus tiga seed (subset diagnostik)", ["C", "n", "Rentang °C", "RH %", "Awan %"],
          [[f"C{r['selected_cluster']}", str(r['n_windows']), f"{r['temperature_range']:.2f}", f"{r['humidity_mean']:.2f}", f"{r['cloud_mean']:.2f}"]
           for r in agreement["core_profiles"]])
    core_by_cluster = {r["selected_cluster"]: r["n_windows"] for r in agreement["core_profiles"]}
    para("Proporsi anggota model utama yang tetap berlabel sama pada seluruh seed adalah " +
         "; ".join(f"C{r['cluster']}: {core_by_cluster.get(r['cluster'], 0)}/{r['n_windows']} "
                   f"({core_by_cluster.get(r['cluster'], 0)/r['n_windows']*100:.2f}%)" for r in profiles) +
         ". Perbedaan ini menunjukkan bahwa ketidakpastian batas kelompok tidak merata. Ini diagnostik representasi, "
         "bukan bukti bahwa suatu kondisi cuaca secara fisik lebih tidak pasti.")
    figure("cnn_seed_stability.png", "ARI dari tiga pasangan encoder pada tiap arsitektur. Kesepakatan label tidak merupakan akurasi terhadap kategori meteorologi.")
    heading("F. Kontribusi CNN dibanding representasi sederhana", True)
    names = {"cnn1d": "CNN 1D", "untrained_cnn1d": "CNN acak", "pca_input": "PCA input", "summary_statistics": "Statistik 28"}
    table("Pengelompokan 1D dan pembanding", ["Representasi", "k", "Sil. val*", "Sil. input test"],
          [[names[r['representation']], str(r['k']), f"{r['validation_silhouette']:.3f}", f"{r['test_silhouette_common_input']:.3f}"]
           for r in pattern["representations"] if r["representation"] in names])
    para("*Silhouette validation dihitung pada ruang masing-masing dan digunakan memilih k, bukan merangking representasi lintas ruang. "
         "Kolom terakhir menggunakan ruang input terstandardisasi yang sama. Selisih CNN 1D terhadap PCA pada ruang bersama kecil "
         "dan belum diuji signifikansinya. Representasi sederhana juga menangkap struktur dominan; karena itu tidak diklaim CNN "
         "menemukan pola unik yang tidak dapat ditemukan metode lain. Manfaat yang didukung hasil adalah rekonstruksi temporal, "
         "visualisasi representasi, dan penelusuran kelompok kembali ke cuaca asli dengan ketidakpastian yang dinyatakan.")
    heading("G. Aktivasi dan eksperimen 2D", True)
    para("Visualisasi aktivasi disejajarkan dengan kurva suhu pada satu contoh representatif yang dipilih dari centroid, "
         "sehingga posisi respons dapat ditelusuri kembali ke waktu aslinya. Namun, setiap filter menerima gabungan tujuh "
         "variabel. Besarnya aktivasi tidak ditafsirkan sebagai pengaruh kausal, feature importance, atau bukti bahwa satu "
         "filter memiliki fungsi meteorologis tertentu. Evaluasi keseluruhan window tetap menjadi dasar kesimpulan.")
    figure("cnn1d_temporal_response.png", "Respons delapan kanal awal CNN 1D dan suhu pada contoh representatif C1. Encoder menerima tujuh variabel sekaligus; kesesuaian posisi tidak membuktikan fungsi meteorologis satu filter.")
    para(f"CAE 2D mencapai MSE test {mse2:.5f}, dibanding citra rerata train {benchmarks['train_mean_image']:.5f}. "
         f"ARI antar-encoder 2D rata-rata {pattern['encoder_seed_ARI']['cnn2d']['mean']:.3f}. Nilai ini menjelaskan kestabilan partisi "
         "untuk citra suhu ternormalisasi dan tidak dapat digunakan menyatakan 2D lebih baik daripada 1D multivariat. "
         "Bentuk diagonal serta simetri sudah ada sebelum model belajar. Perbedaan input, hilangnya level suhu, dan kompresi citra "
         "membatasi makna meteorologis perbandingan antararsitektur.")
    figure("cnn2d_reconstruction_error.png", "Eksperimen pendamping 2D: input, rekonstruksi, dan galat absolut dengan rentang warna 0–1. Ini hubungan pasangan waktu, bukan peta geografis.")
    heading("V. KETERBATASAN DAN REPRODUKSIBILITAS")
    para("Tahun 2025 telah dibaca dalam EDA sebelumnya, sehingga evaluasi tetap bersifat eksploratif dan bukan holdout yang sepenuhnya "
         "belum dilihat. Pemilihan checkpoint, seed utama, dan k pada eksperimen ini memakai validation; keputusan protokol tetap perlu "
         "dikonfirmasi pada periode eksternal untuk klaim generalisasi. Data hanya berasal dari satu grid dan empat tahun, sementara "
         "model meteorologi sumber belum teridentifikasi. Tidak ada validasi stasiun, label kelas cuaca, atau ground truth anomali.")
    para("Window 48 jam memiliki fase awal tetap dan belum dibandingkan secara sistematis dengan panjang/offset lain. "
         "Tiga seed hanya memberi pemeriksaan awal ketidakpastian, bukan estimasi lengkap distribusi model. Kandidat k dibatasi 2–6; "
         "hasil k=2 tidak membuktikan jumlah kelas alami adalah dua. Banyak run mencapai batas 150 epoch; hasil tidak diklaim optimum global. "
         "Perbaikan rekonstruksi juga tidak otomatis menyelesaikan ketidakstabilan embedding.")
    para("Seluruh angka dan grafik dibangun dari CSV/tensor tersimpan, dengan metadata window, scaler train, checkpoint per seed, "
         "label tiap window, kandidat k validation, profil kelompok, dan manifest SHA-256. Kode sumber, lingkungan paket, serta "
         "script pembangun PDF disertakan dalam repository. Gambar utama berasal dari PNG 300 dpi. "
         "Laporan 01–06 merupakan dokumentasi historis tahap persiapan; laporan ini menetapkan tujuan aktif representasi dan pencarian pola.")
    heading("VI. KESIMPULAN")
    para(f"CAE 1D dapat mempertahankan variasi temporal cuaca pada rekonstruksi, dengan MSE test {mse1:.5f} dan profil "
         "kelompok yang dapat dijelaskan melalui rentang suhu, kelembapan, awan, dan frekuensi jam basah. Dua kelompok terpilih "
         "merangkum perbedaan kondisi yang lebih lembap/berawan dan kondisi dengan variasi suhu lebih besar. "
         f"Sebanyak {agreement['percentage']:.2f}% window memperoleh label sama pada tiga seed, tetapi ARI rata-rata {ari1:.3f} "
         "menunjukkan batas kelompok masih sensitif terhadap encoder. Pola deskriptif memiliki dukungan angka, "
         "sementara klaim dua kelas cuaca alami, keunggulan universal CNN, atau konsistensi sempurna tidak didukung. "
         "Penelitian menghasilkan visualisasi pola beserta ketidakpastiannya; konfirmasi lintas periode, seed, dan panjang window "
         "menjadi langkah berikutnya yang terpisah dari hasil sekarang.")
    heading("DAFTAR PUSTAKA")
    references = [
        "[1] Open-Meteo, “Historical Weather API.” https://open-meteo.com/en/docs/historical-weather-api, diakses 5 November 2026.",
        "[2] G. E. Hinton dan R. R. Salakhutdinov, “Reducing the Dimensionality of Data with Neural Networks,” Science, vol. 313, no. 5786, hlm. 504–507, 2006. doi:10.1126/science.1127647.",
        "[3] P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,” Journal of Computational and Applied Mathematics, vol. 20, hlm. 53–65, 1987. doi:10.1016/0377-0427(87)90125-7.",
        "[4] L. Hubert dan P. Arabie, “Comparing partitions,” Journal of Classification, vol. 2, hlm. 193–218, 1985. doi:10.1007/BF01908075.",
        "[5] J.-P. Eckmann, S. O. Kamphorst, dan D. Ruelle, “Recurrence Plots of Dynamical Systems,” Europhysics Letters, vol. 4, no. 9, hlm. 973–977, 1987. doi:10.1209/0295-5075/4/9/004.",
        "[6] P. Vijay, “Timeseries anomaly detection using an Autoencoder,” Keras Code Examples, 2020. https://keras.io/examples/timeseries/timeseries_anomaly_detection/, diakses 5 November 2026.",
        "[7] scikit-learn developers, “Principal component analysis” dan “Clustering performance evaluation.” https://scikit-learn.org/stable/modules/decomposition.html dan https://scikit-learn.org/stable/modules/clustering.html, diakses 5 November 2026."]
    for reference in references:
        blocks.append({"type": "reference", "text": reference})
    heading("LAMPIRAN: PETA ARTEFAK")
    para("Sumber utama: src/stage6_cnn_experiments.py; src/stage7_visual_interpretation.py; src/build_ieee_report.py; "
         "src/verify_cnn_results.py. Data bersih: data/processed/bandung_weather_hourly_clean.csv. "
         "Perintah lengkap dan versi dependency tersedia dalam README.md dan requirements.txt.")
    para("Angka utama: output/cnn/results.json dan pattern_results.json. Pemeriksaan pola: cluster_weather_profiles.csv, "
         "cluster_stability.csv, encoder_seed_window_agreement.csv, consensus_core_weather_profiles.csv, "
         "cluster_model_comparison.csv, dan latent_weather_associations.csv. Penelusuran waktu memakai window_metadata.csv "
         "dan window_cluster_assignments.csv. Manifest eksperimen dan pola menyediakan hash setiap artefak.")
    para("Laporan PDF dan Markdown pendamping bernama sama dihasilkan bersama dari artefak numerik melalui script pembangun laporan. "
         "Versi sebelum revisi tersimpan di reports/archive; keluaran eksperimen tiga epoch berada di output/cnn/legacy_smoke_test. "
         f"Repository pengembangan: {REPO_URL}. Versi yang menghasilkan laporan ini ditentukan oleh hash kode dan artefak dalam manifest pendamping.")
    return blocks


def markdown(blocks):
    lines = ["# "+TITLE, "", AUTHORS, "", AFFILIATION, ""]
    for block in blocks:
        kind = block["type"]
        if kind in {"heading", "subheading"}:
            lines.append(("### " if kind == "subheading" else "## ")+block["text"])
        elif kind == "table":
            lines += ["**"+block["caption"]+"**", "", "| "+" | ".join(block["columns"])+" |",
                      "| "+" | ".join(["---"]*len(block["columns"]))+" |"]
            lines += ["| "+" | ".join(row)+" |" for row in block["rows"]]
        elif kind == "figure":
            lines += [f"![{block['caption']}](../output/cnn/{Path(block['path']).name})", "", block["caption"]]
        else:
            lines.append(block["text"])
        lines.append("")
    return "\n".join(lines)


def render_pdf(blocks, path):
    fonts = Path("C:/Windows/Fonts")
    for name, filename in (("StudyTimes", "times.ttf"), ("StudyTimes-Bold", "timesbd.ttf"), ("StudyTimes-Italic", "timesi.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts/filename)))
    pdfmetrics.registerFontFamily("StudyTimes", normal="StudyTimes", bold="StudyTimes-Bold", italic="StudyTimes-Italic", boldItalic="StudyTimes-Bold")
    width, height = A4
    margin, bottom, gap = 19*mm, 21*mm, 5*mm
    column = (width-2*margin-gap)/2
    styles = {
        "body": ParagraphStyle("body", fontName="StudyTimes", fontSize=10, leading=11.7,
                               alignment=TA_JUSTIFY, spaceAfter=5, splitLongWords=True, allowWidows=0, allowOrphans=0),
        "heading": ParagraphStyle("heading", fontName="StudyTimes-Bold", fontSize=10, leading=12,
                                  alignment=TA_CENTER, spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "subheading": ParagraphStyle("subheading", fontName="StudyTimes-Italic", fontSize=10, leading=12,
                                     spaceBefore=6, spaceAfter=4, keepWithNext=True),
        "caption": ParagraphStyle("caption", fontName="StudyTimes", fontSize=8, leading=9.3,
                                  alignment=TA_CENTER, spaceBefore=3, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="StudyTimes", fontSize=7.5, leading=8.6, alignment=TA_LEFT),
        "reference": ParagraphStyle("reference", fontName="StudyTimes", fontSize=8, leading=9.5, spaceAfter=5, splitLongWords=True),
    }
    def para(text, style="body"):
        escaped = html.escape(text)
        # Add hyperlink annotations without depending on a network connection.
        escaped = re.sub(r"https://[^\s,]+", lambda m: f'<link href="{m.group(0).rstrip(".")}" color="#194f90">{m.group(0)}</link>', escaped)
        return Paragraph(escaped, styles[style])
    def on_page(canvas, doc):
        canvas.setTitle(TITLE)
        canvas.setAuthor(AUTHORS)
        canvas.setSubject("Visualisasi dan pencarian pola cuaca; bukan forecasting")
        canvas.saveState()
        canvas.setFont("StudyTimes", 8)
        canvas.setFillColor(colors.HexColor("#444444"))
        canvas.drawCentredString(width/2, 12*mm, str(doc.page))
        if doc.page == 1:
            title_style = ParagraphStyle("title", fontName="StudyTimes", fontSize=24, leading=27, alignment=TA_CENTER)
            title = Paragraph(html.escape(TITLE), title_style)
            _, h = title.wrap(width-2*margin, 150)
            top = height-18*mm
            title.drawOn(canvas, margin, top-h)
            canvas.setFont("StudyTimes", 10)
            canvas.drawCentredString(width/2, top-h-20, "Ishak Palentino Napitupulu — 13524022")
            canvas.drawCentredString(width/2, top-h-33, "Marcel Luther Sitorus — 13524063")
            canvas.drawCentredString(width/2, top-h-48, AFFILIATION)
        canvas.restoreState()
    document = BaseDocTemplate(str(path), pagesize=A4, leftMargin=margin, rightMargin=margin,
                               topMargin=19*mm, bottomMargin=bottom, title=TITLE, author=AUTHORS,
                               pageCompression=1, allowSplitting=1)
    body_height = height-bottom-19*mm
    first_top = height-195
    first_frames = [Frame(margin+i*(column+gap), bottom, column, first_top-bottom,
                          leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id=f"first{i}") for i in range(2)]
    body_frames = [Frame(margin+i*(column+gap), bottom, column, body_height,
                         leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id=f"body{i}") for i in range(2)]
    document.addPageTemplates([PageTemplate(id="first", frames=first_frames, onPage=on_page, autoNextPageTemplate="body"),
                               PageTemplate(id="body", frames=body_frames, onPage=on_page)])
    story = []
    for block in blocks:
        kind = block["type"]
        if kind in {"paragraph", "reference", "heading", "subheading"}:
            story.append(para(block["text"], "body" if kind == "paragraph" else kind))
        elif kind == "table":
            cells = [[para(text, "cell") for text in block["columns"]]]
            cells += [[para(text, "cell") for text in row] for row in block["rows"]]
            widths = [column/len(block["columns"])]*len(block["columns"])
            if len(widths) == 3:
                widths = [column*.18, column*.54, column*.28]
            elif len(widths) == 4:
                widths = [column*.36, column*.15, column*.23, column*.26]
            t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), .6, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), .4, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), .6, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story.append(KeepTogether([para(block["caption"], "caption"), t, Spacer(1, 5)]))
        elif kind == "figure":
            with PILImage.open(block["path"]) as image:
                ratio = image.height/image.width
            image_width = column
            image_height = image_width*ratio
            if image_height > 390:
                image_width *= 390/image_height
                image_height = 390
            picture = Image(block["path"], width=image_width, height=image_height, hAlign="CENTER")
            story.append(KeepTogether([Spacer(1, 4), picture, para(block["caption"], "caption")]))
    document.build(story)


def main():
    check_manifest("experiment_manifest.json")
    check_manifest("pattern_manifest.json")
    blocks = build_content()
    source = markdown(blocks)
    (REPORTS/f"{STEM}.md").write_text(source, encoding="utf-8")
    (REPORTS/"07_cnn_paper.md").write_text(source, encoding="utf-8")
    start = next(i for i, b in enumerate(blocks) if b.get("text") == "IV. HASIL DAN PEMBAHASAN")
    end = next(i for i, b in enumerate(blocks) if b.get("text") == "VI. KESIMPULAN")
    (REPORTS/"08_pattern_analysis_validation.md").write_text(
        "# Analisis pola dan validasi representasi CNN\n\n"
        "Ringkasan hasil dan batas interpretasi dari laporan utama. Metodologi lengkap dan daftar pustaka "
        f"tersedia pada [{STEM}.md]({STEM}.md).\n\n" +
        markdown(blocks[start:end]).split("## IV. HASIL DAN PEMBAHASAN", 1)[1], encoding="utf-8")
    pdf = REPORTS/f"{STEM}.pdf"
    render_pdf(blocks, pdf)
    shutil.copy2(pdf, REPORTS/"07_cnn_paper.pdf")
    files = [pdf, REPORTS/f"{STEM}.md", REPORTS/"07_cnn_paper.md", REPORTS/"07_cnn_paper.pdf",
             REPORTS/"08_pattern_analysis_validation.md"]
    json_write(REPORTS/"report_manifest.json", {"status": "passed", "canonical_pdf": pdf.name,
        "pattern_manifest_sha256": sha256(OUT/"pattern_manifest.json"),
        "builder_sha256": sha256(Path(__file__)),
        "artifacts": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "layout": "A4, two columns, 10pt embedded Times New Roman; IEEE-style course report",
        "figure_count": sum(b["type"] == "figure" for b in blocks),
        "table_count": sum(b["type"] == "table" for b in blocks)})
    print(f"REPORT UPDATED: {pdf}")


if __name__ == "__main__":
    main()
