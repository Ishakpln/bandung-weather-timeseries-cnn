# VISUALISASI DATA TIME-SERIES MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN)

Ishak Palentino Napitupulu — 13524022; Marcel Luther Sitorus — 13524063. Sistem Cerdas TF4018, Institut Teknologi Bandung, 2026.

## ABSTRAK
Penelitian ini mengeksplorasi visualisasi representasi temporal meteorologi Bandung menggunakan convolutional autoencoder (CAE) 1D dan 2D. Dataset berisi 35.064 observasi per jam selama 2022–2025, dengan tujuh fitur numerik. CNN 1D mempelajari rekonstruksi window multivariat 48 jam; CNN 2D mempelajari rekonstruksi citra kemiripan rekuren temperatur berukuran 32×32. Pembagian kronologis adalah train 2022–2023, validation 2024, test 2025. Pada eksperimen awal tiga epoch, MSE test CNN 1D adalah 0,8924 pada ruang data terstandardisasi, sedangkan CNN 2D 0,0908 pada ruang citra kemiripan; nilai tersebut tidak dapat dibandingkan langsung. Aktivasi, kernel, rekonstruksi, dan embedding PCA berhasil divisualisasikan. Kontribusi penelitian berupa penyajian dan pembacaan visual input, rekonstruksi, aktivasi, kernel, serta embedding; penelitian tidak menargetkan penemuan pola meteorologi baru.

Kata kunci: time-series, meteorologi, convolutional autoencoder, recurrence similarity, interpretabilitas.

## 1. PENDAHULUAN
### 1.1 Latar Belakang
Visualisasi garis membantu mengamati tren meteorologi, tetapi kurang menonjolkan hubungan antarkondisi yang terpisah dalam waktu. CNN dapat mempelajari transformasi representasi dari window deret waktu; visualisasi aktivasi diperlukan agar fitur yang dipelajari tidak diperlakukan sebagai penjelasan kausal.
### 1.2 Rumusan Masalah
Informasi apa yang dapat diperoleh dari visualisasi input, feature maps, rekonstruksi, dan embedding CNN 1D? Bagaimana citra kemiripan rekuren dan aktivasi CNN 2D menyajikan hubungan antarwaktu? Bagaimana kedua pendekatan berbeda dari sisi keterbacaan dan kehilangan informasi?
### 1.3 Tujuan Penelitian
Menghasilkan dan menginterpretasikan visualisasi data time-series meteorologi menggunakan representasi CNN 1D dan CNN 2D; membandingkan bentuk dan keterbacaan visualisasi, sementara galat rekonstruksi hanya memeriksa keberlangsungan pembelajaran.
### 1.4 Batasan Penelitian
Observasi Bandung 2022–2025; tujuh fitur kontinu untuk 1D, hanya temperatur untuk 2D; window 48 jam; eksperimen awal tiga epoch pada CPU. Tidak ada forecasting, klaim sebab-akibat, atau generalisasi lintas lokasi.
### 1.5 Manfaat Penelitian
Memberikan contoh pipeline yang dapat direproduksi untuk melihat data meteorologi melalui grafik konvensional, respons filter temporal, citra kemiripan, dan representasi laten; tidak mensyaratkan penemuan fenomena baru.

## 2. TINJAUAN PUSTAKA
### 2.1 Data Time-Series
Data time-series adalah pengukuran berurutan dengan dependensi temporal; pola cuaca dapat memiliki siklus harian dan variasi musiman.
### 2.2 Visualisasi Data Time-Series
Line plot, moving average, distribusi, dan korelasi memberi konteks; matriks kemiripan rekuren menyoroti pasangan waktu dengan keadaan serupa.
### 2.3 Convolutional Neural Network (CNN)
Konvolusi 1D mempelajari pola lokal sepanjang waktu; konvolusi 2D mempelajari struktur lokal citra. Autoencoder meminimalkan galat rekonstruksi dan menghasilkan representasi internal, tetapi aktivasi tinggi tidak otomatis membuktikan makna meteorologis.
### 2.4 Penelitian Terdahulu
Eckmann, Kamphorst, dan Ruelle (1987) memperkenalkan recurrence plots. Wang dan Oates (2015) mengeksplorasi pengkodean deret waktu menjadi citra untuk klasifikasi. Fawaz dkk. (2019) mengulas deep learning untuk klasifikasi time-series; penelitian ini berbeda karena menitikberatkan visualisasi dan reconstruction learning.

## 3. METODOLOGI PENELITIAN
### 3.1 Alur Penelitian
Dataset → audit dan EDA → split kronologis → standardisasi train-only → window 48 jam → (a) CNN 1D atau (b) citra kemiripan rekuren + CNN 2D → reconstruction loss → visualisasi aktivasi, kernel, embedding → interpretasi.
### 3.2 Dataset dan Sumber Data
Dataset `data/raw/BandungRawWeather.csv` mengikuti format Open-Meteo Historical Weather API (https://open-meteo.com/en/docs/historical-weather-api); data hasil audit 35.064 jam, 2022–2025, Asia/Jakarta. Fitur: temperature_2m, relative_humidity_2m, dew_point_2m, precipitation, pressure_msl, cloud_cover, wind_speed_10m. Weather_code tidak diperlakukan sebagai besaran kontinu.
### 3.3 Eksplorasi dan Pemahaman Data
Pipeline tahap 1–5 memeriksa frekuensi, duplikasi, missing values, distribusi, korelasi, dan pola harian/bulanan. Audit tersimpan melaporkan tidak ada missing timestamp atau duplikasi; angka perlu diverifikasi ulang pada environment lain.
### 3.4 Preprocessing Data
StandardScaler hanya difit pada 2022–2023; validation 2024 dan test 2025 ditransformasikan menggunakan parameter yang sama. Window 48 jam tanpa overlap dipilih untuk percobaan awal; jumlah window 365/183/182. Citra 2D dibentuk dari temperatur setiap window yang disampel ke 32 titik dan dinormalisasi min–max per-window: S(i,j)=exp(-|u_i-u_j|/0,25). Ini *continuous recurrence similarity*, bukan recurrence plot biner konvensional. Normalisasi per-window menghilangkan informasi level temperatur absolut.
### 3.5 Perancangan dan Penerapan CNN
CNN 1D: Conv1d(7,16,k=5)→ReLU→MaxPool→Conv1d(16,8,k=3)→ReLU→AdaptiveAvgPool(8)→Upsample(48)→Conv1d(8,16)→ReLU→Conv1d(16,7). CNN 2D: Conv2d(1,8,k=3)→ReLU→MaxPool→Conv2d(8,8,k=3)→ReLU→AdaptiveAvgPool(4×4)→Upsample(32×32)→Conv2d(8,8)→ReLU→Conv2d(8,1)→Tanh. Optimizer Adam, learning rate 0,001, batch 64, seed 42, tiga epoch, model terbaik berdasarkan validation MSE. Kedua decoder menggunakan interpolasi dan konvolusi, bukan deconvolution.
### 3.6 Visualisasi Data dan Hasil Model
Visualisasi meliputi EDA terdahulu, citra kemiripan rekuren, input-rekonstruksi, kernel awal, aktivasi layer pertama dan kedua, respons aktivasi CNN 1D yang disejajarkan dengan temperatur, peta aktivasi CNN 2D, error rekonstruksi citra, training loss, dan proyeksi PCA embedding test. Semua grafik CNN ditempatkan pada `output/cnn`. Kernel 2D ditampilkan dalam bentuk koefisien terflatten; bentuk tersebut tidak langsung memetakan waktu.
### 3.7 Evaluasi Model
Evaluasi utama bersifat deskriptif-visual: apakah representasi memperlihatkan respons yang bervariasi, dapat dipetakan ke posisi waktu, dan dapat ditafsirkan tanpa mengubahnya menjadi klaim kausal. MSE reconstruction hanya pemeriksaan pendukung dan dihitung terpisah pada ruang numerik 1D dan citra 2D. PCA hanya proyeksi, bukan bukti cluster yang stabil. Uji multi-seed dan variasi window tidak menjadi syarat untuk tujuan deskriptif awal, tetapi membatasi generalisasi.
### 3.8 Perangkat dan Tools
Python, pandas, NumPy, scikit-learn, Matplotlib, PyTorch 2.10.0+cpu. Pengujian dilakukan pada runtime CPU lingkungan eksekusi, bukan GPU pengguna.

## 4. HASIL DAN PEMBAHASAN
### 4.1 Gambaran Umum Dataset
35.064 pengamatan hourly selama empat tahun. Tidak ada missing values menurut audit awal. Pola hujan bersifat sparse: 73,25% observasi memiliki presipitasi nol menurut laporan repository.
### 4.2 Hasil Preprocessing
Data train/validation/test dibagi secara kronologis. Sebanyak 365/183/182 window nonoverlap 48 jam digunakan; scaler dan parameter eksperimen disimpan untuk reproduksibilitas.
### 4.3 Hasil Visualisasi Time-Series
Lima belas grafik EDA dari repository dipertahankan sebagai konteks. Grafik `recurrence_example.png` menampilkan temperatur 48 jam dan matriks kemiripan 32×32. Nilai diagonal matriks selalu 1 karena setiap titik dibandingkan dengan dirinya sendiri; simetri matriks juga berasal dari rumus transformasi, bukan hasil pembelajaran CNN. Kesamaan warna pada pasangan indeks waktu menunjukkan kedekatan *temperatur ternormalisasi dalam window*, bukan kesamaan seluruh kondisi cuaca. Normalisasi per-window membuang tingkat temperatur absolut.

### 4.4 Hasil Penerapan dan Evaluasi CNN
Eksperimen baseline tiga epoch menghasilkan MSE train/validation/test CNN 1D: 0,9179 / 1,0842 / 0,8924. CNN 2D: 0,09177 / 0,09128 / 0,09081. MSE 1D berasal dari tujuh variabel terstandardisasi, sedangkan MSE 2D dari matriks kemiripan temperatur; nilai-nilai tersebut **tidak** boleh digunakan untuk menentukan model yang lebih baik.

**Gambar hasil visualisasi yang dapat diperiksa:**

- Gambar 1 — `output/cnn/recurrence_example.png`: temperatur dan matriks kemiripan; memperlihatkan representasi pasangan waktu sebelum CNN.
- Gambar 2 — `output/cnn/cnn1d_temporal_response.png`: kurva temperatur dan respons delapan filter pertama pada sumbu waktu yang sama; memudahkan pemeriksaan *di bagian mana* respons meningkat, bukan identifikasi otomatis makna filter.
- Gambar 3 — `output/cnn/cnn1d_intermediate_layers.png`: perbandingan layer pertama (48 posisi) dan kedua (24 posisi setelah pooling); menunjukkan perubahan resolusi representasi temporal.
- Gambar 4 — `output/cnn/cnn1d_reconstruction_lines.png`: input dan rekonstruksi temperatur serta kelembapan terstandardisasi; selisih visual menunjukkan keterbatasan rekonstruksi baseline.
- Gambar 5 — `output/cnn/cnn2d_feature_map_grid.png`: enam peta aktivasi awal; memperlihatkan respons spasial berbeda terhadap matriks pasangan waktu.
- Gambar 6 — `output/cnn/cnn2d_intermediate_layers.png`: citra input dan aktivasi dua tingkat CNN 2D; membantu melihat perubahan struktur yang direpresentasikan.
- Gambar 7 — `output/cnn/cnn2d_reconstruction_error.png`: input, rekonstruksi, dan galat absolut per piksel pada ruang citra kemiripan.
- Gambar 8–9 — `output/cnn/cnn1d_pca.png` dan `cnn2d_pca.png`: proyeksi dua dimensi embedding test yang diberi warna berdasarkan bulan; pola sebaran tidak cukup untuk menyimpulkan cluster atau seasonality.

Seluruh gambar CNN adalah keluaran model checkpoint baseline yang tersimpan. Gambar 2–7 ditambahkan melalui `src/stage7_visual_interpretation.py` tanpa pelatihan ulang, memakai contoh window 48 jam pertama pada tahun 2025. Oleh sebab itu, pembacaan rinci grafik contoh tidak dapat digeneralisasi ke semua window.

### 4.5 Insight dan Interpretasi
Hasil visualisasi memperlihatkan **tiga cara membaca data yang berbeda**. Pertama, grafik time-series mempertahankan nilai fisik dan urutan waktu, sehingga perubahan temperatur dapat ditelusuri secara langsung. Kedua, CNN 1D menampilkan intensitas respons beberapa filter sepanjang waktu. Posisi aktivasi dapat disejajarkan dengan data asli, tetapi besarnya respons belum menunjukkan bahwa filter memiliki satu fungsi meteorologis tertentu. Ketiga, matriks kemiripan dan CNN 2D menampilkan hubungan antara *pasangan posisi waktu*. Bentuk garis atau blok pada matriks adalah pola kedekatan temperatur yang ditentukan oleh transformasi; feature maps memperlihatkan bagaimana layer CNN merespons struktur visual itu, bukan bukti pola baru.

Visualisasi dua layer memperlihatkan perbedaan resolusi dan kanal aktivasi. Rekonstruksi menunjukkan informasi yang dipertahankan dan yang tidak terpulihkan secara sempurna pada baseline. PCA memberi gambaran ringkas penyebaran embedding, namun tidak memberikan makna fisik langsung bagi sumbu PC1/PC2. **Kontribusi yang dapat didukung hasil aktual adalah perbedaan sudut pandang visual dan keterbatasan interpretasinya, bukan penemuan meteorologi baru atau keunggulan arsitektur.**

### 4.6 Diskusi
Eksperimen menghasilkan visualisasi yang dapat dibaca pada tiga tingkat: data asli, transformasi kemiripan, dan respons model. CNN 1D lebih langsung disejajarkan dengan urutan waktu; CNN 2D menekankan hubungan antarpasangan waktu, namun hanya memakai temperatur dan menghilangkan amplitudo absolut akibat normalisasi. Perbandingan bersifat deskriptif, bukan kompetisi skor. Belum ada evaluasi kestabilan lintas seed atau variasi window; EDA historis juga telah melihat periode test. Hal tersebut membatasi generalisasi, tetapi tidak membatalkan tujuan mendeskripsikan keluaran visual model.

## 5. KESIMPULAN DAN SARAN
### 5.1 Kesimpulan
Convolutional autoencoder 1D dan 2D menghasilkan visualisasi yang saling melengkapi: CNN 1D memetakan respons filter terhadap posisi waktu, sedangkan pendekatan 2D menampilkan struktur kemiripan temperatur antarwaktu dan respons filter pada citra tersebut. Grafik rekonstruksi, aktivasi antar-layer, dan PCA menambah sudut pandang terhadap representasi, dengan keterbatasan interpretasi masing-masing. Tujuan penelitian tercapai pada tingkat demonstrasi dan pembacaan visual baseline; penelitian ini tidak bertujuan menemukan pola meteorologi baru.
### 5.2 Saran
Perluas contoh visualisasi pada beberapa musim dan kondisi cuaca; tambahkan interaksi untuk memilih window dan filter; pertimbangkan RP biner/GAF dan citra multivariat jika diperlukan. Pelatihan lebih lama, variasi seed/window, dan evaluasi kuantitatif tambahan bersifat pengembangan opsional, bukan prasyarat untuk menjelaskan hasil visualisasi saat ini.

## DAFTAR PUSTAKA
[1] J.-P. Eckmann, S. O. Kamphorst, and D. Ruelle, “Recurrence Plots of Dynamical Systems,” Europhysics Letters, vol. 4, no. 9, pp. 973–977, 1987. DOI: 10.1209/0295-5075/4/9/004.
[2] Z. Wang and T. Oates, “Imaging Time-Series to Improve Classification and Imputation,” Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), 2015, pp. 3939–3945.
[3] H. I. Fawaz et al., “Deep learning for time series classification: a review,” Data Mining and Knowledge Discovery, vol. 33, pp. 917–963, 2019. DOI: 10.1007/s10618-019-00619-1.
[4] Open-Meteo, “Historical Weather API,” https://open-meteo.com/en/docs/historical-weather-api (accessed November 2026).

## LAMPIRAN (JIKA DIPERLUKAN)
Kode `src/stage6_cnn_experiments.py` dan `src/stage7_visual_interpretation.py`; manifest `output/cnn/visual_interpretation_manifest.json`; parameter `output/cnn/results.json` dan `scaler.json`; checkpoint `.pt`; gambar pada `output/cnn`; EDA sebelumnya pada `output/figures`. Pembagian tugas anggota tidak ditetapkan karena belum diberikan.
