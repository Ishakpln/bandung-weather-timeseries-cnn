# Laporan Eksperimen CNN 1D dan Insight

## Konfigurasi
- Data: `data/processed/bandung_weather_hourly_clean.csv` (35.064 baris, 2022-01-01 00:00–2025-12-31 23:00 WIB).
- Split kalender: train 2022–2023; validation 2024; test 2025. Window dibentuk mandiri di masing-masing segmen; tidak ada window lintas batas.
- Target: `temperature_2m` pada t+h (h ∈ {1,6,24}); input observasi t−L+1,...,t (L ∈ {24,48,72}).
- Fitur: 7 variabel meteorologis kontinu (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `pressure_msl`, `cloud_cover`, `wind_speed_10m`); 6 fitur kalender siklik; `weather_code` one-hot dengan kategori yang ada di train (10 kategori). Total 23 kanal pada data train (7+6+15). Kalender diketahui secara deterministik. Semua fitur cuaca berasal dari timestamp input ≤ t.
- StandardScaler X dan y di-fit hanya pada train. Metrik target dikembalikan ke °C.
- CNN: Conv1D 16 filter kernel 5 → ReLU → Conv1D 32 filter kernel 3 → ReLU → GlobalAveragePooling1D → Dense 16 → ReLU → Dropout 0,15 → Dense 1. Adam, lr 0,001; MSE; batch 256; maksimum 10 epoch; seed 42; early stopping patience 3; training windows stride 3 untuk efisiensi, evaluasi validation/test stride 1.
- MAE dan RMSE dilaporkan pada test. Model per horizon dipilih menurut MAE validation, bukan test.

## Hasil
Tabel lengkap di `output/cnn/tables/cnn_experiment_results.csv`. CNN dibandingkan dengan baseline persistence: prediksi suhu pada target menggunakan suhu pada waktu terakhir input (t).

| Horizon | Lookback | Val MAE (°C) | Test MAE (°C) | Test RMSE (°C) | Persistence MAE (°C) | Persistence RMSE (°C) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0,889 | 0,869 | 1,087 | 0,745 | 1,001 |
| 1 | 48 | 1,664 | 1,433 | 1,813 | 0,746 | 1,001 |
| 1 | 72 | 2,443 | 2,116 | 2,709 | 0,745 | 1,001 |
| 6 | 24 | 1,119 | 1,085 | 1,360 | 3,304 | 4,049 |
| 6 | 48 | 2,014 | 1,741 | 2,326 | 3,303 | 4,049 |
| 6 | 72 | 2,513 | 2,176 | 2,698 | 3,301 | 4,046 |
| 24 | 24 | 1,115 | 1,088 | 1,370 | 0,960 | 1,310 |
| 24 | 48 | 1,712 | 1,528 | 1,983 | 0,961 | 1,311 |
| 24 | 72 | 2,318 | 2,004 | 2,564 | 0,960 | 1,308 |

## Insight dan interpretasi
1. **Lookback 24 jam paling baik dalam validation pada seluruh horizon yang diuji.** Pada eksperimen ini, penambahan lookback 48/72 jam tidak meningkatkan generalisasi; error meningkat. Ini hasil konfigurasi/model saat ini, bukan bukti bahwa konteks lebih panjang selalu buruk.
2. **Horizon 6 jam adalah kasus ketika CNN mengungguli baseline persistence secara jelas.** CNN terbaik (L=24) memperoleh MAE 1,085 °C vs persistence 3,304 °C dan RMSE 1,360 °C vs 4,049 °C. Persistence satu jam terakhir memang baseline yang lemah untuk enam jam ke depan.
3. **Horizon 1 jam: persistence lebih baik daripada CNN.** MAE persistence 0,745 °C dibanding CNN 0,869 °C (L=24). CNN belum menunjukkan nilai tambah atas baseline sederhana pada prediksi sangat dekat.
4. **Horizon 24 jam: persistence juga sedikit lebih baik.** MAE persistence 0,960 °C dibanding CNN 1,088 °C (L=24). Model perlu dibandingkan dengan seasonal-naive (suhu pada jam yang sama sehari sebelumnya), serta dituning lebih sistematis.
5. Error meningkat untuk lookback yang lebih panjang, konsisten di tiga horizon. Ini dapat terkait kapasitas model, optimisasi, representasi fitur, atau jumlah epoch; tidak boleh disimpulkan sebagai hukum data tanpa uji lanjutan.
6. Rata-rata temperatur tahunan data: 22,584 °C (2022), 23,050 °C (2023), 23,221 °C (2024), 23,245 °C (2025). Perbedaan deskriptif ini bukan klaim tren iklim jangka panjang; hanya empat tahun satu grid reanalysis.

## Batasan dan QA
- Dataset sudah dieksplorasi pada tahap EDA termasuk tahun 2025. Oleh karena itu, 2025 kronologis sebagai test, tetapi **bukan holdout yang benar-benar belum pernah dilihat peneliti**. Paper wajib menyatakan risiko ini.
- Eksperimen memakai 10 epoch dan stride 3 pada subset window train sebagai kompromi komputasi; model sederhana, belum ada tuning sistematis atau beberapa seed.
- Loss history per konfigurasi tidak tersimpan konsisten saat batch run dihentikan oleh batas eksekusi. Tidak disajikan kurva training/validation loss yang seolah berasal dari run final. Metrik prediksi, CSV, model checkpoints, dan grafik aktual-vs-prediksi tersimpan.
- Baseline persistence perlu diperluas menjadi seasonal-naive (t−24) dan linear/AR baseline. Kesimpulan model belum boleh disebut unggul umum.
- Offline reanalysis bukan real-time forecast; nilai historis dapat direvisi dan precipitation merupakan jumlah jam sebelumnya sesuai definisi Open-Meteo.
