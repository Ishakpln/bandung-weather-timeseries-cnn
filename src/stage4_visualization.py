"""Tahap 4: publication-resolution figures and auditable plot data."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from common import (CONTINUOUS, FIGURES, LABELS, REPORTS, TABLES, begin_stage,
                    finish_stage, json_write, load_clean, markdown_table,
                    raw_argument, resolve_raw, save_table)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
SHORT = ["Suhu", "Kelembapan", "Titik embun", "Hujan", "Tekanan MSL", "Awan", "Angin"]


def summaries(frame):
    indexed = frame.set_index("time")
    daily = indexed.resample("D").agg(
        n_hours=("temperature_2m", "size"), temperature_mean=("temperature_2m", "mean"),
        temperature_min=("temperature_2m", "min"), temperature_max=("temperature_2m", "max"),
        humidity_mean=("relative_humidity_2m", "mean"),
        precipitation_sum=("precipitation", lambda x: x.sum(min_count=24)),
        wet_hours=("precipitation", lambda x: x.gt(0).sum()))
    daily["diurnal_range"] = daily.temperature_max - daily.temperature_min
    daily["year"], daily["month"] = daily.index.year, daily.index.month
    monthly = indexed.resample("MS").agg(
        n_hours=("temperature_2m", "size"), temperature_mean=("temperature_2m", "mean"),
        temperature_std=("temperature_2m", "std"), temperature_min=("temperature_2m", "min"),
        temperature_max=("temperature_2m", "max"), humidity_mean=("relative_humidity_2m", "mean"),
        precipitation_sum=("precipitation", "sum"),
        wet_hours=("precipitation", lambda x: x.gt(0).sum()),
        cloud_mean=("cloud_cover", "mean"))
    monthly["year"], monthly["month"] = monthly.index.year, monthly.index.month
    monthly["wet_pct"] = monthly.wet_hours / monthly.n_hours * 100
    annual = frame.groupby("year").agg(
        n_hours=("temperature_2m", "size"), temperature_mean=("temperature_2m", "mean"),
        temperature_std=("temperature_2m", "std"), humidity_mean=("relative_humidity_2m", "mean"),
        precipitation_sum=("precipitation", "sum"),
        wet_hours=("precipitation", lambda x: x.gt(0).sum()),
        wind_mean=("wind_speed_10m", "mean"), cloud_mean=("cloud_cover", "mean"))
    annual["wet_pct"] = annual.wet_hours / annual.n_hours * 100
    annual["rain_mm_per_day"] = annual.precipitation_sum / (annual.n_hours / 24)
    month = frame.groupby("month").agg(
        n_hours=("temperature_2m", "size"), temperature_mean=("temperature_2m", "mean"),
        temperature_std=("temperature_2m", "std"), humidity_mean=("relative_humidity_2m", "mean"),
        wet_pct=("precipitation", lambda x: x.gt(0).mean() * 100))
    month["mean_monthly_rain_mm"] = monthly.groupby("month").precipitation_sum.mean()
    month["std_monthly_rain_across_years_mm"] = monthly.groupby("month").precipitation_sum.std()
    month["mean_rain_mm_per_day"] = frame.groupby("month").precipitation.sum() / (month.n_hours / 24)
    hour = frame.groupby("hour").agg(
        n_hours=("temperature_2m", "size"), temperature_mean=("temperature_2m", "mean"),
        temperature_q25=("temperature_2m", lambda x: x.quantile(.25)),
        temperature_q75=("temperature_2m", lambda x: x.quantile(.75)),
        humidity_mean=("relative_humidity_2m", "mean"),
        rain_mean_mm=("precipitation", "mean"), wet_pct=("precipitation", lambda x: x.gt(0).mean() * 100))
    # Retrospective, season/hour-conditioned anomaly candidates. No values are removed.
    grouped = frame.groupby(["month", "hour"]).temperature_2m
    median = grouped.transform("median")
    absolute_deviation = (frame.temperature_2m - median).abs()
    mad = absolute_deviation.groupby([frame.month, frame.hour]).transform("median")
    score = .6744897501960817 * (frame.temperature_2m - median) / mad.replace(0, np.nan)
    anomalies = frame[["time", "temperature_2m", "month", "hour"]].copy()
    anomalies["month_hour_median_C"] = median
    anomalies["month_hour_mad_C"] = mad
    anomalies["modified_z"] = score
    anomalies["candidate"] = score.abs() > 3.5
    residual = frame.temperature_2m - frame.groupby(["month", "hour"]).temperature_2m.transform("mean")
    autocorrelation = pd.DataFrame({"lag_hours": range(1, 169)})
    autocorrelation["temperature_raw"] = [frame.temperature_2m.autocorr(lag) for lag in autocorrelation.lag_hours]
    autocorrelation["temperature_month_hour_residual"] = [residual.autocorr(lag) for lag in autocorrelation.lag_hours]
    autocorrelation["precipitation"] = [frame.precipitation.autocorr(lag) for lag in autocorrelation.lag_hours]
    return daily, monthly, annual, month, hour, anomalies, autocorrelation


def run(raw=None):
    raw = resolve_raw(raw)
    original_hash = begin_stage(4, raw)
    frame = load_clean()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15,
                  rc={"font.family": "DejaVu Sans", "axes.spines.top": False,
                      "axes.spines.right": False, "savefig.dpi": 300})
    daily, monthly, annual, month, hour, anomalies, acf = summaries(frame)
    artifacts, figure_items = [], []
    for name, values in [("daily_summary", daily), ("monthly_summary", monthly),
                         ("annual_summary", annual), ("monthly_pattern", month),
                         ("hourly_pattern", hour)]:
        artifacts.append(save_table(name, values.reset_index()))
    artifacts.append(save_table("temperature_anomaly_scores", anomalies))
    artifacts.append(save_table("temperature_anomaly_candidates", anomalies.loc[anomalies.candidate]))
    artifacts.append(save_table("autocorrelation", acf))

    def save(fig, name, caption, source_tables):
        fig.suptitle("Bandung | data hourly 2022–2025 | WIB (UTC+7)", fontsize=10, y=1.015)
        fig.tight_layout()
        path = FIGURES / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white",
                    metadata={"Title": caption, "Software": "Bandung weather analysis; Matplotlib"})
        plt.close(fig)
        with Image.open(path) as pic:
            width, height = pic.size
        artifacts.append(path)
        figure_items.append({"filename": path.name, "caption": caption, "source_tables": source_tables,
                             "width_px": width, "height_px": height, "dpi": 300})

    def dates(ax):
        ax.xaxis.set_major_locator(mdates.YearLocator(tz=frame.time.dt.tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y", tz=frame.time.dt.tz))
        ax.set_xlabel("Waktu lokal (WIB)")

    for variable, daily_column, filename, title in [
        ("temperature_2m", "temperature_mean", "01_temperature_timeseries", "Time-series suhu"),
        ("relative_humidity_2m", "humidity_mean", "02_humidity_timeseries", "Time-series kelembapan")]:
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.plot(frame.time, frame[variable], color="#9ab6c4", alpha=.35, lw=.3, label="Hourly (seluruh data)")
        ax.plot(daily.index, daily[daily_column], color="#005f73", lw=.8, label="Rata-rata harian")
        ax.set(title=title, ylabel=LABELS[variable])
        dates(ax)
        ax.legend(loc="upper left", ncol=2)
        save(fig, filename, f"{title}; data hourly dan rata-rata per tanggal lokal.", ["daily_summary.csv"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    axes[0].plot(daily.index, daily.precipitation_sum, color="#0077b6", lw=.65)
    axes[0].set(title="Presipitasi per tanggal timestamp", ylabel="Jumlah (mm/tanggal)")
    dates(axes[0])
    axes[1].bar(monthly.index, monthly.precipitation_sum, width=23, color="#0096c7")
    axes[1].set(title="Presipitasi bulanan berdasarkan label timestamp", ylabel="Jumlah (mm/bulan)")
    dates(axes[1])
    axes[2].bar(annual.index.astype(str), annual.precipitation_sum, color="#005f73")
    axes[2].set(title="Perbandingan empat tahun", ylabel="Jumlah (mm/tahun)", xlabel="Tahun")
    for i, total in enumerate(annual.precipitation_sum):
        axes[2].text(i, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9)
    axes[2].margins(y=.15)
    save(fig, "03_precipitation_analysis", "Agregasi presipitasi: tanggal, bulan, tahun. Pengelompokan mengikuti timestamp yang tersedia.",
         ["daily_summary.csv", "monthly_summary.csv", "annual_summary.csv"])

    wet = frame.precipitation > 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    counts = [int((~wet).sum()), int(wet.sum())]
    axes[0].bar(["0 mm", "> 0 mm"], counts, color=["#adb5bd", "#0077b6"])
    for i, count in enumerate(counts):
        axes[0].text(i, count, f"{count:,}\n({count / len(frame) * 100:.1f}%)", ha="center", va="bottom")
    axes[0].margins(y=.2)
    axes[0].set(title="Jam tanpa/dengan presipitasi", ylabel="Jumlah jam")
    axes[1].hist(frame.loc[wet, "precipitation"], bins=45, color="#0077b6")
    axes[1].set(yscale="log", title="Distribusi hanya jam basah", xlabel="Presipitasi (mm per jam tercatat)", ylabel="Jumlah jam (skala log)")
    save(fig, "04_precipitation_distribution", "Nol ditampilkan terpisah; histogram positif memakai sumbu jumlah logaritmik.", ["eda_summary.json"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    sns.boxplot(data=daily, x="month", y="temperature_mean", ax=axes[0], color="#90c4cc", showfliers=True,
                flierprops={"markersize": 2})
    axes[0].set(title="Distribusi rata-rata suhu harian per bulan", xlabel="Bulan", ylabel="Suhu harian rata-rata (°C)")
    axes[0].set_xticks(range(12), MONTHS)
    for year, group in monthly.groupby("year"):
        axes[1].plot(group.month, group.temperature_mean, marker="o", ms=3, lw=1.2, label=str(year))
    axes[1].plot(month.index, month.temperature_mean, color="black", lw=2.1, label="Gabungan seluruh jam")
    axes[1].set(title="Rata-rata bulanan setiap tahun", xlabel="Bulan", ylabel="Suhu (°C)", xticks=range(1, 13))
    axes[1].set_xticklabels(MONTHS)
    axes[1].legend(ncol=5, fontsize=8)
    save(fig, "05_monthly_temperature_pattern", "Boxplot rata-rata harian: kotak Q1–Q3, garis median, whisker 1,5 IQR; panel bawah empat tahun terpisah.",
         ["daily_summary.csv", "monthly_summary.csv", "monthly_pattern.csv"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    for year, group in frame.groupby("year"):
        profile = group.groupby("hour").temperature_2m.mean()
        axes[0].plot(profile.index, profile.values, lw=.9, alpha=.7, label=str(year))
    axes[0].fill_between(hour.index, hour.temperature_q25, hour.temperature_q75, color="#94d2bd", alpha=.35, label="IQR observasi hourly")
    axes[0].plot(hour.index, hour.temperature_mean, color="black", lw=2, label="Rata-rata gabungan")
    axes[0].set(title="Pola suhu menurut jam lokal", ylabel="Suhu (°C)", xticks=range(0, 24, 2))
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(hour.index, hour.humidity_mean, color="#0077b6", marker="o", ms=3)
    axes[1].set(title="Pola kelembapan menurut jam lokal", xlabel="Jam WIB", ylabel="Kelembapan rata-rata (%)", xticks=range(0, 24, 2))
    save(fig, "06_hourly_temperature_humidity", "Suhu dan kelembapan per jam; pita adalah rentang kuartil observasi, bukan confidence interval.", ["hourly_pattern.csv"])

    for x, y, filename, title in [
        ("temperature_2m", "relative_humidity_2m", "07_temperature_vs_humidity", "Suhu vs kelembapan"),
        ("cloud_cover", "precipitation", "08_cloud_cover_vs_precipitation", "Tutupan awan vs presipitasi"),
        ("relative_humidity_2m", "precipitation", "09_humidity_vs_precipitation", "Kelembapan vs presipitasi")]:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        density = ax.hexbin(frame[x], frame[y], gridsize=45, mincnt=1, bins="log", cmap="viridis", linewidths=0)
        fig.colorbar(density, ax=ax, label="Jumlah observasi per sel (skala log)")
        pearson = frame[x].corr(frame[y])
        spearman = frame[x].corr(frame[y], method="spearman")
        ax.set(title=f"{title}\nPearson r={pearson:.3f}; Spearman ρ={spearman:.3f}; n={len(frame):,}",
               xlabel=LABELS[x], ylabel=LABELS[y])
        save(fig, filename, f"{title}; hexbin seluruh data, termasuk nol hujan; korelasi simultan bersifat deskriptif.",
             ["correlation_pearson.csv", "correlation_spearman.csv"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for method, ax in zip(["pearson", "spearman"], axes):
        matrix = frame[CONTINUOUS].corr(method=method)
        sns.heatmap(matrix, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    square=True, xticklabels=SHORT, yticklabels=SHORT, cbar_kws={"shrink": .7})
        ax.set_title(method.capitalize() + " | n=35.064 per pasangan")
        ax.tick_params(axis="x", labelrotation=55)
        ax.tick_params(axis="y", labelrotation=0)
    save(fig, "10_correlation_heatmaps", "Korelasi tujuh variabel kontinu; kode WMO dan fitur kalender dikecualikan.",
         ["correlation_pearson.csv", "correlation_spearman.csv"])

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for variable, ax in zip(CONTINUOUS, axes.flat):
        ax.hist(frame[variable], bins=40, color="#558b9b", edgecolor="white", lw=.25)
        ax.axvline(frame[variable].median(), color="#ae2012", lw=1.1, label="Median")
        ax.set(xlabel=LABELS[variable], ylabel="Jumlah jam")
        if variable == "precipitation":
            ax.set_yscale("log")
            ax.set_ylabel("Jumlah jam (log)")
        ax.legend(fontsize=7)
    codes = frame.weather_code.value_counts().sort_index()
    axes.flat[-1].bar(codes.index.astype(str), codes.values, color="#5f6f52")
    axes.flat[-1].set(xlabel="Kode cuaca WMO (kategori)", ylabel="Jumlah jam")
    axes.flat[-1].tick_params(axis="x", labelrotation=90, labelsize=7)
    save(fig, "11_variable_distributions", "Histogram seluruh variabel kontinu dan frekuensi kategori cuaca; hujan memakai sumbu jumlah log.",
         ["descriptive_statistics.csv", "weather_code_frequency.csv"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    month_hour_temp = frame.pivot_table(index="month", columns="hour", values="temperature_2m", aggfunc="mean")
    wet_heatmap = frame.assign(wet=wet.astype(float) * 100).pivot_table(index="month", columns="hour", values="wet", aggfunc="mean")
    for matrix, ax, cmap, title, unit in [
        (month_hour_temp, axes[0], "YlOrRd", "Rata-rata suhu per kombinasi bulan dan jam", "°C"),
        (wet_heatmap, axes[1], "Blues", "Persentase jam basah (>0 mm) per kombinasi bulan dan jam", "% jam basah")]:
        sns.heatmap(matrix, ax=ax, cmap=cmap, yticklabels=MONTHS, cbar_kws={"label": unit})
        ax.set(title=title, xlabel="Jam WIB", ylabel="Bulan")
    artifacts.append(save_table("month_hour_temperature", month_hour_temp.rename_axis("month").reset_index()))
    artifacts.append(save_table("month_hour_wet_pct", wet_heatmap.rename_axis("month").reset_index()))
    save(fig, "12_month_hour_patterns", "Heatmap bulan–jam dari seluruh observasi; persentase menggunakan jumlah jam pada sel masing-masing.",
         ["month_hour_temperature.csv", "month_hour_wet_pct.csv"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(month.index, month.mean_monthly_rain_mm, color="#0077b6")
    axes[0].set(title="Rata-rata jumlah bulanan dari empat tahun", xlabel="Bulan", ylabel="Presipitasi (mm/bulan)", xticks=range(1, 13))
    axes[0].set_xticklabels(MONTHS, rotation=45)
    axes[1].bar(hour.index, hour.wet_pct, color="#005f73")
    axes[1].set(title="Frekuensi jam basah menurut jam lokal", xlabel="Jam WIB (label akhir interval)", ylabel="Jam basah (%)", xticks=range(0, 24, 2))
    save(fig, "13_monthly_hourly_rain_patterns", "Presipitasi bulanan dirata-ratakan atas empat total tahunan; jam basah memakai ambang >0 mm.",
         ["monthly_pattern.csv", "hourly_pattern.csv"])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for column, label, color in [("temperature_raw", "Suhu asli", "#ae2012"),
                                  ("temperature_month_hour_residual", "Suhu dikurangi rerata bulan–jam", "#ee9b00"),
                                  ("precipitation", "Presipitasi", "#0077b6")]:
        ax.plot(acf.lag_hours, acf[column], label=label, color=color, lw=1.4)
    ax.axhline(0, color="black", lw=.6)
    ax.set(title="Autokorelasi deskriptif (tanpa uji signifikansi IID)", xlabel="Lag (jam)", ylabel="Korelasi dengan lag", xticks=[1, 24, 48, 72, 96, 120, 144, 168])
    ax.legend(fontsize=8)
    save(fig, "14_autocorrelation", "Autokorelasi lag 1–168 jam; residual bulan–jam dihitung retrospektif hanya untuk EDA.", ["autocorrelation.csv"])

    candidates = anomalies.loc[anomalies.candidate]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(frame.time, frame.temperature_2m, lw=.35, alpha=.5, color="#8d99ae", label="Suhu hourly")
    ax.scatter(candidates.time, candidates.temperature_2m, color="#ae2012", s=11, label=f"Kandidat |modified z| > 3,5 (n={len(candidates)})", zorder=3)
    ax.set(title="Kandidat anomali relatif terhadap bulan dan jam yang sama", ylabel="Suhu (°C)")
    dates(ax)
    ax.legend(fontsize=8)
    save(fig, "15_temperature_anomaly_candidates", "Median/MAD per bulan–jam; kandidat bukan bukti kesalahan data atau kejadian ekstrem terverifikasi.",
         ["temperature_anomaly_scores.csv", "temperature_anomaly_candidates.csv"])

    manifest_path = FIGURES / "figure_manifest.json"
    json_write(manifest_path, figure_items)
    artifacts.append(manifest_path)
    figure_table = pd.DataFrame([{k: row[k] for k in ("filename", "width_px", "height_px", "dpi", "caption")}
                                 for row in figure_items])
    report = REPORTS / "04_visualization_guide.md"
    report.write_text(f"""# Tahap 4 — Panduan visualisasi

Seluruh {len(figure_items)} grafik PNG telah dirender pada **300 dpi**, dengan ukuran fisik dan
label sumbu untuk penggunaan dalam paper. Seluruh observasi digunakan; tidak ada subsampling.
File dan caption: `output/figures/figure_manifest.json`. Tabel sumber tersedia di `output/tables/`.

{markdown_table(figure_table, 0)}

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
""", encoding="utf-8")
    artifacts.append(report)
    finish_stage(4, raw, original_hash, artifacts, figures=len(figure_items), dpi=300,
                 temperature_anomaly_candidates=len(candidates))
    return figure_items


if __name__ == "__main__":
    run(raw_argument(__doc__).parse_args().raw)
