"""Find, validate, and visualize weather-window patterns in saved CNN representations.

All scaling, PCA, and cluster centroids are fit on train; k is chosen on validation.
Seed stability is label agreement, not evidence of meteorological ground truth.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import CONTINUOUS, ROOT, json_write, load_clean, read_json, resolve_raw, sha256
from stage6_cnn_experiments import AE1D, AE2D, OUT, evaluate

PARTS = ("train", "validation", "test")
DESCRIPTORS = ["temperature_mean", "temperature_range", "humidity_mean", "rain_sum", "wet_pct", "cloud_mean", "wind_mean"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def check_manifest(name):
    manifest = read_json(OUT/name)
    if manifest["status"] != "passed" or manifest["raw_sha256"] != sha256(resolve_raw()):
        raise RuntimeError(f"Invalid prerequisite {name}")
    for filename, digest in manifest["code"].items():
        if sha256(ROOT/filename) != digest:
            raise RuntimeError(f"Code changed: {filename}; rerun prerequisite")
    for filename, digest in manifest["artifacts"].items():
        if not (OUT/filename).exists() or sha256(OUT/filename) != digest:
            raise RuntimeError(f"Artifact changed: {filename}")
    return manifest


def scale_representation(arrays, with_std=True):
    scaler = StandardScaler(with_std=with_std).fit(arrays["train"])
    return {part: scaler.transform(x) for part, x in arrays.items()}, scaler


def silhouette(values, labels):
    return float(silhouette_score(values, labels)) if 1 < len(np.unique(labels)) < len(labels) else None


def fit_groups(arrays, common_space, train_metadata, name):
    """No test-dependent selection; insufficient occupancy rejects a candidate k."""
    # PCA input already comes from standardized weather. Preserve its variance;
    # standardizing PCs again would implicitly whiten the baseline.
    scaled, scaler = scale_representation(arrays, with_std=name != "pca_input")
    candidates, fitted = [], {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(scaled["train"])
        labels = {part: model.predict(x) for part, x in scaled.items()}
        train_counts = np.bincount(labels["train"], minlength=k)
        val_counts = np.bincount(labels["validation"], minlength=k)
        valid = bool(train_counts.min() >= 10 and val_counts.min() >= 3)
        score = silhouette(scaled["validation"], labels["validation"])
        candidates.append({"representation": name, "k": k, "eligible": valid,
            "min_train_cluster_n": int(train_counts.min()), "min_validation_cluster_n": int(val_counts.min()),
            "validation_silhouette": score,
            "validation_silhouette_common_input": silhouette(common_space["validation"], labels["validation"])})
        fitted[k] = model
    valid_rows = [r for r in candidates if r["eligible"] and r["validation_silhouette"] is not None]
    if not valid_rows:
        raise RuntimeError(f"No supported partition for {name}; do not invent weather classes")
    chosen = max(valid_rows, key=lambda r: r["validation_silhouette"])
    model = fitted[chosen["k"]]
    original = {part: model.predict(x) for part, x in scaled.items()}
    # Stable descriptive numbering is determined using train temperature range only.
    order = sorted(range(chosen["k"]), key=lambda c: train_metadata.loc[original["train"] == c, "temperature_range"].mean())
    mapping = {old: new for new, old in enumerate(order)}
    labels = {part: np.array([mapping[c] for c in values]) for part, values in original.items()}
    summary = {"representation": name, "k": chosen["k"],
        "validation_silhouette": chosen["validation_silhouette"],
        "test_silhouette": silhouette(scaled["test"], labels["test"]),
        "test_silhouette_common_input": silhouette(common_space["test"], labels["test"]),
        "test_cluster_counts": np.bincount(labels["test"], minlength=chosen["k"]).tolist()}
    return labels, scaled, scaler, model, order, summary, candidates


def main():
    check_manifest("experiment_manifest.json")
    (OUT/"pattern_manifest.json").unlink(missing_ok=True)
    torch.set_num_threads(2)
    frame = load_clean()
    results = read_json(OUT/"results.json")
    metadata = pd.read_csv(OUT/"window_metadata.csv")
    meta = {part: metadata.loc[metadata.split == part].reset_index(drop=True) for part in PARTS}
    x1 = {part: np.load(OUT/f"cnn1d_{part}_input.npy") for part in PARTS}
    x2 = {part: np.load(OUT/f"cnn2d_{part}_input.npy") for part in PARTS}
    flat = {part: x.reshape(len(x), -1) for part, x in x1.items()}
    pca_input = PCA(n_components=96, svd_solver="full").fit(flat["train"])
    reps = {"cnn1d": {p: np.load(OUT/f"cnn1d_{p}_embedding.npy") for p in PARTS},
            "cnn2d": {p: np.load(OUT/f"cnn2d_{p}_embedding.npy") for p in PARTS},
            "untrained_cnn1d": {p: np.load(OUT/f"cnn1d_untrained_{p}_embedding.npy") for p in PARTS},
            "pca_input": {p: pca_input.transform(x) for p, x in flat.items()},
            "summary_statistics": {p: np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2)], axis=1) for p, x in x1.items()}}
    artifacts, figures = [], []
    def table(name, data):
        data.to_csv(OUT/name, index=False, float_format="%.10g")
        artifacts.append(name)
    def array(name, data):
        np.save(OUT/name, data)
        artifacts.append(name)
    assignments, scaled_reps, group_models, summaries, all_candidates = {}, {}, {}, [], []
    for name, representation in reps.items():
        labels, scaled, scaler, model, order, summary, candidates = fit_groups(representation, flat, meta["train"], name)
        assignments[name], scaled_reps[name] = labels, scaled
        group_models[name] = (scaler, model, order)
        summaries.append(summary)
        all_candidates.extend(candidates)
        for part in PARTS:
            array(f"{name}_{part}_cluster_labels.npy", labels[part])
        np.savez(OUT/f"{name}_cluster_parameters.npz", mean=scaler.mean_,
                 scale=scaler.scale_ if scaler.scale_ is not None else np.ones_like(scaler.mean_),
                 centers=model.cluster_centers_, order=np.array(order))
        artifacts.append(f"{name}_cluster_parameters.npz")
    table("cluster_model_comparison.csv", pd.DataFrame(summaries))
    table("cluster_k_validation.csv", pd.DataFrame(all_candidates))
    primary = next(r for r in summaries if r["representation"] == "cnn1d")
    k = primary["k"]
    labeled = []
    for part in PARTS:
        m = meta[part].copy()
        m["cluster"] = assignments["cnn1d"][part]+1
        labeled.append(m)
    labeled = pd.concat(labeled, ignore_index=True)
    table("window_cluster_assignments.csv", labeled)
    profile = labeled.groupby(["split", "cluster"]).agg(
        n_windows=("window_id", "size"), temperature_mean=("temperature_mean", "mean"),
        temperature_range=("temperature_range", "mean"), humidity_mean=("humidity_mean", "mean"),
        rain_sum=("rain_sum", "mean"), wet_pct=("wet_pct", "mean"),
        cloud_mean=("cloud_mean", "mean"), wind_mean=("wind_mean", "mean")).reset_index()
    table("cluster_weather_profiles.csv", profile)
    month_counts = pd.crosstab(meta["test"].month, assignments["cnn1d"]["test"]+1).reindex(index=range(1, 13), columns=range(1, k+1), fill_value=0)
    month_pct = month_counts.div(month_counts.sum(axis=1), axis=0)*100
    table("test_cluster_month_counts.csv", month_counts.rename_axis("month").reset_index())
    table("test_cluster_month_percentages.csv", month_pct.rename_axis("month").reset_index())
    # Cross-encoder-seed agreement: independent models, same k, train-fit centroids.
    stability_rows, seed_labels, aligned_primary = [], {}, {}
    for tag in ("cnn1d", "cnn2d"):
        fixed_k = next(r["k"] for r in summaries if r["representation"] == tag)
        by_seed = {}
        for seed in results["configuration"]["seeds"]:
            rep = {p: np.load(OUT/f"{tag}_seed{seed}_{p}_embedding.npy") for p in PARTS}
            scaled, _ = scale_representation(rep)
            model = KMeans(n_clusters=fixed_k, n_init=20, random_state=42).fit(scaled["train"])
            by_seed[seed] = model.predict(scaled["test"])
            if tag == "cnn1d":
                train_labels = model.predict(scaled["train"])
                contingency = np.zeros((fixed_k, fixed_k), dtype=int)
                np.add.at(contingency, (assignments[tag]["train"], train_labels), 1)
                rows, columns = linear_sum_assignment(-contingency)
                mapping = dict(zip(columns, rows))
                aligned_primary[seed] = np.array([mapping[c]+1 for c in by_seed[seed]])
        seed_labels[tag] = by_seed
        for first, second in itertools.combinations(by_seed, 2):
            stability_rows.append({"representation": tag, "stability_type": "encoder_seed",
                "seed_a": first, "seed_b": second, "k_fixed": fixed_k,
                "test_ARI": adjusted_rand_score(by_seed[first], by_seed[second])})
    for a, b in itertools.combinations([11, 22, 33, 44, 55], 2):
        predictions = []
        for seed in (a, b):
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled_reps["cnn1d"]["train"])
            predictions.append(model.predict(scaled_reps["cnn1d"]["test"]))
        stability_rows.append({"representation": "cnn1d", "stability_type": "kmeans_initialization",
            "seed_a": a, "seed_b": b, "k_fixed": k,
            "test_ARI": adjusted_rand_score(*predictions)})
    stability = pd.DataFrame(stability_rows)
    table("cluster_stability.csv", stability)
    agreement = meta["test"].copy()
    for seed, labels in aligned_primary.items():
        agreement[f"seed_{seed}_cluster"] = labels
    agreement["selected_cluster"] = assignments["cnn1d"]["test"]+1
    agreement["all_three_seeds_agree"] = np.ptp(np.stack(list(aligned_primary.values())), axis=0) == 0
    table("encoder_seed_window_agreement.csv", agreement)
    core_profile = agreement.loc[agreement.all_three_seeds_agree].groupby("selected_cluster").agg(
        n_windows=("window_id", "size"), temperature_mean=("temperature_mean", "mean"),
        temperature_range=("temperature_range", "mean"), humidity_mean=("humidity_mean", "mean"),
        rain_sum=("rain_sum", "mean"), wet_pct=("wet_pct", "mean"), cloud_mean=("cloud_mean", "mean")).reset_index()
    table("consensus_core_weather_profiles.csv", core_profile)
    # Reconstruction benchmarks all use exactly the same test windows and target space.
    scale = read_json(OUT/"scaler.json")
    weather = frame[CONTINUOUS].to_numpy()
    standardized = (weather-np.array(scale["mean"]))/np.array(scale["scale"])
    train_mask = frame.time.dt.year.to_numpy() <= 2023
    hour_profile = pd.DataFrame(standardized[train_mask], columns=CONTINUOUS).assign(
        hour=frame.hour.to_numpy()[train_mask]).groupby("hour").mean()
    hourly = np.stack([hour_profile.loc[frame.hour.iloc[int(s):int(s)+48]].to_numpy().T for s in meta["test"].start_row])
    rec1 = np.load(OUT/"cnn1d_test_reconstruction.npy")
    rec2 = np.load(OUT/"cnn2d_test_reconstruction.npy")
    comparisons = [
        {"space": "1D standardized weather", "method": "train_mean", "test_mse": float(np.mean(x1["test"]**2))},
        {"space": "1D standardized weather", "method": "train_hour_profile", "test_mse": float(np.mean((x1["test"]-hourly)**2))},
        {"space": "1D standardized weather", "method": "cnn1d", "test_mse": float(np.mean((x1["test"]-rec1)**2))},
        {"space": "2D temperature similarity", "method": "train_mean_image", "test_mse": float(np.mean((x2["test"]-x2["train"].mean(0))**2))},
        {"space": "2D temperature similarity", "method": "cnn2d", "test_mse": float(np.mean((x2["test"]-rec2)**2))}]
    table("reconstruction_benchmarks.csv", pd.DataFrame(comparisons))
    per_feature = results["cnn1d"]["selected_metrics"]["test"]["per_feature"]
    table("cnn1d_per_feature_reconstruction.csv", pd.DataFrame(per_feature))
    physical_rec = rec1*np.array(scale["scale"])[None, :, None]+np.array(scale["mean"])[None, :, None]
    domain_rows = []
    for column, name, upper in ((1, "relative_humidity_2m", 100), (3, "precipitation", None),
                                 (5, "cloud_cover", 100), (6, "wind_speed_10m", None)):
        values = physical_rec[:, column]
        violations = values < -1e-4
        if upper is not None:
            violations |= values > upper+1e-4
        domain_rows.append({"feature": name, "lower_bound": 0, "upper_bound": upper,
            "minimum_reconstructed": float(values.min()), "maximum_reconstructed": float(values.max()),
            "violations_tolerance_1e_4": int(violations.sum()), "evaluated_cells": values.size})
    table("reconstruction_physical_domain.csv", pd.DataFrame(domain_rows))
    error_windows = meta["test"].copy()
    error_windows["reconstruction_mse"] = np.mean((x1["test"]-rec1)**2, axis=(1, 2))
    table("test_window_reconstruction_errors.csv", error_windows.sort_values("reconstruction_mse", ascending=False))
    pca_info, projections, associations = {}, {}, []
    for tag in ("cnn1d", "cnn2d"):
        projection = PCA(n_components=2, svd_solver="full").fit(reps[tag]["train"])
        xy = projection.transform(reps[tag]["test"])
        projections[tag] = xy
        pca_info[tag] = {"fit_split": "train", "explained_variance_ratio_train": projection.explained_variance_ratio_.tolist(),
                        "two_components_variance_pct": float(projection.explained_variance_ratio_.sum()*100)}
        array(f"{tag}_test_pca.npy", xy)
        np.savez(OUT/f"{tag}_pca_parameters.npz", mean=projection.mean_, components=projection.components_)
        artifacts.append(f"{tag}_pca_parameters.npz")
        for descriptor in DESCRIPTORS:
            for component in (0, 1):
                associations.append({"representation": tag, "component": component+1, "descriptor": descriptor,
                    "test_pearson_r": float(np.corrcoef(xy[:, component], meta["test"][descriptor])[0, 1])})
    table("latent_weather_associations.csv", pd.DataFrame(associations))
    # Representative windows are actual test observations nearest a train centroid.
    representatives = []
    for cluster in range(k):
        candidates = np.flatnonzero(assignments["cnn1d"]["test"] == cluster)
        old_cluster = group_models["cnn1d"][2][cluster]
        center = group_models["cnn1d"][1].cluster_centers_[old_cluster]
        chosen = int(candidates[np.argmin(np.sum((scaled_reps["cnn1d"]["test"][candidates]-center)**2, axis=1))])
        representatives.append({"cluster": cluster+1, "test_window_id": chosen,
            "start": meta["test"].iloc[chosen].start, "end": meta["test"].iloc[chosen].end,
            "start_row": int(meta["test"].iloc[chosen].start_row)})
    table("representative_windows.csv", pd.DataFrame(representatives))
    pattern = {"objective": "weather_pattern_exploration_not_forecasting", "selection_split": "validation",
        "primary": primary, "representations": summaries, "reconstruction_benchmarks": comparisons,
        "encoder_seed_ARI": {tag: {
            "mean": float(stability.loc[(stability.representation == tag)&(stability.stability_type == "encoder_seed"), "test_ARI"].mean()),
            "min": float(stability.loc[(stability.representation == tag)&(stability.stability_type == "encoder_seed"), "test_ARI"].min())}
            for tag in ("cnn1d", "cnn2d")}, "pca": pca_info,
        "representatives": representatives, "test_cluster_profiles": profile.loc[profile.split == "test"].to_dict("records"),
        "reconstruction_domain_checks": domain_rows,
        "three_seed_agreement": {"aligned_on": "train labels using Hungarian matching",
            "n_windows": int(agreement.all_three_seeds_agree.sum()),
            "total_test_windows": len(agreement), "percentage": float(agreement.all_three_seeds_agree.mean()*100),
            "core_profiles": core_profile.to_dict("records")},
        "limitations": ["Exploratory results; test year was viewed in EDA", "Clusters are algorithmic partitions, not confirmed weather classes",
            "Silhouettes in different representation spaces are not directly comparable superiority scores",
            "Fixed 48h windows, one location and three encoder seeds", "Month labels refer to window start; windows may span months",
            "No causal attribution, anomaly ground truth, forecast, or long-term climatology"]}
    json_write(OUT/"pattern_results.json", pattern)
    artifacts.append("pattern_results.json")
    make_figures(results, pattern, meta, x1, x2, rec1, rec2, projections, assignments, profile,
                 month_pct, stability, figures, artifacts, frame)
    json_write(OUT/"visual_interpretation_manifest.json", {"figures": figures, "dpi": 300,
        "representative_windows": representatives, "limitations": pattern["limitations"]})
    artifacts.append("visual_interpretation_manifest.json")
    json_write(OUT/"pattern_manifest.json", {"status": "passed", "raw_sha256": sha256(resolve_raw()),
        "experiment_manifest_sha256": sha256(OUT/"experiment_manifest.json"),
        "code": {"src/stage7_visual_interpretation.py": sha256(Path(__file__))},
        "artifacts": {name: sha256(OUT/name) for name in sorted(artifacts)}})
    print(f"STAGE 7 COMPLETE: k={k}, encoder-seed ARI={pattern['encoder_seed_ARI']['cnn1d']['mean']:.3f}, figures={len(figures)}", flush=True)


def make_figures(results, pattern, meta, x1, x2, rec1, rec2, projections, assignments,
                 profile, month_pct, stability, figures, artifacts, frame):
    sns.set_theme(style="whitegrid", context="paper", rc={"font.family": "DejaVu Sans",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
        "axes.spines.right": False})
    palette = sns.color_palette("colorblind", pattern["primary"]["k"])
    k = len(palette)
    def save(fig, name, caption, sources):
        fig.tight_layout(pad=.6)
        fig.savefig(OUT/name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        artifacts.append(name)
        figures.append({"file": name, "caption": caption, "sources": sources, "dpi": 300})
    for tag in ("cnn1d", "cnn2d"):
        fig, ax = plt.subplots(figsize=(3.35, 2.45))
        for color, seed in zip(sns.color_palette("colorblind", 3), results["configuration"]["seeds"]):
            history = pd.read_csv(OUT/f"{tag}_seed{seed}_training.csv")
            ax.plot(history.epoch, history.train_mse, color=color, lw=.8, label=f"Train {seed}")
            ax.plot(history.epoch, history.validation_mse, color=color, lw=.8, ls="--", label=f"Val {seed}")
        ax.set(xlabel="Epoch", ylabel="MSE rekonstruksi", title=tag.upper()+": tiga seed")
        ax.legend(ncol=2)
        save(fig, f"{tag}_loss.png", "Kurva train/validation semua seed; checkpoint dipilih hanya dari validation.",
             [f"{tag}_seed{s}_training.csv" for s in results["configuration"]["seeds"]])
        xy = projections[tag]
        fig, ax = plt.subplots(figsize=(3.35, 2.6))
        points = ax.scatter(xy[:, 0], xy[:, 1], c=meta["test"].month, s=12, cmap="viridis", alpha=.85)
        fig.colorbar(points, ax=ax, label="Bulan awal window", shrink=.8)
        evr = pattern["pca"][tag]["two_components_variance_pct"]
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{tag.upper()} | PCA train: {evr:.1f}% varians")
        save(fig, f"{tag}_pca.png", "PCA difit pada embedding train; warna bulan bukan label kelas cuaca.",
             [f"{tag}_test_pca.npy", "window_metadata.csv"])
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    xy = projections["cnn1d"]
    for c, color in enumerate(palette):
        mask = assignments["cnn1d"]["test"] == c
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, color=color, alpha=.85, label=f"C{c+1} (n={mask.sum()})")
    ax.set(xlabel="PC1", ylabel="PC2", title="Window test dan kelompok CNN 1D")
    ax.legend()
    save(fig, "pattern_embedding.png", "Kelompok ditetapkan di embedding penuh; proyeksi dua dimensi hanya untuk tampilan.",
         ["cnn1d_test_cluster_labels.npy", "cnn1d_test_pca.npy"])
    scale = read_json(OUT/"scaler.json")
    original = x1["test"]*np.array(scale["scale"])[None, :, None]+np.array(scale["mean"])[None, :, None]
    reconstructed = rec1*np.array(scale["scale"])[None, :, None]+np.array(scale["mean"])[None, :, None]
    hours = np.arange(48)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.1), sharex=True)
    for c, color in enumerate(palette):
        mask = assignments["cnn1d"]["test"] == c
        for column, ax in zip((0, 1), axes):
            curves = original[mask, column]
            median = np.median(curves, axis=0)
            q1, q3 = np.quantile(curves, [.25, .75], axis=0)
            ax.plot(hours, median, color=color, lw=1, label=f"C{c+1}")
            ax.fill_between(hours, q1, q3, color=color, alpha=.13)
    axes[0].set(ylabel="Suhu (°C)", title="Profil test: median dan IQR")
    axes[1].set(xlabel="Jam sejak awal window (WIB)", ylabel="Kelembapan (%)", xticks=[0, 12, 24, 36, 47])
    axes[0].legend(ncol=min(k, 3))
    save(fig, "cluster_profiles.png", "Median/IQR dari window cuaca asli tiap kelompok; pita bukan confidence interval.",
         ["window_cluster_assignments.csv", "cnn1d_test_input.npy", "scaler.json"])
    fig, axes = plt.subplots(k, 1, figsize=(3.35, 1.35*k+.55), sharex=True, squeeze=False)
    for row, rep in enumerate(pattern["representatives"]):
        number = rep["test_window_id"]
        ax = axes[row, 0]
        ax.plot(hours, original[number, 0], color=palette[row], lw=1, label="Asli")
        ax.plot(hours, reconstructed[number, 0], color="black", ls="--", lw=.8, label="Rekonstruksi")
        ax.set(ylabel="Suhu (°C)", title=f"C{row+1}: {rep['start'][:10]}")
        if row == 0:
            ax.legend(ncol=2)
    axes[-1, 0].set(xlabel="Jam sejak awal window", xticks=[0, 12, 24, 36, 47])
    save(fig, "representative_windows.png", "Contoh nyata terdekat dengan centroid train pada setiap kelompok test; tidak ada kurva sintetis.",
         ["representative_windows.csv", "cnn1d_test_reconstruction.npy"])
    fig, ax = plt.subplots(figsize=(3.35, 3.15))
    labels = [f"{MONTHS[m-1]} (n={int(pd.crosstab(meta['test'].month, assignments['cnn1d']['test']).sum(axis=1).get(m, 0))})" for m in range(1, 13)]
    sns.heatmap(month_pct, ax=ax, annot=True, fmt=".0f", vmin=0, vmax=100, cmap="Blues",
                xticklabels=[f"C{c}" for c in month_pct.columns], yticklabels=labels,
                cbar_kws={"label": "% window per bulan", "shrink": .65}, annot_kws={"fontsize": 7})
    ax.set(xlabel="Kelompok", ylabel="", title="Distribusi kelompok test 2025")
    ax.tick_params(axis="y", labelrotation=0)
    save(fig, "cluster_month_frequency.png", "Persentase per bulan; label bulan mengikuti awal window; jumlah window dicantumkan.",
         ["test_cluster_month_counts.csv", "test_cluster_month_percentages.csv"])
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.5))
    for tag, ax in zip(("cnn1d", "cnn2d"), axes):
        rows = [r for r in pattern["reconstruction_benchmarks"] if r["space"].startswith("1D" if tag == "cnn1d" else "2D")]
        names = {"train_mean": "Rerata train", "train_hour_profile": "Profil jam train", "cnn1d": "CNN 1D",
                 "train_mean_image": "Citra rerata train", "cnn2d": "CNN 2D"}
        ax.barh([names[r["method"]] for r in rows], [r["test_mse"] for r in rows], color="#397b91")
        ax.set(xlabel="MSE test", title="Ruang cuaca standar" if tag == "cnn1d" else "Ruang citra kemiripan")
    save(fig, "reconstruction_benchmarks.png", "Pembanding hanya dibandingkan dalam ruang target yang sama; seluruh parameter pembanding berasal dari train.",
         ["reconstruction_benchmarks.csv"])
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    data = stability.loc[stability.stability_type == "encoder_seed"]
    for position, tag in enumerate(("cnn1d", "cnn2d")):
        values = data.loc[data.representation == tag, "test_ARI"].to_numpy()
        ax.scatter(np.full(len(values), position), values, s=25, color=["#0072b2", "#d55e00"][position])
        ax.plot([position-.15, position+.15], [values.mean()]*2, color="black", lw=1.5)
    ax.set(xticks=[0, 1], xticklabels=["CNN 1D", "CNN 2D"], ylabel="ARI pasangan seed", ylim=(-.05, 1.05), title="Stabilitas pengelompokan tiga encoder")
    save(fig, "cnn_seed_stability.png", "Tiga pasangan seed per arsitektur; garis menunjukkan rerata; ARI mengukur kesepakatan label, bukan akurasi kelas cuaca.",
         ["cluster_stability.csv"])
    models = {}
    for tag, constructor in (("cnn1d", AE1D), ("cnn2d", AE2D)):
        model = constructor()
        model.load_state_dict(torch.load(OUT/f"{tag}.pt", map_location="cpu", weights_only=True)["state_dict"])
        model.eval()
        models[tag] = model
    first = pattern["representatives"][0]["test_window_id"]
    with torch.no_grad():
        t1 = torch.from_numpy(x1["test"][first:first+1])
        t2 = torch.from_numpy(x2["test"][first:first+1])
        a1 = models["cnn1d"].encoder[:2](t1).numpy()[0]
        a12 = models["cnn1d"].encoder[:4](t1).numpy()[0]
        a2 = models["cnn2d"].encoder[:2](t2).numpy()[0]
        a22 = models["cnn2d"].encoder[:4](t2).numpy()[0]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.4))
    axes[0].plot(hours, original[first, 0], lw=1)
    axes[0].set(ylabel="Suhu (°C)", title="Contoh representatif C1")
    im = axes[1].imshow(a1[:8], aspect="auto", cmap="viridis", extent=[0, 47, 8, 0])
    axes[1].set(xlabel="Posisi waktu (jam)", ylabel="Kanal aktivasi")
    fig.colorbar(im, ax=axes[1], shrink=.7, label="Aktivasi")
    save(fig, "cnn1d_temporal_response.png", "Aktivasi bersamaan dengan suhu; respons mencampur tujuh kanal sehingga bukan atribusi khusus suhu.",
         ["cnn1d.pt", "representative_windows.csv"])
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.5))
    for ax, values, title in zip(axes, (a1, a12), ("Layer 1: 32 kanal, 48 posisi", "Layer 2: 16 kanal, 24 posisi")):
        im = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set(title=title, xlabel="Posisi temporal", ylabel="Kanal")
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "cnn1d_intermediate_layers.png", "Dua kedalaman encoder; skala warna ditampilkan terpisah dan bukan satuan cuaca.", ["cnn1d.pt"])
    kernel = models["cnn1d"].encoder[0].weight.detach().numpy()
    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    limit = np.abs(kernel).max()
    im = ax.imshow(kernel.reshape(32, -1), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    for boundary in range(5, 35, 5):
        ax.axvline(boundary-.5, color="black", lw=.3)
    ax.set(xlabel="7 kanal × 5 koefisien temporal", ylabel="Filter keluaran", title="Bobot konvolusi awal CNN 1D")
    fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "cnn1d_kernels.png", "Bobot pada input terstandardisasi; batas tiap kanal diberi garis; bukan importance atau hubungan kausal.", ["cnn1d.pt"])
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.7), sharex=True)
    for c, ax, label in zip((0, 1), axes, ("Suhu (°C)", "Kelembapan (%)")):
        ax.plot(hours, original[first, c], label="Asli", lw=1)
        ax.plot(hours, reconstructed[first, c], label="Rekonstruksi", ls="--", lw=.9)
        ax.set(ylabel=label)
    axes[0].legend(ncol=2)
    axes[-1].set(xlabel="Jam sejak awal window")
    save(fig, "cnn1d_reconstruction_lines.png", "Rekonstruksi ditampilkan dalam satuan fisik pada window representatif; ini bukan prediksi masa depan.",
         ["cnn1d_test_reconstruction.npy", "scaler.json"])
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 6))
    for ax, values, title in zip(axes, (x2["test"][first, 0], rec2[first, 0], np.abs(rec2[first, 0]-x2["test"][first, 0])),
                                ("Input kemiripan", "Rekonstruksi CNN 2D", "Galat absolut")):
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set(title=title, xlabel="Indeks sampel waktu", ylabel="Indeks sampel waktu")
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "cnn2d_reconstruction_error.png", "Citra kemiripan dan galat pada skala 0–1 yang sama; diagonal/simetri berasal dari transformasi.",
         ["cnn2d_test_input.npy", "cnn2d_test_reconstruction.npy"])
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(a2[i], cmap="viridis", origin="lower")
        ax.set(title=f"Kanal {i+1}", xlabel="Indeks waktu", ylabel="Indeks waktu")
    save(fig, "cnn2d_feature_map_grid.png", "Enam feature maps CNN 2D; koordinat adalah pasangan waktu, bukan lokasi geografis.", ["cnn2d.pt"])
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.3))
    for ax, values, title in zip(axes, (a2[0], a22[0]), ("Layer 1, kanal 1", "Layer 2, kanal 1")):
        im = ax.imshow(values, cmap="viridis", origin="lower")
        ax.set(title=title, xlabel="Indeks waktu", ylabel="Indeks waktu")
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, "cnn2d_intermediate_layers.png", "Resolusi aktivasi 32×32 dan 16×16; skala warna per layer.", ["cnn2d.pt"])
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.3))
    axes[0].plot(hours, original[first, 0], lw=1)
    axes[0].set(xlabel="Jam dalam window", ylabel="Suhu (°C)")
    im = axes[1].imshow(x2["test"][first, 0], vmin=0, vmax=1, cmap="viridis", origin="lower")
    axes[1].set(xlabel="Indeks sampel waktu", ylabel="Indeks sampel waktu")
    fig.colorbar(im, ax=axes[1], shrink=.7)
    save(fig, "recurrence_example.png", "32 sampel aktual dari window 48 jam; min–max menghilangkan level suhu absolut; citra bukan recurrence plot biner.",
         ["cnn2d_test_input.npy", "representative_windows.csv"])


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
