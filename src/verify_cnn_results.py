"""Verify real-data lineage, trained checkpoints, clusters, figures, and paper.

No training, synthetic data, or fitting on held-out test data is performed here.
Run after build_ieee_report.py; renders pages for final visual inspection.
"""
from __future__ import annotations
import itertools
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import torch
from PIL import Image, ImageStat
from pypdf import PdfReader
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from common import CONTINUOUS, ROOT, REPORTS, json_write, load_clean, read_json, require_previous, resolve_raw, sha256
from stage6_cnn_experiments import AE1D, AE2D, OUT, evaluate
from stage7_visual_interpretation import check_manifest
from build_ieee_report import STEM


def main():
    torch.set_num_threads(2)
    checks = []
    def check(name, passed, detail):
        if not passed:
            raise AssertionError(f"{name}: {detail}")
        checks.append({"check": name, "passed": True, "detail": detail})
    raw = resolve_raw()
    require_previous(6, raw)
    for name in ("experiment_manifest.json", "pattern_manifest.json"):
        check_manifest(name)
    check("data_code_artifact_lineage", True, "EDA and CNN prerequisite hashes match")
    results, pattern = read_json(OUT/"results.json"), read_json(OUT/"pattern_results.json")
    frame = load_clean()
    meta = pd.read_csv(OUT/"window_metadata.csv")
    raw_values = frame[CONTINUOUS].to_numpy()
    train_mask = frame.time.dt.year.to_numpy() <= 2023
    scaler = read_json(OUT/"scaler.json")
    check("train_only_scaler", np.allclose(raw_values[train_mask].mean(0), scaler["mean"], rtol=0, atol=1e-10) and
          np.allclose(raw_values[train_mask].std(0), scaler["scale"], rtol=0, atol=1e-10), "Scaler equals 2022–2023 moments only")
    covered = {}
    for part, expected in (("train", 365), ("validation", 183), ("test", 182)):
        m = meta.loc[meta.split == part].reset_index(drop=True)
        indices = [set(range(int(s), int(s)+48)) for s in m.start_row]
        union = set().union(*indices)
        covered[part] = union
        x = np.load(OUT/f"cnn1d_{part}_input.npy")
        original = np.stack([raw_values[int(s):int(s)+48].T for s in m.start_row])
        inverse = x*np.array(scaler["scale"])[None, :, None]+np.array(scaler["mean"])[None, :, None]
        check(f"real_windows_{part}", len(m) == expected and len(union) == expected*48 and
              np.allclose(inverse, original, rtol=0, atol=2e-5), "Actual weather, exact chronological windows, no duplicated hours")
        check(f"timestamp_labels_{part}", all(frame.time.iloc[int(row.start_row)].isoformat() == row.start and
              frame.time.iloc[int(row.start_row)+47].isoformat() == row.end for row in m.itertuples()), "Saved boundaries match original timestamps")
    check("disjoint_splits_and_tail", not any(covered[a]&covered[b] for a, b in itertools.combinations(covered, 2)) and
          len(set(range(len(frame)))-set().union(*covered.values())) == 24, "Splits share no raw rows; incomplete final 24h remains outside CNN only")
    for tag, constructor in (("cnn1d", AE1D), ("cnn2d", AE2D)):
        selected = min(results[tag]["runs"], key=lambda r: r["validation_selection_mse"])
        check(f"validation_only_selection_{tag}", results[tag]["selected_seed"] == selected["seed"] and
              len(results[tag]["runs"]) >= 3, "Primary seed is validation minimum, not test minimum")
        model = constructor()
        checkpoint = torch.load(OUT/f"{tag}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        val_input = np.load(OUT/f"{tag}_validation_input.npy")
        val_rec, _ = evaluate(model, val_input)
        check(f"checkpoint_validation_mse_{tag}", np.isclose(np.mean((val_rec-val_input)**2),
              selected["validation_selection_mse"], rtol=0, atol=1e-6), "Loaded checkpoint reproduces its selection metric")
        x = np.load(OUT/f"{tag}_test_input.npy")
        r, z = evaluate(model, x)
        saved_rec = np.load(OUT/f"{tag}_test_reconstruction.npy")
        check(f"checkpoint_test_artifacts_{tag}", np.allclose(r, saved_rec, rtol=0, atol=1e-6) and
              np.allclose(z, np.load(OUT/f"{tag}_test_embedding.npy"), rtol=0, atol=1e-6), "Fresh inference matches saved reconstruction and embedding")
        check(f"independent_test_mse_{tag}", np.isclose(np.mean((r-x)**2), results[tag]["mse"]["test"], atol=1e-7), "Test MSE recomputed from tensors")
        if tag == "cnn1d":
            physical = r*np.array(scaler["scale"])[None, :, None]+np.array(scaler["mean"])[None, :, None]
            check("physical_reconstruction_domains", physical[:, [1, 3, 5, 6]].min() >= -1e-4 and
                  physical[:, [1, 5]].max() <= 100+1e-4, "RH/cloud in [0,100]; rain/wind nonnegative within numerical tolerance")
        for run in results[tag]["runs"]:
            history = pd.read_csv(OUT/f"{tag}_seed{run['seed']}_training.csv")
            recorded = float(history.loc[history.epoch == run["best_epoch"], "validation_mse"].iloc[0])
            check(f"best_epoch_{tag}_{run['seed']}", abs(recorded-run["validation_selection_mse"]) < 1e-7 and
                  run["validation_selection_mse"] <= history.validation_mse.min()+1.0001e-5,
                  "Selected epoch matches recorded history and min_delta policy")
    labels = np.load(OUT/"cnn1d_test_cluster_labels.npy")
    params = np.load(OUT/"cnn1d_cluster_parameters.npz")
    z = np.load(OUT/"cnn1d_test_embedding.npy")
    train_z = np.load(OUT/"cnn1d_train_embedding.npy")
    train_z64 = train_z.astype(float)
    check("train_latent_scaler", np.allclose(params["mean"], train_z64.mean(0), atol=1e-8) and
          np.allclose(params["scale"], train_z64.std(0), atol=1e-8), "Latent scaling uses train moments only")
    latent_scaler = StandardScaler().fit(train_z)
    frozen_fit = KMeans(n_clusters=pattern["primary"]["k"], n_init=20, random_state=42).fit(latent_scaler.transform(train_z))
    check("train_only_cluster_centroids", np.allclose(frozen_fit.cluster_centers_, params["centers"], atol=1e-5),
          "Re-fitting only train embeddings reproduces saved centroids")
    transformed = (z-params["mean"])/params["scale"]
    nearest = ((transformed[:, None]-params["centers"][None])**2).sum(2).argmin(1)
    order = params["order"].tolist()
    expected_labels = np.array([order.index(int(c)) for c in nearest])
    check("saved_centroid_assignment", np.array_equal(labels, expected_labels), "All labels follow frozen train centroids")
    validation = pd.read_csv(OUT/"cluster_k_validation.csv")
    valid = validation.loc[(validation.representation == "cnn1d") & validation.eligible]
    best = valid.loc[valid.validation_silhouette.idxmax()]
    check("validation_only_k", int(best.k) == pattern["primary"]["k"], "k maximizes eligible validation silhouette")
    assigned = pd.read_csv(OUT/"window_cluster_assignments.csv")
    profiles = pd.read_csv(OUT/"cluster_weather_profiles.csv")
    for row in profiles.itertuples():
        windows = assigned.loc[(assigned.split == row.split)&(assigned.cluster == row.cluster)]
        check(f"physical_profile_{row.split}_C{row.cluster}", len(windows) == row.n_windows and
              all(np.isclose(windows[name].mean(), getattr(row, name), rtol=0, atol=1e-7)
                  for name in ("temperature_mean", "temperature_range", "humidity_mean", "wet_pct", "cloud_mean", "rain_sum")),
              "Profile independently agrees with underlying window descriptors")
    agreement = pd.read_csv(OUT/"encoder_seed_window_agreement.csv")
    agreement_columns = [f"seed_{s}_cluster" for s in results["configuration"]["seeds"]]
    agreement_mask = np.ptp(agreement[agreement_columns].to_numpy(), axis=1) == 0
    check("consensus_without_case_removal", np.array_equal(agreement_mask, agreement.all_three_seeds_agree) and
          int(agreement_mask.sum()) == pattern["three_seed_agreement"]["n_windows"] and len(agreement) == len(labels),
          "Consensus count is correct; all 182 windows retained in primary evaluation")
    reference_train = np.load(OUT/"cnn1d_train_cluster_labels.npy")
    for seed in results["configuration"]["seeds"]:
        train_seed = np.load(OUT/f"cnn1d_seed{seed}_train_embedding.npy")
        test_seed = np.load(OUT/f"cnn1d_seed{seed}_test_embedding.npy")
        latent_scaler = StandardScaler().fit(train_seed)
        fixed_k = pattern["primary"]["k"]
        grouping = KMeans(n_clusters=fixed_k, n_init=20, random_state=42).fit(latent_scaler.transform(train_seed))
        labels_train = grouping.labels_
        contingency = np.zeros((fixed_k, fixed_k), dtype=int)
        np.add.at(contingency, (reference_train, labels_train), 1)
        rows, columns = linear_sum_assignment(-contingency)
        mapping = dict(zip(columns, rows))
        expected = np.array([mapping[c]+1 for c in grouping.predict(latent_scaler.transform(test_seed))])
        check(f"train_only_alignment_seed_{seed}", np.array_equal(expected, agreement[f"seed_{seed}_cluster"]),
              "Cross-seed label alignment uses train correspondence, not test labels")
    stability = pd.read_csv(OUT/"cluster_stability.csv")
    for first, second in itertools.combinations(results["configuration"]["seeds"], 2):
        score = adjusted_rand_score(agreement[f"seed_{first}_cluster"], agreement[f"seed_{second}_cluster"])
        actual = stability.loc[(stability.representation == "cnn1d")&(stability.stability_type == "encoder_seed")&
                               (stability.seed_a == first)&(stability.seed_b == second), "test_ARI"].iloc[0]
        check(f"encoder_ARI_{first}_{second}", abs(score-actual) < 1e-9, "ARI recomputed from per-window labels")
    figures = read_json(OUT/"visual_interpretation_manifest.json")["figures"]
    for item in figures:
        with Image.open(OUT/item["file"]) as im:
            check("figure_"+item["file"], im.format == "PNG" and all(abs(v-300)<.1 for v in im.info["dpi"]) and
                  max(ImageStat.Stat(im.convert("RGB")).stddev) > 5, "Valid nonempty 300-dpi PNG")
    report = read_json(REPORTS/"report_manifest.json")
    check("paper_builder_and_pattern_lineage", report["builder_sha256"] == sha256(ROOT/"src/build_ieee_report.py") and
          report["pattern_manifest_sha256"] == sha256(OUT/"pattern_manifest.json"), "Paper built from current code and pattern results")
    for name, digest in report["artifacts"].items():
        check("paper_hash_"+name, sha256(ROOT/name) == digest, "Paper/source artifact hash matches")
    pdf_path = REPORTS/f"{STEM}.pdf"
    reader = PdfReader(pdf_path)
    text = "\n".join(p.extract_text() for p in reader.pages)
    current_tokens = ("13524022", "13524063", f"{pattern['three_seed_agreement']['percentage']:.2f}%",
                      f"{results['cnn1d']['mse']['test']:.5f}", f"{pattern['encoder_seed_ARI']['cnn1d']['mean']:.3f}")
    check("paper_contains_current_results", all(token in text for token in current_tokens) and
          "0,8924" not in text and "0.8924" not in text, "Current findings and author identifiers present; obsolete smoke-test result absent")
    check("paper_no_placeholder", re.search(r"\b(?:TODO|TBD|NotImplemented|PLACEHOLDER)\b", text) is None,
          "No unfinished placeholders")
    embedded = []
    for page in reader.pages:
        for font in page["/Resources"].get("/Font", {}).values():
            obj = font.get_object()
            descriptor = obj.get("/FontDescriptor")
            if descriptor:
                embedded.append("/FontFile2" in descriptor.get_object())
    check("paper_fonts_and_A4", embedded and all(embedded) and all(abs(float(p.mediabox.width)-595.276)<1 for p in reader.pages),
          "Embedded TrueType fonts and A4 pages")
    preview = ROOT/"output/report_preview"
    preview.mkdir(exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    outside = 0
    for number in range(len(document)):
        page = document[number]
        page.render(scale=1.3).to_pil().save(preview/f"page_{number+1:02d}.png")
        textpage = page.get_textpage()
        for index in range(textpage.count_chars()):
            left, bottom, right, top = textpage.get_charbox(index)
            if right-left > .1 and top-bottom > .1 and (left < -1 or bottom < -1 or right > page.get_width()+1 or top > page.get_height()+1):
                outside += 1
        textpage.close()
        page.close()
    document.close()
    check("all_pdf_pages_render_and_fit", outside == 0 and len(reader.pages) >= 5, "All pages rendered; no text bounding boxes outside page")
    check("raw_unchanged", sha256(raw) == results["configuration"]["raw_sha256"], "Original weather CSV unchanged")
    json_write(OUT/"verification_results.json", {"status": "passed", "scope": "cnn_patterns_and_submission_report",
        "checks_count": len(checks), "checks": checks, "pdf_pages": len(reader.pages), "figures": len(figures),
        "raw_sha256": sha256(raw), "pdf_sha256": sha256(pdf_path), "forecasting_performed": False})
    print(f"CNN/PAPER VERIFIED: {len(checks)} checks, {len(reader.pages)} PDF pages, {len(figures)} figures.")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
