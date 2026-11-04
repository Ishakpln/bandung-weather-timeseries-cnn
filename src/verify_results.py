"""Check scientific integrity and save a verification summary; do not rerun analysis."""
from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from scipy.stats import spearmanr

import common
from common import (FIGURES, MANIFESTS, PROCESSED, REPORTS, TABLES, WEATHER,
                    json_write, raw_argument, read_json, resolve_raw)


def run(raw=None):
    raw = resolve_raw(raw)
    checks = []

    def check(name, condition, detail):
        if not condition:
            raise AssertionError(f"{name}: {detail}")
        checks.append({"check": name, "passed": True, "detail": detail})

    common.require_previous(6, raw)
    check("provenance_and_artifact_hashes", True, "Semua manifest tahap 1–5, kode, raw, dan artefak cocok")
    # Independent reader rather than the production loader.
    with raw.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.reader(handle))
    header = next(i for i, row in enumerate(raw_rows) if row and row[0] == "time")
    observations = raw_rows[header + 1:]
    numeric_raw = np.array([[float(x) for x in row[1:]] for row in observations])
    original_times = [datetime.fromisoformat(row[0]) for row in observations]
    clean = pd.read_csv(PROCESSED / "bandung_weather_hourly_clean.csv")
    parsed = [datetime.fromisoformat(x) for x in clean.time]
    local = [x.replace(tzinfo=None) for x in parsed]
    check("shape_and_raw_values", clean.shape == (35064, 23) and
          np.array_equal(numeric_raw, clean[WEATHER].to_numpy()),
          "35.064×23; seluruh 280.512 nilai cuaca identik")
    check("hourly_period_and_timezone", local == original_times and
          local[0] == datetime(2022, 1, 1) and local[-1] == datetime(2025, 12, 31, 23) and
          all((b-a) == timedelta(hours=1) for a, b in zip(local, local[1:])) and
          all(t.utcoffset() == timedelta(hours=7) for t in parsed), "Hourly utuh; offset +07:00")
    check("missing_infinite_and_leap_day", not clean.isna().any().any() and
          np.isfinite(clean[WEATHER]).all().all() and
          sum(t.date().isoformat() == "2024-02-29" for t in local) == 24,
          "Tidak ada missing/inf; 29 Februari 2024 mencakup 24 jam")
    calendar = {"hour": [t.hour for t in local], "day": [t.day for t in local],
                "month": [t.month for t in local], "year": [t.year for t in local],
                "day_of_week": [t.weekday() for t in local],
                "day_of_year": [t.timetuple().tm_yday for t in local]}
    check("calendar_features", all(np.array_equal(clean[name], values) for name, values in calendar.items()) and
          np.array_equal(clean.is_weekend, [int(t.weekday() >= 5) for t in local]) and
          list(clean.date) == [t.date().isoformat() for t in local], "Kalender cocok dengan datetime Python")
    check("cyclical_features", all(np.allclose(clean[f"{stem}_sin"]**2 + clean[f"{stem}_cos"]**2, 1, atol=1e-14)
                                   for stem in ("hour", "month", "day_of_year")), "Sin²+cos²=1")
    daily = pd.read_csv(TABLES / "daily_summary.csv")
    monthly = pd.read_csv(TABLES / "monthly_summary.csv")
    annual = pd.read_csv(TABLES / "annual_summary.csv")
    rain_total = math.fsum(numeric_raw[:, 3])
    check("aggregation_conservation", len(daily) == 1461 and len(monthly) == 48 and len(annual) == 4 and
          daily.n_hours.eq(24).all() and all(int(t.n_hours.sum()) == len(clean) and
          math.isclose(math.fsum(t.precipitation_sum), rain_total, rel_tol=0, abs_tol=1e-7)
          for t in (daily, monthly, annual)), "1.461 hari, 48 bulan, 4 tahun; jam dan hujan konsisten")
    pearson = pd.read_csv(TABLES / "correlation_pearson.csv", index_col=0).to_numpy()
    spearman = pd.read_csv(TABLES / "correlation_spearman.csv", index_col=0).to_numpy()
    check("independent_correlations", np.allclose(pearson, np.corrcoef(numeric_raw[:, :7], rowvar=False), atol=1e-8) and
          np.allclose(spearman, spearmanr(numeric_raw[:, :7]).statistic, atol=1e-8),
          "Pearson NumPy dan Spearman SciPy cocok dengan tabel Pandas")
    evidence = read_json(TABLES / "pattern_evidence.json")
    scores = pd.read_csv(TABLES / "temperature_anomaly_scores.csv")
    candidates = pd.read_csv(TABLES / "temperature_anomaly_candidates.csv")
    check("insight_evidence", math.isclose(evidence["temperature_mean_C"], numeric_raw[:, 0].mean(), abs_tol=1e-10) and
          math.isclose(evidence["rain_total_timestamp_grouped_mm"], rain_total, abs_tol=1e-7) and
          evidence["wet_hours_gt0"] == int((numeric_raw[:, 3] > 0).sum()) and
          len(candidates) == evidence["anomaly_candidates"] == int(scores.candidate.sum()) and
          np.array_equal(scores.candidate, scores.modified_z.abs() > 3.5), "Angka insight dan kandidat konsisten")
    split = pd.read_csv(TABLES / "cnn_recommended_split_plan.csv")
    boundaries = [(datetime.fromisoformat(a), datetime.fromisoformat(b))
                  for a, b in zip(split.start_local, split.end_local)]
    check("chronological_split_plan", list(split.rows) == [17520, 8784, 8760] and split.rows.sum() == len(clean) and
          all((boundaries[i+1][0] - boundaries[i][1]) == timedelta(hours=1) for i in range(2)),
          "Rencana 2022–2023 / 2024 / 2025 lengkap dan tidak bertumpang tindih")
    figures = read_json(FIGURES / "figure_manifest.json")
    for figure in figures:
        with Image.open(FIGURES / figure["filename"]) as pic:
            check(f"png_{figure['filename']}", pic.format == "PNG" and
                  pic.size == (figure["width_px"], figure["height_px"]) and
                  pic.width >= 1600 and pic.height >= 1200 and
                  all(abs(dpi-300) < .1 for dpi in pic.info.get("dpi", (0, 0))) and
                  max(ImageStat.Stat(pic.convert("RGB")).stddev) > 5,
                  f"{pic.width}×{pic.height} px; 300 dpi; tidak kosong")
    check("required_deliverables", len(figures) == 15 and all((REPORTS / name).is_file() for name in
          ["01_data_validation.md", "02_eda.md", "03_preprocessing.md", "04_visualization_guide.md",
           "05_pattern_analysis_insight.md", "06_cnn_readiness.md"]), "15 PNG dan enam laporan tersedia")

    def rejected(action):
        try:
            action()
        except RuntimeError:
            return True
        return False

    absent = MANIFESTS / "__nonexistent_gate_probe__"
    if absent.exists():
        raise AssertionError("Path probe gate seharusnya tidak ada")
    with patch.object(common, "MANIFESTS", absent):
        check("gate_missing_prerequisite", rejected(lambda: common.require_previous(2, raw)),
              "Tahap 2 ditolak tanpa keberhasilan tahap 1")
    with patch.object(common, "sha256", return_value="0" * 64):
        check("gate_changed_raw", rejected(lambda: common.require_previous(2, raw)), "Hash raw berubah ditolak")
    with patch.object(common, "source_signature", return_value={}):
        check("gate_changed_code", rejected(lambda: common.require_previous(2, raw)), "Kode prasyarat berubah ditolak")
    result = {"status": "passed", "checks_count": len(checks), "checks": checks,
              "raw_sha256": common.sha256(raw), "cnn_training_performed": False}
    json_write(TABLES / "verification_results.json", result)
    print(f"VERIFIKASI LULUS: {len(checks)} pemeriksaan; raw tetap; tahap 1–5 lengkap.")
    return result


if __name__ == "__main__":
    run(raw_argument(__doc__).parse_args().raw)
