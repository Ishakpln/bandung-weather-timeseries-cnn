"""Train CNN representations on real weather windows; no forecasting.

python src/stage6_cnn_experiments.py --epochs 150 --seeds 42 43 44
Checkpoints and primary seed are selected on validation reconstruction MSE.
"""
from __future__ import annotations
import argparse
import copy
import platform
import random
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common import (CONTINUOUS, ROOT, json_write, load_clean, require_previous, resolve_raw, sha256)

OUT = ROOT / "output" / "cnn"
WINDOW = 48
ARCHITECTURE = "pattern_v3_physical_bounds"


class AE1D(nn.Module):
    """48x7 -> 12x8 bottleneck -> 48x7; temporal positions are retained."""
    def __init__(self, channels=7, mean=None, scale=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2), nn.LeakyReLU(.1),
            nn.Conv1d(32, 16, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.Conv1d(16, 8, 4, stride=2, padding=1), nn.LeakyReLU(.1))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.ConvTranspose1d(16, 32, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.Conv1d(32, channels, 5, padding=2))
        # Known physical domains, transformed using train moments only.
        # These constrain model outputs, not observations or input outliers.
        center = np.zeros(channels) if mean is None else np.asarray(mean)
        spread = np.ones(channels) if scale is None else np.asarray(scale)
        low = np.array([-np.inf, 0, -np.inf, 0, -np.inf, 0, 0])
        high = np.array([np.inf, 100, np.inf, np.inf, np.inf, 100, np.inf])
        self.register_buffer("output_lower", torch.tensor((low-center)/spread, dtype=torch.float32)[None, :, None])
        self.register_buffer("output_upper", torch.tensor((high-center)/spread, dtype=torch.float32)[None, :, None])

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        bounded = torch.maximum(torch.minimum(decoded, self.output_upper), self.output_lower)
        return bounded, z


class AE2D(nn.Module):
    """Supplementary temperature-similarity autoencoder; separate target space."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(8, 16, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(16, 8, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(8, 8, 4, stride=2, padding=1), nn.LeakyReLU(.1))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.LeakyReLU(.1),
            nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def recurrence(x):
    """Continuous similarity of 32 real samples; per-window normalization.

    This is not a binary recurrence plot. No new weather observations are made.
    Absolute temperature level is intentionally removed by the transformation.
    """
    ids = np.linspace(0, len(x)-1, 32).round().astype(int)
    a = np.asarray(x[ids], dtype=np.float32)
    a = (a-a.min()) / (np.ptp(a)+1e-6)
    return np.exp(-np.abs(a[:, None]-a[None, :])/.25).astype("float32")


def prepare_data():
    require_previous(6, resolve_raw())
    frame = load_clean()
    if not frame.time.is_unique or not frame.time.is_monotonic_increasing:
        raise ValueError("Timestamp invalid")
    if not frame.time.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Hourly grid incomplete")
    raw = frame[CONTINUOUS].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite input")
    year = frame.time.dt.year.to_numpy()
    masks = {"train": year <= 2023, "validation": year == 2024, "test": year == 2025}
    scaler = StandardScaler().fit(raw[masks["train"]])
    scaled = scaler.transform(raw).astype("float32")
    starts, x1, x2, metadata, coverage = {}, {}, {}, [], []
    for part, mask in masks.items():
        idx = np.flatnonzero(mask)
        if not len(idx) or not np.all(np.diff(idx) == 1):
            raise ValueError(f"Invalid split {part}")
        starts[part] = np.arange(idx[0], idx[-1]-WINDOW+2, WINDOW)
        x1[part] = np.stack([scaled[s:s+WINDOW].T for s in starts[part]])
        x2[part] = np.stack([recurrence(raw[s:s+WINDOW, 0]) for s in starts[part]])[:, None]
        coverage.append({"split": part, "rows": len(idx), "windows": len(starts[part]),
                         "used_hours": len(starts[part])*WINDOW,
                         "tail_hours_not_windowed": len(idx)-len(starts[part])*WINDOW})
        for number, s in enumerate(starts[part]):
            if s+WINDOW-1 > idx[-1]:
                raise AssertionError("Window crosses split")
            b = frame.iloc[s:s+WINDOW]
            metadata.append({"split": part, "window_id": number, "start_row": int(s),
                "start": b.time.iloc[0].isoformat(), "end": b.time.iloc[-1].isoformat(),
                "month": int(b.month.iloc[0]), "year": int(b.year.iloc[0]),
                "temperature_mean": b.temperature_2m.mean(),
                "temperature_range": b.temperature_2m.max()-b.temperature_2m.min(),
                "temperature_std": b.temperature_2m.std(ddof=0),
                "humidity_mean": b.relative_humidity_2m.mean(), "rain_sum": b.precipitation.sum(),
                "wet_pct": b.precipitation.gt(0).mean()*100, "cloud_mean": b.cloud_cover.mean(),
                "wind_mean": b.wind_speed_10m.mean()})
    return frame, scaler, starts, x1, x2, pd.DataFrame(metadata), pd.DataFrame(coverage)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, array, batch=128):
    model.eval()
    rec, latent = [], []
    with torch.no_grad():
        for x in torch.from_numpy(array).split(batch):
            r, z = model(x)
            rec.append(r.numpy())
            latent.append(z.flatten(1).numpy())
    return np.concatenate(rec), np.concatenate(latent)


def train(model, arrays, seed, epochs, patience, batch, lr, tag):
    loader = DataLoader(TensorDataset(torch.from_numpy(arrays["train"])), batch_size=batch,
        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_epoch, wait, checkpoint, history = float("inf"), 0, 0, None, []
    for epoch in range(1, epochs+1):
        model.train()
        total, elements = 0., 0
        for (x,) in loader:
            optimizer.zero_grad(set_to_none=True)
            r, _ = model(x)
            loss = (r-x).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item()*x.numel()
            elements += x.numel()
        r, _ = evaluate(model, arrays["validation"])
        val = float(np.mean((r-arrays["validation"])**2))
        history.append({"epoch": epoch, "train_mse": total/elements, "validation_mse": val})
        if val < best-1e-5:
            best, best_epoch, wait = val, epoch, 0
            checkpoint = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"{tag} seed={seed} epoch={epoch}: train={total/elements:.5f}, val={val:.5f}", flush=True)
        if wait >= patience:
            break
    model.load_state_dict(checkpoint)
    return pd.DataFrame(history), best_epoch, best


def metrics(truth, reconstructed, tag, scaler):
    error = reconstructed-truth
    result = {"mse": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error)))}
    if tag == "cnn1d":
        rows = []
        for c, name in enumerate(CONTINUOUS):
            correlations = [np.corrcoef(a.astype(float), b.astype(float))[0, 1]
                            for a, b in zip(truth[:, c], reconstructed[:, c])
                            if np.ptp(a) > 1e-8 and np.ptp(b) > 1e-8]
            rows.append({"feature": name, "mse_standardized": float(np.mean(error[:, c]**2)),
                "mae_original_units": float(np.mean(np.abs(error[:, c]))*scaler.scale_[c]),
                "rmse_original_units": float(np.sqrt(np.mean(error[:, c]**2))*scaler.scale_[c]),
                "mean_window_temporal_correlation": float(np.mean(correlations)) if correlations else None,
                "nonconstant_windows_for_correlation": len(correlations)})
        result["per_feature"] = rows
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--reuse-checkpoints", action="store_true",
                   help="Recover evaluation after an interrupted run; validate checkpoint lineage")
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or len(set(args.seeds)) < 3:
        p.error("Positive epochs/patience and at least three distinct seeds required")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"experiment_manifest.json").unlink(missing_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frame, scaler, starts, x1, x2, metadata, coverage = prepare_data()
    raw_hash = sha256(resolve_raw())
    artifacts = []
    def register(name):
        artifacts.append(name)
        return OUT/name
    def save_array(name, array):
        np.save(register(name), array)
    metadata.to_csv(register("window_metadata.csv"), index=False)
    coverage.to_csv(register("window_coverage.csv"), index=False)
    json_write(register("scaler.json"), {"columns": CONTINUOUS, "mean": scaler.mean_.tolist(),
               "scale": scaler.scale_.tolist(), "train_period": "2022-2023"})
    for tag, arrays in (("cnn1d", x1), ("cnn2d", x2)):
        for part, array in arrays.items():
            save_array(f"{tag}_{part}_input.npy", array)
    results = {"configuration": {"objective": "representation_learning_and_pattern_analysis",
        "forecasting": False, "architecture_version": ARCHITECTURE, "window_hours": WINDOW,
        "stride_hours": WINDOW, "features": CONTINUOUS, "seeds": args.seeds,
        "epochs_max": args.epochs, "patience": args.patience, "min_delta": 1e-5,
        "batch": args.batch, "lr": args.lr, "optimizer": "Adam", "weight_decay": 1e-5,
        "gradient_clip_norm": 5, "torch": torch.__version__, "sklearn": sklearn.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "threads": 2,
        "deterministic_algorithms": True, "raw_sha256": raw_hash,
        "chronological_split": {"train": "2022-2023", "validation": "2024", "test": "2025"},
        "selection": "minimum validation reconstruction MSE; test never selects model",
        "test_exposure": "2025 examined during prior EDA; results remain exploratory",
        "recurrence": "32 real samples; exp(-abs(per-window-minmax difference)/0.25)",
        "output_constraints_1d": "RH/cloud in [0,100]%; precipitation/wind >=0; bounds scaled using train moments",
        "note": "1D and 2D MSE have different target spaces and must not be ranked together"}}
    for tag, constructor, arrays in (("cnn1d", AE1D, x1), ("cnn2d", AE2D, x2)):
        runs = []
        for seed in args.seeds:
            set_seed(seed)
            model = AE1D(mean=scaler.mean_, scale=scaler.scale_) if tag == "cnn1d" else constructor()
            if seed == args.seeds[0]:
                for part in arrays:
                    _, z = evaluate(model, arrays[part])
                    save_array(f"{tag}_untrained_{part}_embedding.npy", z)
            started = time.perf_counter()
            if args.reuse_checkpoints:
                saved = torch.load(OUT/f"{tag}_seed{seed}.pt", map_location="cpu", weights_only=True)
                if (saved["architecture_version"] != ARCHITECTURE or saved["raw_sha256"] != raw_hash
                        or saved["seed"] != seed):
                    raise RuntimeError("Checkpoint lineage mismatch")
                model.load_state_dict(saved["state_dict"])
                history = pd.read_csv(OUT/f"{tag}_seed{seed}_training.csv")
                epoch, val = saved["best_epoch"], saved["validation_mse"]
            else:
                history, epoch, val = train(model, arrays, seed, args.epochs, args.patience, args.batch, args.lr, tag)
            history.to_csv(register(f"{tag}_seed{seed}_training.csv"), index=False)
            torch.save({"state_dict": model.state_dict(), "architecture_version": ARCHITECTURE,
                "features": CONTINUOUS, "window_hours": WINDOW, "seed": seed,
                "best_epoch": epoch, "validation_mse": val, "raw_sha256": raw_hash},
                register(f"{tag}_seed{seed}.pt"))
            run = {"seed": seed, "epochs_completed": len(history), "best_epoch": epoch,
                   "validation_selection_mse": val,
                   "training_elapsed_seconds": None if args.reuse_checkpoints else round(time.perf_counter()-started, 3), "metrics": {}}
            for part in arrays:
                r, z = evaluate(model, arrays[part])
                run["metrics"][part] = metrics(arrays[part], r, tag, scaler)
                save_array(f"{tag}_seed{seed}_{part}_embedding.npy", z)
                if part == "test":
                    save_array(f"{tag}_seed{seed}_test_reconstruction.npy", r)
            runs.append(run)
            print(f"{tag} seed={seed} complete: best epoch={epoch}, val={val:.5f}, test={run['metrics']['test']['mse']:.5f}", flush=True)
        selected = min(runs, key=lambda r: r["validation_selection_mse"])
        seed = selected["seed"]
        for suffix in [".pt", "_training.csv"]:
            shutil.copy2(OUT/f"{tag}_seed{seed}{suffix}", register(f"{tag}{suffix}"))
        for part in arrays:
            shutil.copy2(OUT/f"{tag}_seed{seed}_{part}_embedding.npy", register(f"{tag}_{part}_embedding.npy"))
        shutil.copy2(OUT/f"{tag}_seed{seed}_test_reconstruction.npy", register(f"{tag}_test_reconstruction.npy"))
        results[tag] = {"selected_seed": seed, "samples": {k: len(v) for k, v in arrays.items()},
            "selected_metrics": selected["metrics"], "runs": runs,
            "mse": {part: selected["metrics"][part]["mse"] for part in arrays},
            "test_mse_mean_across_seeds": float(np.mean([r["metrics"]["test"]["mse"] for r in runs])),
            "test_mse_std_across_seeds": float(np.std([r["metrics"]["test"]["mse"] for r in runs], ddof=1))}
    json_write(register("results.json"), results)
    if sha256(resolve_raw()) != raw_hash:
        raise AssertionError("Raw data changed")
    json_write(OUT/"experiment_manifest.json", {"status": "passed", "raw_sha256": raw_hash,
        "processed_sha256": sha256(ROOT/"data/processed/bandung_weather_hourly_clean.csv"),
        "code": {"src/stage6_cnn_experiments.py": sha256(Path(__file__))},
        "artifacts": {name: sha256(OUT/name) for name in sorted(artifacts)}})
    print("STAGE 6 COMPLETE: representations, three seeds, chronological windows, train-only scaler.", flush=True)


if __name__ == "__main__":
    main()
