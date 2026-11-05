"""Run EDA stages; add --with-cnn for representations, patterns, and the paper."""
import subprocess
import sys
from pathlib import Path
from common import raw_argument, resolve_raw
import stage1_validation
import stage2_eda
import stage3_preprocessing
import stage4_visualization
import stage5_patterns


def main():
    parser = raw_argument(__doc__)
    parser.add_argument("--with-cnn", action="store_true", help="Train CNN, analyze patterns, build and verify the report")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    raw = resolve_raw(args.raw)
    for module in (stage1_validation, stage2_eda, stage3_preprocessing,
                   stage4_visualization, stage5_patterns):
        module.run(raw)
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root/"src/verify_results.py"), "--raw", str(raw)], check=True)
    if args.with_cnn:
        for script, extra in [("stage6_cnn_experiments.py", ["--epochs", str(args.epochs)]),
                              ("stage7_visual_interpretation.py", []), ("build_ieee_report.py", []),
                              ("verify_cnn_results.py", [])]:
            subprocess.run([sys.executable, str(root/"src"/script), *extra], check=True, cwd=root)
        print("SELESAI: EDA, representasi CNN, pola, dan laporan IEEE. Tidak ada forecasting.")
    else:
        print("SELESAI: tahap 1–5; gunakan --with-cnn untuk eksperimen dan laporan lengkap.")


if __name__ == "__main__":
    main()
