"""Run stages 1 through 5 in order; stop immediately if a stage fails."""
from common import raw_argument, resolve_raw
import stage1_validation
import stage2_eda
import stage3_preprocessing
import stage4_visualization
import stage5_patterns


def main():
    args = raw_argument(__doc__).parse_args()
    raw = resolve_raw(args.raw)
    for module in (stage1_validation, stage2_eda, stage3_preprocessing,
                   stage4_visualization, stage5_patterns):
        module.run(raw)
    print("SELESAI: tahap 1–5. Tidak ada training, arsitektur, atau prediksi CNN.")


if __name__ == "__main__":
    main()
