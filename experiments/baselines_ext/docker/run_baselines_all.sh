#!/usr/bin/env bash
# Build each baseline's Docker image and run it on the five in-domain datasets.
# Results go to experiments/out/baselines_ext.csv in the bind-mounted repo.
#
# Usage (from the repo root):
#     bash experiments/baselines_ext/docker/run_baselines_all.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA="$REPO/datasets"
OUT="$REPO/experiments/out/baselines_ext"
mkdir -p "$OUT"

DATASETS=(abide eeg_eye_state synth_neuro clinvar_sample pfam_proteins)

# ---------------------------------------------------------------- SPuManTE
echo "=== building spumante image ==="
docker build -f experiments/baselines_ext/docker/Dockerfile.spumante \
             -t same-bench/spumante:latest \
             experiments/baselines_ext/docker
for ds in "${DATASETS[@]}"; do
  echo "--- spumante on $ds ---"
  docker run --rm \
    -v "$DATA:/data:ro" \
    -v "$OUT:/out" \
    same-bench/spumante:latest \
    --input /data/"$ds".csv --dataset "$ds" --alpha 0.05 \
    --out /out/spumante_"$ds".csv \
    || echo "  (spumante failed on $ds)"
done

# ---------------------------------------------------------------- LAMP
echo "=== building lamp image ==="
docker build -f experiments/baselines_ext/docker/Dockerfile.lamp \
             -t same-bench/lamp:latest \
             experiments/baselines_ext/docker
for ds in "${DATASETS[@]}"; do
  echo "--- lamp on $ds ---"
  docker run --rm \
    -v "$DATA:/data:ro" \
    -v "$OUT:/out" \
    same-bench/lamp:latest \
    --input /data/"$ds".csv --dataset "$ds" --alpha 0.05 \
    --out /out/lamp_"$ds".csv \
    || echo "  (lamp failed on $ds)"
done

# ---------------------------------------------------------------- WYlight
echo "=== building wylight image ==="
docker build -f experiments/baselines_ext/docker/Dockerfile.wylight \
             -t same-bench/wylight:latest \
             experiments/baselines_ext/docker
for ds in "${DATASETS[@]}"; do
  echo "--- wylight on $ds ---"
  docker run --rm \
    -v "$DATA:/data:ro" \
    -v "$OUT:/out" \
    same-bench/wylight:latest \
    --input /data/"$ds".csv --dataset "$ds" --alpha 0.05 \
    --out /out/wylight_"$ds".csv \
    || echo "  (wylight failed on $ds)"
done

# ---------------------------------------------------------------- Kingfisher
echo "=== building kingfisher image ==="
docker build -f experiments/baselines_ext/docker/Dockerfile.kingfisher \
             -t same-bench/kingfisher:latest \
             experiments/baselines_ext/docker
for ds in "${DATASETS[@]}"; do
  echo "--- kingfisher on $ds ---"
  docker run --rm \
    -v "$DATA:/data:ro" \
    -v "$OUT:/out" \
    same-bench/kingfisher:latest \
    --input /data/"$ds".csv --dataset "$ds" --alpha 0.05 \
    --out /out/kingfisher_"$ds".csv \
    || echo "  (kingfisher failed on $ds)"
done

# ---------------------------------------------------------------- Merge
echo "=== merging per-dataset per-baseline CSVs ==="
python3 "$REPO/experiments/baselines_ext/docker/merge_results.py" "$OUT"
echo "done. Merged CSV at $OUT/all_baselines_ext.csv"
