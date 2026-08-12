#!/usr/bin/env bash
# Stage 4 — leakage probe. Labels are permuted before splitting, so test
# macro-F1 must fall to chance (~0.5 binary). Anything higher means the
# pipeline leaks. ~30 min. Separate output directory on purpose.
set -u
GPU=${GPU:-0}
DATA=${DATA:-data}
mkdir -p logs

CUDA_VISIBLE_DEVICES=$GPU python -m nidsbench.run \
  --data-dir "$DATA" --out results_v2/probe \
  --tasks binary --splits random --models mlp xgboost \
  --subsample 200000 --subsample-by entity --shuffle-labels \
  --cache-dir cache_probe \
  2>&1 | tee -a logs/probe.log

echo
echo "STAGE 4 COMPLETE. Every macro-F1 above should sit near chance."
echo "Send results_v2/probe/results.csv"
