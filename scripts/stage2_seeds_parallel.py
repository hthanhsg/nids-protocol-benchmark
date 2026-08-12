#!/usr/bin/env bash
# Stage 2 (parallel variant) — seed variance, written to its OWN directory tree
# so it can run alongside Stage 1 without two processes appending to the same
# results.csv. Merge afterwards with merge_results.py.
#
#   MPH=4 GPU=1 ./stage2_seeds_parallel.sh
#
# MPH must match Stage 1 (4), or seeds 1/2 sample differently from seed 42 and
# the "seed variance" is contaminated by a sampling change. Cache is keyed on
# min_per_host, so warm cache_seeds with the same value first:
#
#   python warm_cache.py --data-dir data --cache-dir cache_seeds \
#       --subsample 1000000 --subsample-by entity --min-per-host 4 --seeds 1 2
set -u
GPU=${GPU:-1}
DATA=${DATA:-data}
N=${N:-1000000}
MPH=${MPH:-4}
mkdir -p logs

for S in 1 2; do
  for T in binary multiclass; do
    echo "=== seed $S / $T  (min_per_host=$MPH)  $(date '+%F %T') ==="
    CUDA_VISIBLE_DEVICES=$GPU python -m nidsbench.run \
      --data-dir "$DATA" --out "results_v2_seeds/${T}" --tasks "$T" \
      --models xgboost mlp transformer autoencoder \
      --subsample "$N" --subsample-by entity --min-per-host "$MPH" \
      --seeds "$S" --cache-dir cache_seeds \
      2>&1 | tee -a "logs/seed${S}_${T}.log"
  done
done
echo "STAGE 2 COMPLETE -> results_v2_seeds/"