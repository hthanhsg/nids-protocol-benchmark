#!/usr/bin/env bash
# Stage 1 across two GPUs, after the cache has been warmed.
#
#   1) python warm_cache.py --data-dir data --cache-dir cache_entity \
#          --subsample 1000000 --subsample-by entity
#   2) ./run_parallel.sh
#
# binary goes to the smaller GPU, multiclass to the larger (more classes ->
# slightly wider heads). Both read the shared parquet cache, so neither touches
# the raw CSVs and peak RAM stays low. Separate --out trees mean the two
# processes never append to the same results.csv.
set -u
DATA=${DATA:-data}
N=${N:-1000000}
MPH=${MPH:-4}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
mkdir -p logs

CUDA_VISIBLE_DEVICES=$GPU_A python -m nidsbench.run \
  --data-dir "$DATA" --out results_v2/binary --tasks binary \
  --models xgboost mlp transformer autoencoder \
  --subsample "$N" --subsample-by entity --min-per-host "$MPH" \
  --cache-dir cache_entity --save-predictions \
  > logs/binary.log 2>&1 &
PID_A=$!
echo "binary   -> GPU $GPU_A  (pid $PID_A)  logs/binary.log"

CUDA_VISIBLE_DEVICES=$GPU_B python -m nidsbench.run \
  --data-dir "$DATA" --out results_v2/multiclass --tasks multiclass \
  --models xgboost mlp transformer autoencoder \
  --subsample "$N" --subsample-by entity --min-per-host "$MPH" \
  --cache-dir cache_entity --save-predictions \
  > logs/multiclass.log 2>&1 &
PID_B=$!
echo "multi    -> GPU $GPU_B  (pid $PID_B)  logs/multiclass.log"
echo
echo "watch with:  tail -f logs/binary.log logs/multiclass.log"

wait $PID_A; echo "binary finished $(date '+%F %T')"
wait $PID_B; echo "multiclass finished $(date '+%F %T')"

# The graph model runs alone: it is the fastest of the five and benefits from
# having a whole GPU rather than competing with a tabular job.
echo "=== graph  $(date '+%F %T') ==="
CUDA_VISIBLE_DEVICES=$GPU_B python -m nidsbench.run \
  --data-dir "$DATA" --out results_v2/graph --tasks binary multiclass \
  --models egraphsage --subsample "$N" --subsample-by entity --min-per-host "$MPH" \
  --cache-dir cache_entity --save-predictions \
  2>&1 | tee logs/graph.log

echo
echo "STAGE 1 COMPLETE. Send results_v2/*/{results,per_class}.csv and logs/*.log"