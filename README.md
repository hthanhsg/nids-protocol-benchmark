# nids-protocol-benchmark

Controlled benchmark accompanying the systematic survey **"Deep Learning for
Flow-Based Network Intrusion Detection (2022–2026)"**.

The survey audits how the field evaluates flow-based NIDS models and argues that
reported performance differences between studies often reflect the *evaluation
protocol* rather than the *architecture*. This repository contains the
controlled experiment that measures that effect directly: three protocol factors
(train/test split, identity features, task) crossed with five model families
over eight standardised NetFlow corpora, with data, preprocessing, seed and
hyper-parameters held constant so that any difference between cells is
attributable to the protocol.

If you use this code or the results table, please cite the paper (see
`CITATION.cff`).

## What is here

```
nidsbench/            the benchmark package
  data.py             loading, entity-level sampling, encoding
  splits.py           random / temporal / disjoint-IP splits (with feasibility guards)
  models.py           XGBoost, MLP, tabular Transformer, Autoencoder
  graph.py            E-GraphSAGE over a host graph
  metrics.py          macro-F1, macro-F1 over learnable classes, per-class F1, FPR
  run.py              the sweep driver (resumable; one CSV row per run)
scripts/
  warm_cache.py       build the sample cache once before parallel runs
  run_parallel.sh     Stage 1: the main sweep across two GPUs
  stage2_seeds_parallel.sh   Stage 2: seeds 1 and 2
  stage3_sampling_check.sh   Stage 3: flow-sampling control
  stage4_probe.sh     Stage 4: label-permutation leakage probe
  merge_results.py    combine result directories into one table
  measure_cost.py     parameters, size, latency, throughput per model family
  make_figures.py     regenerate the paper's data figures from all_results.csv
  diagnose.py         separate leakage / degenerate-split / easy-dataset causes
  inspect_split.py    print the exact split a run uses; verify no label leaks
results/
  all_results.csv     the complete 360-run results table (1,080 rows incl. 3 seeds)
  family_by_year.csv  architecture-family counts per year (survey corpus)
CITATION.cff
environment.yml
```

## Datasets

The experiment uses the eight standardised NetFlow corpora from the NF-* group
(v2 and v3 of NF-BoT-IoT, NF-ToN-IoT, NF-CSE-CIC-IDS2018/NF-CICIDS2018, and
NF-UNSW-NB15). **These are third-party datasets and are not redistributed here.**
Obtain them from their original source (the University of Queensland NIDS
datasets) and place the CSVs in a `data/` directory:

```
data/
  NF-BoT-IoT-v2.csv
  NF-BoT-IoT-v3.csv
  NF-ToN-IoT-v2.csv
  NF-ToN-IoT-v3.csv
  NF-CSE-CIC-IDS2018-v2.csv
  NF-CICIDS2018-v3.csv
  NF-UNSW-NB15-v2.csv
  NF-UNSW-NB15-v3.csv
```

## Reproducing the experiment

```bash
# 1. environment
conda env create -f environment.yml
conda activate nidsbench

# 2. build the sample cache once (avoids two jobs reading the raw CSVs at once)
python scripts/warm_cache.py --data-dir data --cache-dir cache_entity \
    --subsample 1000000 --subsample-by entity --min-per-host 4

# 3. Stage 1 — the main sweep (two GPUs; ~15 h)
bash scripts/run_parallel.sh

# 4. Stage 2 — seed variance (seeds 1 and 2)
python scripts/warm_cache.py --data-dir data --cache-dir cache_seeds \
    --subsample 1000000 --subsample-by entity --min-per-host 4 --seeds 1 2
MPH=4 GPU=1 bash scripts/stage2_seeds_parallel.sh

# 5. Stages 3 and 4 — controls
bash scripts/stage3_sampling_check.sh   # flow-sampling control
bash scripts/stage4_probe.sh            # label-permutation leakage probe

# 6. combine everything into one table
python scripts/merge_results.py results_v2 results_v2_seeds -o merged
```

Every run appends one row to a `results.csv`; the sweep is resumable, so an
interrupted job continues where it stopped. The released `results/all_results.csv`
is the merged output of the above.

## Regenerating the figures and cost table

```bash
python scripts/make_figures.py --results results/all_results.csv --outdir images
python scripts/measure_cost.py --out cost.csv        # run on the target GPU
```

## Verifying there is no leakage

```bash
# labels are permuted before splitting; test macro-F1 must fall to chance
bash scripts/stage4_probe.sh

# inspect the exact split a run uses and confirm no label column is a feature
python scripts/inspect_split.py --data-dir data --dataset NF-BoT-IoT-v3 \
    --cache-dir cache_entity --subsample 1000000 --subsample-by entity \
    --min-per-host 4 --split disjoint_ip --task binary
```

## Notes on the results table

`results/all_results.csv` has one row per completed run (1,080 rows = 5 models ×
3 seeds × the feasible protocol cells). Key columns:

- `test_macro_f1` and `test_macro_f1_learnable` — the latter restricted to
  classes present in both the train and test folds, which matters under temporal
  and disjoint splits where whole classes can fall on one side.
- `n_entities_train/val/test` — source-host counts per split; a test fold with
  very few entities produces scores that look fine but mean little.
- `val_uninformative`, `classes_missing_in_*` — flags for degenerate cells.
- `shuffle_labels` — marks the leakage-probe runs.

## License

MIT (see `LICENSE`). The datasets are not covered by this license and remain
under their original terms.
