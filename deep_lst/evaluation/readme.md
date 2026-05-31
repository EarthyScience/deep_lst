# Evaluation pipeline

```
inference → baselines → eval (fast + benchmark) → across-sites → comparison
```

All plots are generated in one `run_evaluation.py` call per (experiment, eval) pair.
Results: `results/{experiment}/{version}/{eval}/plots/{period_tag}/`
Metrics: `results/{experiment}/{version}/{eval}/metrics_*.nc`

---

## Plot flags quick-reference

| Flag | Tier | What it shows |
|---|---|---|
| `timeseries` | fast | Region-mean LST over test period ± spatial std; all baselines as dashed overlays |
| `diurnal_mean` | fast | Mean LST by hour of day; all baselines as dashed overlays |
| `forecast_gif` | fast | Animated pred vs obs map over horizon |
| `fingerprint` | fast | (DOY × hour) heatmap: model pred + obs + all baselines in one figure |
| `error_growth` | fast | RMSE vs lead time; baseline RMSE as horizontal reference lines |
| `error_growth_daily` | fast | Mean RMSE per day bucket (day 1/2/3); baseline references |
| `error_overview` | bench | Boxplot: model vs each baseline (region_time RMSE distribution) |
| `error_fingerprint` | bench | \|pred-obs\| and \|baseline-obs\| fingerprints side by side |
| `improvement_fingerprint` | bench | ΔMAE fingerprint per baseline (neg = model better) |
| `diurnal` | bench | Diurnal RMSE/bias lines, model + all baselines in one panel |
| `diurnal_delta` | bench | ΔT/h by hour: model + obs + all baselines in one panel |
| `diurnal_delta_error` | bench | ΔT/h error (pred − obs) by hour: model + all baselines in one panel |
| `power_spectrum` | bench | Spatial power spectrum (log-log): model + obs + all baselines |
| `bias_maps` | bench | Spatial bias maps: model + all baselines |
| `improvement_map` | bench | RMSE improvement: multi-panel figure, one panel per baseline |

**Baseline names:** `era5`  `clim_hourly`  `clim_smooth`  `persistence`

---

## Step 1 — Inference

Inference uses a two-phase approach: a CPU setup job creates the output zarr store,
then a GPU array job fills it in parallel chunks. `scripts/inference/run_inference.sh`
submits both with automatic `afterok` dependency — no manual wiring needed.

**Key parameters**

| Parameter | Required | Notes |
|---|---|---|
| `experiment` | yes | e.g. `mlp_rollout_L_KG_mse_h6_v4` |
| `version` | no | defaults to `version_0` |
| `eval` | yes | eval config name — see `conf/eval/` |
| `inference_horizon` | yes | rollout horizon in hours; must be ≥ trained horizon |
| `best` | no | `true` = load `best.ckpt`; default loads `last.ckpt` |
| `--array=0-N` | no | N+1 GPU chunks; defaults to `0-3` |
| `--time=`, `--mem=`, `--gres=`, `--partition=` | no | SLURM overrides for the array job |

### Dense eval region (e.g. `dahra`)

```bash
for eval in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa yangambi bouamir; do
    bash scripts/inference/run_inference.sh \
        experiment=mlp_rollout_L_MT_mse_h4_v5_large_l72_mtg \
        version=version_0 \
        eval=${eval} \
        inference_horizon=200 \
        best=true
done
```

### Sparse training-site region (e.g. `train_KG_L_fullyear`)

```bash
bash scripts/inference/run_inference.sh \
    experiment=mlp_rollout_L_MT_mse_h4_v5_large_l72_mtg \
    version=version_0 \
    eval=train_mtg \
    inference_horizon=200 \
    best=true
```

Output: `results/{experiment}/{version}/{eval}/forecast.zarr`


---

## Step 2 — Baselines (once per eval region)

### ERA5

Single SLURM array job — all 7 point-site eval regions in parallel:

```bash
sbatch scripts/benchmarks/extract_era5_baseline.sh
```

Output: `assets/era5/baselines/baseline_{eval}.nc`

**Training-site eval regions** (e.g. `train_KG_L_fullyear`, `train_KG_L`) do not have
dedicated eval zarr stores. Pass `--store_spatial` and `--index_name` to reuse the
training-region stores:

```bash
uv run scripts/benchmarks/extract_era5_baseline.py \
    --eval train_KG_L_fullyear \
    --store_spatial KG_L \
    --index_name KG_L \
    --test_year 2024 \
    --overwrite
```

### SEVIRI climatology

Point-site eval regions:

```bash
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    sbatch scripts/benchmarks/extract_seviri_climatology.sh ${EVAL} 2024 7
done
```

Training-site eval regions — pass `--store_spatial` to use the training-region store:

```bash
uv run scripts/benchmarks/extract_seviri_climatology.py \
    --eval train_KG_L_fullyear \
    --store_spatial KG_L \
    --test_year 2024 \
    --window_days 7 \
    --overwrite
```

Output: `assets/climatology/hourly/climatology_{eval}.nc`
        `assets/climatology/smoothed_hourly/climatology_{eval}.nc`

### SEVIRI persistence

Point-site eval regions:

```bash
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    uv run scripts/benchmarks/extract_seviri_persistence.py \
        --eval ${EVAL} \
        --test_year 2024
done
```

Training-site eval regions — pass `--store_spatial` to use the training-region store:

```bash
uv run scripts/benchmarks/extract_seviri_persistence.py \
    --eval train_KG_L_fullyear \
    --store_spatial KG_L \
    --test_year 2024 \
    --overwrite
```

Output: `assets/seviri/baselines/persistence_{eval}.nc`

### Baseline fingerprint check

```bash
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    sbatch scripts/benchmarks/evaluate_baselines.sh ${EVAL}
done
```

Output: `results/baselines/{eval}/fingerprint.png`

---

## Step 3 — Fast + Benchmark evaluation

Run after Step 2. Always generates all plots. Benchmark plots are skipped
automatically when no baselines are loaded.

### Training-site eval (post-inference, with baselines)

After running inference with `eval=train_KG_L_fullyear` and creating baselines
(Step 2), evaluate over the first 3 days of init times (72 h horizon):

```bash
uv run scripts/evaluation/run_evaluation.py \
    --experiment mlp_rollout_L_KG_crps_lb6_bptt_v3 \
    --version version_0 \
    --eval train_KG_L_fullyear \
    --lead_time 6h \
    --baselines era5 clim_hourly clim_smooth \
    --temporal test \
    --overwrite
```

Or via SLURM:

```bash
sbatch scripts/evaluation/run_evaluation.sh \
    --experiment mlp_rollout_L_KG_crps_lb6_bptt_v3 \
    --version version_0 \
    --eval dahra \
    --lead_time 12h \
    --baselines era5 clim_hourly clim_smooth persistence \
    --temporal test \
    --overwrite
```

Just the error-growth plot for a specific site and time period:

```bash
uv run scripts/evaluation/run_evaluation.py \
    --experiment mlp_rollout_L_KG_crps_lb6_bptt_v3 \
    --version version_0 \
    --eval train_KG_L_fullyear \
    --baselines era5 clim_hourly clim_smooth \
    --temporal test \
    --plots error_growth error_growth_daily
```

### All experiments × all eval regions

```bash
bash scripts/evaluation/run_evaluation_experiments.sh mlp_diag \
    --lead_time 3h \
    --baselines era5 clim_hourly clim_smooth \
    --overwrite

bash scripts/evaluation/run_evaluation_experiments.sh mlp_rollout \
    --lead_time 3h \
    --baselines era5 clim_hourly clim_smooth \
    --overwrite
```

### One model × all eval regions

```bash
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    uv run scripts/evaluation/run_evaluation.py \
        --experiment mlp_rollout_L_KG_v1 \
        --version version_1 \
        --eval ${EVAL} \
        --lead_time 3h \
        --baselines era5 clim_hourly clim_smooth \
        --overwrite
done
```

### Single experiment / single eval (quick test)

```bash
uv run scripts/evaluation/run_evaluation.py \
    --experiment mlp_rollout_L_KG_v1 \
    --version version_1 \
    --eval dahra \
    --lead_time 3h \
    --baselines era5 clim_hourly clim_smooth \
    --overwrite
```

Restrict to a specific test period:

```bash
uv run scripts/evaluation/run_evaluation.py \
    --experiment mlp_rollout_L_KG_v1 \
    --version version_1 \
    --eval dahra \
    --lead_time 3h \
    --baselines era5 clim_hourly clim_smooth \
    --temporal test_period_01
```

Customise the GIF:

```bash
uv run scripts/evaluation/run_evaluation.py \
    --experiment mlp_rollout_L_KG_v1 --version version_1 --eval dahra \
    --gif_init_time 2024-01-07T06:00 --gif_lead_times 1h 6h 12h 24h --gif_fps 4
```

Available periods: `test_period_00` … `test_period_25` (`assets/temporal/`).

---

## Step 4 — Across-sites evaluation

Summarises one model's performance across all eval regions in a single figure.
Requires `metrics_region_time.nc` and `metrics_region_time_{baseline}.nc` from
Step 3 (saved automatically when `--baselines` are provided).

**Plots:** paired boxplots — one group per region, one box per source (model +
selected baseline). Shows median error and temporal spread per site.

```bash
# All three baselines
uv run scripts/evaluation/run_across_sites.py \
    --experiment mlp_rollout_L_KG_v1 \
    --version version_1 \
    --baseline era5 clim_hourly clim_smooth \
    --lead_time 3h \
    --overwrite
```

Output: `results/{experiment}/{version}/across_sites/cross_site_rmse_lt{lead_time}_vs_{baseline}.png`

---

## Step 5 — Cross-experiment comparison

Requires `metrics_diurnal_mean.nc` from Step 3 for all experiments.

Group by sample size (S / M / L):

```bash
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    uv run scripts/evaluation/run_comparison.py \
        --eval ${EVAL} --version version_1 \
        --diag    mlp_diag_S_KG_v1    mlp_diag_M_KG_v1    mlp_diag_L_KG_v1 \
        --rollout mlp_rollout_S_KG_v1 mlp_rollout_M_KG_v1 mlp_rollout_L_KG_v1 \
        --group_by sample_size --metric rmse --lead_time 3h
done
```

Group by stratification (KG vs KGLC):

```bash
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    uv run scripts/evaluation/run_comparison.py \
        --eval ${EVAL} --version version_1 \
        --diag    mlp_diag_L_KG_v1    mlp_diag_L_KGLC_v1 \
        --rollout mlp_rollout_L_KG_v1 mlp_rollout_L_KGLC_v1 \
        --group_by stratification --metric rmse --lead_time 3h
done
```

Subset of plots: add `--plots boxplot diurnal_delta power_spectrum` (default: all three).
