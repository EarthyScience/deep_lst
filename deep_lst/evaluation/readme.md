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

**Baseline names:** `era5`  `clim_hourly`  `clim_smooth`

---

## Step 1 — Inference

```bash
# Default horizon (3 h), 3 test periods, all experiments × all eval regions
bash scripts/inference/run_inference_experiments.sh mlp_diag
bash scripts/inference/run_inference_experiments.sh mlp_rollout

# Long horizon (72 h)
SLURM_TIME=12:00:00 bash scripts/inference/run_inference_experiments.sh mlp_diag 72
SLURM_TIME=12:00:00 bash scripts/inference/run_inference_experiments.sh mlp_rollout 72
```

Wall time: ~2 h (24 h horizon), ~6 h (72 h), ~10 h (120 h).

---

## Step 2 — Baselines (once per eval region)

### ERA5

Single SLURM array job — all 7 eval regions in parallel:

```bash
sbatch scripts/benchmarks/extract_era5_baseline.sh
```

Output: `assets/era5/baselines/baseline_{eval}.nc`

### SEVIRI climatology

```bash
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    sbatch scripts/benchmarks/extract_seviri_climatology.sh ${EVAL} 2024 7
done
```

Output: `assets/climatology/hourly/climatology_{eval}.nc`
        `assets/climatology/smoothed_hourly/climatology_{eval}.nc`

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
