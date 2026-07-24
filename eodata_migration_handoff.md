# deep_lst → eodata-main migration & eval rewrite — handoff

Session of 2026-07-23/24. deep_lst now runs end-to-end on **eodata main** (full unextracted
`.eozarr` stores, no sampled stores). This doc is the warm-start reference for the next work:
variable-drop experiments, the SEVIRI quality mask, and calibrated-uncertainty (CRPS) training.

---

## 1. What changed (data layer)

The dataset was rewritten around eodata main's reader API. Core idea: **load full → normalize →
select**, so dropping a channel is config-only, never a stats recompute.

- **Preload** (`deep_lst/dataset/dataset.py::preload_shared`): per (anchor, modality) it reads the
  full hour axis from the `.eozarr` store, normalizes with a per-modality stats zarr, then selects
  the config `load` variables. Cached to `/Net/Groups/BGI/tscratch/mwessel/tmp/`.
- **Anchors**: training on the global **MajorTomGrid** (`indices_mtg.npz`), eval on the native
  **equiangular seviri** grid (holdout `indices_eval_*.npz`). `deep_lst/utils/spatial.py` maps any
  anchor → each modality's store row (identity for seviri-on-seviri, NN otherwise).
- **Axis trim**: cache holds only 2016–2026 (`PRELOAD_START_DAY`/`PRELOAD_END_DAY`), ~9 GB/region.
- **Statistics** (`scripts/statistics/compute_statistics.py`): base stats over **all** store
  channels + prognostic increments + a `computed` group (solar_zenith) + `variables` name attrs on
  every group. One modality per SLURM job.

### Cache addressing (why the filenames look odd)
The model never looks up a cache by name — it **recomputes a fingerprint** from the data config
(anchors + per-modality store/load/stats-mtime/axis) and that hash *is* the filename:
`preload_<spatial>_<fingerprint16>.pt`. Same config → same hash → cache hit. Any change to
anchors/store/load/stats/axis → new hash → rebuild (never a silent stale reuse). The `<spatial>`
prefix is cosmetic.

---

## 2. Eval path (all off sampled stores)

- **Forecast writer** (`deep_lst/evaluation/inference.py`): keeps the dense/sparse fork (dense
  `(init,lead,lat,lon)` for box regions, sparse `(…,pixel)` for scattered MTG). Coords from the
  preload's `cell_lonlat` + `pixel_grid`; `init_time` from absolute T1H; denorm via
  `StatsZarrNormalizer`; land-water mask from the preloaded LANDCOV.
- **Baselines** (`deep_lst/baselines/`, refactored to one file per baseline): `base.py`
  (`BaselineExtractor`, `ModalityBaselineExtractor`), `persistence.py`, `climatology.py`
  (`smooth` flag), `era5.py`, `era5land.py`. **All source from the preload** — persistence/
  climatology from seviri LST, cross-modality (era5_land skt) via `(modality, var)`. Coords always
  on the seviri grid so baselines match forecast.zarr. attrs come from the store's own metadata.
  Driver: `scripts/baselines/run_baselines.py --region <cfg> --test_year 2024 [--overwrite]`.
- **Evaluator** (`deep_lst/evaluation/evaluator.py`): `_attach_class_coords` reads KG classes from
  the npz + `pixel_grid` (verified identical to the old sampled-store result). `era5_land`
  registered in `_BASELINE_REGISTRY`.

**Run eval**: `sbatch scripts/evaluation/run_evaluation.sh --experiment <exp> --version version_0
--eval <region> --lead_time 32h --temporal test --baselines era5_land clim_smooth clim_hourly
persistence`. Spatial = the 9 box regions; temporal extrapolation = `--eval train_mtg`.

---

## 3. Variable inventory & how to run drop/add experiments

Store variables **currently unused** (available to add):

| store | loaded | unused |
|---|---|---|
| era5 `data` | sp,u10,v10,ssrd,strd,tp,tcwv,tcc | **sf** |
| era5 `data_static_categorical` | tvh,tvl,lsm | **slt** |
| era5_land `data` | stl1,stl2,swvl1,swvl2,t2m,d2m,skt | **e,pev,sshf,slhf,snowc,fal,src** |

**Two levels of change:**
- **Drop/add a model input** = edit the model role (`conf/model/mlp_rollout.yaml` `inputs`/
  `context_vars`), keep it in `load`. → **cache reused** (fingerprint keys on `load`, not roles),
  new smaller/larger model. Fast experiment path.
- **Drop/add from `load`** = changes the cache fingerprint → **one rebuild**.

**Forcing vs prognostic** (for adding vars):
- **Forcing** (context): load + `context_vars`. → cache rebuild, **no stats recompute** (base
  stats already cover all store channels; categoricals need none).
- **Prognostic**: load + `prognostics`. → cache rebuild **plus a stats recompute** (the
  `increments` group is only built for configured prognostics).
- **Open scientific question**: which era5/era5_land vars are available as *forcing at forecast
  time*. Forecast-available → forcing; not → must become prognostic. `sshf`/`slhf` (surface energy
  balance) are the most physically relevant unused era5_land vars for LST.

---

## 4. SEVIRI quality mask (implemented, deterministic training)

`_seviri_load` now caches `metadata: [quality_flag, standard_error]` **raw** (the normalizer skips
the metadata leaf — no metadata stats group). A `_quality_mask` block
(`field: standard_error, se_max: 1.8, target: LST`) is referenced by train/val seviri, **not test**
(eval scores all valid obs). `dataset.py::_apply_quality_mask` NaNs the target where SE > se_max,
at `__getitem__` (config-only threshold, no rebuild to re-tune), cloned so the shared cache is
never mutated. Metadata is loaded for masking/CRPS, never packed into the model.

**Why SE, not the flag**: `quality_flag` uses **two encodings** across the axis — PUM bit-packed
(≤2020: 10014/14238 = Nominal/Above-Nominal) vs simplified (≥2021: 101/102/103). `== 103` would
mask nothing pre-2021 and silently bias train (mostly PUM) vs test (2024 simplified).
`standard_error` is one continuous Kelvin quantity, stable across the 2021 boundary (mean
~1.2–1.4 K every year), so one absolute threshold behaves identically everywhere — and it's the
quantity CRPS needs. Cloud/invalid is already NaN in the store mask (valid LST ~30%); SE exists
only where LST is valid. "Nominal" is a retrieval-*condition* label (emissivity/atmosphere), not an
error percentile — so use an **absolute** cut, never a subset percentile.

**Deferred**: recompute seviri stats under the mask (second-order — dropping ~2–10% barely moves
LST mean/std).

---

## 5. CRPS / calibrated uncertainty — design fork for next session

Goal: a model with **fair CRPS / calibrated aleatoric uncertainty**. Key reasoning settled here:

- `standard_error` is the **retrieval/observation error**, NOT the forecast uncertainty to model.
  A plain CRPS head calibrates the aleatoric spread of `p(LST | inputs)`, which is a **convolution**
  of genuine forecast uncertainty (LST underdetermination, forcing error) with retrieval noise.
- **Decision**: divide the retrieval error out — treat `obs ~ N(true_LST, SE²)` with `SE` known,
  and have the model predict `p(true_LST | inputs)`. Then the calibrated spread is the forecast
  aleatoric uncertainty, not the retrieval noise. This is why SE is loaded into the cache.
- A CRPS head on deterministic inputs does **not** capture **initial-condition** or **epistemic/
  model** uncertainty — those need input perturbation / ensembles. Whether to also model those is
  the next-experiment framing decision (not yet made).
- Dropping high-SE pixels (the quality mask) is a *crude* version of dividing out retrieval noise —
  fine for the deterministic models, but for CRPS prefer keeping the pixels and using SE in the
  likelihood.

---

## 6. Deferred / known issues

- **`eval_mtg` region** (400-px holdout): `eval_indices_path` double-prefix bug (`eval_mtg` →
  `indices_eval_eval_mtg.npz`, doesn't exist). Box regions + `train_mtg` are fine.
- **Baseline name collision**: `_eval_fname('mtg')` and `_eval_fname('eval_mtg')` both → `eval_mtg.nc`.
  Only bites if both `train_mtg` and the `eval_mtg` region are extracted (currently only train_mtg).
- **`weathergen_landv0.py`** (Shams comparison) still reads an old sampled store — not migrated.
- **Temporal npz** files are 2016-relative (`STORE_EPOCH_OFFSET` shifts them); regenerate as
  absolute T1H eventually.
- **Orphan caches** on tscratch (pre-naming/pre-skt/pre-metadata) are deletable once the current
  named caches are confirmed.
- **eval_name convention** (`io.py::_eval_fname`) is overloaded (routing key + label) — cleanup
  deferred.
- **odc-geo / zarr** lock bumps (0.5.x / 3.2.1) from the migration; upstream ask: eodata declare
  `odc-geo>=0.5`.

---

## 7. Verification trail (what was checked)

Axis trim (cache == old slice), MTG/seviri mapping, statistics (≤3% vs old npz), the **0.0-loss
bug** (FVC NaN → unimputed `step_X_dyn` → gradient poisoning; fixed by `nan_to_num` in the forward;
model then converged epoch 56, val 0.081), solar standardization, both forecast paths (dense
passage + sparse eval_mtg), baselines (E1 bit-identical distribution, E2 skt vs store, structural
all regions), evaluator class-coords (identical to old), quality mask (SE-threshold fires
train-only). Verify scripts live in the session scratchpad.
