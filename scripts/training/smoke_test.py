"""Pre-flight smoke test for the MLP training pipeline.

Validates config loading, path resolution, and zarr store accessibility for
each experiment scenario WITHOUT requiring statistics to be up-to-date.
Statistics files are checked for existence only, not for variable coverage.

Run from the project root:
    python scripts/training/smoke_test.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eodata.eo_zarr import EOZarrStore

from deep_lst.utils.config import merge_configs, open_config
from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

SCENARIOS = [
    ("mlp_diag",    ["mlp_diag_L_KG_v1"]),
    ("mlp_diag",    ["mlp_diag_M_KG_v1"]),
    ("mlp_diag",    ["mlp_diag_S_KG_v1"]),
    ("mlp_diag",    ["mlp_diag_L_KG_v1", "eval_antananarivo"]),
    ("mlp_diag",    ["mlp_diag_L_KG_v1", "eval_dahra"]),
    ("mlp_diag",    ["mlp_diag_L_KG_v1", "eval_east_africa"]),
    ("mlp_diag",    ["mlp_diag_L_KG_v1", "eval_gobabeb"]),
    ("mlp_diag",    ["mlp_diag_L_KG_v1", "eval_johannesburg"]),
    ("mlp_diag",    ["mlp_diag_L_KG_v1", "eval_sahel_niger"]),
    ("mlp_diag",    ["mlp_diag_L_KG_v1", "eval_w_africa"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1"]),
    ("mlp_rollout", ["mlp_rollout_M_KG_v1"]),
    ("mlp_rollout", ["mlp_rollout_S_KG_v1"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1", "eval_antananarivo"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1", "eval_dahra"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1", "eval_east_africa"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1", "eval_gobabeb"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1", "eval_johannesburg"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1", "eval_sahel_niger"]),
    ("mlp_rollout", ["mlp_rollout_L_KG_v1", "eval_w_africa"]),
]

REQUIRED_DATASET_KEYS = ["spatial", "temporal"]
REQUIRED_TOP_KEYS = ["modality_ids", "train_dataset", "val_dataset", "test_dataset", "model", "training"]
DATASET_SPLITS = ["train_dataset", "val_dataset", "test_dataset"]


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"    [{status}] {label}{suffix}")
    return condition


def check_warn(label, condition, detail=""):
    status = PASS if condition else WARN
    suffix = f"  ({detail})" if detail else ""
    print(f"    [{status}] {label}{suffix}")
    return condition  # warnings don't count as failures


def validate_scenario(config_name, overrides):
    label = config_name
    if overrides:
        label += " --override " + " ".join(overrides)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    failures = 0

    # ── 1. Config loading ────────────────────────────────────────
    print("  [Config]")
    try:
        base = PATHS.config_path(config_name)
        if overrides:
            override_paths = [PATHS.config_root / config_name / "overrides" / f"{o}.yml" for o in overrides]
            config = merge_configs(base, override_paths)
        else:
            config = open_config(base)
        if not check("Config loads without error", True):
            failures += 1
    except Exception as e:
        check("Config loads without error", False, str(e))
        return failures + 1  # can't continue without a config

    # Required top-level keys
    for key in REQUIRED_TOP_KEYS:
        if not check(f"  '{key}' present", key in config):
            failures += 1

    # ── 2. Dataset key validation ────────────────────────────────
    print("  [Dataset keys]")
    modality_ids = config.get("modality_ids", [])

    for split in DATASET_SPLITS:
        ds = config.get(split, {})
        for key in REQUIRED_DATASET_KEYS:
            present = key in ds
            detail = repr(ds.get(key, "<missing>"))
            # test_dataset.temporal is intentionally absent without an eval override
            if not present and split == "test_dataset" and key == "temporal":
                check_warn(f"  {split}.{key}  (warn — requires eval override)", present, detail)
            elif not check(f"  {split}.{key}", present, detail):
                failures += 1
        for mod in modality_ids:
            mod_cfg = ds.get(mod, {})
            for field in ["inputs", "statistics", "normalization"]:
                present = field in mod_cfg
                if not check(f"  {split}.{mod}.{field}", present):
                    failures += 1

    # ── 3. File existence ────────────────────────────────────────
    print("  [Paths]")

    def check_file(path, label):
        p = Path(path)
        exists = p.exists()
        if not check(label, exists, str(p)):
            return 1
        return 0

    for split in DATASET_SPLITS:
        ds = config.get(split, {})
        spatial = ds.get("spatial")
        temporal = ds.get("temporal")

        # Spatial indices
        if spatial is not None:
            names = spatial if isinstance(spatial, list) else [spatial]
            for mod in modality_ids:
                for name in names:
                    failures += check_file(
                        PATHS.spatial_indices_path(mod, name),
                        f"  {split}: spatial indices [{mod}] '{name}'"
                    )

        # Temporal indices
        if temporal is not None:
            periods = temporal if isinstance(temporal, list) else [temporal]
            for period in periods:
                failures += check_file(
                    PATHS.temporal_indices_path(period),
                    f"  {split}: temporal indices '{period}'"
                )

        # Statistics (warn only — may not be up to date)
        for mod in modality_ids:
            stats_name = ds.get(mod, {}).get("statistics")
            if stats_name:
                path = PATHS.statistics_path(mod, stats_name)
                check_warn(
                    f"  {split}: statistics [{mod}] '{stats_name}'  (warn — may be stale)",
                    path.exists(),
                    str(path)
                )

    # ── 4. Zarr store accessibility ──────────────────────────────
    print("  [Stores]")
    for mod in modality_ids:
        store_path = PATHS.store_path(mod)
        if not check(f"  Store path exists [{mod}]", store_path.exists(), str(store_path)):
            failures += 1
            continue

        # Zarr v3 requires zarr.json at the root; without it open_group raises
        # "No group found". Check for it explicitly to give an actionable error.
        meta = store_path / "zarr.json"
        if not meta.exists():
            check(
                f"  Store openable  [{mod}]",
                False,
                f"zarr.json missing at store root — store was never initialized "
                f"(chunks may exist but metadata was not written)",
            )
            failures += 1
            continue

        try:
            EOZarrStore(store_path, mode="r")
            check(f"  Store openable  [{mod}]", True)
        except Exception as e:
            check(f"  Store openable  [{mod}]", False, str(e))
            failures += 1

    return failures


def main():
    total_failures = 0
    results = []

    for config_name, overrides in SCENARIOS:
        failures = validate_scenario(config_name, overrides)
        total_failures += failures
        label = config_name + (" --override " + " ".join(overrides) if overrides else "")
        results.append((label, failures == 0))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for label, passed in results:
        status = PASS if passed else FAIL
        print(f"  [{status}] {label}")

    print(f"\nTotal failures: {total_failures}")
    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
