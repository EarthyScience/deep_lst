"""Run matrix of the hydrostates ladder, shared by the hydrostates_* scripts: the data
conditions in ladder order with their figure labels and colours, the seeds, the run-name
convention, the solar windows and the folder of the cross-run products.

Two matrices share the conditions and the run-name convention. v16 (2026-10-09) trained on
mtg_new with all five conditions and three seeds; v16.2 (2026-10-16) trained on the
validity-decorrelated mtg_hydrostates with three conditions and five seeds. The module-level
SEEDS and run_name default are v16, so the v01/v02 products stay reproducible."""

from pathlib import Path

OUT_DIR = Path("results/independent/hydrostates")
REFERENCE = "full"  # the paired reference of every ladder
CONDITIONS = {  # data config -> (label, colour), ladder order
    "full_noswvl": ("− swvl (no soil moisture)", "#7b3294"),
    "full": ("full (no GloFAS)", "#0b0b0b"),
    "full_rzsm": ("+ RZSM", "#2a78d6"),
    "full_rzsm_dis": ("+ RZSM + discharge", "#eb6834"),
    "full_glofas": ("+ RZSM + discharge + runoff", "#1baf7a"),
}
SEEDS = (42, 7, 23)
VERSIONS = {  # matrix -> seeds, conditions, training selection (for in_training)
    "v16": {"seeds": SEEDS, "conditions": tuple(CONDITIONS), "training": "mtg_new"},
    "v16.2": {
        "seeds": (42, 7, 23, 11, 5),
        "conditions": ("full_noswvl", "full", "full_rzsm"),
        "training": "mtg_hydrostates",
    },
}
WINDOWS = ("day", "night")
# Vegetation state at init (masks.GREEN_BINS, ten 0.1-wide bins of relative greenness) x solar window.
GREEN_WINDOWS = [f"green_d{k}_{window}" for k in range(1, 11) for window in WINDOWS]


def run_name(data, seed, version="v16"):
    """Experiment name of one matrix cell; seed 42 carries the bare `hydrostates` tag."""
    tag = "hydrostates" if seed == 42 else f"hydrostates_s{seed}"
    return f"mlp_smoothl1_h24_{data}_{version}_{tag}"
