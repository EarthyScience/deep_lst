#!/bin/bash
# GloFAS hydrology ladder x seeds, MLP only (2026-10-09). 5 data configs x 3 seeds = 15 cells:
#
#   full_noswvl    no hydrological state: full minus ERA5-Land swvl1/swvl2 (added 2026-10-13;
#                  same 3-modality cache as full)
#   full           no glofas (3-modality cache preload_mtg_new_b91b60fefa33b664.pt)
#   full_rzsm      + swir                 (root-zone soil wetness)
#   full_rzsm_dis  + swir + dis24         (+ routed river discharge)
#   full_glofas    + swir + dis24 + rowe  (+ local runoff)
#
# Every glofas config shares one preload cache (the cache depends on what is LOADED, not on
# which variables become inputs). It is written by the first glofas job to run; launch this
# matrix only after that file exists, or every cell re-reads all four stores for hours.
#
# Model settings reproduce the existing v16 e70 runs (lb168, integer statics, no incremental
# loss, 70 epochs) under the tag "hydrostates"; seed 42 keeps the bare tag, other seeds get a
# _s<seed> suffix. All cells are fresh runs (the earlier e70 no-glofas run stays outside the
# matrix as a reference). Cells whose experiment directory already exists are skipped, so the
# script can be re-run to fill gaps. Paired design: judge the per-seed differences between rungs.
#
# Validation stays on the training pixels (_spatial_val=mtg_new) for every rung, as in the 12
# cells of 2026-10-09; conf/train.yaml now defaults to the spatial holdout eval_mtg_new_val.
#
# Usage: bash scripts/training/launch_glofas_ablation.sh [--dry-run] [--force]
#   --force   launch even if the glofas preload cache is missing

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

DRY_RUN=false
FORCE=false
for arg in "$@"; do
    [ "$arg" == "--dry-run" ] && DRY_RUN=true
    [ "$arg" == "--force" ] && FORCE=true
done

GLOFAS_CACHE=/Net/Groups/BGI/tscratch/mwessel/tmp/preload_mtg_new_508763c408d21147.pt
if [ ! -f "${GLOFAS_CACHE}" ] && ! $FORCE && ! $DRY_RUN; then
    echo "glofas preload cache not found: ${GLOFAS_CACHE}"
    echo "wait for the running glofas job to write it (or pass --force)."
    exit 1
fi

MLP_LAUNCH="scripts/training/train_mlp.sh"
MLP_ARGS="model=mlp_rollout loss=smoothl1 horizon=h24 _lookback=168 model.static_encoding=integer model.incremental_loss=false training.max_epochs=70 _spatial_val=mtg_new"
MLP_TIME="10:00:00"
BASE_TAG="hydrostates"

DATA_CONFIGS="full_noswvl full full_rzsm full_rzsm_dis full_glofas"
SEEDS="42 7 23"

submit () {
    local time=$1 launcher=$2; shift 2
    if $DRY_RUN; then
        echo "[dry] sbatch --time=${time} ${launcher} $*"
    else
        local job
        job=$(sbatch --parsable --time=${time} ${launcher} "$@")
        echo "job ${job}: $*"
    fi
}

for data in ${DATA_CONFIGS}; do
    for seed in ${SEEDS}; do
        tag=${BASE_TAG}
        [ "${seed}" != "42" ] && tag="${BASE_TAG}_s${seed}"
        name="mlp_smoothl1_h24_${data}_v16_${tag}"
        if [ -d "experiments/${name}" ]; then
            echo "skip ${name}: experiment directory exists"
            continue
        fi
        submit ${MLP_TIME} ${MLP_LAUNCH} ${MLP_ARGS} data=${data} seed=${seed} tag=${tag}
    done
done
echo "Done. Monitor: squeue -u \$USER"
