#!/bin/bash
# Hydrostates v16.2 (2026-10-16): the hydrology ladder on the validity-decorrelated selection.
# 3 data configs x 5 seeds = 15 cells, MLP recipe of the v16 e70 runs (lb168, integer statics,
# no incremental loss, 70 epochs):
#
#   full_noswvl    no hydrological state: full minus ERA5-Land swvl1/swvl2
#   full           no glofas
#   full_rzsm      + swir (root-zone soil wetness)
#
# Trains on mtg_hydrostates (3000 sites, KG_major x MODIS x biome x elevation strata, validity
# weighted) and validates / tests on its spatial holdout eval_mtg_hydrostates_val (500 sites,
# drawn first). _version=v16.2 puts the selection generation into the run name:
#   mlp_smoothl1_h24_<data>_v16.2_hydrostates[_s<seed>]
#
# Preload caches: full and full_noswvl share the 3-modality cache, full_rzsm has the 4-modality
# one. The seed-42 cell of each family is submitted without dependency and builds the caches
# (training + holdout; measured 2026-10-09/15: 4-modality training cache 2 h 51 min, 500-site
# holdout cache 1 h 19 min), so it gets BUILDER_TIME instead of the 7-9 h a cached 70-epoch run
# needs. Every other cell of the family starts at the earliest CACHE_LEAD minutes after that
# builder started (slurm "after:<job>+<minutes>"), so the caches are written once.
# Cells whose experiment directory exists are skipped, so the script can be re-run to fill gaps.
#
# Usage: bash scripts/training/launch_hydrostates_v16_2.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

DRY_RUN=false
for arg in "$@"; do
    [ "$arg" == "--dry-run" ] && DRY_RUN=true
done

MLP_LAUNCH="scripts/training/train_mlp.sh"
MLP_ARGS="model=mlp objective=rollout loss=smoothl1 horizon=h24 window.lookback=168 model.static_encoding=integer objective.terms.increment=0 training.max_epochs=70 _spatial=mtg_hydrostates _spatial_val=eval_mtg_hydrostates_val _version=v16.2"
MLP_TIME="10:00:00"
BUILDER_TIME="16:00:00"
BASE_TAG="hydrostates"
declare -A CACHE_LEAD=([base]=300 [glofas]=360)   # minutes after the builder's start before the other cells may start

DATA_CONFIGS="full_noswvl full full_rzsm"
SEEDS="42 7 23 11 5"

cache_family () {
    case $1 in
        full_rzsm) echo glofas ;;
        *) echo base ;;
    esac
}

declare -A BUILDER_JOB   # cache family -> job id of the cell that builds the caches

submit () {
    # submit <time> <dependency or ""> <launcher> <overrides...>; prints the job id
    local time=$1 dependency=$2 launcher=$3; shift 3
    local dep_flag=""
    [ -n "${dependency}" ] && dep_flag="--dependency=${dependency}"
    if $DRY_RUN; then
        echo "[dry] sbatch --time=${time} ${dep_flag} ${launcher} $*" >&2
        echo "dry"
    else
        sbatch --parsable --time=${time} ${dep_flag} ${launcher} "$@"
    fi
}

for data in ${DATA_CONFIGS}; do
    family=$(cache_family ${data})
    for seed in ${SEEDS}; do
        tag=${BASE_TAG}
        [ "${seed}" != "42" ] && tag="${BASE_TAG}_s${seed}"
        name="mlp_smoothl1_h24_${data}_v16.2_${tag}"
        if [ -d "experiments/${name}" ]; then
            echo "skip ${name}: experiment directory exists"
            continue
        fi
        if [ -z "${BUILDER_JOB[${family}]}" ]; then
            job=$(submit ${BUILDER_TIME} "" ${MLP_LAUNCH} ${MLP_ARGS} data=${data} seed=${seed} tag=${tag})
            BUILDER_JOB[${family}]=${job}
            echo "job ${job}: ${name} (builds the ${family} caches)"
        else
            job=$(submit ${MLP_TIME} "after:${BUILDER_JOB[${family}]}+${CACHE_LEAD[${family}]}" ${MLP_LAUNCH} ${MLP_ARGS} data=${data} seed=${seed} tag=${tag})
            echo "job ${job}: ${name} (after ${BUILDER_JOB[${family}]} + ${CACHE_LEAD[${family}]} min)"
        fi
    done
done
echo "Done. Monitor: squeue -u \$USER"
