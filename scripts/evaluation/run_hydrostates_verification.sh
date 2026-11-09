#!/bin/bash
# Verification matrix for the hydrostates ladder: the MLP runs (data configs x seeds 42/7/23) on
# train_mtg_new (KG windows, 450G) and eval_mtg (plain + greenness-at-init + FVC windows, 256G),
# through run_climpred.sh, which submits verif_init / verif_pixel / verif_anom + evaluate per pair
# and skips pairs without forecast.zarr. Existing verifications are overwritten.
#
# --cells submits the per-cell verifier (verif_cell.py -> verif/by_cell.nc, the init-level
# product for the mixed models) instead of the climpred chains; --after=<jobid> makes those
# jobs wait (afterok), e.g. for the init-state file of the selection.
#
# 2026-10-12: the 12 GloFAS-ladder runs, baselines scored on their own valid set (unpaired).
# 2026-10-14: `full_noswvl` added; both evals re-verified for all runs with the paired baselines
#             (common.build_ensembles masks the baselines with the model's samples).
# 2026-10-18: --version=v16.2 = the matrix on the validity-decorrelated selection (full_noswvl,
#             full, full_rzsm x seeds 42/7/23/11/5; evals eval_mtg_hydrostates_val + train_mtg_hydrostates).
#
# Usage: bash scripts/evaluation/run_hydrostates_verification.sh [--dry-run] [--cells] [--after=<jobid>]
#            [--data="full full_noswvl"] [--evals="eval_mtg"] [--version=v16.2]

set -e
WORKDIR=${WORKDIR:-/Net/Groups/BGI/people/mwessel/deep_lst}   # override with WORKDIR=<worktree> to run from another checkout
cd ${WORKDIR}

DRY_RUN=""
CELLS=false
AFTER=""
VERSION=v16
DATA_CONFIGS=""
EVALS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --cells)   CELLS=true ;;
        --after=*) AFTER="${arg#--after=}" ;;
        --data=*)  DATA_CONFIGS="${arg#--data=}" ;;
        --evals=*) EVALS="${arg#--evals=}" ;;
        --version=*) VERSION="${arg#--version=}" ;;
        *)         echo "unknown argument: $arg"; exit 1 ;;
    esac
done

case ${VERSION} in  # defaults per matrix (scripts/evaluation/hydrostates_runs.VERSIONS)
    v16)   SEEDS="42 7 23";        DEFAULT_DATA="full_noswvl full full_rzsm full_rzsm_dis full_glofas"
           DEFAULT_EVALS="train_mtg_new eval_mtg" ;;
    v16.2) SEEDS="42 7 23 11 5";   DEFAULT_DATA="full_noswvl full full_rzsm"
           DEFAULT_EVALS="train_mtg_hydrostates eval_mtg_hydrostates_val" ;;
    *)     echo "unknown version: ${VERSION}"; exit 1 ;;
esac
DATA_CONFIGS=${DATA_CONFIGS:-${DEFAULT_DATA}}
EVALS=${EVALS:-${DEFAULT_EVALS}}

declare -A CELL_MEM=([eval_mtg]=48G [eval_mtg_new_val]=64G [train_mtg_new]=160G
    [eval_mtg_hydrostates_val]=64G [train_mtg_hydrostates]=160G)  # four error cubes of (1460, 241, n_pixels)
HOUR_WINDOWS="h00_04 h04_08 h08_12 h12_16 h16_20 h20_24"  # masks.HOUR_BINS: diurnal cycle, holdout files only
declare -A CELL_WINDOWS=(
    [eval_mtg]="all day night ${HOUR_WINDOWS}"
    [eval_mtg_new_val]="all day night ${HOUR_WINDOWS}"
    [train_mtg_new]="all day night"
    [eval_mtg_hydrostates_val]="all day night amplitude"   # v04: day/night rows + diurnal amplitude error, no hour bins
    [train_mtg_hydrostates]="all day night amplitude"
)
PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && export PYTHONPATH=${WORKDIR}"

submit_cells () {  # submit_cells <experiment> <eval>
    local exp=$1 eval_name=$2 tag="${1}_${2}"
    if [ ! -d "results/${exp}/version_0/${eval_name}/forecast.zarr" ]; then
        echo "SKIP ${tag}: no forecast.zarr"
        return
    fi
    if [ -n "${DRY_RUN}" ]; then
        echo "[dry] vcell ${tag} (${CELL_MEM[$eval_name]}${AFTER:+, afterok:${AFTER}}): windows ${CELL_WINDOWS[$eval_name]}"
        return
    fi
    local dependency=()
    [ -n "${AFTER}" ] && dependency=(--dependency=afterok:${AFTER} --kill-on-invalid-dep=yes)
    local job
    job=$(sbatch --parsable "${dependency[@]}" --partition=big --mem=${CELL_MEM[$eval_name]} --cpus-per-task=4 \
        --time=03:00:00 --job-name="vcell_${tag}" --output="${WORKDIR}/scripts/out/vcell_${tag}.%j" \
        --wrap="${PREAMBLE} && python3 -m deep_lst.evaluation_climpred.verif_cell --experiment ${exp} --eval_name ${eval_name} --windows ${CELL_WINDOWS[$eval_name]}")
    echo "${exp} (${eval_name}): verif_cell ${job}"
}

for data in ${DATA_CONFIGS}; do
    for seed in ${SEEDS}; do
        tag=hydrostates
        [ "${seed}" != "42" ] && tag="hydrostates_s${seed}"
        exp="mlp_smoothl1_h24_${data}_${VERSION}_${tag}"
        if $CELLS; then
            for eval_name in ${EVALS}; do submit_cells ${exp} ${eval_name}; done
        else
            bash scripts/evaluation/run_climpred.sh ${DRY_RUN} --experiment=${exp} ${EVALS} | grep -v "^Done"
        fi
    done
done
echo "Done. Monitor: squeue -u \$USER"
