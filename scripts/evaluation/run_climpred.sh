#!/bin/bash

# Usage: bash scripts/evaluation/run_climpred.sh [--dry-run] [--experiment=<name>]
#            [--after=<jobid>[:<jobid>...]] [--months="1 2 3 4"] [--mem=<size>] [eval_name ...]
#
#   bash scripts/evaluation/run_climpred.sh zambia train_mtg_new sahel_niger yangambi

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

EXPERIMENT=v18_mlp_smoothl1_h24_full
VERSION="version_0"

FULL_WINDOWS="all day night kg_arid kg_tropical kg_temperate kg_arid_day kg_arid_night kg_tropical_day kg_tropical_night kg_temperate_day kg_temperate_night"
PLAIN_WINDOWS="all day night"
# Annual-mean FVC deciles (masks.FVC_GROUPS), total + day + night each: 30 windows. Every window
# stacks one masked copy of the forecast cube (~37 GB per window at 3000 px), so these are
# affordable on the 400-px holdout only; on train_mtg_new the FVC rows come from by_pixel.nc.
FVC_WINDOWS=$(for d in 1 2 3 4 5 6 7 8 9 10; do echo -n "fvc_d${d} fvc_d${d}_day fvc_d${d}_night "; done)
# Vegetation state at init (masks.GREEN_BINS, needs assets/static/fvc_init_state_<spatial>.nc): 30 windows.
GREEN_WINDOWS=$(for d in 1 2 3 4 5 6 7 8 9 10; do echo -n "green_d${d} green_d${d}_day green_d${d}_night "; done)

EVAL_NAMES=(train_mtg_new sahel_niger yangambi zambia)
declare -A WINDOWS=(
    [train_mtg_new]="${FULL_WINDOWS}"
    #[eval_mtg_new_val]="${PLAIN_WINDOWS} ${GREEN_WINDOWS} ${FVC_WINDOWS}"  # 500-site spatial holdout of the spatialval runs
    #[train_mtg_hydrostates]="${FULL_WINDOWS}"
    #[eval_mtg_hydrostates_val]="${PLAIN_WINDOWS} ${GREEN_WINDOWS} ${FVC_WINDOWS}"  # 500-site spatial holdout, same windows
    [sahel_niger]="${PLAIN_WINDOWS}"
    [yangambi]="${PLAIN_WINDOWS}"
    [zambia]="${PLAIN_WINDOWS}"
)

declare -A VERIF_MEM=(
    [train_mtg_new]=500G
    #[eval_mtg_new_val]=320G  # 63 windows x 500 px
    #[train_mtg_hydrostates]=450G
    #[eval_mtg_hydrostates_val]=320G  # 63 windows x 500 px
    [sahel_niger]=500G
    [yangambi]=500G
    [zambia]=1000G  # 16k px: 22.9 GB per cube; the per-pixel sums add ~200 GB after climpred
)

MONTHS="1 2 3 4 5 6 7 8 9 10 11 12"  # restrict to months (1-12) to add the month windows; all 12 = no-op

EVAL_MEM=96G  # evaluate loads the forecast store for the site figures (28 GB for the dense cases)
DIURNAL_MEM=200G  # verif_diurnal: three aligned cubes + per-lead slabs, no climpred (zambia ~100 GB)
VERIF_PARTITION=big

DRY_RUN=false
AFTER=""
MEM=""
SELECTED=()
for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --experiment=*) EXPERIMENT="${arg#--experiment=}" ;;
        --after=*)      AFTER="${arg#--after=}" ;;
        --months=*)     MONTHS="${arg#--months=}" ;;
        --mem=*)        MEM="${arg#--mem=}" ;;
        *)              SELECTED+=("$arg") ;;
    esac
done
[ ${#SELECTED[@]} -gt 0 ] && EVAL_NAMES=("${SELECTED[@]}")
VERIF_DEP=()
[ -n "${AFTER}" ] && VERIF_DEP=(--dependency=afterok:${AFTER} --kill-on-invalid-dep=yes)

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && export PYTHONPATH=${WORKDIR}"

# Animated maps (gifs.py): every evaluate renders the bias over the forecast days. The
# day-1/day-10 pairs over the months need one more verif_pixel run with the twelve month
# windows (all + m01-m12 = thirteen stacked copies, fits the eval's VERIF_MEM), submitted
# for these regions only -- the holdouts are too sparse to read as maps.
GIF_EVALS="train_mtg_new"

for eval_name in "${EVAL_NAMES[@]}"; do
    region=$(sed -E 's/_s[0-9]+$//' <<< "${eval_name}")  # stride reruns share the region's windows
    windows=${WINDOWS[$region]:-}
    mem=${MEM:-${VERIF_MEM[$eval_name]:-}}
    if [ -z "${windows}" ] || [ -z "${mem}" ]; then
        echo "SKIP ${eval_name}: no WINDOWS/VERIF_MEM entry"
        continue
    fi
    if [ -z "${AFTER}" ] && [ ! -d "${WORKDIR}/results/${EXPERIMENT}/${VERSION}/${eval_name}/forecast.zarr" ]; then
        echo "SKIP ${eval_name}: no forecast.zarr yet (run the inference first)"
        continue
    fi
    tag="${EXPERIMENT}_${eval_name}"
    ARGS="--experiment ${EXPERIMENT} --eval_name ${eval_name} --windows ${windows} --months ${MONTHS} --version ${VERSION}"
    gifs=false
    [[ " ${GIF_EVALS} " == *" ${region} "* ]] && gifs=true
    gif_note=""
    $gifs && gif_note=" + month windows (month gifs)"
    if $DRY_RUN; then
        echo "[dry] ${EXPERIMENT} (${eval_name}, ${mem}${AFTER:+, afterok:${AFTER}}): verif_init/verif_pixel/verif_anom/verif_diurnal + evaluate${gif_note}, windows: ${windows}"
        continue
    fi
    JOB_INIT=$(sbatch --parsable "${VERIF_DEP[@]}" --partition=${VERIF_PARTITION} --mem=${mem} --cpus-per-task=4 --time=06:00:00 \
        --job-name="vinit_${tag}" \
        --output="${WORKDIR}/scripts/out/vinit_${tag}.%j" \
        --wrap="${PREAMBLE} && python3 -m deep_lst.evaluation_climpred.verif_init ${ARGS}")
    JOB_PIXEL=$(sbatch --parsable "${VERIF_DEP[@]}" --partition=${VERIF_PARTITION} --mem=${mem} --cpus-per-task=4 --time=06:00:00 \
        --job-name="vpix_${tag}" \
        --output="${WORKDIR}/scripts/out/vpix_${tag}.%j" \
        --wrap="${PREAMBLE} && python3 -m deep_lst.evaluation_climpred.verif_pixel ${ARGS}")
    JOB_ANOM=$(sbatch --parsable "${VERIF_DEP[@]}" --partition=${VERIF_PARTITION} --mem=${mem} --cpus-per-task=4 --time=06:00:00 \
        --job-name="vanom_${tag}" \
        --output="${WORKDIR}/scripts/out/vanom_${tag}.%j" \
        --wrap="${PREAMBLE} && python3 -m deep_lst.evaluation_climpred.verif_anom ${ARGS}")
    JOB_DIURNAL=$(sbatch --parsable "${VERIF_DEP[@]}" --partition=${VERIF_PARTITION} --mem=${DIURNAL_MEM} --cpus-per-task=4 --time=03:00:00 \
        --job-name="vdiur_${tag}" \
        --output="${WORKDIR}/scripts/out/vdiur_${tag}.%j" \
        --wrap="${PREAMBLE} && python3 -m deep_lst.evaluation_climpred.verif_diurnal --experiment ${EXPERIMENT} --eval_name ${eval_name} --version ${VERSION}")
    EVAL_DEPS="${JOB_INIT}:${JOB_PIXEL}:${JOB_ANOM}:${JOB_DIURNAL}"
    JOB_MONTHLY=""
    if $gifs; then
        JOB_MONTHLY=$(sbatch --parsable "${VERIF_DEP[@]}" --partition=${VERIF_PARTITION} --mem=${mem} --cpus-per-task=4 --time=06:00:00 \
            --job-name="vpixm_${tag}" \
            --output="${WORKDIR}/scripts/out/vpixm_${tag}.%j" \
            --wrap="${PREAMBLE} && python3 -m deep_lst.evaluation_climpred.verif_pixel --experiment ${EXPERIMENT} --eval_name ${eval_name} --windows all --monthly --version ${VERSION}")
        EVAL_DEPS="${EVAL_DEPS}:${JOB_MONTHLY}"
    fi
    JOB_EVAL=$(sbatch --parsable --dependency=afterok:${EVAL_DEPS} \
        --kill-on-invalid-dep=yes --partition=${VERIF_PARTITION} --mem=${EVAL_MEM} --cpus-per-task=2 --time=02:00:00 \
        --job-name="eval_${tag}" \
        --output="${WORKDIR}/scripts/out/eval_${tag}.%j" \
        --wrap="${PREAMBLE} && python3 -m deep_lst.evaluation_climpred.evaluate --experiment ${EXPERIMENT} --eval_name ${eval_name} --months ${MONTHS} --version ${VERSION}")
    echo "${EXPERIMENT} (${eval_name}): verifs ${JOB_INIT},${JOB_PIXEL},${JOB_ANOM},${JOB_DIURNAL}${JOB_MONTHLY:+,${JOB_MONTHLY}} -> evaluate ${JOB_EVAL}"
done
echo "Done. Monitor: squeue -u \$USER"
