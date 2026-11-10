#!/bin/bash
# Inference for the deep_lst benchmarks on the 400-site eval_mtg holdout, so they can be compared
# against the WeatherGenerator land model on the same pixels: the 108 pixels of
# assets/baselines/WGL/v1/weathergen_landv0.nc are all eval_mtg sites.
#
# Caveat to carry into the write-up: these models were trained on `mtg_new`, which overlaps
# eval_mtg by 20 of its 400 sites (5%; 3 of the 108 WGL pixels), so this is mildly flattering
# to deep_lst. The clean version needs models trained on `mtg` (disjoint from eval_mtg), which
# is what scripts/assets/launch_statistics_mtg.sh + tag=mtgtrain prepare.
#
# Init times thinned to 00 UTC (daily); 7-day (168 h) rollout, matching the WGL v0 store.
#
# Usage: bash scripts/inference/run_eval_mtg_holdout.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

EXPERIMENTS=(
    mlp_smoothl1_h24_full_v16_lb168_int_noinc_e70
)
EVAL_NAME=eval_mtg
INFER_HORIZON=168

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && DRY_RUN=true

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"

# The preload cache is keyed on the eval region and stores, not the model, so two jobs on the
# same region race on the identical <cache>.tmp file and one renames it away from the other
# (dataset.py:451-453). Chain them so the first builds the cache and the rest reuse it.
PREVIOUS_JOB=""
for exp in "${EXPERIMENTS[@]}"; do
    CMD="srun python3 ${WORKDIR}/scripts/inference/run_inference.py eval=${EVAL_NAME} experiment=${exp} version=version_0 best=true inference_horizon=${INFER_HORIZON} 'splits.test.init_hours=[0]'"
    if $DRY_RUN; then
        echo "[dry] ${exp} (eval=${EVAL_NAME}):"; echo "    ${CMD}"
    else
        DEPENDENCY=""
        [ -n "$PREVIOUS_JOB" ] && DEPENDENCY="--dependency=afterany:${PREVIOUS_JOB}"
        JOB=$(sbatch --parsable ${DEPENDENCY} --partition=gpu --gres=gpu:1 \
            --cpus-per-task=32 --mem=256GB --nodes=1 --time=08:00:00 \
            --output="${WORKDIR}/scripts/out/inf_${EVAL_NAME}_${exp}.%j" \
            --job-name="inf_holdout_${exp%%_*}" \
            --wrap="${PREAMBLE} && ${CMD}")
        echo "${exp} (eval=${EVAL_NAME}): ${JOB} ${DEPENDENCY}"
        PREVIOUS_JOB=${JOB}
    fi
done
echo "Done. Monitor: squeue -u \$USER"
