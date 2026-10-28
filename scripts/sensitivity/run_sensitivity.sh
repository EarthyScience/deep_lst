#!/bin/bash
# Input sensitivity of one trained run on the CPU partition (deep_lst/sensitivity, 2026-10-22).
#
# One SLURM job per permutation block, each on a full `work` node, running
# scripts/sensitivity/run_sensitivity.py, which
#   1. loads the run's saved config.yml and best checkpoint (deep_lst.utils.config.load_model),
#   2. builds the eval dataset from the same preload cache the inference used (every stored
#      input of the eval sites for 2016-2025, normalised), inits thinned to 00 UTC of the test
#      year with test_dataset.init_hours=[0] (365 inits x 500 holdout sites; all four init
#      hours would take four times longer on CPU),
#   3. for every stored temporal input (ERA5, ERA5-Land, SEVIRI FVC, GloFAS when loaded; computed
#      features and statics have no day axis) shuffles that input's days per cell -- within
#      calendar months (block=month: day-to-day anomalies only) or across the whole record
#      (block=all: seasonal cycle too) -- runs the 240 h rollout for every init and writes
#      results/<experiment>/version_0/<eval>/sensitivity/permute_days_<block>/<modality>.<variable>.zarr,
#   4. scores every store against the run's own forecast.zarr on the samples where both forecasts
#      and the observation are valid: scores.nc (per pixel) and summary.csv (pooled per solar
#      window and forecast day), deep_lst/sensitivity/scores.py.
# Existing stores are skipped, so a killed job can simply be resubmitted. About 15 inputs x
# 8 batches x 56 s on 64 cores = 2 h per block. With --score only step 4 runs (the stores
# exist and are skipped): a small allocation, minutes; use it after a change of the scorer.
#
# Usage: bash scripts/sensitivity/run_sensitivity.sh <experiment> <eval_name> [--score] [<block> ...]
#   bash scripts/sensitivity/run_sensitivity.sh mlp_smoothl1_h24_full_v16.2_hydrostates eval_mtg_hydrostates_val
#   bash scripts/sensitivity/run_sensitivity.sh <experiment> <eval_name> month      # one block only
#   bash scripts/sensitivity/run_sensitivity.sh <experiment> <eval_name> --score    # re-score both blocks

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
EXPERIMENT=$1
EVAL=$2
shift 2
SCORE_ONLY=false
BLOCKS=""
for arg in "$@"; do
    case "$arg" in
        --score) SCORE_ONLY=true ;;
        *)       BLOCKS="${BLOCKS} ${arg}" ;;
    esac
done
BLOCKS=${BLOCKS:-month all}

THREADS=48   # torch intra-op threads; the remaining 16 cores feed the loader workers
RESOURCES="--partition=work --nodes=1 --cpus-per-task=64 --mem=120G --time=08:00:00"
if $SCORE_ONLY; then
    THREADS=4
    RESOURCES="--partition=small,work --nodes=1 --cpus-per-task=4 --mem=32G --time=01:00:00"
fi
PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS} PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"

for block in ${BLOCKS}; do
    CMD="srun python3 ${WORKDIR}/scripts/sensitivity/run_sensitivity.py experiment=${EXPERIMENT} eval=${EVAL} \
        version=version_0 best=true sensitivity.block=${block} test_dataset.init_hours=[0] analysis.num_workers=16"
    JOB=$(sbatch --parsable ${RESOURCES} \
        --job-name="sens_${block}_${EXPERIMENT}" --output="${WORKDIR}/scripts/out/sens_${EVAL}_${EXPERIMENT}_${block}.%j" \
        --wrap="${PREAMBLE} && ${CMD}")
    echo "job ${JOB}: ${EXPERIMENT} eval=${EVAL} block=${block}$($SCORE_ONLY && echo ' (score only)')"
done
