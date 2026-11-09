#!/bin/bash
# Holdout inference (eval_mtg_new_val, 500 sites) for the v18 ablation runs on the CPU `work`
# partition (2026-10-23): the GPU queue was two days deep, and the 3.5M-parameter MLP over
# 500 pixels x 1464 inits x 240 h fits a 64-core node in a few hours. Init times thinned to
# 00/06/12/18 UTC; 10-day (240 h) rollout, as scripts/inference/run_temporal_extrapolation.sh.
# The holdout preload cache exists for these load lists, so the jobs run concurrently.
#
# Usage: bash scripts/inference/run_v18_holdout_cpu.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

EXPERIMENTS="v18_mlp_smoothl1_h24_full v18_mlp_smoothl1_h24_noera5static_nostl v18_mlp_smoothl1_h24_noera5static_skt"
EVAL_NAME=eval_mtg_new_val
INFER_HORIZON=240

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && DRY_RUN=true

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"

for exp in ${EXPERIMENTS}; do
    CMD="srun python3 ${WORKDIR}/scripts/inference/run_inference.py eval=${EVAL_NAME} experiment=${exp} version=version_0 best=true inference_horizon=${INFER_HORIZON} 'test_dataset.init_hours=[0, 6, 12, 18]'"
    if $DRY_RUN; then
        echo "[dry] ${exp} (eval=${EVAL_NAME}):"; echo "    ${CMD}"
    else
        JOB=$(sbatch --parsable --partition=work --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=200G --time=24:00:00 \
            --output="${WORKDIR}/scripts/out/inf_${EVAL_NAME}_${exp}.%j" \
            --job-name="inf_val_${exp#v18_mlp_smoothl1_h24_}" \
            --wrap="${PREAMBLE} && ${CMD}")
        echo "${exp} (eval=${EVAL_NAME}): ${JOB}"
    fi
done
echo "Done. Monitor: squeue -u \$USER"
