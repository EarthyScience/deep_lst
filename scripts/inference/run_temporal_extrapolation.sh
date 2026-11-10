#!/bin/bash
# Submit one GPU inference job per experiment, each on the spatial selection it was
# trained on (conf/eval/<name>.yaml; eval_name keys the output dir and baselines).
# Init times thinned to 00/06/12/18 UTC; medium-range 10-day (240 h) rollout.
#
# Usage: bash scripts/inference/run_temporal_extrapolation.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

EXPERIMENT=v18_mlp_smoothl1_h24_full_lstdiag
EVALS="train_mtg_new eval_mtg_new_val"

INFER_HORIZON=240
DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && DRY_RUN=true

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"

for eval_name in ${EVALS}; do
    exp=${EXPERIMENT}
    CMD="srun python3 ${WORKDIR}/scripts/inference/run_inference.py eval=${eval_name} experiment=${exp} version=version_0 best=true inference_horizon=${INFER_HORIZON} 'splits.test.init_hours=[0, 6, 12, 18]'"
    if $DRY_RUN; then
        echo "[dry] ${exp} (eval=${eval_name}):"; echo "    ${CMD}"
    else
        JOB=$(sbatch --parsable --partition=gpu --gres=gpu:1 \
            --cpus-per-task=32 --mem=256GB --nodes=1 --time=06:00:00 \
            --output="${WORKDIR}/scripts/out/inf_${eval_name}_${exp#mlp_rollout}.%j" \
            --job-name="inf_tm_${exp#mlp_rollout}" \
            --wrap="${PREAMBLE} && ${CMD}")
        echo "${exp} (eval=${eval_name}): ${JOB}"
    fi
done
echo "Done. Monitor: squeue -u \$USER"
