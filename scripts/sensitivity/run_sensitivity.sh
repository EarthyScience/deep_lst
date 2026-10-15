#!/bin/bash
# Submit one sensitivity job (GPU): every temporal input of the run permuted one at a time on one
# eval, then scored. Extra arguments are passed to run_sensitivity.py as Hydra overrides.
#
# Usage: bash scripts/sensitivity/run_sensitivity.sh <experiment> <eval> [hydra overrides...]
#   bash scripts/sensitivity/run_sensitivity.sh mlp_smoothl1_h24_full_v16_hydrostates eval_mtg
#   bash scripts/sensitivity/run_sensitivity.sh mlp_smoothl1_h24_full_v16_hydrostates eval_mtg \
#       'sensitivity.variables=[era5_land.swvl1,era5_land.swvl2]' sensitivity.block=all

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
EXPERIMENT=$1
EVAL=$2
shift 2
OVERRIDES="$*"

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"
CMD="srun python3 ${WORKDIR}/scripts/sensitivity/run_sensitivity.py experiment=${EXPERIMENT} eval=${EVAL} version=version_0 best=true ${OVERRIDES}"

JOB=$(sbatch --parsable --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=256GB --nodes=1 --time=12:00:00 \
    --output="${WORKDIR}/scripts/out/sens_${EVAL}_${EXPERIMENT}.%j" --job-name="sens_${EVAL}" \
    --wrap="${PREAMBLE} && ${CMD}")
echo "job ${JOB}: ${EXPERIMENT} eval=${EVAL} ${OVERRIDES}"
