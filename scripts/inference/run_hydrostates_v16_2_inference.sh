#!/bin/bash
# Inference for the hydrostates v16.2 matrix (2026-10-18): 15 MLP runs x 2 evals.
#
#   train_mtg_hydrostates      3000 training sites, 2024, 6-hourly inits, 240 h
#   eval_mtg_hydrostates_val   500-site validity-decorrelated spatial holdout, same inits/horizon
#
# Defaults from conf/infer.yaml (init_hours 0/6/12/18, inference_horizon 240). The preload
# caches of both selections were written by the two training builders (7517885 base family,
# 7517895 rzsm), so every job loads them; no chaining needed. Cells whose forecast.zarr exists
# are skipped, so the script can be re-run to fill gaps.
#
# Usage: bash scripts/inference/run_hydrostates_v16_2_inference.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && DRY_RUN=true

DATA_CONFIGS="full_noswvl full full_rzsm"
SEEDS="42 7 23 11 5"
EVALS="eval_mtg_hydrostates_val train_mtg_hydrostates"

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"

for data in ${DATA_CONFIGS}; do
    for seed in ${SEEDS}; do
        tag=hydrostates
        [ "${seed}" != "42" ] && tag="hydrostates_s${seed}"
        exp="mlp_smoothl1_h24_${data}_v16.2_${tag}"
        for eval_name in ${EVALS}; do
            if [ -d "results/${exp}/version_0/${eval_name}/forecast.zarr" ]; then
                echo "skip ${exp} eval=${eval_name}: forecast.zarr exists"
                continue
            fi
            cmd="srun python3 ${WORKDIR}/scripts/inference/run_inference.py eval=${eval_name} experiment=${exp} version=version_0 best=true"
            if $DRY_RUN; then
                echo "[dry] ${exp} eval=${eval_name}"
                continue
            fi
            job=$(sbatch --parsable --partition=gpu --gres=gpu:1 \
                --cpus-per-task=32 --mem=256GB --nodes=1 --time=08:00:00 \
                --output="${WORKDIR}/scripts/out/inf_${eval_name}_${exp}.%j" \
                --job-name="inf_${eval_name}" \
                --wrap="${PREAMBLE} && ${cmd}")
            echo "job ${job}: ${exp} eval=${eval_name}"
        done
    done
done
echo "Done. Monitor: squeue -u \$USER"
