#!/bin/bash
# Inference matrix for the hydrostates ladder (2026-10-12): 12 MLP runs x 2 evals.
#
#   train_mtg_new  3000 training sites, 2024, 6-hourly inits, 240 h   (training-site skill)
#   eval_mtg       400 MTG holdout sites, same inits/horizon            (spatial generalisation)
#
# Defaults from conf/infer.yaml (init_hours 0/6/12/18, inference_horizon 240) match the existing
# e70 inference on train_mtg_new (1460 inits x 241 leads). Cells whose forecast.zarr exists are
# skipped, so the script can be re-run to fill gaps.
#
# Preload caches: both mtg_new caches exist. The eval_mtg cache for the glofas configs is
# built by the first glofas eval_mtg job; the other glofas eval_mtg jobs are chained behind it
# (afterany) so they load it instead of each re-reading the stores.
#
# Usage: bash scripts/inference/run_hydrostates_inference.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && DRY_RUN=true

DATA_CONFIGS="full_noswvl" # "full full_rzsm full_rzsm_dis full_glofas"
SEEDS="42 7 23"
EVALS="train_mtg_new eval_mtg"

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"

submit () {  # submit <experiment> <eval> [dependency-job]
    local exp=$1 eval_name=$2 after=${3:-}
    local cmd="srun python3 ${WORKDIR}/scripts/inference/run_inference.py eval=${eval_name} experiment=${exp} version=version_0 best=true"
    local dependency=""
    [ -n "${after}" ] && dependency="--dependency=afterany:${after}"
    if $DRY_RUN; then
        echo "[dry] ${exp} eval=${eval_name} ${dependency}"
        echo "    ${cmd}"
        echo ""
    else
        local job
        job=$(sbatch --parsable ${dependency} --partition=gpu --gres=gpu:1 \
            --cpus-per-task=32 --mem=256GB --nodes=1 --time=08:00:00 \
            --output="${WORKDIR}/scripts/out/inf_${eval_name}_${exp}.%j" \
            --job-name="inf_${eval_name}" \
            --wrap="${PREAMBLE} && ${cmd}")
        echo "job ${job}: ${exp} eval=${eval_name} ${dependency}"
        echo "${job}"
    fi
}

CACHE_BUILDER=""   # first glofas eval_mtg job; later glofas eval_mtg jobs wait for it
for data in ${DATA_CONFIGS}; do
    for seed in ${SEEDS}; do
        tag=hydrostates
        [ "${seed}" != "42" ] && tag="hydrostates_s${seed}"
        exp="mlp_smoothl1_h24_${data}_v16_${tag}"
        for eval_name in ${EVALS}; do
            if [ -d "results/${exp}/version_0/${eval_name}/forecast.zarr" ]; then
                echo "skip ${exp} eval=${eval_name}: forecast.zarr exists"
                continue
            fi
            after=""
            if [ "${eval_name}" == "eval_mtg" ] && [ "${data}" != "full" ]; then
                after="${CACHE_BUILDER}"
            fi
            job=$(submit "${exp}" "${eval_name}" "${after}" | tee /dev/stderr | tail -n 1)
            if [ "${eval_name}" == "eval_mtg" ] && [ "${data}" != "full" ] && [ -z "${CACHE_BUILDER}" ] && ! $DRY_RUN; then
                CACHE_BUILDER="${job}"
            fi
        done
    done
done
echo "Done. Monitor: squeue -u \$USER"
