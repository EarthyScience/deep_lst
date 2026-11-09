#!/bin/bash
# Submit one GPU inference job per eval region.
#
# bash scripts/inference/run_spatial_extrapolation.sh experiment=transformer_smoothl1_h24_full_v16_lb168_int_noinc_e70_fullattn version=version_0 best=true
#
# CPU alternative when the GPU queue is deep (the MLP fits a `work` node; threads feed torch):
# bash scripts/inference/run_spatial_extrapolation.sh --partition=work --gres= --cpus=64 --threads=48 --mem=256GB --time=36:00:00 experiment=<run> version=version_0 best=true

set -e

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
EVALS=(train_mtg_new.yaml yangambi.yaml sahel_niger.yaml zambia.yaml)

JOB_TIME="48:00:00"
JOB_PARTITION="gpu"
JOB_GRES="gpu:A100:1"
JOB_MEM="256GB"
STRIDE=""
JOB_CPUS=32
JOB_THREADS=1
HYDRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --time=*)      JOB_TIME="${arg#--time=}" ;;
        --partition=*) JOB_PARTITION="${arg#--partition=}" ;;
        --gres=*)      JOB_GRES="${arg#--gres=}" ;;
        --mem=*)       JOB_MEM="${arg#--mem=}" ;;
        --stride=*)    STRIDE="${arg#--stride=}" ;;
        --cpus=*)      JOB_CPUS="${arg#--cpus=}" ;;
        --threads=*)   JOB_THREADS="${arg#--threads=}" ;;
        *)             HYDRA_ARGS+=("$arg") ;;
    esac
done

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=${JOB_THREADS} MKL_NUM_THREADS=${JOB_THREADS} OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"
HYDRA_STR="${HYDRA_ARGS[*]}"

GRES_FLAG=""
[ -n "${JOB_GRES}" ] && GRES_FLAG="--gres=${JOB_GRES}"

for eval in "${EVALS[@]}"; do
    name="${eval%.yaml}"
    STRIDE_ARGS=""
    if [ -n "${STRIDE}" ]; then
        name="${name}_s${STRIDE}"
        STRIDE_ARGS="eval_name=${name} test_dataset.temporal=test_stride${STRIDE}"
    fi
    JOB=$(sbatch --parsable \
        --partition=${JOB_PARTITION} ${GRES_FLAG} \
        --cpus-per-task=${JOB_CPUS} --mem=${JOB_MEM} --nodes=1 --time=${JOB_TIME} \
        --output="${WORKDIR}/scripts/out/inference_${name}.%j" \
        --job-name="inf_${name}" \
        --wrap="${PREAMBLE} && srun python3 ${WORKDIR}/scripts/inference/run_inference.py eval=${eval} ${STRIDE_ARGS} ${HYDRA_STR}")
    echo "${name}: ${JOB}"
done
