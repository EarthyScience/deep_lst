#!/bin/bash
# Inference for the ablation campaign on the mtg_new TRAINING sites, VALIDATION period
# (eval=eval_mtg_new: 3000 sites, midmonth 2023, temporal=val_2023b10) — ablation selection
# stays off the 2024 test year. 240 h rollout, inits thinned to 00/06/12/18 UTC.
#
# Jobs run UNCHAINED: the preload cache is keyed by anchors + load lists (not the temporal
# split), so the mtg_new training caches are hit (3-modality and the glofas-family 4-modality
# one). If stats zarrs were regenerated since, the fingerprint changes and parallel jobs would
# race on the <cache>.tmp rebuild — chain them then.
#
# Usage: bash scripts/inference/run_ablation_inference.sh [--dry-run] [experiment ...]
#        (no experiment args = the 26 rows of the five supplement tables; the row lists
#        live in scripts/evaluation/run_ablation_eval.sh)

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

DEFAULT_EXPERIMENTS=(
    transformer_smoothl1_h24_full_v14_lb24_emb
    transformer_smoothl1_h24_full_v14_lb72_emb
    transformer_smoothl1_h24_full_v14_lb168_emb
    mlp_smoothl1_h24_full_v14_lb24_emb
    mlp_smoothl1_h24_full_v14_lb72_emb
    mlp_smoothl1_h24_full_v14_lb168_emb
    transformer_smoothl1_h24_full_v14_lb24_int
    mlp_smoothl1_h24_full_v14_lb24_int
    transformer_smoothl1_h24_full_v14_lb168_emb_noinc
    transformer_smoothl1_h24_full_v14_lb168_emb_noroll
    transformer_smoothl1_h24_full_v14_lb168_emb_notf
    transformer_mae_h24_full_v14_lb168_emb
    transformer_mae_h24_full_v14_lb168_emb_noinc
    transformer_mae_h24_full_v14_lb168_emb_noroll
    transformer_mae_h24_full_v14_lb168_emb_notf
    transformer_mse_h24_full_v14_lb168_emb
    transformer_mse_h24_full_v14_lb168_emb_noinc
    transformer_mse_h24_full_v14_lb168_emb_noroll
    transformer_mse_h24_full_v14_lb168_emb_notf
    transformer_smoothl1_h24_full_glofas_v14_lb168_int_noinc
    transformer_smoothl1_h24_noera5_v14_lb168_int_noinc
    transformer_smoothl1_h24_noera5land_v14_lb168_int_noinc
    transformer_smoothl1_h24_full_v14_lb168_int_noinc
    transformer_smoothl1_h24_nosevaux_v14_lb168_int_noinc
    transformer_smoothl1_h12_full_v14_lb168_int_noinc
    transformer_smoothl1_h72_full_v14_lb168_int_noinc
)
EVAL_NAME=eval_mtg_new
INFER_HORIZON=240

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && { DRY_RUN=true; shift; }
EXPERIMENTS=("${@:-${DEFAULT_EXPERIMENTS[@]}}")

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=${WORKDIR} && \
    unset SLURM_MEM_PER_GPU SLURM_MEM_PER_CPU"

# 3000 px x 480 inits: the final model's train_mtg_new run (1460 inits) took 6.3 h on an
# A100, so ~2 h here, ~4 h on an A40; 8 h walltime as in run_temporal_extrapolation.sh.
for exp in "${EXPERIMENTS[@]}"; do
    CMD="srun python3 ${WORKDIR}/scripts/inference/run_inference.py eval=${EVAL_NAME} experiment=${exp} version=version_0 best=true inference_horizon=${INFER_HORIZON} 'test_dataset.init_hours=[0, 6, 12, 18]'"
    if $DRY_RUN; then
        echo "[dry] ${exp} (eval=${EVAL_NAME}):"; echo "    ${CMD}"
    else
        JOB=$(sbatch --parsable --partition=gpu --gres=gpu:1 \
            --cpus-per-task=32 --mem=256GB --nodes=1 --time=08:00:00 \
            --output="${WORKDIR}/scripts/out/inf_${EVAL_NAME}_${exp}.%j" \
            --job-name="inf_val_${exp%%_*}" \
            --wrap="${PREAMBLE} && ${CMD}")
        echo "${exp}: ${JOB}"
    fi
done
echo "Done. Monitor: squeue -u \$USER"
