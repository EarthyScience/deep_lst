#!/bin/bash
# Ablation evaluation on eval_mtg_new (the 3000 mtg_new training sites, midmonth-2023
# validation blocks). Per row: inference -> the three climpred verifs (afterok on the
# inference) -> evaluate; one tables job afterany on all evaluates recompiles the five
# supplement matrices (rows still missing render as MISSING, so a partial rerun is fine).
#
# Prerequisites (both once per eval name):
#   assets/spatial/indices_eval_mtg_new.npz -> indices_mtg_new.npz (class file symlink)
#   2023 baselines: sbatch scripts/assets/run_baselines.sh --region eval_mtg_new --test_year 2023 --out_key eval_mtg_new
#   (the mtg_new.nc files train_mtg_new uses are materialised on the 2024 test axis only)
#
# Usage: bash scripts/evaluation/run_ablation_eval.sh [--dry-run] [experiment ...]
#        (experiment args restrict inference + verification; the tables always list all rows)

set -eo pipefail
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

EVAL_NAME=eval_mtg_new
INFER=scripts/inference/run_ablation_inference.sh
CLIMPRED=scripts/evaluation/run_climpred.sh

# Supplement table rows (paper order); cells shared between tables are the same run.
BACKBONE_LOOKBACK="transformer_smoothl1_h24_full_v14_lb24_emb transformer_smoothl1_h24_full_v14_lb72_emb transformer_smoothl1_h24_full_v14_lb168_emb mlp_smoothl1_h24_full_v14_lb24_emb mlp_smoothl1_h24_full_v14_lb72_emb mlp_smoothl1_h24_full_v14_lb168_emb"
STATIC="transformer_smoothl1_h24_full_v14_lb24_emb transformer_smoothl1_h24_full_v14_lb24_int mlp_smoothl1_h24_full_v14_lb24_emb mlp_smoothl1_h24_full_v14_lb24_int"
LOSS="transformer_smoothl1_h24_full_v14_lb168_emb transformer_smoothl1_h24_full_v14_lb168_emb_noinc transformer_smoothl1_h24_full_v14_lb168_emb_noroll transformer_smoothl1_h24_full_v14_lb168_emb_notf transformer_mae_h24_full_v14_lb168_emb transformer_mae_h24_full_v14_lb168_emb_noinc transformer_mae_h24_full_v14_lb168_emb_noroll transformer_mae_h24_full_v14_lb168_emb_notf transformer_mse_h24_full_v14_lb168_emb transformer_mse_h24_full_v14_lb168_emb_noinc transformer_mse_h24_full_v14_lb168_emb_noroll transformer_mse_h24_full_v14_lb168_emb_notf"
DATA="transformer_smoothl1_h24_full_glofas_v14_lb168_int_noinc transformer_smoothl1_h24_noera5_v14_lb168_int_noinc transformer_smoothl1_h24_noera5land_v14_lb168_int_noinc transformer_smoothl1_h24_full_v14_lb168_int_noinc transformer_smoothl1_h24_nosevaux_v14_lb168_int_noinc"
HORIZON="transformer_smoothl1_h12_full_v14_lb168_int_noinc transformer_smoothl1_h24_full_v14_lb168_int_noinc transformer_smoothl1_h72_full_v14_lb168_int_noinc"

TABLES="python3 -m deep_lst.evaluation_climpred.tables_ablation --eval_name ${EVAL_NAME}"
TABLE_CMDS="${TABLES} --experiments ${BACKBONE_LOOKBACK} --out tables/ablation_backbone_lookback \
 && ${TABLES} --experiments ${STATIC} --out tables/ablation_static \
 && ${TABLES} --experiments ${LOSS} --out tables/ablation_loss \
 && ${TABLES} --experiments ${DATA} --out tables/ablation_data \
 && ${TABLES} --experiments ${HORIZON} --out tables/ablation_horizon"

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && { DRY_RUN=true; shift; }
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
else
    # union of the table rows, first occurrence wins
    EXPERIMENTS=($(tr ' ' '\n' <<< "${BACKBONE_LOOKBACK} ${STATIC} ${LOSS} ${DATA} ${HORIZON}" | awk '!seen[$0]++'))
fi

for kind in persistence climatology_hourly climatology_smoothed_hourly era5_land; do
    if [ ! -f "assets/baselines/${kind}/${EVAL_NAME}.nc" ]; then
        echo "MISSING assets/baselines/${kind}/${EVAL_NAME}.nc -- run the baselines first (see header)"
        $DRY_RUN || exit 1
    fi
done

PREAMBLE="cd ${WORKDIR} && source ${WORKDIR}/.venv/bin/activate && export PYTHONPATH=${WORKDIR}"

if $DRY_RUN; then
    bash ${INFER} --dry-run "${EXPERIMENTS[@]}"
    for exp in "${EXPERIMENTS[@]}"; do
        bash ${CLIMPRED} --dry-run --experiment="${exp}" --after=INFERENCE "${EVAL_NAME}"
    done
    echo "[dry] tables (afterany on all evaluates): ${TABLE_CMDS}"
    exit 0
fi

EVAL_JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
    INF_JOB=$(bash ${INFER} "${exp}" | sed -n "s/^${exp}: \([0-9]*\)$/\1/p")
    [ -n "${INF_JOB}" ] || { echo "no inference job for ${exp}"; exit 1; }
    EVAL_JOB=$(bash ${CLIMPRED} --experiment="${exp}" --after="${INF_JOB}" "${EVAL_NAME}" | sed -n 's/.*-> evaluate \([0-9]*\)$/\1/p')
    [ -n "${EVAL_JOB}" ] || { echo "no evaluate job for ${exp}"; exit 1; }
    EVAL_JOBS+=("${EVAL_JOB}")
    echo "${exp}: inference ${INF_JOB} -> evaluate ${EVAL_JOB}"
done

DEP=$(IFS=:; echo "${EVAL_JOBS[*]}")
JOB_TABLES=$(sbatch --parsable --dependency=afterany:${DEP} --partition=work --mem=16G --cpus-per-task=2 --time=00:30:00 \
    --job-name="ablation_tables" \
    --output="${WORKDIR}/scripts/out/ablation_tables_${EVAL_NAME}.%j" \
    --wrap="${PREAMBLE} && ${TABLE_CMDS}")
echo "tables: ${JOB_TABLES} (afterany on ${#EVAL_JOBS[@]} evaluates)"
echo "Done. Monitor: squeue -u \$USER"
