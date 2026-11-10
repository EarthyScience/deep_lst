#!/bin/bash
# Supplement ablation launcher (2026-08-31). Three blocks, 19 runs, no job chaining:
# the shared preload cache preload_mtg_new_b91b60fefa33b664.pt is verified present.
#
#   Block 1  backbone x lookback {24,72,168}, embedding statics          -> 6 runs
#   Block 2  backbone x static encoding at lb24 (integer only; the
#            embedding cells are block 1's lb24 runs)                    -> 2 runs
#   Block 3  loss components {full,noinc,noroll,notf} x loss functions
#            {smoothl1,mae,mse} on the transformer lb168_emb base;
#            full x smoothl1 = block 1's lb168_emb run                   -> 11 runs
#
# Batch convention (matches v14): global batch 512 everywhere.
# Transformers run DDP on 4 GPUs with per-rank batch 128 (lb168 at 512x1 OOMs an A40);
# MLPs run 1 GPU at batch 512.
#
# max_epochs=40 uniform (sizes the cosine decay): the v14 lb168 transformer at 80 was
# walltime-killed at epoch 49 mid-decay; 40 finishes inside 24h and matches the tuned
# e40 precedent. Walltime: transformer 24h, mlp 12h (measured ~28 vs ~8 min/epoch).
#
# Usage: bash scripts/training/launch_ablation.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && DRY_RUN=true

TF_LAUNCH="scripts/training/train_transformer.sh"
MLP_LAUNCH="scripts/training/train_mlp.sh"
TF_ARGS="model=transformer objective=rollout training.batch_size=128 training.max_epochs=40"
MLP_ARGS="model=mlp objective=rollout training.max_epochs=40"
BASE="loss=smoothl1 horizon=h24 data=full"
TF_TIME="24:00:00"
MLP_TIME="12:00:00"

submit () {
    local gres=$1 time=$2 launcher=$3; shift 3
    if $DRY_RUN; then
        echo "[dry] sbatch --time=${time} --gres=gpu:${gres} ${launcher} $*"
    else
        local job
        job=$(sbatch --parsable --time=${time} --gres=gpu:${gres} ${launcher} "$@")
        echo "job ${job}: $*"
    fi
}

# ---- Block 1: backbone x lookback (embedding statics) ----
for lb in 24 72 168; do
    submit 4 ${TF_TIME} ${TF_LAUNCH} ${TF_ARGS} ${BASE} window.lookback=${lb} tag=lb${lb}_emb
    submit 1 ${MLP_TIME} ${MLP_LAUNCH} ${MLP_ARGS} ${BASE} window.lookback=${lb} tag=lb${lb}_emb
done

# ---- Block 2: backbone x static encoding at lb24 (integer variants) ----
submit 4 ${TF_TIME} ${TF_LAUNCH} ${TF_ARGS} ${BASE} window.lookback=24 model.static_encoding=integer tag=lb24_int
submit 1 ${MLP_TIME} ${MLP_LAUNCH} ${MLP_ARGS} ${BASE} window.lookback=24 model.static_encoding=integer tag=lb24_int

# ---- Block 3: loss components x loss functions (transformer lb168_emb base) ----
for loss in smoothl1 mae mse; do
    for comp in full noinc noroll notf; do
        [ "${loss}" == "smoothl1" ] && [ "${comp}" == "full" ] && continue
        overrides=""
        [ "${comp}" == "noinc" ]  && overrides="objective.terms.increment=0"
        [ "${comp}" == "noroll" ] && overrides="objective.terms.rollout=0"
        [ "${comp}" == "notf" ]   && overrides="objective.terms.teacher_forcing=0"
        tag="lb168_emb"
        [ "${comp}" != "full" ] && tag="lb168_emb_${comp}"
        submit 4 ${TF_TIME} ${TF_LAUNCH} ${TF_ARGS} loss=${loss} horizon=h24 data=full window.lookback=168 ${overrides} tag=${tag}
    done
done

echo "Done. Monitor: squeue -u \$USER"
