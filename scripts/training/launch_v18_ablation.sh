#!/bin/bash
# v18 ablation (2026-10-22/23): MLP runs on mtg_new / eval_mtg_new_val with the default recipe
# (lb168, integer statics, no incremental loss, 70 epochs; conf defaults since 2026-10-23), one
# cell per (model config, data config):
#
#   mlp_rollout:full                 no. 0, baseline (full since v18 = minus the ERA5 statics)
#   mlp_rollout:noera5static_nostl   no. 1: no. 0 minus ERA5-Land stl1, stl2
#   mlp_rollout:noera5static_skt     no. 2: no. 1 plus ERA5-Land skt
#   mlp_rollout:full_lstdiag         no. 4: the diagnostic regression (LST diagnostic only, history as context)
#   mlp_rollout:full_joint           joint head, init_fill none (gated at inference via the history nowcast)
#   mlp_gated:full_joint             joint head trained with init_fill gated (analysis head)
#   mlp_always:full_joint            joint head trained with init_fill always (analysis head)
#
# All cells share the 3-modality preload cache of full (the load lists are unchanged), so they
# are submitted without dependencies. Run names: v18_<_model_name>_smoothl1_h24_<data>, where
# _model_name is mlp for mlp_rollout and the variant name otherwise. Cells whose experiment
# directory exists are skipped.
#
# Usage: bash scripts/training/launch_v18_ablation.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

DRY_RUN=false
for arg in "$@"; do
    [ "$arg" == "--dry-run" ] && DRY_RUN=true
done

MLP_LAUNCH="scripts/training/train_mlp.sh"
MLP_ARGS="loss=smoothl1 horizon=h24 _spatial=mtg_new _spatial_val=eval_mtg_new_val _version=v18"
MLP_TIME="12:00:00"

CELLS="mlp_rollout:full mlp_rollout:noera5static_nostl mlp_rollout:noera5static_skt mlp_rollout:full_lstdiag mlp_rollout:full_joint mlp_gated:full_joint mlp_always:full_joint"

model_label () {
    case $1 in
        mlp_rollout) echo mlp ;;
        *) echo $1 ;;
    esac
}

for cell in ${CELLS}; do
    model=${cell%%:*}
    data=${cell##*:}
    name="v18_$(model_label ${model})_smoothl1_h24_${data}"
    if [ -d "experiments/${name}" ]; then
        echo "skip ${name}: experiment directory exists"
        continue
    fi
    if $DRY_RUN; then
        echo "[dry] sbatch --time=${MLP_TIME} ${MLP_LAUNCH} model=${model} ${MLP_ARGS} data=${data}"
    else
        job=$(sbatch --parsable --time=${MLP_TIME} ${MLP_LAUNCH} model=${model} ${MLP_ARGS} data=${data})
        echo "job ${job}: ${name}"
    fi
done
echo "Done. Monitor: squeue -u \$USER"
