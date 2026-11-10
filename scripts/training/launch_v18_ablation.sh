#!/bin/bash
# v18 ablation (2026-10-22/23): MLP runs on mtg_new / eval_mtg_new_val with the default recipe
# (lb168, integer statics, no incremental loss, 70 epochs; conf defaults since 2026-10-23), one
# cell per (objective config, data config):
#
#   rollout:full                 no. 0, baseline (full since v18 = minus the ERA5 statics)
#   rollout:noera5static_nostl   no. 1: no. 0 minus ERA5-Land stl1, stl2
#   rollout:noera5static_skt     no. 2: no. 1 plus ERA5-Land skt
#   rollout:full_lstdiag         no. 4: the diagnostic regression (LST diagnostic only, history as context)
#   rollout:full_joint           joint head, init_fill none (gated at inference via the history nowcast)
#   rollout_gated:full_joint     joint head trained with init_fill gated (analysis head)
#   rollout_always:full_joint    joint head trained with init_fill always (analysis head)
#
# All cells share the 3-modality preload cache of full (the load lists are unchanged), so they
# are submitted without dependencies. Run names: v18_mlp_<objective>_smoothl1_h24_<data>. The
# campaign was trained before the run name carried the objective, as
# v18_<mlp|mlp_gated|mlp_always>_smoothl1_h24_<data> (the former model configs mlp_rollout,
# mlp_gated, mlp_always). Cells whose experiment directory exists under either name are skipped.
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
MLP_ARGS="model=mlp loss=smoothl1 horizon=h24 _spatial=mtg_new _spatial_val=eval_mtg_new_val _version=v18"
MLP_TIME="12:00:00"

CELLS="rollout:full rollout:noera5static_nostl rollout:noera5static_skt rollout:full_lstdiag rollout:full_joint rollout_gated:full_joint rollout_always:full_joint"

legacy_model_label () {
    case $1 in
        rollout) echo mlp ;;
        rollout_gated) echo mlp_gated ;;
        rollout_always) echo mlp_always ;;
    esac
}

for cell in ${CELLS}; do
    objective=${cell%%:*}
    data=${cell##*:}
    name="v18_mlp_${objective}_smoothl1_h24_${data}"
    legacy_name="v18_$(legacy_model_label ${objective})_smoothl1_h24_${data}"
    if [ -d "experiments/${name}" ] || [ -d "experiments/${legacy_name}" ]; then
        echo "skip ${name}: experiment directory exists"
        continue
    fi
    if $DRY_RUN; then
        echo "[dry] sbatch --time=${MLP_TIME} ${MLP_LAUNCH} objective=${objective} ${MLP_ARGS} data=${data}"
    else
        job=$(sbatch --parsable --time=${MLP_TIME} ${MLP_LAUNCH} objective=${objective} ${MLP_ARGS} data=${data})
        echo "job ${job}: ${name}"
    fi
done
echo "Done. Monitor: squeue -u \$USER"
