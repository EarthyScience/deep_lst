#!/bin/bash
# Statistics only, for one spatial selection. Unlike launch_assets.sh this does NOT re-extract
# the region baselines: those are physical (Kelvin) series tied to an eval region, and adding a
# new stats selection leaves them untouched.
#
# Needed because train.yaml resolves stats_spatial from _spatial, so training on
# indices_mtg.npz requires *_stats_mtg.zarr to exist. `mtg` is the 3000-site selection that is
# disjoint from the 400-site eval_mtg holdout (verified: 0 shared sites), which is what makes a
# fair comparison against the wgl model possible.
#
#   bash scripts/assets/launch_statistics_mtg.sh [spatial]
#   e.g. bash scripts/assets/launch_statistics_mtg.sh mtg

SPATIAL=${1:-mtg}

for MODALITY in seviri era5 era5_land; do
    JOB_ID=$(sbatch --parsable \
        --job-name=stats_${MODALITY}_${SPATIAL} \
        scripts/assets/compute_statistics_distributed.sh \
        ${MODALITY} ${SPATIAL})
    echo "stats_${MODALITY}: job ${JOB_ID} (spatial=${SPATIAL})"
done

echo
echo "When all three finish, assets/statistics/ should hold:"
for MODALITY in seviri era5 era5_land; do
    echo "  ${MODALITY}_stats_${SPATIAL}.zarr"
done
echo "Then launch training with _spatial=${SPATIAL} (stats_spatial follows it automatically)."
