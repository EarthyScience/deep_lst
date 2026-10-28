#!/bin/bash
# Regenerate all derived assets after (re)sampling the spatial/temporal indices:
#   1. statistics per modality (conf/statistics.yaml drives stores, pixel selection,
#      statistics temporal split, prognostic increments)
#   2. baseline extraction per eval region — submitted with afterok on ALL statistics
#      jobs, because baselines read region preloads that are normalized with (and
#      cache-fingerprinted on) the fresh stats zarrs.
#
#   bash scripts/assets/launch_assets.sh [spatial] [test_year]
#   e.g. bash scripts/assets/launch_assets.sh sites 2024   # stats over indices_sites.npz

SPATIAL=${1:-}      # empty -> conf/statistics.yaml default (currently mtg_new)
TEST_YEAR=${2:-2024}

STATS_JOBS=()
for MODALITY in seviri era5 era5_land; do
    JOB_ID=$(sbatch --parsable \
        --job-name=stats_${MODALITY} \
        scripts/assets/compute_statistics_distributed.sh \
        ${MODALITY} ${SPATIAL})
    STATS_JOBS+=(${JOB_ID})
    echo "stats_${MODALITY}: job ${JOB_ID} (spatial=${SPATIAL:-<yaml default>})"
done

DEPENDENCY=afterok:$(IFS=:; echo "${STATS_JOBS[*]}")

for REGION in dahra antananarivo gobabeb johannesburg bouamir yangambi passage sahel_niger west_africa eval_mtg train_mtg_new; do
    JOB_ID=$(sbatch --parsable \
        --dependency=${DEPENDENCY} \
        --job-name=baselines_${REGION} \
        scripts/assets/run_baselines.sh --region ${REGION} --test_year ${TEST_YEAR} --overwrite)
    echo "baselines_${REGION}: job ${JOB_ID} (${DEPENDENCY})"
done
