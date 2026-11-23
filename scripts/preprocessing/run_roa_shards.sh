#!/bin/bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --job-name=roa_shards

# One array task of the Rain over Africa preprocessing: writes its share of the shards of
# roa_land_expected_hourly_acc.zarr (step "shards" of preprocess_roa.py). Submitted by
# scripts/preprocessing/launch_roa_preprocessing.sh.
#
# The shards 4200 to 5113 hold the hours 2016-01-01T01 to 2026-01-01T00 (printed by the setup
# step). They are split evenly over the array tasks. One shard needs about 90 s and 4 GB.
# A task that is repeated skips the shards that are already written.

FIRST_SHARD=4200
LAST_SHARD=5113
N_TASKS=20      # must match --array above

n_shards=$((LAST_SHARD - FIRST_SHARD + 1))
shards_per_task=$(((n_shards + N_TASKS - 1) / N_TASKS))
first=$((FIRST_SHARD + SLURM_ARRAY_TASK_ID * shards_per_task))
last=$((first + shards_per_task - 1))
if [ ${last} -gt ${LAST_SHARD} ]; then
    last=${LAST_SHARD}
fi

echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname): shards ${first} to ${last}"
bash /Net/Groups/BGI/people/mwessel/deep_lst/scripts/preprocessing/preprocess_roa.sh shards \
    --first_shard ${first} --last_shard ${last}
