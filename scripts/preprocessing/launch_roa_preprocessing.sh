#!/bin/bash
# Submit the full Rain over Africa preprocessing to Slurm:
#
#   1. an array job that writes all shards of roa_land_expected_hourly_acc.zarr
#      (scripts/preprocessing/run_roa_shards.sh, 20 tasks)
#   2. one job that registers the written hours in _indices, started only if every array
#      task ended without error
#
# Run "bash scripts/preprocessing/preprocess_roa.sh setup" once before. Logs go to scripts/out/.
# If an array task fails, the indices job stays pending and has to be cancelled; submit this
# script again after fixing the cause: finished shards are skipped.
#
# Usage: bash scripts/preprocessing/launch_roa_preprocessing.sh

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
OUT_DIR=${WORKDIR}/scripts/out

shards_job=$(sbatch --parsable \
    --output="${OUT_DIR}/roa_shards_%A_%a.out" \
    ${WORKDIR}/scripts/preprocessing/run_roa_shards.sh)
echo "job ${shards_job}: roa_shards (array)"

indices_job=$(sbatch --parsable --partition=work --job-name=roa_indices \
    --cpus-per-task=1 --mem=8GB --time=01:00:00 \
    --dependency=afterok:${shards_job} \
    --output="${OUT_DIR}/roa_indices_%j.out" \
    --wrap="bash ${WORKDIR}/scripts/preprocessing/preprocess_roa.sh indices")
echo "job ${indices_job}: roa_indices (after job ${shards_job})"
echo "Monitor: squeue -u \$USER"
