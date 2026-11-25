#!/bin/bash -x

#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32Gb
#SBATCH --time=00:30:00
#SBATCH --output=scripts/out/compute_statistics_%x.%j.out
#SBATCH --job-name=compute_statistics

# One modality per job; everything else (stores, MTG pixel selection, statistics temporal
# split, prognostic increment variables) comes from conf/statistics.yaml.
#
# No GPU needed. Measured on spatial=mtg_new (3000 pixels, 120 statistics days), 2026-11-24,
# 8 threads: seviri 2 min; the reads of the other modalities take 1 to 5 min. The requests
# below leave ample headroom; a larger pixel selection would need more memory.
#
# Usage:
#   sbatch --job-name=stats_seviri    scripts/assets/compute_statistics_distributed.sh seviri
#   sbatch --job-name=stats_era5      scripts/assets/compute_statistics_distributed.sh era5 sites
#   sbatch --job-name=stats_era5_land scripts/assets/compute_statistics_distributed.sh era5_land
#
# Optional arg $2: spatial selection name (assets/spatial/indices_<spatial>.npz);
# empty -> conf/statistics.yaml default.
# Optional arg $3: number of threads that read the statistics days (default 8).

export PYTHONUNBUFFERED=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR

MODALITY=${1:?Usage: sbatch compute_statistics_distributed.sh <seviri|era5|era5_land> [spatial] [num_workers]}
SPATIAL=${2:-}      # empty -> conf/statistics.yaml default
NUM_WORKERS=${3:-8}

SPATIAL_OVERRIDE=""
if [ -n "${SPATIAL}" ]; then
    SPATIAL_OVERRIDE="spatial=${SPATIAL}"
fi

srun ${WORKDIR}/.venv/bin/python ${WORKDIR}/scripts/assets/compute_statistics.py \
    modality=${MODALITY} \
    num_workers=${NUM_WORKERS} \
    ${SPATIAL_OVERRIDE}
