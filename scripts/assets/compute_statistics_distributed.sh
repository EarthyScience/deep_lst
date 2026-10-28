#!/bin/bash -x

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32Gb
#SBATCH --time=02:00:00
#SBATCH --output=scripts/out/compute_statistics_%x.%j.out
#SBATCH --job-name=compute_statistics

# One modality per job; everything else (stores, MTG pixel selection, statistics temporal
# split, prognostic increment variables) comes from conf/statistics.yaml.
#
# Resources measured on spatial=mtg (3000 pixels, 375 batches), 2026-08-26:
#   seviri 20 min, era5_land 22 min, era5 67 min; peak RSS ~12 GB; ~2.7 cores in use
#   (num_workers=0, serial reads). Requests below leave roughly 2x headroom on time and
#   memory; a larger pixel selection would need them raised proportionally.
#
# Usage:
#   sbatch --job-name=stats_seviri    scripts/assets/compute_statistics_distributed.sh seviri
#   sbatch --job-name=stats_era5      scripts/assets/compute_statistics_distributed.sh era5 sites
#   sbatch --job-name=stats_era5_land scripts/assets/compute_statistics_distributed.sh era5_land
#
# Optional arg $2: spatial selection name (assets/spatial/indices_<spatial>.npz);
# empty -> conf/statistics.yaml default.
# Optional arg $3: DataLoader worker count (default 0 — serial reads; raise once
# fork + zarr 3.2 async stores are confirmed safe).

export PYTHONUNBUFFERED=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR

MODALITY=${1:?Usage: sbatch compute_statistics_distributed.sh <seviri|era5|era5_land> [spatial] [num_workers]}
SPATIAL=${2:-}      # empty -> conf/statistics.yaml default
NUM_WORKERS=${3:-0}

SPATIAL_OVERRIDE=""
if [ -n "${SPATIAL}" ]; then
    SPATIAL_OVERRIDE="spatial=${SPATIAL}"
fi

srun ${WORKDIR}/.venv/bin/python ${WORKDIR}/scripts/assets/compute_statistics.py \
    modality=${MODALITY} \
    num_workers=${NUM_WORKERS} \
    ${SPATIAL_OVERRIDE}
