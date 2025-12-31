#!/bin/bash -x

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120Gb
#SBATCH --time=24:00:00
#SBATCH --output=scripts/out/compute_statistics.%j
#SBATCH --job-name=compute_statistics

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR

EXPERIMENT="global"
MODALITY="era5"

${WORKDIR}/.venv/bin/python3 ${WORKDIR}/deep_lst/scripts/tools/compute_statistics.py \
    --modality $MODALITY \
    --statistics "assets/${MODALITY}/${EXPERIMENT}/statistics/statistics.npz" \
    --store "/Net/Groups/BGI/work_5/scratch/mwessel/wegen-stores/${MODALITY}_test2.zarr" \
    --indices "assets/${MODALITY}/${EXPERIMENT}/indices/indices_5p.npz" \
    --batch_size 1024 \
    --num_workers 26

