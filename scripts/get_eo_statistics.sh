#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64Gb
#SBATCH --time=04:00:00
#SBATCH --output=scripts/out/eo_stats.%j
#SBATCH --job-name=eo_stats

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

cd $WORKDIR

${WORKDIR}/.venv/bin/python3 ${WORKDIR}/deep_lst/get_eo_statistics.py \
    --modality $MODALITY \
    --variable $VAR \
    --experiment "global" \
    --stratification "month_hour" \
    --time_fraction 0.01 \
    --space_fraction 0.01 \

