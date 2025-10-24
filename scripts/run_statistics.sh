#!/bin/bash -x

#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128Gb
#SBATCH --time=01:00:00
#SBATCH --output=scripts/out/eo_stats.%j
#SBATCH --job-name=eo_stats

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

cd $WORKDIR

${WORKDIR}/.venv/bin/python3 ${WORKDIR}/run_statistics.py