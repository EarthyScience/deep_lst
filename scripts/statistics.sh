#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32Gb
#SBATCH --time=00:30:00
#SBATCH --output=scripts/out/eo_stats.%j
#SBATCH --job-name=eo_stats

WORKDIR=/Net/Groups/BGI/people/mwessel/lst-regression

cd $WORKDIR

# writes seviri data into the wedata-store
${WORKDIR}/.venv/bin/python3 ${WORKDIR}/code/statistics.py