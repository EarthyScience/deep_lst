#!/bin/bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=scripts/out/sampling_distributions%j
#SBATCH --job-name=sampling_distributions

# Submit from the project root:
#   sbatch scripts/exploration/run_sampling_distributions.sh

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"

python3 ${WORKDIR}/deep_lst/exploration/sampling_distributions.py
