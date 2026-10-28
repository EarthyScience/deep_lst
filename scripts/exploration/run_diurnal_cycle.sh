#!/bin/bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=scripts/out/diurnal_cycle_%j
#SBATCH --job-name=diurnal_cycle

# Submit from the project root:
#   sbatch scripts/exploration/run_diurnal_cycle.sh

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"

python3 ${WORKDIR}/deep_lst/exploration/diurnal_cycle.py
