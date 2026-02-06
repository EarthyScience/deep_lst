#!/bin/bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=scripts/out/merge_fc.%j
#SBATCH --job-name=merge_fc

# Usage: sbatch --dependency=afterok:<job_id> run_merge.sh <config> <version>
# Or just: sbatch run_merge.sh <config> <version>

CONFIGNAME=${1:-mlp_rollout_multi}
VERSION=${2:-version_0}

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"

python3 ${WORKDIR}/scripts/analysis/merge_forecasts.py \
    --config ${CONFIGNAME} \
    --version ${VERSION} \
    --delete_chunks
