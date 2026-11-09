#!/bin/bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --output=scripts/out/run_baselines%j
#SBATCH --job-name=run_baselines

# All arguments are forwarded to run_baselines.py.
# --region is the eval CONFIG name (conf/eval/<region>.yaml), not a sampled-store key.
# The first run per region rebuilds the (skt) preload cache (~40 min), then extracts.
#
# Usage:
#   sbatch scripts/assets/run_baselines.sh --region eval_mtg_new_val --test_year 2024 --overwrite --force_sparse
#
# Submit for all regions (after fresh statistics): bash scripts/assets/launch_assets.sh

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"

python3 ${WORKDIR}/scripts/assets/run_baselines.py "$@"
