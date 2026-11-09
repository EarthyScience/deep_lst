#!/bin/bash
# Submit the residual-analysis GAM pipeline as 3 dependent SLURM stages (run this on the login node):
#   build (parallel per-region table -> parquet) -> fit array (6 tasks) -> aggregate
# Each fit task is one (model, response); they run concurrently.
#
# We call the env's python DIRECTLY (not `mamba run`) so the 6 parallel array tasks don't contend on
# mamba's cache lock; rpy2 needs R_HOME, set here and propagated via --export=ALL.
set -euo pipefail

cd /Net/Groups/BGI/people/mwessel/deep_lst
ENV=/Net/Groups/BGI/scratch/mwessel/miniforge3/envs/deep_lst_rgam
export R_HOME="$ENV/lib/R"
export PATH="$ENV/bin:$PATH"
export PYTHONUNBUFFERED=1

SCRIPT=deep_lst/exploration/residual_analysis.py
PY="$ENV/bin/python -u $SCRIPT"
OUT=/Net/Groups/BGI/people/mwessel/deep_lst/scripts/out
TABLE=/Net/Groups/BGI/people/mwessel/deep_lst/results/gam/era5_allregions/table.parquet
COMMON="--partition=work --export=ALL"

# Skip the (slow, I/O-bound) build if the table parquet already exists; pass --force-build to rebuild.
if [[ -f "$TABLE" && "${1:-}" != "--force-build" ]]; then
  echo "build       = SKIPPED ($TABLE exists; pass --force-build to rebuild)"
  dep=""
else
  build=$(sbatch --parsable $COMMON --job-name=gam_build --output=$OUT/gam_%x_%j.log \
    --cpus-per-task=9 --mem=96G --time=02:00:00 \
    --wrap "$PY build")
  echo "build       = $build"
  dep="--dependency=afterok:$build"
fi

fit=$(sbatch --parsable $dep $COMMON --job-name=gam_fit \
  --output=$OUT/gam_%x_%A_%a.log --array=0-17 \
  --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --wrap "$PY fit \$SLURM_ARRAY_TASK_ID")
echo "fit array   = $fit (tasks 0-17: 9 regions x day/night)"

agg=$(sbatch --parsable --dependency=afterok:$fit $COMMON --job-name=gam_aggregate --output=$OUT/gam_%x_%j.log \
  --cpus-per-task=1 --mem=8G --time=00:20:00 \
  --wrap "$PY aggregate")
echo "aggregate   = $agg"

plt=$(sbatch --parsable --dependency=afterok:$agg $COMMON --job-name=gam_plot --output=$OUT/gam_%x_%j.log \
  --cpus-per-task=1 --mem=8G --time=00:10:00 \
  --wrap "$PY plot")
echo "plot        = $plt (overlay curves per term)"
