#!/bin/bash
# Monthly climatologies from the full seviri store, one job each on the work partition:
#
#   fvc_monthly_climatology.py           daily leaf, 12 x (fvc_mean, fvc_std, valid_days)
#   lst_validity_monthly_climatology.py  hourly leaf, 12 x lst_validity (% valid hours)
#
# Output: /Net/Groups/BGI/work_5/scratch/mwessel/data/derived/{FVC,LSTvalidity}_monthly_climatology
# The hourly leaf is ~1 TB on disk, hence the longer walltime for the validity job.
#
# Usage: bash scripts/assets/launch_climatologies.sh [--dry-run]

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd ${WORKDIR}

DRY_RUN=false
[ "${1:-}" == "--dry-run" ] && DRY_RUN=true

PYTHON="PYTHONPATH=${WORKDIR} ${WORKDIR}/.venv/bin/python"

submit () {  # submit <name> <mem> <time> <script>
    local name=$1 mem=$2 time=$3 script=$4
    local cmd="cd ${WORKDIR} && ${PYTHON} scripts/assets/${script}"
    if $DRY_RUN; then
        echo "[dry] sbatch --partition=work --job-name=${name} --mem=${mem} --time=${time} --wrap=\"${cmd}\""
    else
        local job
        job=$(sbatch --parsable --partition=work --job-name=${name} --cpus-per-task=4 --mem=${mem} --time=${time} \
            --output="${WORKDIR}/scripts/out/${name}_%j.out" --wrap="${cmd}")
        echo "job ${job}: ${name}"
    fi
}

submit fvc_climatology          32G 12:00:00   fvc_monthly_climatology.py
submit lst_validity_climatology 48G 2-00:00:00 lst_validity_monthly_climatology.py
echo "Done. Monitor: squeue -u \$USER"
