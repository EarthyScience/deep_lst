#!/bin/bash -x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=scripts/out/run_fc_chunk%a.%j
#SBATCH --job-name=run_fc_par
#SBATCH --array=0-2  # Adjust based on number of chunks (0 to N-1)

# Usage: sbatch run_forecast_parallel.sh <config> <version> <start_date> <end_date> [skip_mask]
# Example: sbatch run_forecast_parallel.sh madagaskar_rollout version_0 "2017-12-15" "2017-12-18"
# Example with no cloud mask: sbatch run_forecast_parallel.sh madagaskar_rollout version_0 "2017-12-15" "2017-12-18" true
# This will split 3 days into 3 chunks (one per array task)

CONFIGNAME=${1:-mlp_rollout_multi}
VERSION=${2:-version_0}
START_DATE=${3:-"2017-12-15"}
END_DATE=${4:-"2017-12-18"}
SKIP_MASK=${5:-"false"}  # pass "true" to disable consistent cloud mask

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"

# Calculate time chunk for this array task
# Using Python to handle date arithmetic - output as tab-separated to handle spaces in values
CHUNK_TIMES=$(python3 -c "
from datetime import datetime, timedelta

start = datetime.strptime('${START_DATE}', '%Y-%m-%d')
end = datetime.strptime('${END_DATE}', '%Y-%m-%d')
total_days = (end - start).days
n_tasks = int('${SLURM_ARRAY_TASK_MAX}') - int('${SLURM_ARRAY_TASK_MIN}') + 1
task_id = int('${SLURM_ARRAY_TASK_ID}')

days_per_task = total_days / n_tasks
chunk_start = start + timedelta(days=task_id * days_per_task)
chunk_end = start + timedelta(days=(task_id + 1) * days_per_task)

# Output tab-separated for reliable parsing
print(f\"{chunk_start.strftime('%Y-%m-%d %H:%M')}\t{chunk_end.strftime('%Y-%m-%d %H:%M')}\")
")

# Parse tab-separated output
CHUNK_START=$(echo "$CHUNK_TIMES" | cut -f1)
CHUNK_END=$(echo "$CHUNK_TIMES" | cut -f2)

echo "Task ${SLURM_ARRAY_TASK_ID}: Processing '${CHUNK_START}' to '${CHUNK_END}'"

MASK_FLAG=""
if [ "$SKIP_MASK" = "true" ]; then
    MASK_FLAG="--no_consistent_mask"
fi

srun python3 ${WORKDIR}/scripts/analysis/run_forecast.py \
    --config ${CONFIGNAME} \
    --version ${VERSION} \
    --start_time "${CHUNK_START}" \
    --end_time "${CHUNK_END}" \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    ${MASK_FLAG}
