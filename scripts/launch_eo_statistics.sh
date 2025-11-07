#!/usr/bin/env bash

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR

STORE_DIR="/Net/Groups/BGI/work_5/scratch/mwessel/wegen-stores/lst_test.zarr"
MODALITY=seviri

if [ "$MODALITY" = "era5" ]; then
    VARS=("stl1" "swvl1" "u10" "v10" "sp" "t2m" "d2m" "strd" "str" "ssr" "crr" "tcwv" "skt" "tvh" "tvl")
else
    VARS=("LST" "DEM" "standard_error" "AL-NI-DH" "FVC" "LANDCOV")
fi

wait_for_job() {
  local jid="$1"
  # Return immediately if empty (first job)
  [[ -z "$jid" ]] && return 0
  echo "Waiting for job $jid to finish..."

  scontrol wait_job "$jid"
}

last_jobid=""
for v in "${VARS[@]}"; do
  wait_for_job "$last_jobid"
  out=$(sbatch --export=ALL,VAR="$v",MODALITY="$MODALITY",STORE_DIR="$STORE_DIR" scripts/get_eo_statistics.sh)
  last_jobid=$(sed -E 's/[^0-9]*([0-9]+).*/\1/' <<< "$out")
  echo "Submitted $v as job ${last_jobid}"
done