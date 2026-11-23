#!/bin/bash
# Download the Rain over Africa (RoA) expected rain rate, 2016-2025, as a raw mirror on tscratch.
#
# Source:  s3://rainoverafrica/data/roa_<year>.zarr  (public bucket, no credentials needed)
#          Amell et al. (2025), https://doi.org/10.1029/2025JD044595, https://github.com/SEE-GEO/roa
# Subset:  y_hat_mu (expected rain rate, mm/h, 15 min, SEVIRI 3 km grid)
#          + coordinates time, x, y, latitude, longitude.
#          Skipped: y_hat_tau (quantiles), acq_time, platform.
# Target:  /Net/Groups/BGI/tscratch/mwessel/data/rain_over_africa/roa_<year>.zarr
# Volume:  ~3.2 TB in ~11.2 million files, ~1.5 days through the proxy (estimated 2026-11-13).
#
# The stores keep the original .zmetadata, which still lists the skipped arrays. Open with
#   xr.open_zarr(store, drop_variables=["y_hat_tau", "quantile_level", "acq_time", "platform"])
# y_hat_mu contains inf and absurdly large values; mask to e.g. 0-200 mm/h before use.
#
# Usage (inside tmux; safe to rerun, files that are already there are skipped):
#   ml proxy
#   conda activate ml_basic     # provides the AWS CLI (pip package "awscli") and python
#   bash scripts/downloads/download_roa.sh              # 2016-2025
#   bash scripts/downloads/download_roa.sh 2016 2017    # selected years only
#
# Progress and the per-year completeness check go to <target>/download.log.

set -u

BUCKET=s3://rainoverafrica/data
DEST=/Net/Groups/BGI/tscratch/mwessel/data/rain_over_africa
ARRAYS="time x y latitude longitude y_hat_mu"
CHUNKS_PER_STEP=32          # y_hat_mu: one time step = 8 x 4 spatial chunks
PARALLEL_REQUESTS=64        # AWS CLI default is 10; the proxy saturates at about 64
AWS_OPTS="--no-sign-request --only-show-errors"

if [ $# -gt 0 ]; then
    YEARS="$*"
else
    YEARS=$(seq 2016 2025)
fi

if [ -z "${https_proxy:-}" ]; then
    echo "No https_proxy set. Run 'ml proxy' first." >&2
    exit 1
fi
if ! aws --version 2>&1 | grep -q '^aws-cli/'; then
    echo "'aws' is not the AWS CLI. Run 'conda activate ml_basic' first." >&2
    exit 1
fi

mkdir -p ${DEST}
LOG=${DEST}/download.log

# Parallelism lives in a config file next to the data, ~/.aws/config stays untouched.
export AWS_CONFIG_FILE=${DEST}/aws_config
aws configure set default.s3.max_concurrent_requests ${PARALLEL_REQUESTS}

log () {
    echo "$(date '+%F %T') $*" | tee -a ${LOG}
}

count_time_steps () {  # count_time_steps <.zmetadata>
    python -c "
import json, sys
metadata = json.load(open(sys.argv[1]))['metadata']
print(metadata['time/.zarray']['shape'][0])
" $1
}

incomplete_years=""
for year in ${YEARS}; do
    store=roa_${year}.zarr
    transfer_ok=true
    log "${store}: start"

    # A killed run leaves partial downloads behind, named <chunk>.<8 hex characters>.
    if [ -d ${DEST}/${store}/y_hat_mu ]; then
        find ${DEST}/${store}/y_hat_mu -type f \
            -regex '.*/[0-9]+\.[0-9]+\.[0-9]+\.[0-9a-fA-F]+' -delete
    fi

    for meta_file in .zattrs .zgroup .zmetadata; do
        aws s3 cp ${AWS_OPTS} ${BUCKET}/${store}/${meta_file} ${DEST}/${store}/${meta_file} \
            || transfer_ok=false
    done
    for array in ${ARRAYS}; do
        aws s3 sync ${AWS_OPTS} ${BUCKET}/${store}/${array}/ ${DEST}/${store}/${array}/ \
            || transfer_ok=false
    done

    # Completeness: every time step of y_hat_mu must have all its chunk files.
    n_steps=$(count_time_steps ${DEST}/${store}/.zmetadata)
    n_expected=$((n_steps * CHUNKS_PER_STEP))
    n_found=$(find ${DEST}/${store}/y_hat_mu -type f -regex '.*/[0-9]+\.[0-9]+\.[0-9]+' | wc -l)
    if ${transfer_ok} && [ "${n_found}" -eq "${n_expected}" ]; then
        log "${store}: complete, ${n_found} chunks for ${n_steps} time steps"
    else
        log "${store}: INCOMPLETE, ${n_found} of ${n_expected} chunks (transfer_ok=${transfer_ok})"
        incomplete_years="${incomplete_years} ${year}"
    fi
done

if [ -n "${incomplete_years}" ]; then
    log "Finished with incomplete years:${incomplete_years}. Rerun the script to fill the gaps."
    exit 1
fi
log "Finished, all years complete."
