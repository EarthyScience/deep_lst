#!/bin/bash
# Preprocess the Rain over Africa raw mirror into roa_land_expected_hourly_acc.zarr.
# See scripts/preprocessing/preprocess_roa.py for the steps.
#
# The script needs zarr 3.4 and eodata v0.6.1 (store template and land grid). Both are in the
# environment of the eodata_main worktree, not in the deep_lst environment (zarr 3.2,
# eodata v0.3). EOD_CACHE_DIR points to the land grid that was cached for the FVC stores.
#
# Usage, from anywhere:
#   bash scripts/preprocessing/preprocess_roa.sh setup
#   bash scripts/preprocessing/preprocess_roa.sh shards --first_shard <n> --last_shard <m>
#   bash scripts/preprocessing/preprocess_roa.sh indices

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
EODATA_MAIN=/Net/Groups/BGI/people/mwessel/eodata_main

export EOD_CACHE_DIR=${EODATA_MAIN}/scripts/deep_lst_stores/cache
export PYTHONPATH=${WORKDIR}

${EODATA_MAIN}/.venv/bin/python -u ${WORKDIR}/scripts/preprocessing/preprocess_roa.py "$@"
