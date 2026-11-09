#!/bin/bash
# Figures of the input sensitivity of one run (deep_lst/sensitivity/plots.py), once the
# summary.csv of each block exists (run_sensitivity.sh). Written next to the stores, into the
# run's results/<experiment>/version_0/<eval>/sensitivity/ folder:
#   relevance_<eval>.pdf        inputs ranked by the RMSE increase when permuted,
#                               panels = block (month, all) x forecast day (1, 10)
#   relevance_leads_<eval>.pdf  the same per forecast day as a heatmap (month block)
# and prints the ranked table. Runs on the login node in seconds.
#
# Usage: bash scripts/sensitivity/plot_sensitivity.sh <experiment> <eval_name> [<out_dir>]
#   bash scripts/sensitivity/plot_sensitivity.sh mlp_smoothl1_h24_full_v16.2_hydrostates eval_mtg_hydrostates_val
# Several experiments (e.g. seeds) in quotes, "exp_a exp_b", are drawn together with their mean;
# the figures then go to the first experiment's folder unless <out_dir> is given.

set -e
WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
EXPERIMENTS=$1
EVAL=$2
OUT=${3:-}
cd ${WORKDIR}
source ${WORKDIR}/.venv/bin/activate
MPLBACKEND=Agg python scripts/sensitivity/plot_sensitivity.py --experiment ${EXPERIMENTS} --eval_name ${EVAL} \
    --blocks month all --days 1 10 ${OUT:+--out ${OUT}}
