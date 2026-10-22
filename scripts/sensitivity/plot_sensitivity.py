"""Figures of the input sensitivity of a trained run (deep_lst/sensitivity/plots.py).

    python scripts/sensitivity/plot_sensitivity.py --experiment mlp_smoothl1_h24_full_v16.2_hydrostates \\
        --eval_name eval_mtg_hydrostates_val --blocks month all --days 1 10 [--metric kelvin|percent]

Per block, writes relevance_<block>_<eval>.pdf (RMSE increase and amplitude error increase,
day 1 -> day 10) and relevance_leads_<block>_<eval>.pdf (RMSE increase per forecast day) next
to the stores, into the experiment's results/<experiment>/<version>/<eval>/sensitivity/ folder
(--out overrides), and prints the ranked values. The default metric is the increase in Kelvin;
--metric percent gives the increase relative to the unperturbed RMSE, with "_pct" in the file
names. Several --experiment values (e.g. seeds) are averaged, into the first experiment's folder.
"""

import argparse
from pathlib import Path

from deep_lst.sensitivity.forecast import sensitivity_dir
from deep_lst.sensitivity.plots import (
    METRICS,
    increase_table,
    load_relevance,
    plot_leads,
    plot_relevance,
    rank_variables,
)
from deep_lst.sensitivity.scores import AMPLITUDE_WINDOW
from deep_lst.utils.io import ExperimentPaths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", nargs="+", required=True)
    parser.add_argument("--eval_name", required=True)
    parser.add_argument("--version", default="version_0")
    parser.add_argument("--blocks", nargs="+", default=["month", "all"])
    parser.add_argument("--days", type=int, nargs=2, default=[1, 10], help="first and second forecast day drawn")
    parser.add_argument("--window", default="all", help="window of the RMSE column: all | day | night")
    parser.add_argument("--metric", default="kelvin", choices=list(METRICS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths = ExperimentPaths()
    table = load_relevance(paths, args.experiment, args.eval_name, args.version, args.blocks)
    if args.out is None:
        first_block = sensitivity_dir(
            paths, args.experiment[0], args.version, args.eval_name, "permute_days", args.blocks[0]
        )
        out_dir = first_block.parent
    else:
        out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.eval_name if args.metric == "kelvin" else f"{args.eval_name}_pct"
    unit = METRICS[args.metric]["unit"]

    for block in args.blocks:
        order = rank_variables(table, block, args.days[0], args.window, args.metric)
        plot_relevance(
            table, order, block, args.days, out_dir / f"relevance_{block}_{suffix}.pdf", args.window, args.metric
        )
        plot_leads(table, order, block, out_dir / f"relevance_leads_{block}_{suffix}.pdf", args.window, args.metric)
        print(f"=== block {block}: RMSE increase ({unit}) in window {args.window}, days {args.days}")
        print(increase_table(table, block, order, args.days, args.window, args.metric).round(3).to_string())
        amplitude = increase_table(table, block, order, args.days, AMPLITUDE_WINDOW, args.metric)
        if not amplitude.empty:
            print(f"--- amplitude error increase ({unit}), days {args.days}")
            print(amplitude.round(3).to_string())
    print(f"wrote relevance_<block>_{suffix}.pdf and relevance_leads_<block>_{suffix}.pdf to {out_dir}")


if __name__ == "__main__":
    main()
