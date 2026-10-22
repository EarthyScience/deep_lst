"""Figures of the input sensitivity of a trained run (deep_lst/sensitivity/plots.py).

    python scripts/sensitivity/plot_sensitivity.py --experiment mlp_smoothl1_h24_full_v16.2_hydrostates \\
        --eval_name eval_mtg_hydrostates_val --blocks month all --days 1 10

Per block, writes relevance_<block>_<eval>.pdf (RMSE increase and amplitude error increase,
day 1 -> day 10) and relevance_leads_<block>_<eval>.pdf (RMSE increase per forecast day) next
to the stores, into the experiment's results/<experiment>/<version>/<eval>/sensitivity/ folder
(--out overrides), and prints the ranked values. Several --experiment values (e.g. seeds) are
averaged, into the first experiment's folder.
"""

import argparse
from pathlib import Path

from deep_lst.sensitivity.forecast import sensitivity_dir
from deep_lst.sensitivity.plots import load_relevance, plot_leads, plot_relevance, rank_variables, relevance_table
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

    for block in args.blocks:
        order = rank_variables(table, block, args.days[0], args.window)
        plot_relevance(table, order, block, args.days, out_dir / f"relevance_{block}_{args.eval_name}.pdf", args.window)
        plot_leads(table, order, block, out_dir / f"relevance_leads_{block}_{args.eval_name}.pdf", args.window)
        print(f"=== block {block}: RMSE increase (%) in window {args.window}, days {args.days}")
        print(relevance_table(table, block, order, args.days, args.window).round(2).to_string())
        amplitude = relevance_table(table, block, order, args.days, AMPLITUDE_WINDOW)
        if not amplitude.empty:
            print(f"--- amplitude error increase (%), days {args.days}")
            print(amplitude.round(2).to_string())
    print(f"wrote relevance_<block>_{args.eval_name}.pdf and relevance_leads_<block>_{args.eval_name}.pdf to {out_dir}")


if __name__ == "__main__":
    main()
