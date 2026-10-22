"""Figures of the input sensitivity of a trained run (deep_lst/sensitivity/plots.py).

    python scripts/sensitivity/plot_sensitivity.py --experiment mlp_smoothl1_h24_full_v16.2_hydrostates \\
        --eval_name eval_mtg_hydrostates_val --blocks month all --days 1 10

Writes relevance_<eval>.pdf and relevance_leads_<eval>.pdf next to the stores, into the
experiment's results/<experiment>/<version>/<eval>/sensitivity/ folder (--out overrides), and
prints the ranked relevance of the first block and day. Several --experiment values (e.g.
seeds) are drawn together with their mean, into the first experiment's folder.
"""

import argparse
from pathlib import Path

from deep_lst.sensitivity.forecast import sensitivity_dir
from deep_lst.sensitivity.plots import load_relevance, plot_leads, plot_relevance, rank_variables, relevance_table
from deep_lst.utils.io import ExperimentPaths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", nargs="+", required=True)
    parser.add_argument("--eval_name", required=True)
    parser.add_argument("--version", default="version_0")
    parser.add_argument("--blocks", nargs="+", default=["month", "all"])
    parser.add_argument("--days", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--window", default="all", help="solar window of the summary rows: all | day | night")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths = ExperimentPaths()
    table = load_relevance(paths, args.experiment, args.eval_name, args.version, args.blocks, args.window)
    order = rank_variables(table, args.blocks[0], args.days[0])

    if args.out is None:
        first_block = sensitivity_dir(
            paths, args.experiment[0], args.version, args.eval_name, "permute_days", args.blocks[0]
        )
        out_dir = first_block.parent
    else:
        out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.eval_name if args.window == "all" else f"{args.eval_name}_{args.window}"
    plot_relevance(table, order, args.blocks, args.days, out_dir / f"relevance_{suffix}.pdf")
    plot_leads(table, order, args.blocks[0], out_dir / f"relevance_leads_{suffix}.pdf")

    print(f"relevance (%), block {args.blocks[0]}, day {args.days[0]}, window {args.window}:")
    print(relevance_table(table, args.blocks[0], args.days[0], order).round(2).to_string())
    print(f"wrote relevance_{suffix}.pdf and relevance_leads_{suffix}.pdf to {out_dir}")


if __name__ == "__main__":
    main()
