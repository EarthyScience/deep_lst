"""Forecast-limit-by-month tables + two-panel plot for one experiment.

Forecast limit = first forecast-day bucket where daily-bucketed RMSE skill vs the
baseline drops below zero, per (region, month, day/night). Average skill only, no
uncertainty; months by init_time. Writes a tidy CSV (load for plotting), two LaTeX
tables (daytime, nighttime), and the two-panel plot.

uv run scripts/evaluation/run_fc_limit_months.py \\
    --experiment mlp_smoothl1_h24_v7_era5L --version version_0 \\
    --baseline clim_smooth --temporal test
"""

import argparse
import json

import pandas as pd

from deep_lst.evaluation.fc_limits import (
    build_class_records,
    build_records,
    write_csv,
    write_latex_tables,
    write_validity_tables,
)
from deep_lst.evaluation.plots.fc_limits import (
    forecast_limit_class_heatmap,
    forecast_limit_month_bubbles,
    forecast_limit_month_heatmap,
    forecast_limit_validity_scatter,
)
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# (display label, eval_name). The training holdout is drawn as the black dashed series.
REGION_SPECS = [
    ("Dahra", "dahra"),
    ("Antananarivo", "antananarivo"),
    ("Bouamir", "bouamir"),
    ("Gobabeb", "gobabeb"),
    ("Johannesburg", "johannesburg"),
    ("Passage", "passage"),
    ("Sahel", "sahel_niger"),
    ("West Africa", "west_africa"),
    ("Yangambi", "yangambi"),
    ("Training sites", "train_mtg"),
]
TRAINING_LABEL = "Training sites"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--baseline", default="clim_smooth")
    p.add_argument("--temporal", nargs="+", default=["test"])
    p.add_argument("--regions", nargs="+", default=None, help="Subset of eval_names; default all.")
    p.add_argument(
        "--plot-only", action="store_true", help="Re-render the heatmap from the existing CSV; skip the compute."
    )
    p.add_argument(
        "--by-class",
        default=None,
        help="Class coord (e.g. modis_lc_classes) to pool one site's pixels by, instead of regions.",
    )
    p.add_argument("--class-eval-name", default="train_mtg", help="Site whose pixels are pooled in --by-class mode.")
    return p.parse_args()


def run_by_class(args, fc_dir, figures_dir):
    """Forecast limit per (land-cover class, month, day/night) for one site, pooling its pixels."""
    class_key = args.by_class
    labels = json.loads(PATHS.eval_meta_path().read_text())[f"{class_key}_labels"]
    stem = f"{args.experiment}_{args.version}_{args.baseline}_{class_key}"

    records = build_class_records(
        args.experiment, args.version, args.class_eval_name, class_key, args.baseline, labels, args.temporal
    )
    if not records:
        print("No class records computed.")
        return
    df = pd.DataFrame.from_records(records)

    # x-axis order: classes by ascending class id.
    class_order = df[["region", "class_id"]].drop_duplicates().sort_values("class_id")["region"].tolist()

    write_csv(records, fc_dir / f"{stem}.csv")
    write_latex_tables(
        records,
        class_order,
        fc_dir,
        stem,
        caption_prefix=f"Forecast limit (days) by month and {class_key} for {args.experiment} vs {args.baseline}",
        label_prefix=f"tab:{stem}",
    )
    write_validity_tables(
        records,
        class_order,
        PATHS.tables_dir(args.experiment, args.version, "validity"),
        stem,
        caption_prefix=f"Clear-sky validity (\\%) by month and {class_key} for {args.experiment}",
        label_prefix=f"tab:{stem}_validity",
    )

    horizon_days = float(df["limit_days"].max())
    forecast_limit_class_heatmap(df, class_order, horizon_days, figures_dir / f"{stem}.pdf")
    print("\nDone.")


def main():
    args = parse_args()
    region_specs = REGION_SPECS
    if args.regions is not None:
        region_specs = [(label, name) for label, name in REGION_SPECS if name in args.regions]

    fc_dir = PATHS.tables_dir(args.experiment, args.version, "fc_limits")
    figures_dir = PATHS.figures_dir(args.experiment, args.version)
    stem = f"{args.experiment}_{args.version}_{args.baseline}"
    csv_path = fc_dir / f"{stem}.csv"

    if args.by_class:
        run_by_class(args, fc_dir, figures_dir)
        return

    if args.plot_only:
        if not csv_path.exists():
            print(f"No CSV to re-plot: {csv_path}. Run once without --plot-only first.")
            return
        df = pd.read_csv(csv_path)
    else:
        records = build_records(args.experiment, args.version, region_specs, args.baseline, args.temporal)
        if not records:
            print("No records computed.")
            return
        df = pd.DataFrame.from_records(records)

    # x-axis order: training holdout first (x=0), then eval regions by descending latitude.
    lat_by_region = df.groupby("region")["latitude"].first()
    eval_regions = [region for region in lat_by_region.index if region != TRAINING_LABEL]
    region_order = [TRAINING_LABEL] + sorted(eval_regions, key=lat_by_region.get, reverse=True)

    if not args.plot_only:
        write_csv(records, csv_path)
        write_latex_tables(
            records,
            region_order,
            fc_dir,
            stem,
            caption_prefix=f"Forecast limit (days) by month for {args.experiment} vs {args.baseline}",
            label_prefix=f"tab:{stem}",
        )
        write_validity_tables(
            records,
            region_order,
            PATHS.tables_dir(args.experiment, args.version, "validity"),
            stem,
            caption_prefix=f"Clear-sky validity (\\%) by month for {args.experiment}",
            label_prefix=f"tab:{stem}_validity",
        )

    horizon_days = float(df["limit_days"].max())
    forecast_limit_month_heatmap(df, region_order, TRAINING_LABEL, horizon_days, figures_dir / f"{stem}.pdf")
    forecast_limit_month_bubbles(df, region_order, TRAINING_LABEL, horizon_days, figures_dir / f"{stem}_validity.pdf")
    forecast_limit_validity_scatter(df, figures_dir / f"{stem}_corr.pdf")
    print("\nDone.")


if __name__ == "__main__":
    main()
