"""Cross-experiment ablation table: total scores per run from the saved scores/*.nc.

Compiles one row per experiment -- RMSE, bias, centered ACC and anomaly R2 at the
requested forecast days (default: day 1 and day 10) for one window -- into a csv + tex
pair for the supplement ablation tables. Reads only what verif_* + evaluate already
wrote (scores/skill_daily.nc, scores/skill.nc, verif/anom.nc); no climpred, no slurm.

RMSE and bias come from skill_daily.nc unchanged. The day-level R2 is exact:
1 - sum_l(n*mse) / sum_l(n*var) over the day's hourly leads. The day-level ACC is the
count-weighted mean of the per-lead centered correlations -- correlations do not pool
like MSE, so it is a summary, not a re-verification; the per-lead curves stay the
reference. A run whose scores are missing (still training, typo) renders as a MISSING
row and is announced, so partial tables can be compiled while runs trickle in.

Run: python -m deep_lst.evaluation_climpred.tables_ablation \
        --experiments <exp> [<exp> ...] --eval_name <name> [--out tables/ablation]
"""

import argparse
from pathlib import Path

import xarray as xr

from deep_lst.utils.io import ExperimentPaths

from .derive import pooled_mean
from .tables import MISSING, _emit_csv, _emit_latex

METRICS = ["rmse", "bias", "acc", "r2"]
METRIC_HEADS = ["RMSE", "Bias", "ACC", "$R^2$"]
METRIC_FORMATS = ["{:.2f}", "{:+.2f}", "{:.2f}", "{:.2f}"]


def experiment_values(paths, experiment, version, eval_name, window, days):
    """One run's row: METRICS at each requested day; None for unverified days."""
    results_dir = paths.results_dir(experiment, version, eval_name)
    skill_daily = xr.open_dataset(results_dir / "scores" / "skill_daily.nc", decode_timedelta=False)
    skill = xr.open_dataset(results_dir / "scores" / "skill.nc", decode_timedelta=False)
    anom = xr.open_dataset(results_dir / "verif" / "anom.nc", decode_timedelta=False)

    day = xr.DataArray((skill["lead"].values - 1) // 24 + 1, dims="lead", name="day")
    counts = skill["n"]
    mse_daily = pooled_mean(skill["rmse_model"] ** 2, counts, dim="lead", group=day)
    var_daily = pooled_mean(anom["anomaly_obs_variance"], counts, dim="lead", group=day)
    r2_daily = 1.0 - mse_daily / var_daily
    acc_daily = pooled_mean(skill["acc_centered"], counts, dim="lead", group=day)

    values = []
    verified_days = set(skill_daily["day"].values)
    for requested in days:
        if requested not in verified_days:
            values += [None] * len(METRICS)
            continue
        values.append(float(skill_daily["rmse_model"].sel(window=window, day=requested)))
        values.append(float(skill_daily["bias_model"].sel(window=window, day=requested)))
        values.append(float(acc_daily.sel(window=window, day=requested)))
        values.append(float(r2_daily.sel(window=window, day=requested)))
    return values


def build_rows(args, paths):
    """(group, run name, values) per experiment; all-None values where scores are absent."""
    group = f"{args.eval_name} ({args.window})"
    rows = []
    for experiment in args.experiments:
        try:
            values = experiment_values(paths, experiment, args.version, args.eval_name, args.window, args.days)
        except (FileNotFoundError, KeyError) as error:
            print(f"MISSING {experiment}: {error}")
            values = [None] * (len(METRICS) * len(args.days))
        rows.append((group, experiment, values))
    return rows


def _cells(values):
    formatted = []
    for index, value in enumerate(values):
        fmt = METRIC_FORMATS[index % len(METRIC_FORMATS)]
        formatted.append(MISSING if value is None else fmt.format(value))
    return formatted


def _tex_escaped(rows):
    """Run names and eval names carry underscores; escape them for the tex emit only."""
    escaped = []
    for group, label, values in rows:
        escaped.append((group.replace("_", r"\_"), label.replace("_", r"\_"), values))
    return escaped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", required=True, help="run names under results/")
    parser.add_argument("--eval_name", required=True, help="eval region the runs were verified on")
    parser.add_argument("--version", default="version_0")
    parser.add_argument("--window", default="all", help="verification window for the totals")
    parser.add_argument("--days", nargs="+", type=int, default=[1, 10], help="forecast days to report")
    parser.add_argument("--out", default="tables/ablation", help="output path prefix (.csv/.tex appended)")
    args = parser.parse_args()
    paths = ExperimentPaths()

    rows = build_rows(args, paths)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.parent / f"{out.name}.csv"
    tex_path = out.parent / f"{out.name}.tex"

    columns = [f"{metric}_day{requested}" for requested in args.days for metric in METRICS]
    _emit_csv(csv_path, columns, rows)

    day_heads = " & ".join(rf"\multicolumn{{{len(METRICS)}}}{{c}}{{day {requested}}}" for requested in args.days)
    metric_heads = " & ".join([" & ".join(METRIC_HEADS)] * len(args.days))
    header = rf" & {day_heads} \\" + "\n" + rf"Run & {metric_heads} \\"
    caption = (
        f"Ablation scores, {args.window} window: RMSE and bias (K), centered anomaly correlation "
        "and anomaly $R^2$ per forecast day. ACC is the count-weighted mean over the day's hourly "
        "leads; $R^2$ pools MSE and anomaly variance over them exactly."
    )
    colspec = "l" + "r" * len(columns)
    _emit_latex(tex_path, colspec, header, _tex_escaped(rows), caption, _cells)

    for _, label, values in rows:
        print(f"{label}: " + " ".join(_cells(values)))
    print(f"Wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
