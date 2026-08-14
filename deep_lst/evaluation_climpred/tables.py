"""Score tables from the derived per-day scores. Reads scores/, writes tables/.

No arithmetic here -- everything is already pooled in derive.py. Each table is built
once as raw values and emitted twice, as .csv (full precision, for downstream use)
and .tex (rounded, for the paper), so the two can never disagree.
"""

import csv

from .masks import WINDOW_LABELS

MISSING = "--"  # window registered in MASKS but never verified

PLAIN_WINDOWS = ("all", "day", "night", "diurnal")  # time-of-day only, no row group


def _row_groups(pooled):
    """Row groups derived from the verified windows.

    Every base window (no _day/_night suffix, not a plain time-of-day window)
    becomes one group with total/day/night rows; members that were not verified
    render as MISSING. Ordered by WINDOW_LABELS first, unknown windows after."""
    available = _available(pooled)
    bases = [w for w in available if w not in PLAIN_WINDOWS and not w.endswith("_day") and not w.endswith("_night")]
    ordered = [w for w in WINDOW_LABELS if w in bases]
    ordered += sorted(w for w in bases if w not in WINDOW_LABELS)
    groups = {}
    for window in ordered:
        label = WINDOW_LABELS.get(window, window)
        groups[label] = {"total": window, "day": f"{window}_day", "night": f"{window}_night"}
    return groups


def _available(pooled):
    return {str(window) for window in pooled["window"].values}


def _score_rows(pooled, days):
    """(group, label, values) with raw floats; None where the window was not verified.

    Columns: rmse and bias at the nowcast day, then at the medium-range day."""
    nowcast, medium = days
    available = _available(pooled)
    rows = []
    for group, members in _row_groups(pooled).items():
        for label, window in members.items():
            if window not in available:
                rows.append((group, label, [None] * 4))
                continue
            values = [
                float(pooled["rmse_model"].sel(window=window, day=nowcast)),
                float(pooled["bias_model"].sel(window=window, day=nowcast)),
                float(pooled["rmse_model"].sel(window=window, day=medium)),
                float(pooled["bias_model"].sel(window=window, day=medium)),
            ]
            rows.append((group, label, values))
    return rows


def _skill_rows(pooled, bootstrap_daily, days, baselines):
    """(group, label, values) with skill and its CI bounds, three floats per cell."""
    available = _available(pooled)
    rows = []
    for group, members in _row_groups(pooled).items():
        for label, window in members.items():
            values = []
            for day in days:
                for baseline in baselines:
                    if window not in available:
                        values.extend([None, None, None])
                        continue
                    selection = dict(window=window, day=day, baseline=baseline)
                    values.extend(
                        [
                            float(pooled["rmse_ss"].sel(**selection)),
                            float(bootstrap_daily.sel(quantile=0.025, **selection)),
                            float(bootstrap_daily.sel(quantile=0.975, **selection)),
                        ]
                    )
            rows.append((group, label, values))
    return rows


def _emit_csv(path, columns, rows):
    """One flat file: group, row, then the raw values at full precision."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "row"] + columns)
        for group, label, values in rows:
            writer.writerow([group, label] + ["" if v is None else v for v in values])


def _emit_latex(path, colspec, header, rows, caption, cells):
    """booktabs tabular -- needs \\usepackage{booktabs}. `cells` formats one row."""
    lines = [r"\begin{table}[t]", r"\centering", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule"]
    span = len(colspec.replace("|", ""))
    current = None
    for group, label, values in rows:
        if group != current:
            lines.append(rf"\multicolumn{{{span}}}{{l}}{{\textit{{{group}}}}} \\")
            current = group
        lines.append(rf"\quad {label} & " + " & ".join(cells(values)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def _score_cells(values):
    if values[0] is None:
        return [MISSING] * 4
    rmse_nowcast, bias_nowcast, rmse_medium, bias_medium = values
    return [f"{rmse_nowcast:.2f}", f"{bias_nowcast:+.2f}", f"{rmse_medium:.2f}", f"{bias_medium:+.2f}"]


def _skill_cells(values):
    cells = []
    for start in range(0, len(values), 3):
        skill, low, high = values[start : start + 3]
        cells.append(MISSING if skill is None else f"{skill:.2f} ({low:.2f}--{high:.2f})")
    return cells


def write_score_table(pooled, days, tables_dir):
    """RMSE and bias per region x time of day, at both lead windows."""
    nowcast, medium = days
    columns = [f"rmse_d{nowcast}", f"bias_d{nowcast}", f"rmse_d{medium}", f"bias_d{medium}"]
    rows = _score_rows(pooled, days)
    _emit_csv(tables_dir / "scores.csv", columns, rows)
    header = (
        f"& \\multicolumn{{2}}{{c}}{{Nowcast (day {nowcast})}}"
        f" & \\multicolumn{{2}}{{c}}{{Medium range (day {medium})}} \\\\\n"
        "& RMSE [K] & Bias [K] & RMSE [K] & Bias [K] \\\\"
    )
    _emit_latex(
        tables_dir / "scores.tex",
        "lrrrr",
        header,
        rows,
        "Forecast error by Koeppen-Geiger group and time of day. Bias is signed (forecast $-$ observed).",
        _score_cells,
    )


def write_skill_table(pooled, bootstrap_daily, days, tables_dir, baselines, labels):
    """RMSE-SS against every baseline, with its bootstrap CI."""
    nowcast, medium = days
    columns = [f"{name}_d{day}_{part}" for day in days for name in baselines for part in ("ss", "lo", "hi")]
    rows = _skill_rows(pooled, bootstrap_daily, days, baselines)
    _emit_csv(tables_dir / "skill.csv", columns, rows)
    per_day = " & ".join(labels[name] for name in baselines)
    n_baselines = len(baselines)
    header = (
        f"& \\multicolumn{{{n_baselines}}}{{c}}{{Nowcast (day {nowcast})}}"
        f" & \\multicolumn{{{n_baselines}}}{{c}}{{Medium range (day {medium})}} \\\\\n"
        f"& {per_day} & {per_day} \\\\"
    )
    _emit_latex(
        tables_dir / "skill.tex",
        "l" + "r" * (2 * len(baselines)),
        header,
        rows,
        "RMSE skill score against each baseline, with 95\\% bootstrap confidence intervals over init days.",
        _skill_cells,
    )


def write_verif_tables(pooled, bootstrap_daily, days, tables_dir, baselines, labels):
    write_score_table(pooled, days, tables_dir)
    write_skill_table(pooled, bootstrap_daily, days, tables_dir, baselines, labels)
