"""Score tables from the derived per-day scores. Reads scores/, writes tables/.

Everything is already pooled in derive.py; the only arithmetic here is the
medium-vs-nowcast change in the skill table. Each table is built once as raw values
and emitted twice, as .csv (full precision, for downstream use) and .tex (rounded,
for the paper), so the two can never disagree.
"""

import csv

from .masks import COV_BINS, WINDOW_LABELS

MISSING = "--"  # window registered in MASKS but never verified

SHORT_BASELINE_LABELS = {"clim_smooth": "Clim.", "persistence": "Persis.", "era5land": "SKT"}

PLAIN_WINDOWS = ("all", "day", "night", "diurnal")  # time-of-day only, no row group

# Short row labels for the standalone coverage tables (one group, one row per bin).
COVERAGE_ROW_LABELS = {
    "cov_lt10": r"$<$ 10\%",
    "cov_10_25": r"10-25\%",
    "cov_25_50": r"25-50\%",
    "cov_50_75": r"50-75\%",
    "cov_75_90": r"75-90\%",
    "cov_gt90": r"$>$ 90\%",
}


def _row_groups(pooled):
    """Row groups derived from the verified windows.

    Every base window (no _day/_night suffix, not a plain time-of-day window)
    becomes one group with total/day/night rows; members that were not verified
    render as MISSING. Ordered by WINDOW_LABELS first, unknown windows after."""
    available = _available(pooled)
    bases = [
        w
        for w in available
        if w not in PLAIN_WINDOWS and w not in COV_BINS and not w.endswith("_day") and not w.endswith("_night")
    ]
    ordered = [w for w in WINDOW_LABELS if w in bases]
    ordered += sorted(w for w in bases if w not in WINDOW_LABELS)
    groups = {}
    for window in ordered:
        label = WINDOW_LABELS.get(window, window)
        groups[label] = {"total": window, "day": f"{window}_day", "night": f"{window}_night"}
    if not groups and {"all", "day", "night"} & available:
        # Homogeneous case-study regions verify only the plain windows; render them
        # as a single group so the tables are not empty.
        groups["All"] = {"total": "all", "day": "day", "night": "night"}
    return groups


def _available(pooled):
    return {str(window) for window in pooled["window"].values}


def _score_rows(pooled, days, groups):
    """(group, label, values) with raw floats; None where the window was not verified.

    Columns: rmse and bias at the nowcast day, at the medium-range day, then the
    medium-vs-nowcast change (relative for rmse, absolute for bias)."""
    nowcast, medium = days
    available = _available(pooled)
    rows = []
    for group, members in groups.items():
        for label, window in members.items():
            if window not in available:
                rows.append((group, label, [None] * 6))
                continue
            rmse_nowcast = float(pooled["rmse_model"].sel(window=window, day=nowcast))
            bias_nowcast = float(pooled["bias_model"].sel(window=window, day=nowcast))
            rmse_medium = float(pooled["rmse_model"].sel(window=window, day=medium))
            bias_medium = float(pooled["bias_model"].sel(window=window, day=medium))
            rmse_change = 100.0 * (rmse_medium - rmse_nowcast) / rmse_nowcast
            bias_change = bias_medium - bias_nowcast
            values = [rmse_nowcast, bias_nowcast, rmse_medium, bias_medium, rmse_change, bias_change]
            rows.append((group, label, values))
    return rows


def _skill_rows(pooled, days, baselines, groups):
    """(group, label, values): skill per baseline at both days, then the change [%].

    The change is the medium-range skill relative to the nowcast skill."""
    nowcast, medium = days
    available = _available(pooled)
    rows = []
    for group, members in groups.items():
        for label, window in members.items():
            if window not in available:
                rows.append((group, label, [None] * 3 * len(baselines)))
                continue
            skill_nowcast = [
                float(pooled["rmse_ss"].sel(window=window, day=nowcast, baseline=baseline)) for baseline in baselines
            ]
            skill_medium = [
                float(pooled["rmse_ss"].sel(window=window, day=medium, baseline=baseline)) for baseline in baselines
            ]
            change = [
                100.0 * (at_medium - at_nowcast) / at_nowcast
                for at_nowcast, at_medium in zip(skill_nowcast, skill_medium)
            ]
            rows.append((group, label, skill_nowcast + skill_medium + change))
    return rows


def _emit_csv(path, columns, rows):
    """One flat file: group, row, then the raw values at full precision."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "row"] + columns)
        for group, label, values in rows:
            writer.writerow([group, label] + ["" if v is None else v for v in values])


def _emit_latex(path, colspec, header, rows, caption, cells, highlight=False):
    """booktabs tabular -- needs \\usepackage{booktabs}. `cells` formats one row.

    `highlight` bolds the group headers and shades each group's first (total) row;
    the shading needs \\usepackage[table]{xcolor}."""
    lines = [r"\begin{table}[t]", r"\centering", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header, r"\midrule"]
    span = len(colspec.replace("|", ""))
    group_style = r"\textbf" if highlight else r"\textit"
    current = None
    for group, label, values in rows:
        if group != current:
            lines.append(rf"\multicolumn{{{span}}}{{l}}{{{group_style}{{{group}}}}} \\")
            if highlight:
                lines.append(r"\rowcolor[gray]{0.9}")
            current = group
        lines.append(rf"\quad {label} & " + " & ".join(cells(values)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def _score_cells(values):
    if values[0] is None:
        return [MISSING] * 6
    rmse_nowcast, bias_nowcast, rmse_medium, bias_medium, rmse_change, bias_change = values
    return [
        f"{rmse_nowcast:.2f}",
        f"{bias_nowcast:+.2f}",
        f"{rmse_medium:.2f}",
        f"{bias_medium:+.2f}",
        f"{rmse_change:+.1f}",
        f"{bias_change:+.2f}",
    ]


def _score_cells_nodiff(values):
    if values[0] is None:
        return [MISSING] * 4
    rmse_nowcast, bias_nowcast, rmse_medium, bias_medium = values
    return [f"{rmse_nowcast:.2f}", f"{bias_nowcast:+.2f}", f"{rmse_medium:.2f}", f"{bias_medium:+.2f}"]


def _skill_cells(values):
    """Two-decimal skill scores, then the one-decimal changes."""
    if values[0] is None:
        return [MISSING] * len(values)
    n_scores = 2 * len(values) // 3
    scores = [f"{value:.2f}" for value in values[:n_scores]]
    changes = [f"{value:.1f}" for value in values[n_scores:]]
    return scores + changes


def write_score_table(pooled, days, tables_dir):
    """RMSE and bias per region x time of day, at both lead windows plus their change."""
    nowcast, medium = days
    columns = [f"rmse_d{nowcast}", f"bias_d{nowcast}", f"rmse_d{medium}", f"bias_d{medium}"]
    columns += ["rmse_diff_pct", "bias_diff"]
    rows = _score_rows(pooled, days, _row_groups(pooled))
    _emit_csv(tables_dir / "scores.csv", columns, rows)
    header = (
        "& \\multicolumn{2}{c}{Nowcast range} & \\multicolumn{2}{c}{Medium range}"
        " & \\multicolumn{2}{c}{Difference} \\\\\n"
        "& RMSE [K] & Bias [K] & RMSE [K] & Bias [K] & RMSE [\\%] & Bias [K] \\\\"
    )
    caption = (
        "Forecast scores by Koeppen-Geiger group and time of day, pooled over lead times in a "
        f"nowcasting (day {nowcast}, lead hours {24 * nowcast - 23}-{24 * nowcast}) and medium-range "
        f"(day {medium}, lead hours {24 * medium - 23}-{24 * medium}) window. Bias computes as "
        "forecast $-$ observed. The difference displays the medium-range $-$ nowcast range scores "
        "(percentage for RMSE, absolute for Bias)."
    )
    _emit_latex(
        tables_dir / "scores.tex",
        "lrrrr|rr",
        header,
        rows,
        caption,
        _score_cells,
        highlight=True,
    )


def write_skill_table(pooled, days, tables_dir, baselines):
    """RMSE-SS against every baseline at both lead windows, and the change between them."""
    nowcast, medium = days
    columns = [f"{name}_d{day}" for day in days for name in baselines]
    columns += [f"{name}_diff_pct" for name in baselines]
    rows = _skill_rows(pooled, days, baselines, _row_groups(pooled))
    _emit_csv(tables_dir / "skill.csv", columns, rows)
    per_group = " & ".join(SHORT_BASELINE_LABELS[name] for name in baselines)
    n_baselines = len(baselines)
    header = (
        f"& \\multicolumn{{{n_baselines}}}{{c}}{{Nowcast range}}"
        f" & \\multicolumn{{{n_baselines}}}{{c}}{{Medium range}}"
        f" & \\multicolumn{{{n_baselines}}}{{c}}{{Difference [\\%]}} \\\\\n"
        f"& {per_group} & {per_group} & {per_group} \\\\"
    )
    caption = (
        "RMSE skill score against one of three baselines that is the diurnal smooth climatology, "
        "diurnal persistence and Era5Land skin temperature (SKT) in a nowcasting "
        f"(day {nowcast}, lead hours {24 * nowcast - 23}-{24 * nowcast}) and medium-range "
        f"(day {medium}, lead hours {24 * medium - 23}-{24 * medium}) window. "
        "The percentage difference is computed relative to the nowcast."
    )
    _emit_latex(
        tables_dir / "skill.tex",
        "l" + "|".join(["r" * n_baselines] * 3),
        header,
        rows,
        caption,
        _skill_cells,
        highlight=True,
    )


def write_coverage_tables(pooled, days, tables_dir, baselines):
    """Standalone scores and skill across the lookback-coverage bins (one group,
    total windows only); skipped entirely when no bin was verified."""
    if not any(window in _available(pooled) for window in COV_BINS):
        return
    nowcast, medium = days
    groups = {"Coverage (total)": {label: window for window, label in COVERAGE_ROW_LABELS.items()}}

    score_rows = [(group, label, values[:4]) for group, label, values in _score_rows(pooled, days, groups)]
    columns = [f"rmse_d{nowcast}", f"bias_d{nowcast}", f"rmse_d{medium}", f"bias_d{medium}"]
    _emit_csv(tables_dir / "scores_coverage.csv", columns, score_rows)
    header = (
        "& \\multicolumn{2}{c}{Nowcast range} & \\multicolumn{2}{c}{Medium range} \\\\\n"
        "& RMSE [K] & Bias [K] & RMSE [K] & Bias [K] \\\\"
    )
    caption = (
        "Forecast scores by clear-sky coverage of the lookback window, pooled over the same "
        f"nowcasting (day {nowcast}) and medium-range (day {medium}) windows as the main score table."
    )
    _emit_latex(tables_dir / "scores_coverage.tex", "lrrrr", header, score_rows, caption, _score_cells_nodiff)

    skill_rows = _skill_rows(pooled, days, baselines, groups)
    skill_columns = [f"{name}_d{day}" for day in days for name in baselines]
    skill_columns += [f"{name}_diff_pct" for name in baselines]
    _emit_csv(tables_dir / "skill_coverage.csv", skill_columns, skill_rows)
    per_group = " & ".join(SHORT_BASELINE_LABELS[name] for name in baselines)
    n_baselines = len(baselines)
    skill_header = (
        f"& \\multicolumn{{{n_baselines}}}{{c}}{{Nowcast range}}"
        f" & \\multicolumn{{{n_baselines}}}{{c}}{{Medium range}}"
        f" & \\multicolumn{{{n_baselines}}}{{c}}{{Difference [\\%]}} \\\\\n"
        f"& {per_group} & {per_group} & {per_group} \\\\"
    )
    skill_caption = (
        "RMSE skill score against the three baselines by clear-sky coverage of the lookback "
        "window, at the same lead windows as the main skill table. The percentage difference "
        "is computed relative to the nowcast."
    )
    _emit_latex(
        tables_dir / "skill_coverage.tex",
        "l" + "|".join(["r" * n_baselines] * 3),
        skill_header,
        skill_rows,
        skill_caption,
        _skill_cells,
    )


def write_verif_tables(pooled, days, tables_dir, baselines):
    write_score_table(pooled, days, tables_dir)
    write_skill_table(pooled, days, tables_dir, baselines)
    write_coverage_tables(pooled, days, tables_dir, baselines)
