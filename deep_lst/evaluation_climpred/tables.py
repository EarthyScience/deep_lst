"""Score tables from the derived per-day scores. Reads scores/, writes tables/.

Everything is already pooled in derive.py; the only arithmetic here is the
medium-vs-nowcast change in the skill table. Each table is built once as raw values
and emitted twice, as .csv (full precision, for downstream use) and .tex (rounded,
for the paper), so the two can never disagree.
"""

import csv

from .masks import WINDOW_LABELS

MISSING = "--"  # window registered in MASKS but never verified

SHORT_BASELINE_LABELS = {"clim_smooth": "Clim.", "persistence": "Persis.", "era5land": "SKT"}

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


def _skill_rows(pooled, days, baselines):
    """(group, label, values): skill per baseline at both days, then the change [%].

    The change is the medium-range skill relative to the nowcast skill."""
    nowcast, medium = days
    available = _available(pooled)
    rows = []
    for group, members in _row_groups(pooled).items():
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


def write_skill_table(pooled, days, tables_dir, baselines):
    """RMSE-SS against every baseline at both lead windows, and the change between them."""
    nowcast, medium = days
    columns = [f"{name}_d{day}" for day in days for name in baselines]
    columns += [f"{name}_diff_pct" for name in baselines]
    rows = _skill_rows(pooled, days, baselines)
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


def write_verif_tables(pooled, days, tables_dir, baselines):
    write_score_table(pooled, days, tables_dir)
    write_skill_table(pooled, days, tables_dir, baselines)
