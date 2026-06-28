"""Generic LaTeX table assembly, shared by the score tables and the forecast-limit tables."""


def latex_table(rows, columns, caption, label, display_map=None):
    """Assemble a LaTeX table from pre-formatted string cells.

    rows         {row_key: {column: cell_str}} -- cells already formatted.
    columns      ordered column names (also the header labels).
    display_map  optional {row_key: pretty_label}; row_key is used when absent.
    """
    display_map = display_map or {}
    col_spec = "c|" + "|".join("c" for _ in columns)
    header = " & " + " & ".join(columns) + r" \\"
    lines = [r"\begin{table}[]", r"    \centering", f"    \\begin{{tabular}}{{{col_spec}}}", "    " + header]
    for row_key, row in rows.items():
        display = display_map.get(row_key, row_key)
        cells = " & ".join(row[column] for column in columns)
        lines.append(f"    {display} & {cells} \\\\")
    lines += [r"    \end{tabular}", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def write_month_tables(records, col_order, out_dir, stem, caption_prefix, label_prefix, cell, months, kinds):
    """Two col x month LaTeX tables (one per day/night kind); `cell(record) -> str` formats each cell.

    records: tidy dicts keyed by (region, month, daynight). col_order: row order (regions or classes).
    Shared by the forecast-limit, validity and monthly-score tables.
    """
    by_key = {(r["region"], r["month"], r["daynight"]): r for r in records}
    columns = [str(month) for month in months]
    for kind in kinds:
        rows = {}
        for label in col_order:
            cells = {}
            for month in months:
                record = by_key.get((label, month, kind))
                cells[str(month)] = cell(record) if record else "--"
            rows[label] = cells
        tex = latex_table(rows, columns, f"{caption_prefix} ({kind}).", f"{label_prefix}_{kind}")
        path = out_dir / f"{stem}_{kind}.tex"
        path.write_text(tex + "\n")
        print(f"[written] {path}")
