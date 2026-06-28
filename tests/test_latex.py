from deep_lst.evaluation.latex import latex_table


def test_latex_table_assembles_rows_columns_and_display_map():
    rows = {
        "sahel_niger": {"a": "1.0", "b": "2.0"},
        "made_up": {"a": "3.0", "b": "4.0"},  # not in display_map -> key is used verbatim
    }
    out = latex_table(rows, ("a", "b"), "cap", "tab:x", {"sahel_niger": "sahel"})

    expected = "\n".join(
        [
            r"\begin{table}[]",
            r"    \centering",
            r"    \begin{tabular}{c|c|c}",
            r"     & a & b \\",
            r"    sahel & 1.0 & 2.0 \\",
            r"    made_up & 3.0 & 4.0 \\",
            r"    \end{tabular}",
            r"    \caption{cap}",
            r"    \label{tab:x}",
            r"\end{table}",
        ]
    )
    assert out == expected


def test_latex_table_without_display_map_uses_row_keys():
    rows = {"r1": {"col": "v"}}
    out = latex_table(rows, ("col",), "c", "tab:y")
    assert "    r1 & v \\\\" in out
