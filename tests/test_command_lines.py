"""Every Hydra command line in the tracked launchers and in the documentation still composes.

A launcher whose overrides name a key or an option that no longer exists fails at submission
time at the earliest; this test fails instead. Checked are
  - every tracked scripts/**/*.sh: lines that call an entry point (directly, through a
    wrapper launcher or through a shell variable holding one), usage comments included
  - README.md and docs/*.md: fenced code blocks and inline code spans

How a line is read: backslash continuations are joined, shell variables assigned once in the
script are expanded, and the `key=value` tokens after the entry point are composed against
its config. A variable assigned several times (e.g. in a loop over variants) is checked value
by value. A value that is only known at run time (`${seed}`, `<tag>`) is replaced by a
placeholder; for a config group (`data=${data}`) the token is skipped, since the option
cannot be known.
"""

import re
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"

# script name on a command line -> the config it composes
ENTRY_POINTS = {
    "train.py": "train",
    "train_mlp.sh": "train",
    "train_transformer.sh": "train",
    "finetune.py": "finetune",
    "run_finetune.sh": "finetune",
    "run_inference.py": "infer",
    "run_sensitivity.py": "sensitivity",
    "compute_statistics.py": "statistics",
}
ENTRY_POINT = re.compile(r"(?<![\w.-])(" + "|".join(re.escape(name) for name in ENTRY_POINTS) + r")(?![\w.])")
CONFIG_GROUPS = sorted(path.name for path in CONF_DIR.iterdir() if path.is_dir())

ASSIGNMENT = re.compile(r"""(?:^|[\s;&|(])(?:local\s+|export\s+)?([A-Za-z_]\w*)=("[^"]*"|'[^']*'|[^\s;)]*)""")
VARIABLE = re.compile(r"\$\{(\w+)\}|\$(\w+)")
TOKEN = re.compile(r"""'[^']*'|"[^"]*"|\S+""")
OVERRIDE = re.compile(r"([+~]{0,2}[A-Za-z_][\w./]*)=(.*)", re.DOTALL)
SHELL_OPERATORS = ("&&", "||", "|", ";", ">", ">>", "2>&1")


def _tracked_launchers():
    try:
        listing = subprocess.check_output(["git", "ls-files", "scripts"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        paths = [REPO_ROOT / line for line in listing.decode().splitlines()]
    except (OSError, subprocess.CalledProcessError):
        paths = list((REPO_ROOT / "scripts").rglob("*.sh"))
    return sorted(path for path in paths if path.suffix == ".sh" and path.exists())


def _logical_lines(text):
    """Lines with backslash continuations joined; (first line number, line)."""
    lines = []
    buffer = ""
    start = None
    for number, line in enumerate(text.splitlines(), start=1):
        if start is None:
            start = number
        if line.rstrip().endswith("\\"):
            buffer += line.rstrip()[:-1] + " "
            continue
        lines.append((start, buffer + line))
        buffer = ""
        start = None
    if buffer:
        lines.append((start, buffer))
    return lines


def _strip_quotes(value):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _shell_variables(lines):
    """name -> the distinct values assigned to it anywhere in the script."""
    variables = {}
    for _, line in lines:
        for name, value in ASSIGNMENT.findall(line):
            values = variables.setdefault(name, [])
            value = _strip_quotes(value)
            if value not in values:
                values.append(value)
    return variables


def _expand(line, variables):
    """Replace references to variables that have exactly one value, a few levels deep."""

    def substitute(match):
        name = match.group(1) or match.group(2)
        values = variables.get(name, [])
        if len(values) == 1:
            return values[0]
        return match.group(0)

    for _ in range(5):
        expanded = VARIABLE.sub(substitute, line)
        if expanded == line:
            break
        line = expanded
    return line


def _override_tokens(text):
    """The key=value tokens of a command line remainder, up to the first shell operator."""
    tokens = []
    for token in TOKEN.findall(text):
        if token in SHELL_OPERATORS:
            break
        if token[0] not in "\"'":
            token = token.rstrip("\"')")  # the end of a quoted command string or of $( ... )
        match = OVERRIDE.fullmatch(_strip_quotes(token))
        if match:
            tokens.append((match.group(1), match.group(2)))
    return tokens


def _command_lines(lines, variables):
    """(line number, config name, override tokens) of every entry-point call, plus one entry
    per value of each several-valued variable such a call references."""
    found = []
    for number, line in lines:
        expanded = _expand(line, variables)
        match = ENTRY_POINT.search(expanded)
        if match is None:
            continue
        config_name = ENTRY_POINTS[match.group(1)]
        remainder = expanded[match.end() :]
        tokens = _override_tokens(remainder)
        if tokens:
            found.append((number, config_name, tokens))
        for reference in VARIABLE.finditer(remainder):
            name = reference.group(1) or reference.group(2)
            values = variables.get(name, [])
            if len(values) > 1:
                for value in values:
                    found.append((number, config_name, _override_tokens(value)))
    return found


def _compose_errors(command_lines, source):
    errors = []
    for number, config_name, tokens in command_lines:
        overrides = []
        for key, value in tokens:
            unknown_at_test_time = "$" in value or "<" in value
            if unknown_at_test_time and key in CONFIG_GROUPS:
                continue
            if unknown_at_test_time:
                value = "0"
            overrides.append(f"{key}={value}")
        try:
            with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
                compose(config_name=config_name, overrides=overrides)
        except Exception as error:
            reason = str(error).strip().splitlines()[0]
            errors.append(f"{source}:{number}: {config_name} {' '.join(overrides)}\n    {reason}")
    return errors


def _documentation_snippets(text):
    """Fenced code blocks line by line, and inline code spans, as (line number, code)."""
    snippets = []
    in_block = False
    block = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("```"):
            if in_block:
                first_line = number - len(block)
                for start, logical in _logical_lines("\n".join(block)):
                    snippets.append((first_line + start - 1, logical))
                block = []
            in_block = not in_block
            continue
        if in_block:
            block.append(line)
        else:
            for span in re.findall(r"`([^`]+)`", line):
                snippets.append((number, span))
    return snippets


LAUNCHERS = _tracked_launchers()
DOCUMENTS = [REPO_ROOT / "README.md"] + sorted((REPO_ROOT / "docs").glob("*.md"))


@pytest.mark.parametrize("path", LAUNCHERS, ids=[str(path.relative_to(REPO_ROOT)) for path in LAUNCHERS])
def test_launcher_command_lines_compose(path):
    lines = _logical_lines(path.read_text())
    variables = _shell_variables(lines)

    errors = _compose_errors(_command_lines(lines, variables), path.relative_to(REPO_ROOT))

    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("path", DOCUMENTS, ids=[str(path.relative_to(REPO_ROOT)) for path in DOCUMENTS])
def test_documented_command_lines_compose(path):
    snippets = _documentation_snippets(path.read_text())

    errors = _compose_errors(_command_lines(snippets, {}), path.relative_to(REPO_ROOT))

    assert not errors, "\n".join(errors)


def test_the_reader_finds_the_command_lines_it_is_meant_to():
    """Guards the parsing above: a reader that finds nothing would pass every script."""
    script = "\n".join(
        [
            'LAUNCH="scripts/training/train_mlp.sh"',
            'ARGS="loss=smoothl1 horizon=h24"',
            "for comp in a b; do",
            '    [ "${comp}" == "a" ] && extra="objective.terms.increment=1"',
            '    [ "${comp}" == "b" ] && extra="objective.terms.rollout=0"',
            "    sbatch --time=1:00:00 ${LAUNCH} model=mlp ${ARGS} \\",
            "        data=${data} seed=${seed} ${extra}",
            "done",
            "srun python3 run_inference.py eval=gobabeb 'splits.test.init_hours=[0, 6]' && echo done=1",
        ]
    )
    lines = _logical_lines(script)

    found = _command_lines(lines, _shell_variables(lines))

    assert found == [
        (
            6,
            "train",
            [("model", "mlp"), ("loss", "smoothl1"), ("horizon", "h24"), ("data", "${data}"), ("seed", "${seed}")],
        ),
        (6, "train", [("objective.terms.increment", "1")]),
        (6, "train", [("objective.terms.rollout", "0")]),
        (9, "infer", [("eval", "gobabeb"), ("splits.test.init_hours", "[0, 6]")]),
    ]
    assert _compose_errors(found, "example") == []
    stale = [(1, "train", [("model", "mlp_rollout")]), (2, "infer", [("test_dataset.init_hours", "[0]")])]
    assert len(_compose_errors(stale, "example")) == 2
