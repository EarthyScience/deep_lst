# deep_lst — orientation for Claude

Forecasting land surface temperature over Africa from SEVIRI + ERA5/ERA5-Land, PyTorch
Lightning + Hydra + W&B. **This file describes the `develop` branch.**

Facts about the project live in the documentation, not in this file. Read the relevant
document before asking about, or assuming, anything it covers. Where a document and the code
disagree, the code wins; fix the document in the same change.

| Question | Read |
|---|---|
| what the project is, install, the five pipeline commands, the role of every package | `README.md` |
| config groups, run naming, computed features and old checkpoints, pitfalls | `docs/ARCHITECTURE.md` |
| what a configuration key or option means, its default, which options go together | `docs/CONFIGURATION.md` |
| cluster setup, directories, data stores and assets, preload cache, the SLURM job of every stage | `docs/RUNBOOK.md` |
| drawing a new spatial selection and training on it | `docs/ONBOARDING.md` |

The ground truth of what a run did is `experiments/<run>/version_N/config.yml`, not `conf/`.

## Working here

- Run everything from the repo root, with `.venv/bin/python` or `uv run`, never a bare `python`.
- Network needs `ml proxy` (git, uv, wandb).
- Do not switch branches in `../eodata_slim`; it is an editable install (`docs/ARCHITECTURE.md`,
  section 3).
- Do not run checkpoints of the publication tag `DeepLST_v1.0.0` from `develop`
  (`docs/ARCHITECTURE.md`, section 2).
- `assets/`, `experiments/`, `results/`, `plots/`, `tables/`, `notebooks/` and `scripts/out/` are
  gitignored. Don't propose committing them.
- The SLURM launchers under `scripts/` are tracked. Read the actual script before assuming its
  flags.
- Temporary scripts and intermediate outputs go to tscratch, not into the repo.
- `deep_lst/exploration/` is deliberately kept — mention dead code, don't delete it.
