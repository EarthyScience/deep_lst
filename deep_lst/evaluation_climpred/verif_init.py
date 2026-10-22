"""Per-init verification: keeps the init dim, reduces pixels (dim=["pixel"]).

Saves raw material only (mse + counts); pooled curves and buckets are derived in
derive.py. Run: python -m deep_lst.evaluation_climpred.verif_init
"""

import xarray as xr

from .common import VERIFY_KWARGS, build_ensembles, setup_verif, verification_dir
from .stats import per_init_counts


def main():
    args, initilised, obs = setup_verif()
    hindcast, baselines = build_ensembles(args, initilised, obs)

    model_by_init = hindcast.verify(dim=["pixel"], **VERIFY_KWARGS)["LST"].drop_vars("skill", errors="ignore")
    baseline_by_init = baselines.verify(dim=["pixel"], **VERIFY_KWARGS)["LST"].drop_vars("skill", errors="ignore")

    bias_kwargs = dict(VERIFY_KWARGS, metric="me")
    bias_by_init = hindcast.verify(dim=["pixel"], **bias_kwargs)["LST"].drop_vars("skill", errors="ignore")

    out = xr.Dataset(
        {
            "mse_model": model_by_init,
            "mse_baseline": baseline_by_init,
            "bias_model": bias_by_init,
            "n": per_init_counts(initilised, obs, args.windows),
        }
    )
    out.to_netcdf(verification_dir(args, "verif") / "by_init.nc")


if __name__ == "__main__":
    main()
