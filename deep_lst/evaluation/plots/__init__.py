import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless cluster nodes

from deep_lst.evaluation.plots.across_sites import cross_site_acc_boxplot, cross_site_rmse_boxplot
from deep_lst.evaluation.plots.diurnal import diurnal_line
from deep_lst.evaluation.plots.error_growth import (
    error_growth_daily_plot,
    error_growth_plot,
    skill_plot,
)
from deep_lst.evaluation.plots.maps import bias_maps, improvement_map
from deep_lst.evaluation.plots.sweep import sweep_overview
from deep_lst.evaluation.plots.timeseries import timeseries_plot
from deep_lst.evaluation.plots.verification import regression_parity

__all__ = [
    "sweep_overview",
    "diurnal_line",
    "bias_maps",
    "improvement_map",
    "timeseries_plot",
    "error_growth_plot",
    "error_growth_daily_plot",
    "skill_plot",
    "regression_parity",
    "cross_site_rmse_boxplot",
    "cross_site_acc_boxplot",
]
