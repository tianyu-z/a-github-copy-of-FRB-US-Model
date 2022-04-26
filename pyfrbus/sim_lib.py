from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# For mypy typing
from typing import List
from pandas import PeriodIndex, Series


def single_plot(axes, r, c, periods: PeriodIndex, base, sim):
    axes[r, c].plot(range(len(periods)), base, color="C0", label="Baseline")
    axes[r, c].plot(range(len(periods)), sim, color="#AA4700", ls="--", label="Sim")
    xt = range(0, len(periods), ceil(len(periods) / 4))
    axes[r, c].set_xticks(xt)
    axes[r, c].set_xticklabels(periods[xt])
    axes[r, c].set_ylabel("Percent")
    if r == 0 and c == 0:
        axes[r, c].legend(loc="upper right")


def single_fanplot(axes, r, c, periods: PeriodIndex, base, lo70, hi70, lo90, hi90):
    axes[r, c].fill_between(
        range(len(periods)),
        lo90,
        hi90,
        color="none",
        edgecolor="#5A5A5A",
        hatch="|||||||",
        label="90% Confidence Interval",
    )
    axes[r, c].fill_between(
        range(len(periods)),
        lo70,
        hi70,
        color="lightgrey",
        label="70% Confidence Interval",
        edgecolor="#5A5A5A",
    )
    axes[r, c].plot(range(len(periods)), base, color="C0", label="Baseline")
    xt = range(0, len(periods), ceil(len(periods) / 4))
    axes[r, c].set_xticks(xt)
    axes[r, c].set_xticklabels(periods[xt])
    if r == 1 and c == 1:
        axes[r, c].legend(loc="upper left", fontsize="x-small", handlelength=1.0)


def sim_plot(baseline: DataFrame, sim: DataFrame, start: str, end: str) -> None:
    # Pad with 25% history but not more than 6 or less than 2 qtrs
    back_pad = max(min(round((pd.Period(end) - pd.Period(start)).n / 4), 6), 2)
    plot_period = pd.period_range(pd.Period(start) - back_pad, end, freq="Q")
    growth_gdp = sim["xgdp"].pct_change(4) * 100
    growth_inf = sim["pcxfe"].pct_change(4) * 100
    base_growth_gdp = baseline["xgdp"].pct_change(4) * 100
    base_growth_inf = baseline["pcxfe"].pct_change(4) * 100
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    axes[0, 0].set_title("Real GDP Growth, Quarterly Annualized")
    axes[0, 1].set_title("Unemployment Rate")
    axes[1, 0].set_title("Core PCE Inflation, Quarterly Annualized")
    axes[1, 1].set_title("Federal Funds Rate")
    single_plot(
        axes, 0, 0, plot_period, base_growth_gdp[plot_period], growth_gdp[plot_period]
    )
    single_plot(
        axes,
        0,
        1,
        plot_period,
        baseline.loc[plot_period, "lur"],
        sim.loc[plot_period, "lur"],
    )
    single_plot(
        axes, 1, 0, plot_period, base_growth_inf[plot_period], growth_inf[plot_period]
    )
    single_plot(
        axes,
        1,
        1,
        plot_period,
        baseline.loc[plot_period, "rff"],
        sim.loc[plot_period, "rff"],
    )
    fig.tight_layout()
    plt.show()


def stochsim_plot(
    baseline: DataFrame, sims: List[DataFrame], start: str, end: str
) -> None:
    # Pad with 25% history but not more than 6 or less than 2 qtrs
    back_pad = max(min(round((pd.Period(end) - pd.Period(start)).n / 4), 6), 2)
    plot_period = pd.period_range(pd.Period(start) - back_pad, end, freq="Q")
    base_growth_gdp = baseline["xgdp"].pct_change(4) * 100
    base_growth_inf = baseline["pcxfe"].pct_change(4) * 100
    # Convert frame data too
    for sim in sims:
        sim["growth_gdp"] = sim["xgdp"].pct_change(4) * 100
        sim["growth_inf"] = sim["pcxfe"].pct_change(4) * 100

    growth_gdp_lo70 = take_quantile(sims, "growth_gdp", plot_period, 0.15)
    growth_gdp_hi70 = take_quantile(sims, "growth_gdp", plot_period, 0.85)
    growth_inf_lo70 = take_quantile(sims, "growth_inf", plot_period, 0.15)
    growth_inf_hi70 = take_quantile(sims, "growth_inf", plot_period, 0.85)
    lur_lo70 = take_quantile(sims, "lur", plot_period, 0.15)
    lur_hi70 = take_quantile(sims, "lur", plot_period, 0.85)
    rff_lo70 = take_quantile(sims, "rff", plot_period, 0.15)
    rff_hi70 = take_quantile(sims, "rff", plot_period, 0.85)

    # 90s
    growth_gdp_lo90 = take_quantile(sims, "growth_gdp", plot_period, 0.05)
    growth_gdp_hi90 = take_quantile(sims, "growth_gdp", plot_period, 0.95)
    growth_inf_lo90 = take_quantile(sims, "growth_inf", plot_period, 0.05)
    growth_inf_hi90 = take_quantile(sims, "growth_inf", plot_period, 0.95)
    lur_lo90 = take_quantile(sims, "lur", plot_period, 0.05)
    lur_hi90 = take_quantile(sims, "lur", plot_period, 0.95)
    rff_lo90 = take_quantile(sims, "rff", plot_period, 0.05)
    rff_hi90 = take_quantile(sims, "rff", plot_period, 0.95)

    # Drop the growth series
    for sim in sims:
        sim.drop(["growth_gdp", "growth_inf"], axis=1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    axes[0, 0].set_title("Real GDP Growth, Quarterly Annualized")
    axes[0, 1].set_title("Unemployment Rate")
    axes[1, 0].set_title("Core PCE Inflation, Quarterly Annualized")
    axes[1, 1].set_title("Federal Funds Rate")

    single_fanplot(
        axes,
        0,
        0,
        plot_period,
        base_growth_gdp[plot_period],
        growth_gdp_lo70[plot_period],
        growth_gdp_hi70[plot_period],
        growth_gdp_lo90[plot_period],
        growth_gdp_hi90[plot_period],
    )
    single_fanplot(
        axes,
        0,
        1,
        plot_period,
        baseline.loc[plot_period, "lur"],
        lur_lo70[plot_period],
        lur_hi70[plot_period],
        lur_lo90[plot_period],
        lur_hi90[plot_period],
    )
    single_fanplot(
        axes,
        1,
        0,
        plot_period,
        base_growth_inf[plot_period],
        growth_inf_lo70[plot_period],
        growth_inf_hi70[plot_period],
        growth_inf_lo90[plot_period],
        growth_inf_hi90[plot_period],
    )
    single_fanplot(
        axes,
        1,
        1,
        plot_period,
        baseline.loc[plot_period, "rff"],
        rff_lo70[plot_period],
        rff_hi70[plot_period],
        rff_lo90[plot_period],
        rff_hi90[plot_period],
    )
    fig.tight_layout()
    plt.show()


def take_quantile(
    sims: List[DataFrame], seriesname: str, plot_period: PeriodIndex, q: float
) -> Series:
    return DataFrame(map(lambda s: s[seriesname][plot_period], sims)).quantile(q)
