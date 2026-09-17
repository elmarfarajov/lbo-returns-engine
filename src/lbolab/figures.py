"""Investment-committee style figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .assumptions import Deal
from .bridge import ValueBridge
from .model import LBOResult, run_lbo
from .montecarlo import MonteCarloConfig, run_monte_carlo
from .returns import equity_returns
from .sensitivity import shift_deal

if TYPE_CHECKING:
    from .montecarlo import MonteCarloResult

INK = "#1c2733"
MUTED = "#75818c"
GRID = "#e6e9ec"
POSITIVE = "#1f7a5a"
NEGATIVE = "#b3402f"
ACCENT = "#28527a"
SOFT = "#8aa8c8"
DEBT_COLOURS = ["#28527a", "#4b7ba7", "#7ea6c9", "#a9c4dc", "#cfdded"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#c3c9ce",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
PCT_FORMATTER = FuncFormatter(lambda v, _: f"{100 * v:.0f}%")


def sources_and_uses(result: LBOResult) -> Figure:
    su = result.sources_uses
    sources = {f"{name}": amount for name, amount in su.tranche_draws.items() if amount > 0}
    sources["Rollover equity"] = su.rollover_equity
    sources["Sponsor equity"] = su.sponsor_equity
    uses = {
        "Purchase price": su.enterprise_value,
        "Transaction fees": su.transaction_fees,
        "Financing fees": su.financing_fees,
        "Cash to balance sheet": su.cash_to_balance_sheet,
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for column, (label, items) in enumerate([("Sources", sources), ("Uses", uses)]):
        bottom = 0.0
        for index, (name, amount) in enumerate(items.items()):
            if amount <= 0:
                continue
            colour = DEBT_COLOURS[index % len(DEBT_COLOURS)] if column == 0 else [ACCENT, NEGATIVE, "#d98d6a", SOFT][index % 4]
            ax.bar(column, amount, bottom=bottom, color=colour, width=0.55, edgecolor="white")
            if amount > 0.03 * su.total_uses:
                ax.text(
                    column, bottom + amount / 2, f"{name}\n{amount:,.0f}", ha="center", va="center", fontsize=8, color="white"
                )
            bottom += amount
        ax.text(column, bottom * 1.02, f"{label}: {bottom:,.0f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sources", "Uses"])
    ax.set_ylabel(f"{result.deal.currency}")
    entry = result.deal.exit.entry_multiple
    ax.set_title(
        f"Sources and uses: {entry:.1f}x EBITDA entry, {result.deal.total_debt_turns():.2f}x leverage, "
        f"{100 * su.equity_cheque / su.total_uses:.0f}% equity"
    )
    ax.set_ylim(0, bottom * 1.12)
    fig.tight_layout()
    return fig


def deleveraging(result: LBOResult) -> Figure:
    schedule = result.schedule
    years = schedule["year"].to_numpy()
    debt_columns = [c for c in schedule.columns if c.startswith("debt_")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    bottom = np.zeros(len(years))
    for index, column in enumerate(debt_columns):
        values = schedule[column].to_numpy()
        ax1.bar(
            years,
            values,
            bottom=bottom,
            color=DEBT_COLOURS[index % len(DEBT_COLOURS)],
            label=column.replace("debt_", ""),
            width=0.6,
            edgecolor="white",
        )
        bottom += values
    ax1.plot(years, schedule["cash_end"], "o--", color=POSITIVE, ms=4, label="Cash")
    ax1.set_xticks(years)
    ax1.set_xlabel("year")
    ax1.set_ylabel(result.deal.currency)
    ax1.set_title("Capital structure through the hold period")
    ax1.set_ylim(0, float(bottom.max()) * 1.32)
    ax1.legend(fontsize=8, ncol=2, loc="upper center")

    ax2.plot(years, schedule["leverage"], "o-", color=ACCENT, ms=4, label="Net debt / EBITDA")
    limits = [result.deal.covenants.leverage_limit(int(y)) for y in years]
    if any(limit is not None for limit in limits):
        ax2.step(
            years,
            [np.nan if limit is None else limit for limit in limits],
            where="mid",
            color=NEGATIVE,
            lw=1.2,
            ls="--",
            label="Covenant limit",
        )
    ax2.set_xticks(years)
    ax2.set_xlabel("year")
    ax2.set_ylabel("turns of EBITDA")
    twin = ax2.twinx()
    twin.plot(years, schedule["interest_coverage"], "s-", color=POSITIVE, ms=4, label="EBITDA / cash interest")
    twin.set_ylabel("interest coverage (x)")
    twin.grid(False)
    lines = ax2.get_lines() + twin.get_lines()
    ax2.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")
    ax2.set_title("Leverage and coverage versus covenants")
    fig.tight_layout()
    return fig


def value_bridge_chart(bridge: ValueBridge, currency: str = "USD m") -> Figure:
    points = bridge.waterfall_points()
    top = float((points["base"] + points["height"]).max())
    threshold = 0.004 * top
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for index, row in points.iterrows():
        is_total = index in (0, len(points) - 1)
        colour = ACCENT if is_total else (POSITIVE if row["value"] >= 0 else NEGATIVE)
        ax.bar(index, max(row["height"], threshold * 0.3), bottom=row["base"], color=colour, width=0.62, edgecolor="white")
        label = row["height"] if is_total else row["value"]
        if is_total or abs(row["value"]) >= threshold:
            above = is_total or row["value"] >= 0
            y = row["base"] + row["height"] + 0.012 * top if above else row["base"] - 0.045 * top
            ax.text(index, y, f"{label:,.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(-0.08 * top, top * 1.16)
    ax.set_xticks(range(len(points)))
    ax.set_xticklabels([label.replace(" ", "\n") for label in points["label"]], fontsize=8)
    ax.set_ylabel(currency)
    total = bridge.total_value_created
    multiple = (bridge.exit_equity_value + bridge.dividends) / bridge.entry_equity
    ax.set_title(f"Where the {total:,.0f} of equity value came from ({multiple:.2f}x gross money multiple)")
    fig.tight_layout()
    return fig


def sensitivity_heatmaps(grids: dict[str, pd.DataFrame]) -> Figure:
    fig, axes = plt.subplots(1, len(grids), figsize=(5.6 * len(grids), 4.2), squeeze=False)
    for ax, (title, grid) in zip(axes.ravel(), grids.items(), strict=False):
        data = 100 * grid.to_numpy()
        image = ax.imshow(data, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([f"{v:g}" for v in grid.columns])
        ax.set_yticks(range(grid.shape[0]))
        ax.set_yticklabels([f"{v:g}" for v in grid.index])
        ax.set_xlabel(grid.columns.name.replace("_", " "))
        ax.set_ylabel(grid.index.name.replace("_", " "))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8.5)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="IRR (%)")
    fig.tight_layout()
    return fig


def monte_carlo_chart(mc: MonteCarloResult) -> Figure:
    trials = mc.trials
    irr = trials["gross_irr"].to_numpy()
    finite = irr[np.isfinite(irr)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    ax1.hist(finite, bins=45, color=SOFT, edgecolor="white")
    for value, colour, label in [
        (np.median(finite), ACCENT, f"median {100 * np.median(finite):.1f}%"),
        (np.percentile(finite, 5), NEGATIVE, f"5th pct {100 * np.percentile(finite, 5):.1f}%"),
        (0.20, POSITIVE, "20% target"),
    ]:
        ax1.axvline(value, color=colour, lw=1.5, ls="--", label=label)
    ax1.xaxis.set_major_formatter(PCT_FORMATTER)
    ax1.set_xlabel("gross IRR")
    ax1.set_ylabel("trials")
    ax1.set_title(f"Outcome distribution ({len(irr):,} trials)")
    ax1.legend(fontsize=8)

    downturn = trials["recession_year"] > 0
    ax2.hist(
        [finite[~downturn[np.isfinite(irr)]], finite[downturn[np.isfinite(irr)]]],
        bins=30,
        stacked=True,
        color=[SOFT, NEGATIVE],
        edgecolor="white",
        label=["no downturn", "downturn in hold period"],
    )
    ax2.xaxis.set_major_formatter(PCT_FORMATTER)
    ax2.set_xlabel("gross IRR")
    ax2.set_ylabel("trials")
    summary = mc.summary()
    ax2.set_title(
        f"P(loss) {100 * summary['prob_loss']:.1f}% · P(IRR>20%) {100 * summary['prob_above_20pct']:.0f}% · "
        f"P(covenant breach) {100 * summary['prob_covenant_breach']:.0f}%"
    )
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def leverage_risk_return(
    deal: Deal, turns: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0), n_trials: int = 400
) -> tuple[Figure, pd.DataFrame]:
    """The central trade-off: leverage lifts the median return and the chance of breaching."""
    rows = []
    for level in turns:
        try:
            levered = shift_deal(deal, "leverage_turns", level)
            base = equity_returns(run_lbo(levered))
            mc = run_monte_carlo(levered, MonteCarloConfig(n_trials=n_trials))
            summary = mc.summary()
            rows.append(
                {
                    "leverage_turns": level,
                    "base_case_irr": base.gross_irr,
                    "median_irr": summary["median_irr"],
                    "p5_irr": summary["p5_irr"],
                    "prob_loss": summary["prob_loss"],
                    "prob_breach": summary["prob_covenant_breach"],
                    "equity_cheque": base.entry_equity,
                }
            )
        except ValueError:
            continue
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(frame["leverage_turns"], frame["median_irr"], "o-", color=ACCENT, label="median IRR")
    ax.fill_between(
        frame["leverage_turns"], frame["p5_irr"], frame["median_irr"], color=SOFT, alpha=0.35, label="5th percentile to median"
    )
    ax.yaxis.set_major_formatter(PCT_FORMATTER)
    ax.set_xlabel("total leverage at entry (turns of EBITDA)")
    ax.set_ylabel("gross IRR")
    twin = ax.twinx()
    twin.plot(frame["leverage_turns"], frame["prob_breach"], "s--", color=NEGATIVE, label="P(covenant breach)")
    twin.plot(frame["leverage_turns"], frame["prob_loss"], "^:", color="#7a3b2e", label="P(loss)")
    twin.yaxis.set_major_formatter(PCT_FORMATTER)
    twin.set_ylabel("probability")
    twin.grid(False)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper left")
    ax.set_title("How much is leverage really worth?")
    fig.tight_layout()
    return fig, frame


def screen_chart(screen: pd.DataFrame, target_irr: float) -> Figure:
    financeable = screen[screen["max_premium"].notna()].sort_values("max_premium", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.32 * len(financeable) + 1.4)))
    colours = [POSITIVE if value > 0.25 else (SOFT if value > 0 else NEGATIVE) for value in financeable["max_premium"]]
    ax.barh(financeable["ticker"], 100 * financeable["max_premium"], color=colours)
    for y, (value, multiple) in enumerate(
        zip(financeable["max_premium"], financeable["entry_multiple_at_premium"], strict=False)
    ):
        ax.text(100 * value + 0.6, y, f"{100 * value:.0f}%  ({multiple:.1f}x)", va="center", fontsize=8, color=INK)
    ax.set_xlabel("maximum premium to the current share price (%)")
    ax.set_title(f"What a sponsor could pay and still make {100 * target_irr:.0f}% IRR")
    ax.grid(axis="y")
    fig.tight_layout()
    return fig
