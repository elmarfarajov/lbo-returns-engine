"""Two-way sensitivity grids and a solver for break-even inputs.

Sponsors do not present a single IRR; they present the grid that shows how the answer
survives being wrong. Any assumption exposed here can be shifted, and any of the
reported metrics can be gridded against any other.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .assumptions import Deal
from .model import run_lbo
from .returns import equity_returns

METRICS: dict[str, Callable[[Deal], float]] = {}


def _metric(name: str):
    def decorator(func: Callable[[Deal], float]) -> Callable[[Deal], float]:
        METRICS[name] = func
        return func

    return decorator


@_metric("gross_irr")
def _gross_irr(deal: Deal) -> float:
    return equity_returns(run_lbo(deal)).gross_irr


@_metric("net_irr")
def _net_irr(deal: Deal) -> float:
    return equity_returns(run_lbo(deal)).net_irr


@_metric("gross_moic")
def _gross_moic(deal: Deal) -> float:
    return equity_returns(run_lbo(deal)).gross_moic


@_metric("exit_leverage")
def _exit_leverage(deal: Deal) -> float:
    return float(run_lbo(deal).diagnostics["exit_leverage"])


@_metric("min_interest_coverage")
def _min_coverage(deal: Deal) -> float:
    return float(run_lbo(deal).schedule["interest_coverage"].min())


@_metric("covenant_breach")
def _covenant_breach(deal: Deal) -> float:
    return float(run_lbo(deal).diagnostics["covenant_breached"])


def shift_deal(deal: Deal, axis: str, value: float) -> Deal:
    """Return a copy of the deal with one assumption moved to (or by) ``value``."""
    operating, financing, exit_assumptions = deal.operating, deal.financing, deal.exit
    if axis == "entry_multiple":
        return replace(deal, exit=replace(exit_assumptions, entry_multiple=value))
    if axis == "exit_multiple":
        return replace(deal, exit=replace(exit_assumptions, exit_multiple=value))
    if axis == "multiple_change":
        return replace(deal, exit=replace(exit_assumptions, exit_multiple=exit_assumptions.entry_multiple + value))
    if axis == "hold_years":
        years = int(value)
        if years > len(operating.revenue_growth):
            raise ValueError("hold_years exceeds the forecast horizon in the operating assumptions")
        return replace(deal, exit=replace(exit_assumptions, hold_years=years))
    if axis == "leverage_turns":
        funded = [t for t in financing.tranches if not t.is_revolver]
        current = sum(t.opening_balance(deal.entry_ebitda) for t in funded) / deal.entry_ebitda
        scale = value / current if current else 0.0
        tranches = tuple(
            t
            if t.is_revolver
            else replace(t, turns=t.opening_balance(deal.entry_ebitda) / deal.entry_ebitda * scale, amount=None)
            for t in financing.tranches
        )
        return replace(deal, financing=replace(financing, tranches=tranches))
    if axis == "revenue_growth_shift":
        growth = tuple(g + value for g in operating.revenue_growth)
        return replace(deal, operating=replace(operating, revenue_growth=growth))
    if axis == "margin_shift":
        margins = (operating.ebitda_margin[0],) + tuple(m + value for m in operating.ebitda_margin[1:])
        return replace(deal, operating=replace(operating, ebitda_margin=margins))
    if axis == "base_rate_shift":
        curve = tuple(r + value for r in financing.base_rate)
        return replace(deal, financing=replace(financing, base_rate=curve))
    if axis == "cash_sweep_pct":
        return replace(deal, financing=replace(financing, cash_sweep_pct=value))
    if axis == "capex_pct_revenue":
        return replace(deal, operating=replace(operating, capex_pct_revenue=value))
    raise ValueError(f"Unknown sensitivity axis: {axis}")


def sensitivity_grid(
    deal: Deal,
    x_axis: str,
    x_values: Sequence[float],
    y_axis: str,
    y_values: Sequence[float],
    metric: str = "gross_irr",
) -> pd.DataFrame:
    """Grid of a metric over two assumptions; rows are the y axis, columns the x axis."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}")
    evaluate = METRICS[metric]
    grid = np.empty((len(y_values), len(x_values)))
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            try:
                grid[i, j] = evaluate(shift_deal(shift_deal(deal, y_axis, y), x_axis, x))
            except ValueError:
                grid[i, j] = np.nan
    return pd.DataFrame(grid, index=pd.Index(y_values, name=y_axis), columns=pd.Index(x_values, name=x_axis))


def solve_for_target(
    deal: Deal,
    axis: str,
    target: float,
    metric: str = "gross_irr",
    bracket: tuple[float, float] = (0.0, 20.0),
) -> float:
    """Value of one assumption that makes a metric hit a target, e.g. the entry multiple
    that still delivers a 20% IRR. Returns NaN when no value in the bracket does."""
    evaluate = METRICS[metric]

    def objective(value: float) -> float:
        try:
            return evaluate(shift_deal(deal, axis, value)) - target
        except (ValueError, ZeroDivisionError):
            return float("nan")

    # Extreme inputs can make a deal unfinanceable, so scan the bracket and root-find on
    # the first sign change between neighbouring points that both produce a real answer.
    grid = np.linspace(bracket[0], bracket[1], 41)
    values = np.array([objective(x) for x in grid])
    for i in range(len(grid) - 1):
        left, right = values[i], values[i + 1]
        if np.isfinite(left) and np.isfinite(right) and left * right <= 0:
            if left == 0:
                return float(grid[i])
            return float(brentq(objective, grid[i], grid[i + 1], xtol=1e-8, maxiter=200))
    return float("nan")
