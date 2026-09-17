"""One call from a deal file to every analysis in the report."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assumptions import Deal
from .bridge import ValueBridge, attribution_summary, value_bridge
from .model import LBOResult, run_lbo
from .montecarlo import MonteCarloConfig, MonteCarloResult, run_monte_carlo
from .returns import EquityReturns, equity_returns
from .sensitivity import sensitivity_grid, shift_deal, solve_for_target

LEVERS: dict[str, tuple[str, float]] = {
    "Exit multiple +1.0x": ("exit_multiple", 1.0),
    "Exit multiple -1.0x": ("exit_multiple", -1.0),
    "Revenue growth +200bp p.a.": ("revenue_growth_shift", 0.02),
    "Revenue growth -200bp p.a.": ("revenue_growth_shift", -0.02),
    "EBITDA margin +100bp": ("margin_shift", 0.01),
    "EBITDA margin -100bp": ("margin_shift", -0.01),
    "Leverage +1.0x": ("leverage_turns", 1.0),
    "Leverage -1.0x": ("leverage_turns", -1.0),
    "Base rates +200bp": ("base_rate_shift", 0.02),
    "Capex +100bp of revenue": ("capex_pct_revenue", 0.01),
    "Exit one year earlier": ("hold_years", -1.0),
}

BREAK_EVENS: dict[str, tuple[str, float, tuple[float, float]]] = {
    "Entry multiple for a 20% IRR": ("entry_multiple", 0.20, (4.0, 20.0)),
    "Exit multiple for a 20% IRR": ("exit_multiple", 0.20, (3.0, 20.0)),
    "Exit multiple to break even": ("exit_multiple", 0.0, (1.0, 20.0)),
    "Revenue growth for a 20% IRR": ("revenue_growth_shift", 0.20, (-0.20, 0.20)),
    "Margin change for a 20% IRR": ("margin_shift", 0.20, (-0.10, 0.10)),
    "Rate shock the deal survives (0% IRR)": ("base_rate_shift", 0.0, (-0.05, 0.40)),
}


@dataclass
class AnalysisConfig:
    monte_carlo_trials: int = 2_000
    leverage_levels: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0)
    leverage_curve_trials: int = 400
    run_monte_carlo: bool = True
    run_leverage_curve: bool = True
    entry_multiple_range: tuple[float, ...] = ()
    exit_multiple_range: tuple[float, ...] = ()
    index_levels: tuple[float, ...] = ()
    """Total-return index path (1.0 at entry) used for the public market equivalent."""


@dataclass
class DealAnalysis:
    deal: Deal
    result: LBOResult
    returns: EquityReturns
    bridge: ValueBridge
    attribution: pd.DataFrame
    levers: pd.DataFrame
    break_evens: pd.DataFrame
    grids: dict[str, pd.DataFrame]
    monte_carlo: MonteCarloResult | None
    leverage_curve: pd.DataFrame | None
    pme: dict[str, float] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def _default_multiple_range(centre: float) -> tuple[float, ...]:
    return tuple(round(centre + step, 2) for step in (-1.5, -0.75, 0.0, 0.75, 1.5))


def lever_table(deal: Deal, base_irr: float) -> pd.DataFrame:
    """Each driver moved by a standard amount, ranked by how much IRR it buys or costs."""
    rows = []
    for label, (axis, delta) in LEVERS.items():
        if axis == "leverage_turns":
            value = deal.total_debt_turns() + delta
        elif axis == "exit_multiple":
            value = deal.exit.exit_multiple + delta
        elif axis == "hold_years":
            value = deal.exit.hold_years + delta
        elif axis == "capex_pct_revenue":
            value = deal.operating.capex_pct_revenue + delta
        else:
            value = delta
        try:
            shifted = shift_deal(deal, axis, value)
            irr = equity_returns(run_lbo(shifted)).gross_irr
        except ValueError:
            irr = np.nan
        rows.append({"lever": label, "gross_irr": irr, "change_vs_base": irr - base_irr})
    frame = pd.DataFrame(rows)
    return frame.reindex(frame["change_vs_base"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def break_even_table(deal: Deal) -> pd.DataFrame:
    rows = []
    for label, (axis, target, bracket) in BREAK_EVENS.items():
        rows.append({"question": label, "axis": axis, "answer": solve_for_target(deal, axis, target, bracket=bracket)})
    return pd.DataFrame(rows)


def run_analysis(deal: Deal, config: AnalysisConfig | None = None, log: Callable[[str], None] = print) -> DealAnalysis:
    config = config or AnalysisConfig()
    timings: dict[str, float] = {}

    start = time.perf_counter()
    result = run_lbo(deal)
    returns = equity_returns(result)
    bridge = value_bridge(result)
    timings["model"] = time.perf_counter() - start
    log(
        f"  Base case: {returns.gross_moic:.2f}x gross MOIC, {100 * returns.gross_irr:.1f}% gross IRR, "
        f"{100 * returns.net_irr:.1f}% net of fees and carry; exit leverage "
        f"{result.diagnostics['exit_leverage']:.2f}x ({timings['model'] * 1000:.0f} ms)"
    )
    if result.warnings:
        for warning in result.warnings:
            log(f"  ! {warning}")

    start = time.perf_counter()
    levers = lever_table(deal, returns.gross_irr)
    break_evens = break_even_table(deal)
    entry_range = config.entry_multiple_range or _default_multiple_range(deal.exit.entry_multiple)
    exit_range = config.exit_multiple_range or _default_multiple_range(deal.exit.exit_multiple)
    grids = {
        "Gross IRR: entry versus exit multiple": sensitivity_grid(
            deal, "exit_multiple", exit_range, "entry_multiple", entry_range
        ),
        "Gross IRR: leverage versus exit multiple": sensitivity_grid(
            deal, "exit_multiple", exit_range, "leverage_turns", config.leverage_levels
        ),
    }
    timings["sensitivity"] = time.perf_counter() - start
    top = levers.iloc[0]
    log(f"  Biggest single lever: {top['lever']} moves the IRR by {100 * top['change_vs_base']:+.1f} points")

    monte_carlo = leverage_curve = None
    if config.run_monte_carlo:
        start = time.perf_counter()
        monte_carlo = run_monte_carlo(deal, MonteCarloConfig(n_trials=config.monte_carlo_trials))
        timings["monte_carlo"] = time.perf_counter() - start
        summary = monte_carlo.summary()
        log(
            f"  Monte Carlo: median IRR {100 * summary['median_irr']:.1f}%, "
            f"P(loss) {100 * summary['prob_loss']:.1f}%, P(covenant breach) {100 * summary['prob_covenant_breach']:.1f}% "
            f"({timings['monte_carlo']:.1f}s)"
        )
    if config.run_leverage_curve:
        from .figures import leverage_risk_return

        start = time.perf_counter()
        _, leverage_curve = leverage_risk_return(deal, config.leverage_levels, config.leverage_curve_trials)
        timings["leverage_curve"] = time.perf_counter() - start

    pme: dict[str, float] = {}
    if config.index_levels:
        from .returns import direct_alpha, kaplan_schoar_pme

        index = np.asarray(config.index_levels, dtype=float)
        if index.size == returns.sponsor_gross_flows.size:
            pme = {
                "ks_pme": kaplan_schoar_pme(returns.sponsor_gross_flows, index),
                "direct_alpha": direct_alpha(returns.sponsor_gross_flows, index),
                "index_return": float(index[-1] / index[0]) ** (1.0 / deal.exit.hold_years) - 1.0,
            }
            log(f"  Public market equivalent: KS-PME {pme['ks_pme']:.2f}, direct alpha {100 * pme['direct_alpha']:+.1f}%")

    return DealAnalysis(
        deal=deal,
        result=result,
        returns=returns,
        bridge=bridge,
        attribution=attribution_summary(result),
        levers=levers,
        break_evens=break_evens,
        grids=grids,
        monte_carlo=monte_carlo,
        leverage_curve=leverage_curve,
        pme=pme,
        timings=timings,
    )
