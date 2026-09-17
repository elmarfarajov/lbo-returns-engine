"""Monte Carlo on the deal: what is the distribution of outcomes, not just the base case?

Four uncertainties drive an LBO: how fast the business grows, what margin it reaches,
what someone pays for it at exit, and what the debt costs. Growth uncertainty is split
into a persistent component (the business is structurally better or worse than
underwritten) and annual noise, because a permanent growth miss is far more damaging
than a single soft year - leverage turns it into a covenant problem.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .assumptions import Deal
from .model import run_lbo
from .returns import equity_returns


@dataclass(frozen=True)
class MonteCarloConfig:
    n_trials: int = 2_000
    persistent_growth_sd: float = 0.020
    annual_growth_sd: float = 0.015
    margin_sd: float = 0.010
    exit_multiple_sd: float = 1.00
    base_rate_sd: float = 0.010
    exit_multiple_floor: float = 3.0
    recession_probability: float = 0.25
    """Chance of a downturn somewhere in the hold period. Normal-shaped noise alone never
    produces the covenant breaches and write-offs that make leverage dangerous."""
    recession_revenue_shock: float = -0.12
    recession_margin_shock: float = -0.030
    recession_exit_multiple_shock: float = -1.00
    seed: int = 11


@dataclass
class MonteCarloResult:
    trials: pd.DataFrame
    config: MonteCarloConfig

    def summary(self, irr_column: str = "gross_irr") -> dict[str, float]:
        values = self.trials[irr_column].to_numpy()
        finite = values[np.isfinite(values)]
        return {
            "trials": float(len(values)),
            "mean_irr": float(np.mean(finite)),
            "median_irr": float(np.median(finite)),
            "p5_irr": float(np.percentile(finite, 5)),
            "p95_irr": float(np.percentile(finite, 95)),
            "prob_loss": float(np.mean(finite < 0.0)),
            "prob_above_20pct": float(np.mean(finite > 0.20)),
            "prob_above_25pct": float(np.mean(finite > 0.25)),
            "prob_covenant_breach": float(self.trials["covenant_breach"].mean()),
            "prob_recession": float((self.trials["recession_year"] > 0).mean()),
            "median_moic": float(np.nanmedian(self.trials["gross_moic"])),
            "median_exit_leverage": float(np.nanmedian(self.trials["exit_leverage"])),
        }

    def by_scenario(self) -> pd.DataFrame:
        """Outcomes split by whether a downturn hit during the hold period."""
        grouped = self.trials.assign(scenario=np.where(self.trials["recession_year"] > 0, "downturn", "no downturn"))
        return grouped.groupby("scenario").agg(
            trials=("gross_irr", "size"),
            median_irr=("gross_irr", "median"),
            p5_irr=("gross_irr", lambda s: float(np.nanpercentile(s, 5))),
            prob_loss=("gross_irr", lambda s: float(np.nanmean(s < 0))),
            prob_breach=("covenant_breach", "mean"),
            median_moic=("gross_moic", "median"),
        )


def _perturb(deal: Deal, rng: np.random.Generator, config: MonteCarloConfig) -> tuple[Deal, dict[str, float]]:
    years = len(deal.operating.revenue_growth)
    persistent = rng.normal(0.0, config.persistent_growth_sd)
    annual = rng.normal(0.0, config.annual_growth_sd, years)
    margin_shift = rng.normal(0.0, config.margin_sd)
    exit_multiple = max(rng.normal(deal.exit.exit_multiple, config.exit_multiple_sd), config.exit_multiple_floor)
    rate_shift = rng.normal(0.0, config.base_rate_sd)

    growth = [g + persistent + a for g, a in zip(deal.operating.revenue_growth, annual, strict=True)]
    # The margin path ramps to its target, so a shift is phased in the same way.
    steps = len(deal.operating.ebitda_margin) - 1
    margins = [deal.operating.ebitda_margin[0]] + [
        m + margin_shift * (i + 1) / steps for i, m in enumerate(deal.operating.ebitda_margin[1:])
    ]

    recession_year = 0
    if rng.random() < config.recession_probability:
        recession_year = int(rng.integers(1, deal.exit.hold_years + 1))
        growth[recession_year - 1] += config.recession_revenue_shock
        if recession_year < len(growth):  # a partial rebound the year after
            growth[recession_year] += -0.5 * config.recession_revenue_shock
        for year in range(recession_year, len(margins)):
            decay = max(1.0 - 0.4 * (year - recession_year), 0.2)
            margins[year] += config.recession_margin_shock * decay
        exit_multiple = max(exit_multiple + config.recession_exit_multiple_shock, config.exit_multiple_floor)

    growth, margins = tuple(growth), tuple(margins)
    perturbed = replace(
        deal,
        operating=replace(deal.operating, revenue_growth=growth, ebitda_margin=margins),
        financing=replace(deal.financing, base_rate=tuple(r + rate_shift for r in deal.financing.base_rate)),
        exit=replace(deal.exit, exit_multiple=exit_multiple),
    )
    draws = {
        "persistent_growth": persistent,
        "margin_shift": margin_shift,
        "exit_multiple": exit_multiple,
        "base_rate_shift": rate_shift,
        "recession_year": float(recession_year),
    }
    return perturbed, draws


def run_monte_carlo(deal: Deal, config: MonteCarloConfig | None = None) -> MonteCarloResult:
    config = config or MonteCarloConfig()
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, float]] = []
    for _ in range(config.n_trials):
        perturbed, draws = _perturb(deal, rng, config)
        try:
            result = run_lbo(perturbed)
            returns = equity_returns(result)
            records.append(
                {
                    **draws,
                    "gross_irr": returns.gross_irr,
                    "net_irr": returns.net_irr,
                    "gross_moic": returns.gross_moic,
                    "exit_equity_value": returns.exit_equity_value,
                    "exit_leverage": float(result.diagnostics["exit_leverage"]),
                    "max_leverage": result.max_leverage,
                    "min_coverage": float(result.schedule["interest_coverage"].min()),
                    "covenant_breach": float(result.diagnostics["covenant_breached"]),
                    "liquidity_warning": float(bool(result.warnings)),
                }
            )
        except ValueError:
            records.append({**draws, "gross_irr": np.nan, "net_irr": np.nan, "gross_moic": np.nan, "covenant_breach": 1.0})
    return MonteCarloResult(trials=pd.DataFrame(records), config=config)
