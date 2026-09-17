"""Private equity return analytics: IRR, MOIC, the equity waterfall and public market equivalents.

Gross returns flatter sponsors: they ignore the fees and carried interest an investor
actually pays. Every figure here is reported gross *and* net, and against the public
market through the Kaplan-Schoar PME, which answers the only question a limited
partner really has - would the same cash in an index fund have done better?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from .assumptions import WaterfallAssumptions
from .model import LBOResult


def npv(rate: float, cash_flows: ArrayLike) -> float:
    flows = np.asarray(cash_flows, dtype=float)
    periods = np.arange(flows.size)
    return float(np.sum(flows / (1.0 + rate) ** periods))


def irr(cash_flows: ArrayLike, low: float = -0.999, high: float = 10.0) -> float:
    """Annual IRR of equally spaced flows, or NaN when no sign change makes one exist."""
    flows = np.asarray(cash_flows, dtype=float)
    if flows.size < 2 or not (np.any(flows > 0) and np.any(flows < 0)):
        return float("nan")
    f_low, f_high = npv(low, flows), npv(high, flows)
    if f_low * f_high > 0:
        return float("nan")
    return float(brentq(lambda r: npv(r, flows), low, high, xtol=1e-12, maxiter=200))


def xirr(dates: ArrayLike, cash_flows: ArrayLike) -> float:
    """IRR for flows on actual dates, using the 365-day convention."""
    days = np.asarray([(d - np.asarray(dates)[0]).astype("timedelta64[D]").astype(float) for d in np.asarray(dates)])
    flows = np.asarray(cash_flows, dtype=float)
    if not (np.any(flows > 0) and np.any(flows < 0)):
        return float("nan")

    def value(rate: float) -> float:
        return float(np.sum(flows / (1.0 + rate) ** (days / 365.0)))

    if value(-0.999) * value(10.0) > 0:
        return float("nan")
    return float(brentq(value, -0.999, 10.0, xtol=1e-12, maxiter=200))


def moic(cash_flows: ArrayLike) -> float:
    flows = np.asarray(cash_flows, dtype=float)
    invested = -flows[flows < 0].sum()
    return float(flows[flows > 0].sum() / invested) if invested > 0 else float("nan")


def carried_interest(
    contributions: ArrayLike, distributions: ArrayLike, waterfall: WaterfallAssumptions, hold_years: float
) -> tuple[float, float]:
    """Carry on a whole-of-deal (European) waterfall, returning (carry, hurdle profit).

    Investors first receive their capital plus a preferred return. With a full catch-up
    the general partner then takes 100% of the next dollars until it holds its carry
    share of total profit, after which profits split at the carry rate. Beyond the
    catch-up the algebra collapses to carry x profit, which is a useful sanity check.
    """
    contributions = np.asarray(contributions, dtype=float)
    distributions = np.asarray(distributions, dtype=float)
    periods = np.arange(contributions.size)
    compounded = contributions * (1.0 + waterfall.preferred_return) ** (hold_years - periods)
    hurdle_profit = float(compounded.sum() - contributions.sum())
    profit = float(distributions.sum() - contributions.sum())
    if profit <= hurdle_profit:
        return 0.0, hurdle_profit
    if not waterfall.catch_up:
        return waterfall.carry * (profit - hurdle_profit), hurdle_profit
    catch_up_amount = waterfall.carry * hurdle_profit / (1.0 - waterfall.carry)
    if profit - hurdle_profit <= catch_up_amount:
        return profit - hurdle_profit, hurdle_profit
    return waterfall.carry * profit, hurdle_profit


@dataclass(frozen=True)
class EquityReturns:
    entry_equity: float
    sponsor_equity: float
    exit_equity_value: float
    management_incentive: float
    dividends: float
    gross_irr: float
    gross_moic: float
    net_irr: float
    net_moic: float
    carried_interest: float
    management_fees: float
    hurdle_profit: float
    sponsor_gross_flows: np.ndarray
    sponsor_net_flows: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "entry_equity": self.entry_equity,
            "sponsor_equity": self.sponsor_equity,
            "exit_equity_value": self.exit_equity_value,
            "management_incentive": self.management_incentive,
            "dividends": self.dividends,
            "gross_moic": self.gross_moic,
            "gross_irr": self.gross_irr,
            "net_moic": self.net_moic,
            "net_irr": self.net_irr,
            "carried_interest": self.carried_interest,
            "management_fees": self.management_fees,
        }


def equity_returns(result: LBOResult) -> EquityReturns:
    """Split exit proceeds between management, the sponsor and its investors."""
    deal = result.deal
    waterfall = deal.waterfall
    entry_equity = result.sources_uses.equity_cheque
    equity_value = result.exit.equity_value

    value_creation = max(equity_value - entry_equity, 0.0)
    management_incentive = waterfall.management_pool_pct * value_creation
    investor_exit = equity_value - management_incentive

    share = result.sponsor_share
    gross = result.equity_cash_flows.copy() * share
    gross[-1] = (result.equity_cash_flows[-1] - result.exit.equity_value) * share + investor_exit * share
    dividends = float(result.schedule["dividend"].sum()) * share

    contributions = np.where(gross < 0, -gross, 0.0)
    distributions = np.where(gross > 0, gross, 0.0)
    fees = waterfall.management_fee_pct * result.sources_uses.sponsor_equity
    fee_flows = np.zeros_like(gross)
    if fees > 0:
        fee_flows[:-1] = fees  # charged at the start of each year of the hold period
        fee_flows[0] = fees

    net_contributions = contributions + fee_flows
    carry, hurdle_profit = carried_interest(net_contributions, distributions, waterfall, float(deal.exit.hold_years))
    net_flows = distributions - net_contributions
    net_flows[-1] -= carry

    return EquityReturns(
        entry_equity=entry_equity,
        sponsor_equity=result.sources_uses.sponsor_equity,
        exit_equity_value=equity_value,
        management_incentive=management_incentive,
        dividends=dividends,
        gross_irr=irr(gross),
        gross_moic=moic(gross),
        net_irr=irr(net_flows),
        net_moic=moic(net_flows),
        carried_interest=carry,
        management_fees=float(fee_flows.sum()),
        hurdle_profit=hurdle_profit,
        sponsor_gross_flows=gross,
        sponsor_net_flows=net_flows,
    )


def kaplan_schoar_pme(cash_flows: ArrayLike, index_levels: ArrayLike) -> float:
    """PME = PV of distributions / PV of contributions, discounted by the index itself.

    Above 1.0 the deal beat the index on the same cash flow timing; below 1.0 it did not.
    """
    flows = np.asarray(cash_flows, dtype=float)
    index = np.asarray(index_levels, dtype=float)
    if flows.size != index.size:
        raise ValueError("cash flows and index levels must have the same length")
    discounted = flows / (index / index[0])
    inflows = discounted[discounted > 0].sum()
    outflows = -discounted[discounted < 0].sum()
    return float(inflows / outflows) if outflows > 0 else float("nan")


def direct_alpha(cash_flows: ArrayLike, index_levels: ArrayLike) -> float:
    """Annualised excess return over the index: the IRR of index-compounded flows."""
    flows = np.asarray(cash_flows, dtype=float)
    index = np.asarray(index_levels, dtype=float)
    scaled = flows * (index[-1] / index)
    return irr(scaled)
