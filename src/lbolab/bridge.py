"""Value creation bridge: where did the money actually come from?

Sponsors report an IRR; investors want to know whether it came from running the
business better, from paying down debt, or from selling into a friendlier market. The
decomposition below is an exact algebraic identity, not an approximation:

    exit equity + dividends - entry equity
        = M0 x (Rev_N - Rev_0) x margin_0        (revenue growth)
        + M0 x Rev_N x (margin_N - margin_0)     (margin expansion)
        + (M_N - M0) x EBITDA_N                  (multiple expansion)
        + cumulative levered free cash flow      (cash generation)
        - PIK interest accretion
        - transaction, financing and exit fees

because entry equity = EV_0 - net debt_0 + fees and exit equity = EV_N - net debt_N - fees,
and the movement in net debt is exactly cash generated less PIK accretion less dividends.
The reconciliation error is returned so the identity can be asserted in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .model import LBOResult


@dataclass(frozen=True)
class ValueBridge:
    entry_equity: float
    exit_equity_value: float
    dividends: float
    components: dict[str, float]
    reconciliation_error: float

    @property
    def total_value_created(self) -> float:
        return self.exit_equity_value + self.dividends - self.entry_equity

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"component": list(self.components), "value": list(self.components.values())})
        total = self.total_value_created
        frame["share_of_value"] = frame["value"] / total if total else float("nan")
        return frame

    def waterfall_points(self) -> pd.DataFrame:
        """Cumulative positions for a waterfall chart, from entry equity to exit equity."""
        labels = ["Entry equity", *self.components, "Exit equity + dividends"]
        values = [self.entry_equity, *self.components.values(), 0.0]
        running, bases, heights = self.entry_equity, [], []
        for index, value in enumerate(values):
            if index == 0:
                bases.append(0.0)
                heights.append(self.entry_equity)
            elif index == len(values) - 1:
                bases.append(0.0)
                heights.append(running)
            else:
                bases.append(running if value >= 0 else running + value)
                heights.append(abs(value))
                running += value
        return pd.DataFrame({"label": labels, "base": bases, "height": heights, "value": values})


def value_bridge(result: LBOResult) -> ValueBridge:
    deal = result.deal
    schedule = result.schedule
    operating = deal.operating

    entry_multiple = deal.exit.entry_multiple
    exit_multiple = deal.exit.exit_multiple
    entry_revenue = operating.entry_revenue
    entry_margin = operating.ebitda_margin[0]
    exit_revenue = float(schedule["revenue"].iloc[-1])
    exit_margin = float(schedule["ebitda_margin"].iloc[-1])
    exit_ebitda = float(schedule["ebitda"].iloc[-1])

    components = {
        "Revenue growth": entry_multiple * (exit_revenue - entry_revenue) * entry_margin,
        "Margin expansion": entry_multiple * exit_revenue * (exit_margin - entry_margin),
        "Multiple expansion": (exit_multiple - entry_multiple) * exit_ebitda,
        "Cash generation": float(schedule["levered_fcf"].sum()),
        "PIK accretion": -float(schedule["pik_interest"].sum()),
        "Entry and financing fees": -(result.sources_uses.transaction_fees + result.sources_uses.financing_fees),
        "Exit fees": -result.exit.exit_fees,
    }

    dividends = float(schedule["dividend"].sum())
    total = result.exit.equity_value + dividends - result.sources_uses.equity_cheque
    error = total - sum(components.values())
    return ValueBridge(
        entry_equity=result.sources_uses.equity_cheque,
        exit_equity_value=result.exit.equity_value,
        dividends=dividends,
        components=components,
        reconciliation_error=error,
    )


def attribution_summary(result: LBOResult) -> pd.DataFrame:
    """Value bridge expressed in money, share of value created, and IRR-equivalent terms."""
    bridge = value_bridge(result)
    frame = bridge.to_frame()
    entry = bridge.entry_equity
    hold = deal_hold_years(result)
    frame["irr_equivalent"] = ((entry + frame["value"].cumsum()) / entry) ** (1.0 / hold) - 1.0
    frame["irr_contribution"] = frame["irr_equivalent"].diff().fillna(frame["irr_equivalent"])
    return frame


def deal_hold_years(result: LBOResult) -> float:
    return float(result.deal.exit.hold_years)
