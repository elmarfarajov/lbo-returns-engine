from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lbolab.model import run_lbo
from lbolab.returns import equity_returns
from lbolab.screener import (
    CompanyFinancials,
    ScreenAssumptions,
    _cagr,
    build_deal,
    fetch_company,
    financeable_leverage,
    max_affordable_premium,
    screen_companies,
)

COMPANY = CompanyFinancials(
    ticker="TEST",
    name="Synthetic Co",
    sector="Industrials",
    price=40.0,
    market_cap=1200.0,
    net_debt=300.0,
    revenue=900.0,
    ebitda=180.0,
    capex=27.0,
    depreciation=30.0,
    revenue_cagr=0.04,
)


def test_company_ratios():
    assert COMPANY.enterprise_value == 1500.0
    assert COMPANY.ev_ebitda == pytest.approx(1500.0 / 180.0)
    assert COMPANY.net_leverage == pytest.approx(300.0 / 180.0)
    assert COMPANY.ebitda_margin == pytest.approx(0.2)


def test_build_deal_prices_the_premium_into_the_entry_multiple():
    deal = build_deal(COMPANY, 0.25, ScreenAssumptions())
    expected_ev = 1200.0 * 1.25 + 300.0
    assert deal.exit.entry_multiple == pytest.approx(expected_ev / 180.0)
    assert deal.exit.exit_multiple == pytest.approx(COMPANY.ev_ebitda)
    assert deal.exit.entry_multiple > deal.exit.exit_multiple  # a premium must be earned back
    assert deal.operating.entry_revenue == COMPANY.revenue


def test_build_deal_caps_growth_and_ramps_the_margin():
    fast = replace(COMPANY, revenue_cagr=0.40)
    deal = build_deal(fast, 0.0, ScreenAssumptions(growth_cap=0.12, margin_improvement=0.02, hold_years=4))
    assert deal.operating.revenue_growth == (0.12, 0.12, 0.12, 0.12)
    assert deal.operating.ebitda_margin[0] == pytest.approx(COMPANY.ebitda_margin)
    assert deal.operating.ebitda_margin[-1] == pytest.approx(COMPANY.ebitda_margin + 0.02)


def test_premium_solver_round_trips():
    assumptions = ScreenAssumptions()
    premium, entry_multiple, irr_at_market = max_affordable_premium(COMPANY, assumptions, 0.20)
    achieved = equity_returns(run_lbo(build_deal(COMPANY, premium, assumptions))).gross_irr
    assert achieved == pytest.approx(0.20, abs=1e-5)
    assert entry_multiple == pytest.approx(build_deal(COMPANY, premium, assumptions).exit.entry_multiple)
    assert np.isfinite(irr_at_market)


def test_a_cheaper_company_supports_a_bigger_premium():
    cheap = replace(COMPANY, market_cap=900.0)
    expensive = replace(COMPANY, market_cap=1800.0)
    assumptions = ScreenAssumptions()
    assert max_affordable_premium(cheap, assumptions)[0] > max_affordable_premium(expensive, assumptions)[0]


def test_financeable_leverage_is_cut_back_for_thin_coverage():
    assumptions = ScreenAssumptions(min_interest_coverage=2.0)
    assert financeable_leverage(COMPANY, assumptions) == pytest.approx(5.0)
    costly = replace(assumptions, spread=0.09, base_rate=0.06)
    reduced = financeable_leverage(COMPANY, costly)
    assert reduced < 5.0
    result = run_lbo(build_deal(COMPANY, 0.0, replace(costly, leverage_turns=reduced)))
    assert float(result.schedule["interest_coverage"].iloc[0]) >= 2.0


def test_cagr_handles_yahoo_ordering_and_missing_data():
    assert _cagr(np.array([121.0, 110.0, 100.0])) == pytest.approx(0.10)  # most recent first
    assert _cagr(np.array([100.0])) == pytest.approx(0.03)
    assert _cagr(np.array([])) == pytest.approx(0.03)


class FakeTicker:
    """Minimal stand-in for yfinance.Ticker so the screen can be tested without a network."""

    def __init__(self, symbol: str):
        self.symbol = symbol

    @property
    def info(self):
        if self.symbol == "EMPTY":
            return {"totalRevenue": 0, "ebitda": 0, "marketCap": 0}
        return {
            "totalRevenue": 900e6,
            "ebitda": 180e6,
            "marketCap": 1200e6,
            "totalDebt": 400e6,
            "totalCash": 100e6,
            "currentPrice": 40.0,
            "shortName": f"{self.symbol} Inc",
            "sector": "Industrials",
            "financialCurrency": "USD",
        }

    @property
    def cashflow(self):
        return pd.DataFrame(
            {pd.Timestamp("2025-12-31"): [-27e6, 30e6]},
            index=["Capital Expenditure", "Depreciation And Amortization"],
        )

    @property
    def income_stmt(self):
        return pd.DataFrame(
            {
                pd.Timestamp("2025-12-31"): [900e6],
                pd.Timestamp("2024-12-31"): [850e6],
                pd.Timestamp("2023-12-31"): [800e6],
            },
            index=["Total Revenue"],
        )


def test_fetch_company_parses_a_chain_without_network():
    company = fetch_company("FAKE", ticker_factory=FakeTicker)
    assert company.revenue == pytest.approx(900.0)
    assert company.ebitda == pytest.approx(180.0)
    assert company.net_debt == pytest.approx(300.0)
    assert company.capex == pytest.approx(27.0)
    assert company.depreciation == pytest.approx(30.0)
    assert company.revenue_cagr == pytest.approx((900 / 800) ** 0.5 - 1)
    assert company.name == "FAKE Inc"


def test_fetch_company_rejects_incomplete_fundamentals():
    with pytest.raises(ValueError, match="incomplete fundamentals"):
        fetch_company("EMPTY", ticker_factory=FakeTicker)


def test_screen_reports_one_row_per_ticker_and_skips_failures():
    screen = screen_companies(["FAKE", "EMPTY"], ScreenAssumptions(), 0.20, ticker_factory=FakeTicker)
    assert len(screen) == 2
    assert (screen["status"] == "ok").sum() == 1
    assert screen.loc[screen["ticker"] == "EMPTY", "status"].str.startswith("skipped").all()
    good = screen[screen["status"] == "ok"].iloc[0]
    assert np.isfinite(good["max_premium"])
    assert good["financeable_leverage"] <= 5.0
