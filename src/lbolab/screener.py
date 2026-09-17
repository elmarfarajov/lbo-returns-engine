"""Public-to-private screen: what could a sponsor pay for a listed company?

For each company the screen builds an LBO at a standard structure and solves for the
**maximum premium to the current share price** that still clears a target IRR, assuming
the sponsor exits at today's trading multiple. A high answer means the market is
pricing the business below what leverage and cash generation can support; a negative
answer means the shares already price in more than a buyout can pay.

Two credit tests come first, because a deal that cannot be financed is not a deal:
interest coverage in year one and the leverage ceiling. Leverage is cut back until the
coverage test passes, and the company is skipped if it never does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .assumptions import (
    CovenantPackage,
    Deal,
    DebtTranche,
    ExitAssumptions,
    FinancingAssumptions,
    OperatingAssumptions,
    WaterfallAssumptions,
)
from .model import run_lbo
from .returns import equity_returns


@dataclass(frozen=True)
class CompanyFinancials:
    ticker: str
    name: str
    sector: str
    price: float
    market_cap: float
    net_debt: float
    revenue: float
    ebitda: float
    capex: float
    depreciation: float
    revenue_cagr: float
    currency: str = "USD m"

    @property
    def enterprise_value(self) -> float:
        return self.market_cap + self.net_debt

    @property
    def ev_ebitda(self) -> float:
        return self.enterprise_value / self.ebitda if self.ebitda > 0 else np.nan

    @property
    def net_leverage(self) -> float:
        return self.net_debt / self.ebitda if self.ebitda > 0 else np.nan

    @property
    def ebitda_margin(self) -> float:
        return self.ebitda / self.revenue if self.revenue > 0 else np.nan


@dataclass(frozen=True)
class ScreenAssumptions:
    hold_years: int = 5
    leverage_turns: float = 5.0
    base_rate: float = 0.038
    spread: float = 0.045
    amortisation_pct: float = 0.01
    revolver_turns: float = 0.5
    transaction_fee_pct: float = 0.02
    financing_fee_pct: float = 0.02
    exit_fee_pct: float = 0.01
    cash_sweep_pct: float = 0.9
    tax_rate: float = 0.25
    growth_cap: float = 0.12
    margin_improvement: float = 0.01
    """Margin the sponsor expects to add over the hold period, in percentage points."""
    min_interest_coverage: float = 2.0
    max_leverage_covenant: float = 6.5


def build_deal(company: CompanyFinancials, premium: float, assumptions: ScreenAssumptions) -> Deal:
    """LBO of a listed company bought at ``premium`` to the market price, exited at today's multiple."""
    offer_equity = company.market_cap * (1.0 + premium)
    entry_ev = offer_equity + company.net_debt
    entry_multiple = entry_ev / company.ebitda
    growth = float(np.clip(company.revenue_cagr, 0.0, assumptions.growth_cap))
    years = assumptions.hold_years
    margin = company.ebitda_margin
    margins = tuple(margin + assumptions.margin_improvement * i / years for i in range(years + 1))

    operating = OperatingAssumptions(
        entry_revenue=company.revenue,
        ebitda_margin=margins,
        revenue_growth=tuple([growth] * years),
        capex_pct_revenue=min(max(company.capex / company.revenue, 0.01), 0.15),
        depreciation_pct_revenue=min(max(company.depreciation / company.revenue, 0.01), 0.15),
        tax_rate=assumptions.tax_rate,
        opening_ppe=company.revenue * 0.4,
    )
    tranches = (
        DebtTranche(
            name="Revolver",
            turns=0.0,
            commitment_turns=assumptions.revolver_turns,
            spread=assumptions.spread - 0.01,
            is_revolver=True,
            sweep_priority=1,
            term_years=6,
        ),
        DebtTranche(
            name="Term Loan B",
            turns=assumptions.leverage_turns,
            spread=assumptions.spread,
            amortisation_pct=assumptions.amortisation_pct,
            sweep_priority=2,
            term_years=7,
        ),
    )
    financing = FinancingAssumptions(
        tranches=tranches,
        base_rate=(assumptions.base_rate,),
        transaction_fee_pct=assumptions.transaction_fee_pct,
        financing_fee_pct=assumptions.financing_fee_pct,
        exit_fee_pct=assumptions.exit_fee_pct,
        minimum_cash=max(0.01 * company.revenue, 1.0),
        cash_sweep_pct=assumptions.cash_sweep_pct,
    )
    return Deal(
        name=f"{company.ticker} take-private at {100 * premium:.0f}% premium",
        operating=operating,
        financing=financing,
        exit=ExitAssumptions(hold_years=years, entry_multiple=entry_multiple, exit_multiple=company.ev_ebitda),
        waterfall=WaterfallAssumptions(management_pool_pct=0.08),
        covenants=CovenantPackage(
            max_total_leverage=(assumptions.max_leverage_covenant,),
            min_interest_coverage=assumptions.min_interest_coverage,
        ),
        currency=company.currency,
    )


def financeable_leverage(company: CompanyFinancials, assumptions: ScreenAssumptions) -> float:
    """Highest leverage, in half-turn steps, that keeps year-one interest coverage above the test."""
    for turns in np.arange(assumptions.leverage_turns, 0.5, -0.5):
        trial = replace(assumptions, leverage_turns=float(turns))
        try:
            result = run_lbo(build_deal(company, 0.0, trial))
        except ValueError:
            continue
        if float(result.schedule["interest_coverage"].iloc[0]) >= assumptions.min_interest_coverage:
            return float(turns)
    return float("nan")


def max_affordable_premium(
    company: CompanyFinancials,
    assumptions: ScreenAssumptions,
    target_irr: float = 0.20,
    bounds: tuple[float, float] = (-0.6, 1.5),
) -> tuple[float, float, float]:
    """Solve for the premium at which gross IRR equals the target.

    A negative answer is meaningful: the shares would have to trade *below* today's price
    for a buyout at this structure to clear the hurdle.

    Returns (premium, entry multiple at that premium, IRR at the current market price).
    """

    def objective(premium: float) -> float:
        try:
            return equity_returns(run_lbo(build_deal(company, premium, assumptions))).gross_irr - target_irr
        except (ValueError, ZeroDivisionError):
            return float("nan")

    at_market = objective(0.0)
    irr_at_market = at_market + target_irr if np.isfinite(at_market) else float("nan")
    grid = np.linspace(bounds[0], bounds[1], 43)
    values = [objective(p) for p in grid]
    for i in range(len(grid) - 1):
        left, right = values[i], values[i + 1]
        if np.isfinite(left) and np.isfinite(right) and left * right <= 0:
            premium = float(brentq(objective, grid[i], grid[i + 1], xtol=1e-6, maxiter=100))
            return premium, build_deal(company, premium, assumptions).exit.entry_multiple, irr_at_market
    return float("nan"), float("nan"), irr_at_market


def fetch_company(ticker: str, ticker_factory=None) -> CompanyFinancials:
    """Pull the fundamentals the screen needs, in millions, from Yahoo Finance."""
    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker
    handle = ticker_factory(ticker)
    info = handle.info or {}
    scale = 1e6

    revenue = _to_float(info.get("totalRevenue")) / scale
    ebitda = _to_float(info.get("ebitda")) / scale
    market_cap = _to_float(info.get("marketCap")) / scale
    net_debt = (_to_float(info.get("totalDebt")) - _to_float(info.get("totalCash"))) / scale
    if min(revenue, ebitda, market_cap) <= 0:
        raise ValueError(f"{ticker}: incomplete fundamentals (revenue, EBITDA or market cap missing)")

    cashflow = getattr(handle, "cashflow", None)
    capex = abs(_statement_row(cashflow, ["Capital Expenditure", "Capital Expenditures"])) / scale
    depreciation = abs(_statement_row(cashflow, ["Depreciation And Amortization", "Depreciation"])) / scale
    income = getattr(handle, "income_stmt", None)
    revenues = _statement_series(income, ["Total Revenue", "TotalRevenue"])
    cagr = _cagr(revenues)

    return CompanyFinancials(
        ticker=ticker.upper(),
        name=str(info.get("shortName") or info.get("longName") or ticker),
        sector=str(info.get("sector") or "n/a"),
        price=_to_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        market_cap=market_cap,
        net_debt=net_debt,
        revenue=revenue,
        ebitda=ebitda,
        capex=capex if capex > 0 else 0.03 * revenue,
        depreciation=depreciation if depreciation > 0 else 0.03 * revenue,
        revenue_cagr=cagr,
        currency=f"{info.get('financialCurrency', 'USD')} m",
    )


def screen_companies(
    tickers: list[str],
    assumptions: ScreenAssumptions | None = None,
    target_irr: float = 0.20,
    ticker_factory=None,
) -> pd.DataFrame:
    assumptions = assumptions or ScreenAssumptions()
    rows = []
    for ticker in tickers:
        try:
            company = fetch_company(ticker, ticker_factory=ticker_factory)
        except Exception as exc:  # a single bad ticker must not stop the screen
            rows.append({"ticker": ticker.upper(), "status": f"skipped: {exc}"})
            continue
        turns = financeable_leverage(company, assumptions)
        if not np.isfinite(turns):
            rows.append(
                {
                    "ticker": company.ticker,
                    "name": company.name,
                    "sector": company.sector,
                    "ev_ebitda": company.ev_ebitda,
                    "net_leverage": company.net_leverage,
                    "status": "not financeable: fails interest coverage at any leverage",
                }
            )
            continue
        tuned = replace(assumptions, leverage_turns=turns)
        premium, entry_multiple, irr_at_market = max_affordable_premium(company, tuned, target_irr)
        result = run_lbo(build_deal(company, max(premium, 0.0) if np.isfinite(premium) else 0.0, tuned))
        rows.append(
            {
                "ticker": company.ticker,
                "name": company.name,
                "sector": company.sector,
                "ev_ebitda": company.ev_ebitda,
                "ebitda_margin": company.ebitda_margin,
                "net_leverage": company.net_leverage,
                "revenue_cagr": company.revenue_cagr,
                "financeable_leverage": turns,
                "irr_at_market_price": irr_at_market,
                "max_premium": premium,
                "entry_multiple_at_premium": entry_multiple,
                "exit_leverage": float(result.diagnostics["exit_leverage"]),
                "covenant_breach": bool(result.diagnostics["covenant_breached"]),
                "status": "ok",
            }
        )
    frame = pd.DataFrame(rows)
    if "max_premium" in frame:
        frame = frame.sort_values("max_premium", ascending=False, na_position="last").reset_index(drop=True)
    return frame


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _statement_row(frame, names: list[str]) -> float:
    if frame is None or getattr(frame, "empty", True):
        return 0.0
    for name in names:
        if name in frame.index:
            series = frame.loc[name].dropna()
            if not series.empty:
                return float(series.iloc[0])
    return 0.0


def _statement_series(frame, names: list[str]) -> np.ndarray:
    if frame is None or getattr(frame, "empty", True):
        return np.array([])
    for name in names:
        if name in frame.index:
            return frame.loc[name].dropna().to_numpy(dtype=float)
    return np.array([])


def _cagr(revenues: np.ndarray) -> float:
    """Yahoo returns the most recent year first, so the series is reversed before compounding."""
    series = np.asarray(revenues, dtype=float)[::-1]
    series = series[series > 0]
    if series.size < 2:
        return 0.03
    years = series.size - 1
    return float((series[-1] / series[0]) ** (1.0 / years) - 1.0)
