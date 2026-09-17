"""Validation suite: accounting identities and return mathematics, checked independently.

A financial model is only as trustworthy as the identities it cannot violate. Each check
below either recomputes a number a different way, or compares it with a closed-form
result, and reports the residual.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .assumptions import (
    CovenantPackage,
    Deal,
    DebtTranche,
    ExitAssumptions,
    FinancingAssumptions,
    OperatingAssumptions,
    WaterfallAssumptions,
)
from .examples import apex_rollup, helios_carveout
from .model import run_lbo
from .montecarlo import MonteCarloConfig, run_monte_carlo
from .returns import carried_interest, direct_alpha, equity_returns, irr, kaplan_schoar_pme, xirr
from .screener import CompanyFinancials, ScreenAssumptions, build_deal, max_affordable_premium
from .sensitivity import shift_deal


@dataclass
class Check:
    area: str
    check: str
    reference: str
    result: str
    error: str
    passed: bool
    seconds: float


def _unlevered_deal() -> Deal:
    """All-equity, fee-free, tax-free deal whose return can be computed by hand."""
    return Deal(
        name="All equity control",
        operating=OperatingAssumptions(
            entry_revenue=100.0,
            ebitda_margin=(0.20, 0.20, 0.20, 0.20),
            revenue_growth=(0.0, 0.0, 0.0),
            capex_pct_revenue=0.0,
            depreciation_pct_revenue=0.0,
            tax_rate=0.0,
            opening_ppe=0.0,
        ),
        financing=FinancingAssumptions(
            tranches=(DebtTranche(name="Term loan", amount=0.0, spread=0.05, sweep_priority=1),),
            base_rate=(0.0,),
            transaction_fee_pct=0.0,
            financing_fee_pct=0.0,
            exit_fee_pct=0.0,
            minimum_cash=0.0,
        ),
        exit=ExitAssumptions(hold_years=3, entry_multiple=8.0, exit_multiple=8.0),
        waterfall=WaterfallAssumptions(management_pool_pct=0.0),
        covenants=CovenantPackage(),
    )


def _synthetic_company() -> CompanyFinancials:
    return CompanyFinancials(
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


def run_validation() -> pd.DataFrame:
    checks: list[Check] = []

    def add(area: str, name: str, func: Callable[[], tuple[str, str, str, bool]]) -> None:
        start = time.perf_counter()
        reference, result, error, passed = func()
        checks.append(Check(area, name, reference, result, error, passed, time.perf_counter() - start))

    helios, apex = helios_carveout(), apex_rollup()
    helios_result, apex_result = run_lbo(helios), run_lbo(apex)

    def sources_equal_uses():
        worst = 0.0
        for result in (helios_result, apex_result):
            worst = max(worst, abs(result.sources_uses.total_sources - result.sources_uses.total_uses))
        return "sources - uses = 0", f"{worst:.2e}", f"{worst:.2e}", worst < 1e-9

    def balance_sheet():
        worst = max(float(r.diagnostics["max_balance_check"]) for r in (helios_result, apex_result))
        return "assets - liabilities - equity = 0", f"{worst:.2e}", f"{worst:.2e}", worst < 1e-8

    def cash_flow_statement():
        worst = max(float(r.diagnostics["max_cash_flow_check"]) for r in (helios_result, apex_result))
        return "cash movement ties to the statement", f"{worst:.2e}", f"{worst:.2e}", worst < 1e-8

    def interest_recomputed():
        """Recompute interest independently from the average of opening and closing balances."""
        worst = 0.0
        for deal, result in ((helios, helios_result), (apex, apex_result)):
            schedule = result.schedule
            opening = {t.name: (0.0 if t.is_revolver else t.opening_balance(deal.entry_ebitda)) for t in deal.financing.tranches}
            for _, row in schedule.iterrows():
                year = int(row["year"])
                cash_expected = 0.0
                for tranche in deal.financing.tranches:
                    closing = float(row[f"debt_{tranche.name}"])
                    average = 0.5 * (opening[tranche.name] + closing)
                    rate = tranche.spread + (deal.financing.rate_for(year) if tranche.floating else 0.0)
                    if not tranche.pik:
                        cash_expected += rate * average
                    if tranche.is_revolver:
                        cash_expected += tranche.undrawn_fee * max(tranche.commitment(deal.entry_ebitda) - average, 0.0)
                    opening[tranche.name] = closing
                worst = max(worst, abs(cash_expected - float(row["cash_interest"])))
        return "interest on average balances", f"{worst:.2e}", f"{worst:.2e}", worst < 1e-8

    def interest_closed_form():
        """One tranche, full sweep, no fees or tax: I = r(2B - EBITDA) / (2 - r)."""
        deal = Deal(
            name="Closed form",
            operating=OperatingAssumptions(
                entry_revenue=100.0,
                ebitda_margin=(0.25, 0.25),
                revenue_growth=(0.0,),
                days_sales_outstanding=0.0,
                days_inventory=0.0,
                days_payable=0.0,
                capex_pct_revenue=0.0,
                depreciation_pct_revenue=0.0,
                tax_rate=0.0,
            ),
            financing=FinancingAssumptions(
                tranches=(DebtTranche(name="Term loan", turns=3.0, spread=0.06, sweep_priority=1),),
                base_rate=(0.0,),
                transaction_fee_pct=0.0,
                financing_fee_pct=0.0,
                exit_fee_pct=0.0,
                minimum_cash=0.0,
            ),
            exit=ExitAssumptions(hold_years=1, entry_multiple=8.0, exit_multiple=8.0),
            waterfall=WaterfallAssumptions(management_pool_pct=0.0),
            covenants=CovenantPackage(),
        )
        row = run_lbo(deal).schedule.iloc[0]
        rate, opening, ebitda = 0.06, 75.0, 25.0
        expected = rate * (2 * opening - ebitda) / (2 - rate)
        error = abs(float(row["cash_interest"]) - expected)
        return f"{expected:.10f}", f"{float(row['cash_interest']):.10f}", f"{error:.1e}", error < 1e-10

    def bridge_identity():
        from .bridge import value_bridge

        worst = max(abs(value_bridge(r).reconciliation_error) for r in (helios_result, apex_result))
        return "value bridge sums to equity value change", f"{worst:.2e}", f"{worst:.2e}", worst < 1e-6

    def irr_moic_consistency():
        returns = equity_returns(helios_result)
        implied = returns.gross_moic ** (1.0 / helios.exit.hold_years) - 1.0
        error = abs(implied - returns.gross_irr)
        return "MOIC^(1/n) - 1 with no interim flows", f"{100 * returns.gross_irr:.4f}%", f"{error:.2e}", error < 1e-10

    def irr_closed_form():
        value = irr([-100.0, 0.0, 0.0, 0.0, 0.0, 200.0])
        expected = 2**0.2 - 1
        return (
            f"{100 * expected:.4f}% (doubling in five years)",
            f"{100 * value:.4f}%",
            f"{abs(value - expected):.1e}",
            abs(value - expected) < 1e-9,
        )

    def xirr_matches_irr():
        dates = np.array(["2026-01-01", "2027-01-01", "2028-01-01"], dtype="datetime64[D]")
        flows = [-100.0, 40.0, 80.0]
        annual, dated = irr(flows), xirr(dates, flows)
        return f"IRR {100 * annual:.4f}%", f"XIRR {100 * dated:.4f}%", f"{abs(annual - dated):.1e}", abs(annual - dated) < 2e-3

    def carry_algebra():
        contributions, distributions = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 5 + [250.0]
        carry, hurdle = carried_interest(contributions, distributions, helios.waterfall, 5.0)
        expected = 0.20 * 150.0
        below, _ = carried_interest(contributions, [0.0] * 5 + [120.0], helios.waterfall, 5.0)
        ok = abs(carry - expected) < 1e-9 and below == 0.0 and abs(hurdle - (100 * 1.08**5 - 100)) < 1e-9
        return (
            "carry = 20% of profit beyond the catch-up; 0 below the hurdle",
            f"{carry:.4f} and {below:.1f}",
            f"{abs(carry - expected):.1e}",
            ok,
        )

    def net_below_gross():
        worst_gap = min(equity_returns(r).gross_irr - equity_returns(r).net_irr for r in (helios_result, apex_result))
        return "net IRR <= gross IRR", f"minimum gap {100 * worst_gap:.2f} points", f"{worst_gap:.4f}", worst_gap >= 0

    def pme_unit():
        """An index that compounds at the deal's own IRR must give a PME of exactly one."""
        returns = equity_returns(helios_result)
        index = np.array([(1.0 + returns.gross_irr) ** year for year in range(helios.exit.hold_years + 1)])
        pme = kaplan_schoar_pme(returns.sponsor_gross_flows, index)
        alpha = direct_alpha(returns.sponsor_gross_flows, index)
        ok = abs(pme - 1.0) < 1e-9 and abs(alpha) < 1e-9
        return "PME = 1.00 and direct alpha = 0", f"{pme:.10f} and {alpha:.2e}", f"{abs(pme - 1.0):.1e}", ok

    def unlevered_return():
        deal = _unlevered_deal()
        result = run_lbo(deal)
        returns = equity_returns(result)
        cash = float(result.schedule["levered_fcf"].sum())
        expected_value = deal.exit.exit_multiple * float(result.schedule["ebitda"].iloc[-1]) + cash
        expected_irr = (expected_value / result.sources_uses.equity_cheque) ** (1 / 3) - 1
        error = abs(returns.gross_irr - expected_irr)
        return f"hand calculation {100 * expected_irr:.4f}%", f"{100 * returns.gross_irr:.4f}%", f"{error:.1e}", error < 1e-9

    def leverage_monotone():
        irrs = [equity_returns(run_lbo(shift_deal(helios, "leverage_turns", turns))).gross_irr for turns in (2.0, 3.5, 5.0, 6.0)]
        increasing = all(b > a for a, b in zip(irrs, irrs[1:], strict=False))
        return (
            "IRR rises with leverage while the spread is positive",
            ", ".join(f"{100 * v:.1f}%" for v in irrs),
            "monotone" if increasing else "not monotone",
            increasing,
        )

    def tax_shield():
        low = run_lbo(shift_deal(helios, "leverage_turns", 2.0)).schedule["taxes"].sum()
        high = run_lbo(shift_deal(helios, "leverage_turns", 6.0)).schedule["taxes"].sum()
        return "more debt pays less tax", f"{low:.1f} versus {high:.1f}", f"{low - high:.1f} of shield", high < low

    def covenant_detection():
        stressed = shift_deal(helios, "leverage_turns", 8.0)
        breached = bool(run_lbo(stressed).diagnostics["covenant_breached"])
        clean = bool(helios_result.diagnostics["covenant_breached"])
        return (
            "8.0x breaches, base case does not",
            f"{breached} and {clean}",
            "detected" if breached and not clean else "missed",
            breached and not clean,
        )

    def pik_accretes():
        schedule = apex_result.schedule
        pik = float(schedule["pik_interest"].sum())
        balances = schedule["debt_PIK note"]
        grew = bool(balances.iloc[-1] > balances.iloc[0])
        return "PIK interest accrues into the balance", f"{pik:.1f} accreted", "increasing" if grew else "flat", pik > 0 and grew

    def recap_flow():
        schedule = apex_result.schedule
        dividend = float(schedule["dividend"].sum())
        flows = apex_result.equity_cash_flows
        interim = float(flows[1:-1].sum())
        return (
            "recap dividend reaches the equity flows",
            f"{dividend:.1f} paid, {interim:.1f} received",
            f"{abs(dividend - interim):.1e}",
            abs(dividend - interim) < 1e-9,
        )

    def monte_carlo_reproducible():
        first = run_monte_carlo(helios, MonteCarloConfig(n_trials=120, seed=5)).trials["gross_irr"]
        second = run_monte_carlo(helios, MonteCarloConfig(n_trials=120, seed=5)).trials["gross_irr"]
        equal = bool(np.allclose(first.to_numpy(), second.to_numpy(), equal_nan=True))
        return "same seed gives the same trials", f"{len(first)} trials", "identical" if equal else "different", equal

    def screen_round_trip():
        company = _synthetic_company()
        assumptions = ScreenAssumptions()
        premium, _, _ = max_affordable_premium(company, assumptions, 0.20)
        achieved = equity_returns(run_lbo(build_deal(company, premium, assumptions))).gross_irr
        error = abs(achieved - 0.20)
        return (
            "solved premium reproduces a 20% IRR",
            f"premium {100 * premium:.2f}% gives {100 * achieved:.4f}%",
            f"{error:.1e}",
            error < 1e-5,
        )

    def yaml_matches_code():
        from pathlib import Path

        folder = Path(__file__).resolve().parents[2] / "deals"
        pairs = [("helios_carveout.yaml", helios), ("apex_rollup.yaml", apex)]
        if not folder.exists():
            return "deal files bundled with the repository", "skipped: deals folder not found", "n/a", True
        mismatches = []
        for filename, expected in pairs:
            loaded = Deal.from_yaml(folder / filename)
            if replace(loaded, notes="") != replace(expected, notes=""):
                mismatches.append(filename)
        return (
            "YAML deals equal their Python equivalents",
            f"{len(pairs)} files compared",
            "; ".join(mismatches) or "identical",
            not mismatches,
        )

    add("Accounting", "Sources equal uses", sources_equal_uses)
    add("Accounting", "Balance sheet balances every year", balance_sheet)
    add("Accounting", "Cash flow statement ties to cash", cash_flow_statement)
    add("Accounting", "Interest recomputed from average balances", interest_recomputed)
    add("Accounting", "Circularity solver against a closed form", interest_closed_form)
    add("Attribution", "Value bridge is an exact identity", bridge_identity)
    add("Returns", "IRR consistent with MOIC", irr_moic_consistency)
    add("Returns", "IRR against a closed form", irr_closed_form)
    add("Returns", "XIRR agrees with annual IRR", xirr_matches_irr)
    add("Returns", "Carried interest algebra", carry_algebra)
    add("Returns", "Net returns never exceed gross", net_below_gross)
    add("Returns", "PME of an index at the deal's own IRR", pme_unit)
    add("Model", "All-equity deal against a hand calculation", unlevered_return)
    add("Model", "Leverage raises the IRR", leverage_monotone)
    add("Model", "Interest tax shield", tax_shield)
    add("Model", "Covenant breach detection", covenant_detection)
    add("Model", "PIK interest accretion", pik_accretes)
    add("Model", "Dividend recapitalisation flows", recap_flow)
    add("Risk", "Monte Carlo reproducibility", monte_carlo_reproducible)
    add("Screen", "Premium solver round trip", screen_round_trip)
    add("Inputs", "YAML deals match the code examples", yaml_matches_code)

    return pd.DataFrame([check.__dict__ for check in checks])


def to_markdown(results: pd.DataFrame) -> str:
    cleaned = results.assign(
        **{column: results[column].astype(str).str.replace("|", "/") for column in ("reference", "result", "error")}
    )
    lines = ["| Area | Check | Reference | Result | Residual | Status |", "|---|---|---|---|---|---|"]
    for row in cleaned.itertuples():
        lines.append(
            f"| {row.area} | {row.check} | {row.reference} | {row.result} | {row.error} | {'PASS' if row.passed else 'FAIL'} |"
        )
    return "\n".join(lines)
