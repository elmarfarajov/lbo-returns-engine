"""The LBO engine: sources and uses, operating forecast, debt schedule and three statements.

Two things make this a real model rather than a spreadsheet sketch.

*Circularity.* Interest accrues on the average debt balance, the balance depends on how
much cash is swept, the sweep depends on cash flow after interest, and cash flow after
interest depends on interest. Spreadsheets break this with an iterative calculation
switch; here each year is solved by fixed-point iteration to a tolerance of 1e-10.

*Accounting identities.* Every year the balance sheet is checked
(assets = liabilities + equity) and the cash flow statement is checked against the
movement in cash. If an identity ever failed the model would be wrong, so the checks
are part of the output rather than a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assumptions import Deal, DebtTranche

_MAX_ITERATIONS = 200
_TOLERANCE = 1e-10


@dataclass(frozen=True)
class SourcesUses:
    enterprise_value: float
    debt_raised: float
    rollover_equity: float
    sponsor_equity: float
    transaction_fees: float
    financing_fees: float
    cash_to_balance_sheet: float
    tranche_draws: dict[str, float]

    @property
    def total_sources(self) -> float:
        return self.debt_raised + self.rollover_equity + self.sponsor_equity

    @property
    def total_uses(self) -> float:
        return self.enterprise_value + self.transaction_fees + self.financing_fees + self.cash_to_balance_sheet

    @property
    def equity_cheque(self) -> float:
        return self.sponsor_equity + self.rollover_equity

    def to_frame(self) -> pd.DataFrame:
        sources = [(f"Debt: {name}", amount) for name, amount in self.tranche_draws.items() if amount > 0]
        sources += [("Rollover equity", self.rollover_equity), ("Sponsor equity", self.sponsor_equity)]
        uses = [
            ("Purchase of enterprise", self.enterprise_value),
            ("Transaction fees", self.transaction_fees),
            ("Financing fees", self.financing_fees),
            ("Cash to balance sheet", self.cash_to_balance_sheet),
        ]
        rows = max(len(sources), len(uses))
        sources += [("", 0.0)] * (rows - len(sources))
        uses += [("", 0.0)] * (rows - len(uses))
        return pd.DataFrame(
            {
                "source": [s[0] for s in sources],
                "source amount": [s[1] for s in sources],
                "use": [u[0] for u in uses],
                "use amount": [u[1] for u in uses],
            }
        )


@dataclass(frozen=True)
class ExitSummary:
    exit_ebitda: float
    exit_multiple: float
    enterprise_value: float
    net_debt: float
    exit_fees: float
    equity_value: float


@dataclass
class LBOResult:
    deal: Deal
    sources_uses: SourcesUses
    schedule: pd.DataFrame
    exit: ExitSummary
    equity_cash_flows: np.ndarray
    """Cash flows to the sponsor and rollover holders combined, year 0 to exit."""
    sponsor_share: float
    diagnostics: dict[str, float | int | bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def entry_equity(self) -> float:
        return self.sources_uses.equity_cheque

    @property
    def covenant_breaches(self) -> pd.DataFrame:
        breached = self.schedule[self.schedule["covenant_breach"] != ""]
        return breached[["year", "leverage", "interest_coverage", "covenant_breach"]]

    @property
    def max_leverage(self) -> float:
        return float(self.schedule["leverage"].max())


def _tranche_rate(tranche: DebtTranche, deal: Deal, year: int) -> float:
    return tranche.spread + (deal.financing.rate_for(year) if tranche.floating else 0.0)


def _weighted_average_term(deal: Deal) -> float:
    weights, terms = [], []
    for tranche in deal.financing.tranches:
        amount = tranche.opening_balance(deal.entry_ebitda)
        if amount > 0 and not tranche.is_revolver:
            weights.append(amount)
            terms.append(tranche.term_years)
    if not weights:
        return float(deal.exit.hold_years)
    return float(np.average(terms, weights=weights))


def build_sources_and_uses(deal: Deal) -> SourcesUses:
    ebitda = deal.entry_ebitda
    enterprise_value = deal.entry_enterprise_value
    draws = {t.name: (0.0 if t.is_revolver else t.opening_balance(ebitda)) for t in deal.financing.tranches}
    debt_raised = sum(draws.values())
    transaction_fees = deal.financing.transaction_fee_pct * enterprise_value
    financing_fees = deal.financing.financing_fee_pct * debt_raised
    cash = deal.financing.minimum_cash
    uses = enterprise_value + transaction_fees + financing_fees + cash
    rollover = deal.waterfall.rollover_equity
    sponsor_equity = uses - debt_raised - rollover
    if sponsor_equity <= 0:
        raise ValueError("Capital structure over-funds the purchase price: sponsor equity would be negative")
    return SourcesUses(
        enterprise_value=enterprise_value,
        debt_raised=debt_raised,
        rollover_equity=rollover,
        sponsor_equity=sponsor_equity,
        transaction_fees=transaction_fees,
        financing_fees=financing_fees,
        cash_to_balance_sheet=cash,
        tranche_draws=draws,
    )


def _covenant_flags(
    deal: Deal, year: int, net_debt: float, ebitda: float, cash_interest: float, capex: float, mandatory: float
) -> tuple[str, float, float, float]:
    leverage = net_debt / ebitda if ebitda > 0 else np.inf
    coverage = ebitda / cash_interest if cash_interest > 0 else np.inf
    fixed_charge = (ebitda - capex) / (cash_interest + mandatory) if (cash_interest + mandatory) > 0 else np.inf
    breaches = []
    limit = deal.covenants.leverage_limit(year)
    if limit is not None and leverage > limit + 1e-12:
        breaches.append(f"leverage {leverage:.2f}x > {limit:.2f}x")
    if deal.covenants.min_interest_coverage is not None and coverage < deal.covenants.min_interest_coverage - 1e-12:
        breaches.append(f"interest cover {coverage:.2f}x < {deal.covenants.min_interest_coverage:.2f}x")
    if deal.covenants.min_fixed_charge_coverage is not None and fixed_charge < deal.covenants.min_fixed_charge_coverage - 1e-12:
        breaches.append(f"fixed charge {fixed_charge:.2f}x < {deal.covenants.min_fixed_charge_coverage:.2f}x")
    return "; ".join(breaches), leverage, coverage, fixed_charge


def run_lbo(deal: Deal) -> LBOResult:
    operating, financing, exit_assumptions = deal.operating, deal.financing, deal.exit
    sources_uses = build_sources_and_uses(deal)
    tranches = {t.name: t for t in financing.tranches}
    balances = dict(sources_uses.tranche_draws)
    originals = dict(balances)
    commitments = {t.name: t.commitment(deal.entry_ebitda) for t in financing.tranches}
    revolvers = [t.name for t in financing.tranches if t.is_revolver]
    sweep_order = sorted(
        (t.name for t in financing.tranches if t.sweep_priority is not None),
        key=lambda name: tranches[name].sweep_priority,
    )
    recaps = {r.year: r for r in exit_assumptions.dividend_recaps}

    cash = financing.minimum_cash
    nwc = operating.net_working_capital(operating.entry_revenue)
    ppe = operating.opening_ppe
    goodwill = sources_uses.enterprise_value - nwc - ppe
    deferred_fees = sources_uses.financing_fees
    fee_amortisation = deferred_fees / max(_weighted_average_term(deal), 1.0)
    equity_book = sources_uses.equity_cheque - sources_uses.transaction_fees
    nol = 0.0

    rows: list[dict[str, float | int | str]] = []
    warnings: list[str] = []
    equity_flows = [-sources_uses.equity_cheque]
    total_iterations = 0

    for year in range(1, exit_assumptions.hold_years + 1):
        revenue = operating.revenue(year)
        ebitda = revenue * operating.ebitda_margin[year]
        depreciation = operating.depreciation_pct_revenue * revenue
        capex = operating.capex_pct_revenue * revenue
        nwc_close = operating.net_working_capital(revenue)
        change_in_nwc = nwc_close - nwc

        recap = recaps.get(year)
        recap_draw = recap.incremental_turns * ebitda if recap else 0.0
        amortisation = {
            name: min(tranche.amortisation_pct * originals[name], balances[name])
            for name, tranche in tranches.items()
            if tranche.amortisation_pct > 0
        }
        mandatory = sum(amortisation.values())
        fee_amortisation_year = min(fee_amortisation, deferred_fees)

        ending = dict(balances)
        iterations = 0
        cash_interest = pik_interest = 0.0
        ebt = taxes = net_income = 0.0
        sweep_used = revolver_draw = 0.0
        cash_close = cash
        repayments: dict[str, float] = {}

        while iterations < _MAX_ITERATIONS:
            iterations += 1
            cash_interest = pik_interest = 0.0
            for name, tranche in tranches.items():
                average = 0.5 * (balances[name] + ending[name])
                interest = _tranche_rate(tranche, deal, year) * average
                if tranche.pik:
                    pik_interest += interest
                else:
                    cash_interest += interest
                if tranche.is_revolver:
                    undrawn = max(commitments[name] - average, 0.0)
                    cash_interest += tranche.undrawn_fee * undrawn

            total_interest = cash_interest + pik_interest + fee_amortisation_year
            ebt = ebitda - depreciation - total_interest
            taxes = operating.tax_rate * max(ebt - nol, 0.0)
            net_income = ebt - taxes
            free_cash_flow = ebitda - taxes - change_in_nwc - cash_interest - capex

            # A recapitalisation draws debt and distributes the same amount, so it is
            # cash neutral at the company; only leverage and the equity flow change.
            available = cash + free_cash_flow - mandatory - financing.minimum_cash
            repayments = {}
            revolver_draw = 0.0
            sweep_used = 0.0
            if available < 0:
                need = -available
                for name in revolvers:
                    capacity = max(commitments[name] - balances[name], 0.0)
                    draw = min(need, capacity)
                    revolver_draw += draw
                    repayments[name] = repayments.get(name, 0.0) - draw
                    need -= draw
                    if need <= 0:
                        break
            else:
                sweep_cash = financing.cash_sweep_pct * available
                for name in sweep_order:
                    if sweep_cash <= 0:
                        break
                    outstanding = balances[name] - amortisation.get(name, 0.0) - repayments.get(name, 0.0)
                    repayment = min(sweep_cash, max(outstanding, 0.0))
                    repayments[name] = repayments.get(name, 0.0) + repayment
                    sweep_cash -= repayment
                    sweep_used += repayment

            new_ending = {}
            for name, tranche in tranches.items():
                balance = balances[name] - repayments.get(name, 0.0) - amortisation.get(name, 0.0)
                if tranche.pik:
                    average = 0.5 * (balances[name] + ending[name])
                    balance += _tranche_rate(tranche, deal, year) * average
                if recap and recap.tranche == name:
                    balance += recap_draw
                new_ending[name] = balance

            cash_close = cash + free_cash_flow - mandatory - sweep_used + revolver_draw
            delta = max(abs(new_ending[name] - ending[name]) for name in new_ending)
            ending = new_ending
            if delta < _TOLERANCE:
                break
        else:
            warnings.append(f"Year {year}: interest circularity did not converge")
        total_iterations += iterations

        if cash_close < financing.minimum_cash - 1e-6:
            warnings.append(
                f"Year {year}: liquidity shortfall, cash of {cash_close:.1f} below the {financing.minimum_cash:.1f} minimum"
            )

        dividend = recap_draw
        # Losses shelter future profits: the carryforward absorbs positive income first
        # and grows by any loss made this year.
        nol = max(nol - max(ebt, 0.0), 0.0) + max(-ebt, 0.0)

        ppe_close = ppe + capex - depreciation
        deferred_close = deferred_fees - fee_amortisation_year
        equity_close = equity_book + net_income - dividend
        total_debt = sum(ending.values())
        net_debt = total_debt - cash_close

        assets = cash_close + nwc_close + ppe_close + goodwill + deferred_close
        liabilities_equity = total_debt + equity_close
        cash_flow_check = cash_close - (
            cash
            + (net_income + depreciation + fee_amortisation_year + pik_interest - change_in_nwc)
            - capex
            + (-mandatory - sweep_used + revolver_draw + recap_draw - dividend)
        )

        breach, leverage, coverage, fixed_charge = _covenant_flags(deal, year, net_debt, ebitda, cash_interest, capex, mandatory)
        equity_flows.append(dividend)

        rows.append(
            {
                "year": year,
                "revenue": revenue,
                "revenue_growth": operating.revenue_growth[year - 1],
                "ebitda": ebitda,
                "ebitda_margin": operating.ebitda_margin[year],
                "depreciation": depreciation,
                "ebit": ebitda - depreciation,
                "cash_interest": cash_interest,
                "pik_interest": pik_interest,
                "fee_amortisation": fee_amortisation_year,
                "ebt": ebt,
                "taxes": taxes,
                "net_income": net_income,
                "capex": capex,
                "change_in_nwc": change_in_nwc,
                "unlevered_fcf": ebitda - taxes - capex - change_in_nwc,
                "levered_fcf": ebitda - taxes - capex - change_in_nwc - cash_interest,
                "mandatory_amortisation": mandatory,
                "cash_sweep": sweep_used,
                "revolver_draw": revolver_draw,
                "recap_draw": recap_draw,
                "dividend": dividend,
                "cash_begin": cash,
                "cash_end": cash_close,
                "total_debt": total_debt,
                "net_debt": net_debt,
                "leverage": leverage,
                "interest_coverage": coverage,
                "fixed_charge_coverage": fixed_charge,
                "covenant_breach": breach,
                "nwc": nwc_close,
                "ppe": ppe_close,
                "goodwill": goodwill,
                "deferred_financing_fees": deferred_close,
                "equity_book": equity_close,
                "balance_check": assets - liabilities_equity,
                "cash_flow_check": cash_flow_check,
                "iterations": iterations,
                **{f"debt_{name}": value for name, value in ending.items()},
            }
        )

        balances = ending
        cash, nwc, ppe = cash_close, nwc_close, ppe_close
        deferred_fees, equity_book = deferred_close, equity_close

    schedule = pd.DataFrame(rows)
    final = schedule.iloc[-1]
    exit_ebitda = float(final["ebitda"])
    exit_ev = exit_assumptions.exit_multiple * exit_ebitda
    exit_fees = financing.exit_fee_pct * exit_ev
    net_debt_exit = float(final["net_debt"])
    equity_value = exit_ev - net_debt_exit - exit_fees
    equity_flows[-1] += equity_value

    summary = ExitSummary(
        exit_ebitda=exit_ebitda,
        exit_multiple=exit_assumptions.exit_multiple,
        enterprise_value=exit_ev,
        net_debt=net_debt_exit,
        exit_fees=exit_fees,
        equity_value=equity_value,
    )
    sponsor_share = sources_uses.sponsor_equity / sources_uses.equity_cheque
    diagnostics = {
        "max_balance_check": float(schedule["balance_check"].abs().max()),
        "max_cash_flow_check": float(schedule["cash_flow_check"].abs().max()),
        "interest_iterations": int(total_iterations),
        "entry_leverage": deal.total_debt_turns(),
        "exit_leverage": net_debt_exit / exit_ebitda if exit_ebitda else np.nan,
        "covenant_breached": bool((schedule["covenant_breach"] != "").any()),
    }
    return LBOResult(
        deal=deal,
        sources_uses=sources_uses,
        schedule=schedule,
        exit=summary,
        equity_cash_flows=np.array(equity_flows, dtype=float),
        sponsor_share=sponsor_share,
        diagnostics=diagnostics,
        warnings=warnings,
    )
