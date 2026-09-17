from dataclasses import replace

import numpy as np
import pytest

from lbolab.assumptions import (
    CovenantPackage,
    Deal,
    DebtTranche,
    ExitAssumptions,
    FinancingAssumptions,
    OperatingAssumptions,
    WaterfallAssumptions,
)
from lbolab.model import build_sources_and_uses, run_lbo


def simple_deal(**overrides) -> Deal:
    """Flat business, one term loan, no fees: everything can be checked by hand."""
    defaults = dict(
        entry_revenue=100.0,
        margin=0.25,
        growth=0.0,
        turns=3.0,
        spread=0.06,
        tax_rate=0.0,
        capex=0.0,
        depreciation=0.0,
        sweep=1.0,
        hold_years=3,
        multiple=8.0,
    )
    defaults.update(overrides)
    years = defaults["hold_years"]
    return Deal(
        name="Simple",
        operating=OperatingAssumptions(
            entry_revenue=defaults["entry_revenue"],
            ebitda_margin=tuple([defaults["margin"]] * (years + 1)),
            revenue_growth=tuple([defaults["growth"]] * years),
            days_sales_outstanding=0.0,
            days_inventory=0.0,
            days_payable=0.0,
            capex_pct_revenue=defaults["capex"],
            depreciation_pct_revenue=defaults["depreciation"],
            tax_rate=defaults["tax_rate"],
        ),
        financing=FinancingAssumptions(
            tranches=(DebtTranche(name="Term loan", turns=defaults["turns"], spread=defaults["spread"], sweep_priority=1),),
            base_rate=(0.0,),
            transaction_fee_pct=0.0,
            financing_fee_pct=0.0,
            exit_fee_pct=0.0,
            minimum_cash=0.0,
            cash_sweep_pct=defaults["sweep"],
        ),
        exit=ExitAssumptions(hold_years=years, entry_multiple=defaults["multiple"], exit_multiple=defaults["multiple"]),
        waterfall=WaterfallAssumptions(management_pool_pct=0.0),
        covenants=CovenantPackage(),
    )


def test_sources_equal_uses(helios_result, apex_result):
    for result in (helios_result, apex_result):
        assert result.sources_uses.total_sources == pytest.approx(result.sources_uses.total_uses)


def test_sources_and_uses_components(helios):
    su = build_sources_and_uses(helios)
    assert su.debt_raised == pytest.approx(5.0 * helios.entry_ebitda)
    assert su.transaction_fees == pytest.approx(0.02 * helios.entry_enterprise_value)
    assert su.financing_fees == pytest.approx(0.02 * su.debt_raised)
    assert su.tranche_draws["Revolver"] == 0.0
    assert su.equity_cheque == pytest.approx(su.total_uses - su.debt_raised)
    frame = su.to_frame()
    assert frame["source amount"].sum() == pytest.approx(frame["use amount"].sum())


def test_over_funded_structure_is_rejected(helios):
    with pytest.raises(ValueError, match="over-funds"):
        run_lbo(helios.with_overrides(exit={"entry_multiple": 3.0}))


@pytest.mark.parametrize("deal_name", ["helios", "apex"])
def test_accounting_identities_hold_every_year(deal_name, helios_result, apex_result):
    result = helios_result if deal_name == "helios" else apex_result
    assert result.schedule["balance_check"].abs().max() < 1e-8
    assert result.schedule["cash_flow_check"].abs().max() < 1e-8


def test_interest_matches_an_independent_recomputation(helios, helios_result):
    schedule = helios_result.schedule
    opening = {t.name: (0.0 if t.is_revolver else t.opening_balance(helios.entry_ebitda)) for t in helios.financing.tranches}
    for _, row in schedule.iterrows():
        expected = 0.0
        for tranche in helios.financing.tranches:
            closing = float(row[f"debt_{tranche.name}"])
            average = 0.5 * (opening[tranche.name] + closing)
            rate = tranche.spread + (helios.financing.rate_for(int(row["year"])) if tranche.floating else 0.0)
            expected += rate * average
            if tranche.is_revolver:
                expected += tranche.undrawn_fee * max(tranche.commitment(helios.entry_ebitda) - average, 0.0)
            opening[tranche.name] = closing
        assert float(row["cash_interest"]) == pytest.approx(expected, abs=1e-9)


def test_circularity_converges_quickly(helios_result):
    assert helios_result.schedule["iterations"].max() < 30
    assert not helios_result.warnings


def test_hand_calculated_first_year():
    deal = simple_deal()
    result = run_lbo(deal)
    row = result.schedule.iloc[0]
    ebitda = 25.0
    # Interest on the average balance solves 0.06 x (75 + 75 - (25 - interest)) / 2 = interest
    interest = 0.06 * (75.0 + 75.0 - 25.0) / 2 / (1 - 0.03)
    assert float(row["ebitda"]) == pytest.approx(ebitda)
    assert float(row["cash_interest"]) == pytest.approx(interest, rel=1e-9)
    assert float(row["levered_fcf"]) == pytest.approx(ebitda - interest)
    assert float(row["total_debt"]) == pytest.approx(75.0 - (ebitda - interest))
    assert float(row["cash_end"]) == pytest.approx(0.0)


def test_cash_sweep_share_controls_repayment():
    full = run_lbo(simple_deal(sweep=1.0)).schedule
    half = run_lbo(simple_deal(sweep=0.5)).schedule
    # Sweeping less leaves more debt outstanding, which costs more interest, so the first
    # year's sweep is slightly below half of the full-sweep case rather than exactly half.
    ratio = half["cash_sweep"].iloc[0] / full["cash_sweep"].iloc[0]
    assert 0.46 < ratio < 0.50
    assert half["cash_end"].iloc[0] > full["cash_end"].iloc[0]
    assert half["total_debt"].iloc[-1] > full["total_debt"].iloc[-1]
    assert half["cash_interest"].sum() > full["cash_interest"].sum()


def test_minimum_cash_is_never_swept():
    deal = replace(simple_deal(), financing=replace(simple_deal().financing, minimum_cash=12.0))
    schedule = run_lbo(deal).schedule
    assert (schedule["cash_end"] >= 12.0 - 1e-9).all()


def test_mandatory_amortisation_is_paid_before_the_sweep():
    deal = simple_deal(sweep=0.0)
    amortising = replace(
        deal,
        financing=replace(
            deal.financing,
            tranches=(DebtTranche(name="Term loan", turns=3.0, spread=0.06, amortisation_pct=0.10, sweep_priority=1),),
        ),
    )
    schedule = run_lbo(amortising).schedule
    assert schedule["mandatory_amortisation"].iloc[0] == pytest.approx(0.10 * 75.0)
    assert schedule["cash_sweep"].max() == 0.0
    assert schedule["total_debt"].iloc[-1] == pytest.approx(75.0 - 3 * 7.5)


def test_revolver_is_drawn_when_cash_flow_turns_negative():
    # Capex above EBITDA guarantees the business cannot fund itself.
    stressed = simple_deal(margin=0.04, capex=0.06, turns=3.0, spread=0.12, sweep=1.0)
    deal = replace(
        stressed,
        financing=replace(
            stressed.financing,
            tranches=(
                DebtTranche(name="Revolver", turns=0.0, commitment_turns=1.0, spread=0.05, is_revolver=True, sweep_priority=1),
                DebtTranche(name="Term loan", turns=3.0, spread=0.12, sweep_priority=2),
            ),
            minimum_cash=5.0,
        ),
    )
    schedule = run_lbo(deal).schedule
    assert schedule["revolver_draw"].sum() > 0
    assert schedule["debt_Revolver"].iloc[-1] > 0


def test_pik_interest_accretes_into_the_balance(apex, apex_result):
    schedule = apex_result.schedule
    assert schedule["pik_interest"].sum() > 0
    balances = schedule["debt_PIK note"].to_numpy()
    assert (np.diff(balances) > 0).all()
    pik = next(t for t in apex.financing.tranches if t.pik)
    assert balances[0] == pytest.approx(
        pik.opening_balance(apex.entry_ebitda) * (1 + pik.spread / (1 - pik.spread / 2)), rel=0.05
    )


def test_dividend_recap_raises_debt_and_pays_equity(apex, apex_result):
    schedule = apex_result.schedule
    recap_year = apex.exit.dividend_recaps[0].year
    row = schedule[schedule["year"] == recap_year].iloc[0]
    assert row["recap_draw"] == pytest.approx(1.0 * row["ebitda"])
    assert row["dividend"] == pytest.approx(row["recap_draw"])
    assert apex_result.equity_cash_flows[recap_year] == pytest.approx(row["dividend"])


def test_losses_shelter_later_taxes():
    """A loss-making first year builds a carryforward that shelters the recovery year."""
    deal = simple_deal(hold_years=2)
    turnaround = replace(
        deal,
        operating=replace(
            deal.operating,
            ebitda_margin=(0.05, 0.05, 0.30),
            depreciation_pct_revenue=0.12,
            tax_rate=0.30,
        ),
    )
    schedule = run_lbo(turnaround).schedule
    loss_year, recovery = schedule.iloc[0], schedule.iloc[1]
    assert loss_year["ebt"] < 0
    assert loss_year["taxes"] == 0.0
    assert recovery["ebt"] > 0
    full_rate_tax = 0.30 * recovery["ebt"]
    assert recovery["taxes"] == pytest.approx(0.30 * (recovery["ebt"] + loss_year["ebt"]))
    assert recovery["taxes"] < full_rate_tax


def test_covenant_breach_is_recorded_and_summarised(helios):
    stressed = run_lbo(helios.with_overrides(covenants={"max_total_leverage": (2.0,)}))
    assert stressed.diagnostics["covenant_breached"]
    assert not stressed.covenant_breaches.empty
    assert "leverage" in stressed.covenant_breaches["covenant_breach"].iloc[0]
    assert stressed.max_leverage == pytest.approx(stressed.schedule["leverage"].max())


def test_exit_bridges_enterprise_value_to_equity(helios, helios_result):
    summary = helios_result.exit
    final = helios_result.schedule.iloc[-1]
    assert summary.enterprise_value == pytest.approx(helios.exit.exit_multiple * final["ebitda"])
    assert summary.equity_value == pytest.approx(summary.enterprise_value - final["net_debt"] - summary.exit_fees)
    assert helios_result.equity_cash_flows[-1] == pytest.approx(summary.equity_value)
    assert helios_result.equity_cash_flows[0] == pytest.approx(-helios_result.sources_uses.equity_cheque)


def test_rollover_equity_dilutes_the_sponsor_share(helios, helios_result):
    assert helios_result.sponsor_share < 1.0
    no_rollover = run_lbo(helios.with_overrides(waterfall={"rollover_equity": 0.0}))
    assert no_rollover.sponsor_share == pytest.approx(1.0)
