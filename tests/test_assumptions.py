from dataclasses import replace
from pathlib import Path

import pytest

from lbolab.assumptions import (
    CovenantPackage,
    Deal,
    DebtTranche,
    DividendRecap,
    ExitAssumptions,
    FinancingAssumptions,
    OperatingAssumptions,
    WaterfallAssumptions,
)

DEALS = Path(__file__).resolve().parents[1] / "deals"


def test_entry_ebitda_and_enterprise_value(helios):
    assert helios.entry_ebitda == pytest.approx(420.0 * 0.180)
    assert helios.entry_enterprise_value == pytest.approx(9.5 * 420.0 * 0.180)
    assert helios.total_debt_turns() == pytest.approx(5.0)


def test_revenue_compounds(helios):
    assert helios.operating.revenue(0) == pytest.approx(420.0)
    assert helios.operating.revenue(2) == pytest.approx(420.0 * 1.08 * 1.07)


def test_working_capital_uses_days_and_cost_of_sales():
    operating = OperatingAssumptions(
        entry_revenue=365.0,
        ebitda_margin=(0.2, 0.2),
        revenue_growth=(0.0,),
        gross_margin=0.5,
        days_sales_outstanding=30.0,
        days_inventory=20.0,
        days_payable=10.0,
    )
    # receivables 30/365 x 365 = 30; inventory 20/365 x 182.5 = 10; payables 10/365 x 182.5 = 5
    assert operating.net_working_capital(365.0) == pytest.approx(30.0 + 10.0 - 5.0)


def test_margin_path_must_cover_year_zero_and_each_year():
    with pytest.raises(ValueError, match="one more entry"):
        OperatingAssumptions(entry_revenue=100.0, ebitda_margin=(0.2, 0.2), revenue_growth=(0.05, 0.05))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"turns": 1.0, "amount": 100.0}, "exactly one"),
        ({}, "exactly one"),
        ({"turns": 1.0, "pik": True, "sweep_priority": 1}, "does not participate"),
    ],
)
def test_tranche_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DebtTranche(name="Bad", **kwargs)


def test_tranche_sizing_in_turns_and_absolute_amounts():
    ebitda = 80.0
    assert DebtTranche(name="A", turns=4.5).opening_balance(ebitda) == pytest.approx(360.0)
    assert DebtTranche(name="B", amount=125.0).opening_balance(ebitda) == 125.0
    revolver = DebtTranche(name="RCF", turns=0.0, commitment_turns=0.75, is_revolver=True)
    assert revolver.opening_balance(ebitda) == 0.0
    assert revolver.commitment(ebitda) == pytest.approx(60.0)


def test_base_rate_curve_holds_the_last_value_flat():
    financing = FinancingAssumptions(tranches=(DebtTranche(name="A", turns=1.0),), base_rate=(0.05, 0.04))
    assert financing.rate_for(1) == 0.05
    assert financing.rate_for(2) == 0.04
    assert financing.rate_for(9) == 0.04


def test_covenant_schedule_steps_down_then_holds():
    covenants = CovenantPackage(max_total_leverage=(6.0, 5.0))
    assert covenants.leverage_limit(1) == 6.0
    assert covenants.leverage_limit(2) == 5.0
    assert covenants.leverage_limit(7) == 5.0
    assert CovenantPackage().leverage_limit(1) is None


def test_deal_rejects_short_forecasts_and_unknown_recap_tranche(helios):
    with pytest.raises(ValueError, match="cover every year"):
        replace(helios, exit=replace(helios.exit, hold_years=9))
    with pytest.raises(ValueError, match="unknown tranche"):
        replace(
            helios, exit=replace(helios.exit, dividend_recaps=(DividendRecap(year=2, incremental_turns=1.0, tranche="Nope"),))
        )
    with pytest.raises(ValueError, match="outside the hold period"):
        replace(
            helios,
            exit=replace(helios.exit, dividend_recaps=(DividendRecap(year=9, incremental_turns=1.0, tranche="Term Loan B"),)),
        )


def test_waterfall_and_exit_validation():
    with pytest.raises(ValueError, match="carry"):
        WaterfallAssumptions(carry=1.2)
    with pytest.raises(ValueError, match="hold_years"):
        ExitAssumptions(hold_years=0, entry_multiple=8.0, exit_multiple=8.0)
    with pytest.raises(ValueError, match="multiples"):
        ExitAssumptions(hold_years=3, entry_multiple=0.0, exit_multiple=8.0)


def test_with_overrides_replaces_nested_fields(helios):
    tweaked = helios.with_overrides(exit={"exit_multiple": 11.0}, name="Tweaked")
    assert tweaked.exit.exit_multiple == 11.0
    assert tweaked.exit.entry_multiple == helios.exit.entry_multiple
    assert tweaked.name == "Tweaked"
    assert helios.exit.exit_multiple == 9.5  # the original is untouched


@pytest.mark.parametrize("filename", ["helios_carveout.yaml", "apex_rollup.yaml"])
def test_yaml_deals_load_and_match_the_code_examples(filename):
    from lbolab.examples import apex_rollup, helios_carveout

    loaded = Deal.from_yaml(DEALS / filename)
    expected = helios_carveout() if "helios" in filename else apex_rollup()
    assert replace(loaded, notes="") == replace(expected, notes="")
    assert loaded.notes.split()[:6] == expected.notes.split()[:6]
