import numpy as np
import pytest

from lbolab.assumptions import WaterfallAssumptions
from lbolab.model import run_lbo
from lbolab.returns import (
    carried_interest,
    direct_alpha,
    equity_returns,
    irr,
    kaplan_schoar_pme,
    moic,
    npv,
    xirr,
)

WATERFALL = WaterfallAssumptions(preferred_return=0.08, carry=0.20, catch_up=True)


def test_irr_and_npv_are_consistent():
    flows = [-100.0, 30.0, 40.0, 60.0]
    rate = irr(flows)
    assert npv(rate, flows) == pytest.approx(0.0, abs=1e-10)


def test_irr_closed_forms():
    assert irr([-100.0, 0.0, 0.0, 0.0, 0.0, 200.0]) == pytest.approx(2**0.2 - 1, abs=1e-12)
    assert irr([-100.0, 110.0]) == pytest.approx(0.10, abs=1e-12)
    assert irr([-100.0, 100.0]) == pytest.approx(0.0, abs=1e-12)


def test_irr_returns_nan_without_a_sign_change():
    assert np.isnan(irr([-100.0, -50.0]))
    assert np.isnan(irr([100.0, 50.0]))
    assert np.isnan(irr([-100.0]))


def test_moic_uses_gross_inflows_and_outflows():
    assert moic([-100.0, 20.0, 200.0]) == pytest.approx(2.2)
    assert np.isnan(moic([100.0, 20.0]))


def test_xirr_matches_annual_irr_on_yearly_dates():
    dates = np.array(["2026-06-30", "2027-06-30", "2028-06-29"], dtype="datetime64[D]")
    flows = [-100.0, 25.0, 95.0]
    assert xirr(dates, flows) == pytest.approx(irr(flows), abs=3e-3)


def test_xirr_rewards_earlier_cash():
    early = np.array(["2026-01-01", "2026-07-01"], dtype="datetime64[D]")
    late = np.array(["2026-01-01", "2027-01-01"], dtype="datetime64[D]")
    assert xirr(early, [-100.0, 120.0]) > xirr(late, [-100.0, 120.0])


def test_carry_is_zero_below_the_hurdle():
    carry, hurdle = carried_interest([100.0] + [0.0] * 5, [0.0] * 5 + [130.0], WATERFALL, 5.0)
    assert hurdle == pytest.approx(100 * 1.08**5 - 100)
    assert carry == 0.0


def test_carry_inside_the_catch_up_takes_everything_above_the_hurdle():
    hurdle = 100 * 1.08**5 - 100
    profit = hurdle + 5.0
    carry, _ = carried_interest([100.0] + [0.0] * 5, [0.0] * 5 + [100.0 + profit], WATERFALL, 5.0)
    assert carry == pytest.approx(5.0)


def test_carry_beyond_the_catch_up_is_exactly_the_carry_share():
    carry, _ = carried_interest([100.0] + [0.0] * 5, [0.0] * 5 + [250.0], WATERFALL, 5.0)
    assert carry == pytest.approx(0.20 * 150.0)


def test_without_a_catch_up_carry_applies_only_above_the_hurdle():
    waterfall = WaterfallAssumptions(preferred_return=0.08, carry=0.20, catch_up=False)
    hurdle = 100 * 1.08**5 - 100
    carry, _ = carried_interest([100.0] + [0.0] * 5, [0.0] * 5 + [250.0], waterfall, 5.0)
    assert carry == pytest.approx(0.20 * (150.0 - hurdle))


def test_equity_returns_are_internally_consistent(helios, helios_returns):
    returns = helios_returns
    assert returns.gross_moic == pytest.approx(returns.gross_moic)
    assert returns.gross_irr == pytest.approx(returns.gross_moic ** (1 / helios.exit.hold_years) - 1, abs=1e-12)
    assert returns.net_irr < returns.gross_irr
    assert returns.net_moic < returns.gross_moic
    assert returns.sponsor_gross_flows[0] == pytest.approx(-returns.sponsor_equity)


def test_management_incentive_dilutes_only_the_gain(helios, helios_result, helios_returns):
    expected_pool = helios.waterfall.management_pool_pct * (helios_result.exit.equity_value - helios_returns.entry_equity)
    assert helios_returns.management_incentive == pytest.approx(expected_pool)
    no_pool = equity_returns(run_lbo(helios.with_overrides(waterfall={"management_pool_pct": 0.0})))
    assert no_pool.gross_irr > helios_returns.gross_irr


def test_management_fees_reduce_net_returns(helios, helios_returns):
    with_fees = equity_returns(run_lbo(helios.with_overrides(waterfall={"management_fee_pct": 0.02})))
    assert with_fees.management_fees > 0
    assert with_fees.net_irr < helios_returns.net_irr
    assert with_fees.gross_irr == pytest.approx(helios_returns.gross_irr)


def test_pme_equals_one_when_the_index_earns_the_deal_irr(helios, helios_returns):
    index = np.array([(1 + helios_returns.gross_irr) ** year for year in range(helios.exit.hold_years + 1)])
    assert kaplan_schoar_pme(helios_returns.sponsor_gross_flows, index) == pytest.approx(1.0, abs=1e-9)
    assert direct_alpha(helios_returns.sponsor_gross_flows, index) == pytest.approx(0.0, abs=1e-9)


def test_pme_above_one_when_the_deal_beats_the_index(helios_returns):
    index = np.array([1.05**year for year in range(helios_returns.sponsor_gross_flows.size)])
    assert kaplan_schoar_pme(helios_returns.sponsor_gross_flows, index) > 1.0
    assert direct_alpha(helios_returns.sponsor_gross_flows, index) == pytest.approx(
        (1 + helios_returns.gross_irr) / 1.05 - 1, abs=1e-9
    )


def test_pme_requires_matching_lengths(helios_returns):
    with pytest.raises(ValueError, match="same length"):
        kaplan_schoar_pme(helios_returns.sponsor_gross_flows, [1.0, 1.1])


def test_returns_summary_exposes_every_headline(helios_returns):
    summary = helios_returns.summary()
    for key in ("gross_irr", "net_irr", "gross_moic", "net_moic", "carried_interest", "management_incentive"):
        assert key in summary
