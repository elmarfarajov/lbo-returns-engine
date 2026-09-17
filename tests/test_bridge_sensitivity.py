import numpy as np
import pytest

from lbolab.bridge import attribution_summary, value_bridge
from lbolab.model import run_lbo
from lbolab.returns import equity_returns
from lbolab.sensitivity import METRICS, sensitivity_grid, shift_deal, solve_for_target


@pytest.mark.parametrize("deal_name", ["helios", "apex"])
def test_bridge_is_an_exact_identity(deal_name, helios_result, apex_result):
    result = helios_result if deal_name == "helios" else apex_result
    bridge = value_bridge(result)
    assert abs(bridge.reconciliation_error) < 1e-6
    assert sum(bridge.components.values()) == pytest.approx(bridge.total_value_created, abs=1e-6)


def test_bridge_components_match_their_definitions(helios, helios_result):
    bridge = value_bridge(helios_result)
    schedule = helios_result.schedule
    entry_multiple = helios.exit.entry_multiple
    entry_margin = helios.operating.ebitda_margin[0]
    exit_revenue = float(schedule["revenue"].iloc[-1])
    assert bridge.components["Revenue growth"] == pytest.approx(
        entry_multiple * (exit_revenue - helios.operating.entry_revenue) * entry_margin
    )
    assert bridge.components["Multiple expansion"] == pytest.approx(0.0)  # entry and exit multiples are equal
    assert bridge.components["Cash generation"] == pytest.approx(schedule["levered_fcf"].sum())
    assert bridge.components["PIK accretion"] == pytest.approx(0.0)


def test_multiple_expansion_shows_up_when_the_exit_multiple_moves(helios):
    richer = run_lbo(helios.with_overrides(exit={"exit_multiple": 11.0}))
    bridge = value_bridge(richer)
    exit_ebitda = float(richer.schedule["ebitda"].iloc[-1])
    assert bridge.components["Multiple expansion"] == pytest.approx(1.5 * exit_ebitda)
    assert abs(bridge.reconciliation_error) < 1e-6


def test_waterfall_points_start_and_end_at_the_totals(helios_result):
    bridge = value_bridge(helios_result)
    points = bridge.waterfall_points()
    assert points["height"].iloc[0] == pytest.approx(bridge.entry_equity)
    assert points["height"].iloc[-1] == pytest.approx(bridge.exit_equity_value + bridge.dividends)


def test_attribution_reaches_the_whole_equity_irr(helios, helios_result, helios_returns):
    """The bridge explains value for all equity holders, so its IRR equivalent is the
    pre-dilution return; the sponsor's own IRR is lower by the management pool."""
    frame = attribution_summary(helios_result)
    bridge = value_bridge(helios_result)
    whole_equity_irr = ((bridge.exit_equity_value + bridge.dividends) / bridge.entry_equity) ** (1 / helios.exit.hold_years) - 1
    assert frame["irr_equivalent"].iloc[-1] == pytest.approx(whole_equity_irr, abs=1e-9)
    assert frame["irr_contribution"].sum() == pytest.approx(whole_equity_irr, abs=1e-9)
    assert helios_returns.gross_irr < whole_equity_irr


@pytest.mark.parametrize(
    ("axis", "value", "check"),
    [
        ("entry_multiple", 10.0, lambda d: d.exit.entry_multiple == 10.0),
        ("exit_multiple", 8.0, lambda d: d.exit.exit_multiple == 8.0),
        ("multiple_change", 1.0, lambda d: d.exit.exit_multiple == d.exit.entry_multiple + 1.0),
        ("hold_years", 4, lambda d: d.exit.hold_years == 4),
        ("leverage_turns", 6.0, lambda d: abs(d.total_debt_turns() - 6.0) < 1e-9),
        ("revenue_growth_shift", 0.01, lambda d: abs(d.operating.revenue_growth[0] - 0.09) < 1e-12),
        ("margin_shift", 0.01, lambda d: abs(d.operating.ebitda_margin[1] - 0.195) < 1e-12),
        ("base_rate_shift", 0.02, lambda d: abs(d.financing.base_rate[0] - 0.062) < 1e-12),
        ("cash_sweep_pct", 0.5, lambda d: d.financing.cash_sweep_pct == 0.5),
        ("capex_pct_revenue", 0.05, lambda d: d.operating.capex_pct_revenue == 0.05),
    ],
)
def test_shift_deal_moves_the_right_assumption(helios, axis, value, check):
    shifted = shift_deal(helios, axis, value)
    assert check(shifted)
    assert helios.exit.entry_multiple == 9.5  # the original deal is never mutated


def test_shift_deal_keeps_the_margin_at_entry_fixed(helios):
    shifted = shift_deal(helios, "margin_shift", 0.02)
    assert shifted.operating.ebitda_margin[0] == helios.operating.ebitda_margin[0]


def test_shift_deal_rejects_unknown_axes_and_impossible_holds(helios):
    with pytest.raises(ValueError, match="Unknown sensitivity axis"):
        shift_deal(helios, "vibes", 1.0)
    with pytest.raises(ValueError, match="exceeds the forecast horizon"):
        shift_deal(helios, "hold_years", 9)


def test_sensitivity_grid_shape_and_monotonicity(helios):
    grid = sensitivity_grid(helios, "exit_multiple", [8.0, 9.5, 11.0], "entry_multiple", [9.0, 10.0])
    assert grid.shape == (2, 3)
    assert (np.diff(grid.to_numpy(), axis=1) > 0).all()  # higher exit multiple, higher IRR
    assert (np.diff(grid.to_numpy(), axis=0) < 0).all()  # paying more at entry, lower IRR


def test_every_registered_metric_returns_a_number(helios):
    for name, metric in METRICS.items():
        value = metric(helios)
        assert np.isfinite(value), name


def test_sensitivity_grid_rejects_unknown_metrics(helios):
    with pytest.raises(ValueError, match="Unknown metric"):
        sensitivity_grid(helios, "exit_multiple", [9.0], "entry_multiple", [9.0], metric="alpha")


def test_solve_for_target_round_trips(helios):
    multiple = solve_for_target(helios, "exit_multiple", 0.20, bracket=(4.0, 20.0))
    achieved = equity_returns(run_lbo(shift_deal(helios, "exit_multiple", multiple))).gross_irr
    assert achieved == pytest.approx(0.20, abs=1e-6)


def test_solve_for_target_returns_nan_when_unreachable(helios):
    assert np.isnan(solve_for_target(helios, "exit_multiple", 5.0, bracket=(4.0, 12.0)))
