import numpy as np
import pytest

from lbolab.montecarlo import MonteCarloConfig, run_monte_carlo
from lbolab.sensitivity import shift_deal


@pytest.fixture(scope="module")
def simulation(helios):
    return run_monte_carlo(helios, MonteCarloConfig(n_trials=400, seed=3))


def test_reproducible_with_a_fixed_seed(helios):
    first = run_monte_carlo(helios, MonteCarloConfig(n_trials=80, seed=7)).trials
    second = run_monte_carlo(helios, MonteCarloConfig(n_trials=80, seed=7)).trials
    np.testing.assert_allclose(first["gross_irr"], second["gross_irr"])


def test_different_seeds_give_different_paths(helios):
    first = run_monte_carlo(helios, MonteCarloConfig(n_trials=80, seed=1)).trials["gross_irr"].to_numpy()
    second = run_monte_carlo(helios, MonteCarloConfig(n_trials=80, seed=2)).trials["gross_irr"].to_numpy()
    assert not np.allclose(first, second)


def test_summary_is_a_valid_distribution(simulation):
    summary = simulation.summary()
    assert summary["p5_irr"] <= summary["median_irr"] <= summary["p95_irr"]
    for key in ("prob_loss", "prob_above_20pct", "prob_covenant_breach", "prob_recession"):
        assert 0.0 <= summary[key] <= 1.0
    assert summary["trials"] == 400


def test_recession_frequency_matches_the_configured_probability(helios):
    config = MonteCarloConfig(n_trials=600, seed=9, recession_probability=0.3)
    trials = run_monte_carlo(helios, config).trials
    share = float((trials["recession_year"] > 0).mean())
    assert share == pytest.approx(0.3, abs=0.06)
    assert set(trials.loc[trials["recession_year"] > 0, "recession_year"]).issubset(set(range(1, 6)))


def test_downturns_hurt_returns_and_covenants(simulation):
    scenarios = simulation.by_scenario()
    assert scenarios.loc["downturn", "median_irr"] < scenarios.loc["no downturn", "median_irr"]
    assert scenarios.loc["downturn", "prob_breach"] >= scenarios.loc["no downturn", "prob_breach"]


def test_leverage_raises_both_the_median_and_the_breach_probability(helios):
    config = MonteCarloConfig(n_trials=300, seed=4)
    low = run_monte_carlo(shift_deal(helios, "leverage_turns", 3.5), config).summary()
    high = run_monte_carlo(shift_deal(helios, "leverage_turns", 6.5), config).summary()
    assert high["median_irr"] > low["median_irr"]
    assert high["prob_covenant_breach"] > low["prob_covenant_breach"]


def test_zero_dispersion_reproduces_the_base_case(helios):
    from lbolab.model import run_lbo
    from lbolab.returns import equity_returns

    quiet = MonteCarloConfig(
        n_trials=5,
        persistent_growth_sd=0.0,
        annual_growth_sd=0.0,
        margin_sd=0.0,
        exit_multiple_sd=0.0,
        base_rate_sd=0.0,
        recession_probability=0.0,
    )
    base = equity_returns(run_lbo(helios)).gross_irr
    trials = run_monte_carlo(helios, quiet).trials
    np.testing.assert_allclose(trials["gross_irr"], base, atol=1e-12)
