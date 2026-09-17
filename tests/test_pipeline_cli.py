import numpy as np
import pandas as pd
import pytest

from lbolab.cli import build_parser, main
from lbolab.pipeline import AnalysisConfig, break_even_table, lever_table, run_analysis
from lbolab.report import write_report
from lbolab.validation import run_validation, to_markdown

FAST = AnalysisConfig(
    monte_carlo_trials=60,
    leverage_levels=(4.0, 6.0),
    leverage_curve_trials=40,
    index_levels=(1.0, 1.1, 1.21, 1.331, 1.4641, 1.61051),
)


def test_lever_table_is_sorted_by_impact(helios, helios_returns):
    frame = lever_table(helios, helios_returns.gross_irr)
    impacts = frame["change_vs_base"].abs().to_numpy()
    assert (np.diff(impacts) <= 1e-12).all()
    assert set(frame["lever"]).issuperset({"Exit multiple +1.0x", "Leverage +1.0x", "Base rates +200bp"})
    positive = frame.loc[frame["lever"] == "Exit multiple +1.0x", "change_vs_base"].item()
    negative = frame.loc[frame["lever"] == "Exit multiple -1.0x", "change_vs_base"].item()
    assert positive > 0 > negative


def test_break_even_table_answers_are_consistent(helios):
    frame = break_even_table(helios).set_index("question")
    entry = frame.loc["Entry multiple for a 20% IRR", "answer"]
    assert entry < helios.exit.entry_multiple  # a 20% IRR needs a cheaper entry than the base case
    breakeven_exit = frame.loc["Exit multiple to break even", "answer"]
    assert 0 < breakeven_exit < helios.exit.exit_multiple


def test_run_analysis_produces_every_block(helios):
    analysis = run_analysis(helios, FAST, log=lambda _: None)
    assert analysis.monte_carlo is not None and len(analysis.monte_carlo.trials) == 60
    assert analysis.leverage_curve is not None and len(analysis.leverage_curve) == 2
    assert set(analysis.grids) == {
        "Gross IRR: entry versus exit multiple",
        "Gross IRR: leverage versus exit multiple",
    }
    assert analysis.pme["ks_pme"] > 1.0
    assert analysis.pme["index_return"] == pytest.approx(0.10, abs=1e-9)
    assert abs(analysis.bridge.reconciliation_error) < 1e-6
    assert "model" in analysis.timings


def test_report_writes_html_and_figures(tmp_path, helios):
    analysis = run_analysis(
        helios, AnalysisConfig(monte_carlo_trials=40, leverage_curve_trials=30, leverage_levels=(4.0, 6.0)), log=lambda _: None
    )
    screen = pd.DataFrame([{"ticker": "AAA", "max_premium": 0.3, "entry_multiple_at_premium": 9.0, "irr_at_market_price": 0.25}])
    path = write_report(analysis, tmp_path, screen=screen)
    html = path.read_text(encoding="utf-8")
    for heading in (
        "The deal",
        "Where the value came from",
        "What moves the needle",
        "Monte Carlo",
        "How much is leverage worth",
        "Public-to-private screen",
    ):
        assert heading in html
    for name in (
        "sources_and_uses",
        "deleveraging",
        "value_bridge",
        "sensitivity",
        "monte_carlo",
        "leverage_risk_return",
        "screen",
    ):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 8_000


def test_report_without_monte_carlo_or_screen(tmp_path, helios):
    analysis = run_analysis(helios, AnalysisConfig(run_monte_carlo=False, run_leverage_curve=False), log=lambda _: None)
    html = write_report(analysis, tmp_path).read_text(encoding="utf-8")
    assert "Monte Carlo" not in html
    assert "Public-to-private screen" not in html
    assert "No maintenance covenant is breached" in html


def test_cli_demo_writes_outputs(tmp_path, capsys):
    main(["demo", "--example", "apex", "--out", str(tmp_path), "--trials", "40", "--no-leverage-curve"])
    assert (tmp_path / "report.html").exists()
    schedule = pd.read_csv(tmp_path / "model_schedule.csv")
    assert len(schedule) == 5
    assert schedule["balance_check"].abs().max() < 1e-8
    assert (tmp_path / "monte_carlo_trials.csv").exists()
    assert "Report:" in capsys.readouterr().out


def test_cli_run_reads_a_yaml_deal(tmp_path, capsys):
    from pathlib import Path

    deal_file = Path(__file__).resolve().parents[1] / "deals" / "helios_carveout.yaml"
    main(["run", str(deal_file), "--out", str(tmp_path), "--no-monte-carlo", "--no-leverage-curve", "--index-return", "0"])
    assert (tmp_path / "value_bridge.csv").exists()
    assert "Helios" in capsys.readouterr().out


def test_cli_validate_reports_all_checks(tmp_path, capsys):
    main(["validate", "--out", str(tmp_path / "VALIDATION.md")])
    written = (tmp_path / "VALIDATION.md").read_text(encoding="utf-8")
    assert "PASS" in written and "FAIL" not in written
    assert "Validation report" in written
    assert "| Area | Check |" in capsys.readouterr().out


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.slow
def test_validation_suite_passes():
    results = run_validation()
    assert len(results) >= 20
    assert results["passed"].all(), results.loc[~results["passed"], ["check", "result", "error"]].to_string()
    assert "| Accounting |" in to_markdown(results)
