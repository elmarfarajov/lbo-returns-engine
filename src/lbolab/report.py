"""Self-contained HTML investment memo with every figure embedded."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__, figures
from .pipeline import DealAnalysis

_CSS = """
:root { --ink:#1c2733; --muted:#5d6b78; --line:#e3e7ea; --soft:#f6f8fa; --accent:#28527a;
        --good:#1f7a5a; --bad:#b3402f; }
* { box-sizing: border-box; }
body { margin:0; font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif;
       color:var(--ink); background:#fff; line-height:1.55; }
main { max-width: 1140px; margin: 0 auto; padding: 36px 20px 80px; }
header h1 { font-size: 30px; margin: 0 0 4px; letter-spacing:-0.01em; }
header p { color: var(--muted); margin: 0; }
h2 { font-size: 20px; margin: 42px 0 8px; padding-top: 12px; border-top: 1px solid var(--line); }
p.lead { color: var(--muted); max-width: 840px; }
blockquote { margin: 16px 0; padding: 12px 16px; background: var(--soft); border-left: 3px solid var(--accent);
             color: var(--muted); border-radius: 4px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:12px; margin: 24px 0 8px; }
.kpi { background: var(--soft); border-radius: 10px; padding: 14px 16px; }
.kpi .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.kpi .value { font-size: 22px; font-weight: 650; margin-top: 2px; }
.good { color: var(--good); } .bad { color: var(--bad); }
img { max-width: 100%; height: auto; display:block; margin: 14px 0; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; margin: 12px 0; }
th, td { padding: 5px 9px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { background: var(--soft); font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
footer { margin-top: 56px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 14px; }
"""


ATTRIBUTION_FORMATS = {
    "value": "{:,.1f}",
    "share_of_value": "{:.1%}",
    "irr_equivalent": "{:.1%}",
    "irr_contribution": "{:+.1%}",
}
SCENARIO_FORMATS = {
    "median_irr": "{:.1%}",
    "p5_irr": "{:.1%}",
    "prob_loss": "{:.1%}",
    "prob_breach": "{:.1%}",
    "median_moic": "{:.2f}x",
    "trials": "{:,.0f}",
}
LEVERAGE_FORMATS = {
    "leverage_turns": "{:.1f}x",
    "base_case_irr": "{:.1%}",
    "median_irr": "{:.1%}",
    "p5_irr": "{:.1%}",
    "prob_loss": "{:.1%}",
    "prob_breach": "{:.1%}",
    "equity_cheque": "{:,.0f}",
}
SCREEN_FORMATS = {
    "max_premium": "{:+.1%}",
    "irr_at_market_price": "{:.1%}",
    "ev_ebitda": "{:.1f}x",
    "net_leverage": "{:.2f}x",
    "ebitda_margin": "{:.1%}",
    "revenue_cagr": "{:+.1%}",
    "financeable_leverage": "{:.1f}x",
    "entry_multiple_at_premium": "{:.1f}x",
    "exit_leverage": "{:.2f}x",
}


def _summary_formats(summary: dict) -> dict[str, str]:
    return {key: ("{:.1%}" if "prob" in key or "irr" in key else "{:,.2f}") for key in summary}


def _png(fig: plt.Figure, path: Path) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    data = buffer.getvalue()
    path.write_bytes(data)
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{base64.b64encode(data).decode()}">'


def _table(frame: pd.DataFrame, formats: dict[str, str] | None = None, default: str = "{:,.2f}") -> str:
    formatted = frame.copy()
    formats = formats or {}
    for column in formatted.columns:
        spec = formats.get(str(column), default if pd.api.types.is_float_dtype(formatted[column]) else None)
        if spec:
            formatted[column] = formatted[column].map(lambda v, spec=spec: "" if pd.isna(v) else spec.format(v))
    formatted.columns = [str(c).replace("_", " ") for c in formatted.columns]
    return f'<div class="table-wrap">{formatted.to_html(index=False, border=0, escape=True)}</div>'


def _kpi(label: str, value: str, tone: str = "") -> str:
    classes = f' class="{tone}"' if tone else ""
    return (
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value"{classes}>{html.escape(value)}</div></div>'
    )


def _pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{100 * value:.1f}%"


def write_report(analysis: DealAnalysis, out_dir: Path, screen: pd.DataFrame | None = None, target_irr: float = 0.20) -> Path:
    out_dir = Path(out_dir)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    deal, result, returns = analysis.deal, analysis.result, analysis.returns
    schedule = result.schedule

    images = {
        "sources": _png(figures.sources_and_uses(result), figure_dir / "sources_and_uses.png"),
        "debt": _png(figures.deleveraging(result), figure_dir / "deleveraging.png"),
        "bridge": _png(figures.value_bridge_chart(analysis.bridge, deal.currency), figure_dir / "value_bridge.png"),
        "grids": _png(figures.sensitivity_heatmaps(analysis.grids), figure_dir / "sensitivity.png"),
    }
    if analysis.monte_carlo is not None:
        images["monte_carlo"] = _png(figures.monte_carlo_chart(analysis.monte_carlo), figure_dir / "monte_carlo.png")
    if analysis.leverage_curve is not None and not analysis.leverage_curve.empty:
        images["leverage"] = _png(_leverage_figure(analysis), figure_dir / "leverage_risk_return.png")
    if screen is not None and "max_premium" in screen:
        images["screen"] = _png(figures.screen_chart(screen, target_irr), figure_dir / "screen.png")

    income_rows = [
        "revenue",
        "revenue_growth",
        "ebitda",
        "ebitda_margin",
        "depreciation",
        "ebit",
        "cash_interest",
        "pik_interest",
        "taxes",
        "net_income",
    ]
    cash_rows = [
        "unlevered_fcf",
        "levered_fcf",
        "capex",
        "change_in_nwc",
        "mandatory_amortisation",
        "cash_sweep",
        "revolver_draw",
        "dividend",
        "cash_end",
    ]
    credit_rows = ["total_debt", "net_debt", "leverage", "interest_coverage", "fixed_charge_coverage"]

    def transpose(rows: list[str]) -> pd.DataFrame:
        frame = schedule.set_index("year")[rows].T.reset_index()
        frame.columns = ["line item", *[f"Y{int(y)}" for y in schedule["year"]]]
        frame["line item"] = frame["line item"].str.replace("_", " ")
        return frame

    breaches = result.covenant_breaches
    covenant_note = (
        f"<blockquote>Covenants are breached in {len(breaches)} of {len(schedule)} years: "
        f"{'; '.join(breaches['covenant_breach'])}.</blockquote>"
        if not breaches.empty
        else "<blockquote>No maintenance covenant is breached in the base case. The tightest year leaves "
        f"{100 * _min_headroom(analysis):.0f}% headroom on the leverage test.</blockquote>"
    )

    returns_frame = pd.DataFrame(
        [
            ("Gross money multiple (MOIC)", f"{returns.gross_moic:.2f}x"),
            ("Gross IRR", _pct(returns.gross_irr)),
            ("Net MOIC after fees and carry", f"{returns.net_moic:.2f}x"),
            ("Net IRR to investors", _pct(returns.net_irr)),
            ("Carried interest", f"{returns.carried_interest:,.1f}"),
            ("Preferred return hurdle (profit)", f"{returns.hurdle_profit:,.1f}"),
            ("Management incentive plan", f"{returns.management_incentive:,.1f}"),
            ("Dividends received during the hold", f"{returns.dividends:,.1f}"),
        ],
        columns=["measure", "value"],
    )
    if analysis.pme:
        returns_frame.loc[len(returns_frame)] = ("Kaplan-Schoar PME", f"{analysis.pme['ks_pme']:.2f}")
        returns_frame.loc[len(returns_frame)] = ("Direct alpha versus the index", _pct(analysis.pme["direct_alpha"]))

    sections = [
        f"""
<h2>1. The deal</h2>
<p class="lead">{html.escape(deal.notes or "")}</p>
{_table(result.sources_uses.to_frame(), default="{:,.1f}")}
{images["sources"]}
""",
        f"""
<h2>2. Operating forecast and cash generation</h2>
{_table(transpose(income_rows), default="{:,.1f}")}
{_table(transpose(cash_rows), default="{:,.1f}")}
""",
        f"""
<h2>3. Debt, deleveraging and covenants</h2>
{_table(transpose(credit_rows), default="{:,.2f}")}
{covenant_note}
{images["debt"]}
""",
        f"""
<h2>4. Returns</h2>
<p class="lead">Gross returns are what the asset produced; net returns are what an investor keeps after
management fees and carried interest on a whole-of-deal waterfall with a
{100 * deal.waterfall.preferred_return:.0f}% preferred return and
{"a full catch-up" if deal.waterfall.catch_up else "no catch-up"}.</p>
{_table(returns_frame)}
""",
        f"""
<h2>5. Where the value came from</h2>
<p class="lead">The decomposition below is an exact identity, reconciling to
{analysis.bridge.reconciliation_error:.1e} of the change in equity value.</p>
{images["bridge"]}
{_table(analysis.attribution, formats=ATTRIBUTION_FORMATS)}
""",
        f"""
<h2>6. What moves the needle</h2>
<p class="lead">Each driver is moved by a standard amount, holding everything else constant, and ranked by
the change in gross IRR.</p>
{_table(analysis.levers, formats={"gross_irr": "{:.1%}", "change_vs_base": "{:+.1%}"})}
{_table(analysis.break_evens, formats={"answer": "{:.4f}"})}
""",
        f"""
<h2>7. Sensitivity</h2>
{images["grids"]}
""",
    ]

    if analysis.monte_carlo is not None:
        mc = analysis.monte_carlo
        summary = mc.summary()
        sections.append(
            f"""
<h2>8. Monte Carlo</h2>
<p class="lead">{mc.config.n_trials:,} trials over persistent and annual growth surprises, margin delivery,
the exit multiple and the rate curve, with a {100 * mc.config.recession_probability:.0f}% chance of a downturn
somewhere in the hold period.</p>
{images["monte_carlo"]}
{_table(pd.DataFrame([summary]), formats=_summary_formats(summary))}
{_table(mc.by_scenario().reset_index(), formats=SCENARIO_FORMATS)}
"""
        )
    if analysis.leverage_curve is not None and not analysis.leverage_curve.empty:
        sections.append(
            f"""
<h2>9. How much is leverage worth?</h2>
<p class="lead">Leverage lifts the median return and the probability of breaching a covenant at the same time.
The question is not whether debt raises the IRR, but what it does to the left tail.</p>
{images["leverage"]}
{_table(analysis.leverage_curve, formats=LEVERAGE_FORMATS)}
"""
        )
    if screen is not None and "max_premium" in screen:
        sections.append(
            f"""
<h2>10. Public-to-private screen</h2>
<p class="lead">For each listed company the model solves for the highest premium to the current share price
that still delivers a {100 * target_irr:.0f}% gross IRR, assuming an exit at today's trading multiple and
leverage cut back to whatever passes the interest coverage test.</p>
{images["screen"]}
{_table(screen, formats=SCREEN_FORMATS)}
"""
        )

    kpis = "".join(
        [
            _kpi("Entry / exit multiple", f"{deal.exit.entry_multiple:.1f}x / {deal.exit.exit_multiple:.1f}x"),
            _kpi("Leverage at entry", f"{deal.total_debt_turns():.2f}x"),
            _kpi("Equity cheque", f"{result.sources_uses.equity_cheque:,.0f}"),
            _kpi("Gross MOIC", f"{returns.gross_moic:.2f}x"),
            _kpi("Gross IRR", _pct(returns.gross_irr), "good" if returns.gross_irr >= 0.20 else "bad"),
            _kpi("Net IRR", _pct(returns.net_irr), "good" if returns.net_irr >= 0.15 else "bad"),
            _kpi("Exit leverage", f"{result.diagnostics['exit_leverage']:.2f}x"),
        ]
    )
    timings = ", ".join(f"{name.replace('_', ' ')} {value:.1f}s" for name, value in analysis.timings.items())
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(deal.name)} - LBO analysis</title><style>{_CSS}</style></head>
<body><main>
<header>
  <h1>{html.escape(deal.name)}</h1>
  <p>{deal.exit.hold_years}-year hold &middot; all figures in {html.escape(deal.currency)} &middot; lbolab {__version__}</p>
</header>
<div class="kpis">{kpis}</div>
{"".join(sections)}
<footer>
  Accounting checks: worst balance sheet difference {result.diagnostics["max_balance_check"]:.2e},
  worst cash flow difference {result.diagnostics["max_cash_flow_check"]:.2e},
  {result.diagnostics["interest_iterations"]} iterations to resolve interest circularity.
  Runtime: {html.escape(timings)}. Educational and research output, not investment advice.
</footer>
</main></body></html>"""
    path = out_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


def _leverage_figure(analysis: DealAnalysis) -> plt.Figure:
    """Redraw the leverage risk-return chart from the frame already computed in the pipeline."""
    frame = analysis.leverage_curve
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(frame["leverage_turns"], frame["median_irr"], "o-", color=figures.ACCENT, label="median IRR")
    ax.fill_between(
        frame["leverage_turns"],
        frame["p5_irr"],
        frame["median_irr"],
        color=figures.SOFT,
        alpha=0.35,
        label="5th percentile to median",
    )
    ax.yaxis.set_major_formatter(figures.PCT_FORMATTER)
    ax.set_xlabel("total leverage at entry (turns of EBITDA)")
    ax.set_ylabel("gross IRR")
    twin = ax.twinx()
    twin.plot(frame["leverage_turns"], frame["prob_breach"], "s--", color=figures.NEGATIVE, label="P(covenant breach)")
    twin.plot(frame["leverage_turns"], frame["prob_loss"], "^:", color="#7a3b2e", label="P(loss)")
    twin.yaxis.set_major_formatter(figures.PCT_FORMATTER)
    twin.set_ylabel("probability")
    twin.grid(False)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper left")
    ax.set_title("How much is leverage really worth?")
    fig.tight_layout()
    return fig


def _min_headroom(analysis: DealAnalysis) -> float:
    schedule = analysis.result.schedule
    limits = [analysis.deal.covenants.leverage_limit(int(y)) for y in schedule["year"]]
    headroom = [
        (limit - lev) / limit for limit, lev in zip(limits, schedule["leverage"], strict=True) if limit is not None and limit > 0
    ]
    return min(headroom) if headroom else float("nan")
