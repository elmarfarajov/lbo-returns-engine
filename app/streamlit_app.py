"""Deal cockpit: `streamlit run app/streamlit_app.py`.

Move an assumption and every number moves with it: the debt schedule, the covenant
headroom, the value bridge and the distribution of outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from lbolab.assumptions import Deal
from lbolab.bridge import value_bridge
from lbolab.examples import EXAMPLES
from lbolab.model import run_lbo
from lbolab.montecarlo import MonteCarloConfig, run_monte_carlo
from lbolab.pipeline import break_even_table, lever_table
from lbolab.returns import equity_returns
from lbolab.screener import ScreenAssumptions, screen_companies
from lbolab.sensitivity import sensitivity_grid, shift_deal

st.set_page_config(page_title="LBO Deal Cockpit", layout="wide")
BLUE, GREEN, RED = "#28527a", "#1f7a5a", "#b3402f"


def apply_overrides(deal: Deal, overrides: dict[str, float]) -> Deal:
    for axis, value in overrides.items():
        deal = shift_deal(deal, axis, value)
    return deal


@st.cache_data(show_spinner=False, hash_funcs={Deal: repr})
def simulate(deal: Deal, trials: int, seed: int) -> pd.DataFrame:
    return run_monte_carlo(deal, MonteCarloConfig(n_trials=trials, seed=seed)).trials


st.title("LBO Deal Cockpit")
st.caption("Leveraged buyout modelling, value-creation attribution and outcome risk, from assumptions to returns.")

with st.sidebar:
    st.header("Deal")
    example = st.selectbox("Example", sorted(EXAMPLES), index=0)
    base_deal = EXAMPLES[example]()
    uploaded = st.file_uploader("or upload a deal YAML", type=["yaml", "yml"])
    if uploaded is not None:
        base_deal = Deal.from_dict(yaml.safe_load(uploaded.getvalue().decode("utf-8")))

    st.header("Overrides")
    entry = st.slider("Entry multiple (x EBITDA)", 5.0, 16.0, float(base_deal.exit.entry_multiple), 0.25)
    exit_multiple = st.slider("Exit multiple (x EBITDA)", 4.0, 16.0, float(base_deal.exit.exit_multiple), 0.25)
    leverage = st.slider("Total leverage (turns)", 1.0, 8.0, float(round(base_deal.total_debt_turns(), 2)), 0.25)
    hold = st.slider("Hold period (years)", 2, len(base_deal.operating.revenue_growth), int(base_deal.exit.hold_years))
    growth_shift = st.slider("Revenue growth shift (pp p.a.)", -5.0, 5.0, 0.0, 0.5) / 100.0
    margin_shift = st.slider("Margin shift (pp)", -5.0, 5.0, 0.0, 0.25) / 100.0
    rate_shift = st.slider("Base rate shift (pp)", -2.0, 6.0, 0.0, 0.25) / 100.0
    sweep = st.slider("Cash sweep (% of free cash flow)", 0, 100, int(100 * base_deal.financing.cash_sweep_pct), 5) / 100.0

deal = apply_overrides(
    base_deal,
    {
        "entry_multiple": entry,
        "exit_multiple": exit_multiple,
        "leverage_turns": leverage,
        "hold_years": hold,
        "revenue_growth_shift": growth_shift,
        "margin_shift": margin_shift,
        "base_rate_shift": rate_shift,
        "cash_sweep_pct": sweep,
    },
)

try:
    result = run_lbo(deal)
    returns = equity_returns(result)
    bridge = value_bridge(result)
    error = None
except ValueError as exc:
    result = returns = bridge = None
    error = exc

if error is not None:
    st.error(f"This structure cannot be funded: {error}")
    st.stop()

columns = st.columns(6)
columns[0].metric("Gross MOIC", f"{returns.gross_moic:.2f}x")
columns[1].metric("Gross IRR", f"{100 * returns.gross_irr:.1f}%")
columns[2].metric("Net IRR", f"{100 * returns.net_irr:.1f}%")
columns[3].metric("Equity cheque", f"{result.sources_uses.equity_cheque:,.0f}")
columns[4].metric("Exit leverage", f"{result.diagnostics['exit_leverage']:.2f}x")
columns[5].metric("Covenants", "breached" if result.diagnostics["covenant_breached"] else "clear")
for warning in result.warnings:
    st.warning(warning)

deal_tab, bridge_tab, sensitivity_tab, risk_tab, screen_tab = st.tabs(
    ["Model", "Value bridge", "Sensitivity", "Risk", "Public-to-private screen"]
)

with deal_tab:
    left, right = st.columns([1, 1])
    left.subheader("Sources and uses")
    left.dataframe(result.sources_uses.to_frame().round(1), hide_index=True, use_container_width=True)

    right.subheader("Leverage and coverage")
    credit = go.Figure()
    credit.add_trace(
        go.Scatter(
            x=result.schedule["year"],
            y=result.schedule["leverage"],
            name="net debt / EBITDA",
            mode="lines+markers",
            line={"color": BLUE},
        )
    )
    limits = [deal.covenants.leverage_limit(int(y)) for y in result.schedule["year"]]
    if any(limit is not None for limit in limits):
        credit.add_trace(
            go.Scatter(x=result.schedule["year"], y=limits, name="covenant", mode="lines", line={"color": RED, "dash": "dash"})
        )
    credit.add_trace(
        go.Scatter(
            x=result.schedule["year"],
            y=result.schedule["interest_coverage"],
            name="interest cover",
            mode="lines+markers",
            line={"color": GREEN},
            yaxis="y2",
        )
    )
    credit.update_layout(
        height=340,
        margin={"t": 10, "b": 10},
        yaxis_title="turns",
        yaxis2={"title": "coverage (x)", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": -0.2},
    )
    right.plotly_chart(credit, use_container_width=True)

    st.subheader("Model schedule")
    st.dataframe(
        result.schedule.drop(columns=["balance_check", "cash_flow_check", "iterations"]).round(2),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Balance sheet ties to {result.diagnostics['max_balance_check']:.1e}; cash flow statement to "
        f"{result.diagnostics['max_cash_flow_check']:.1e}; {result.diagnostics['interest_iterations']} iterations "
        "resolved the interest circularity."
    )

with bridge_tab:
    components = bridge.components
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(components) + ["total"],
            x=["Entry equity", *components, "Exit equity + dividends"],
            y=[bridge.entry_equity, *components.values(), 0],
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": BLUE}},
        )
    )
    figure.update_layout(height=460, margin={"t": 20}, yaxis_title=deal.currency)
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(bridge.to_frame().round(1), hide_index=True, use_container_width=True)
    st.caption(f"Reconciliation error {bridge.reconciliation_error:.1e} - the decomposition is an exact identity.")

    st.subheader("What moves the needle")
    st.dataframe(lever_table(deal, returns.gross_irr).round(4), hide_index=True, use_container_width=True)
    st.dataframe(break_even_table(deal).round(4), hide_index=True, use_container_width=True)

with sensitivity_tab:
    metric = st.selectbox("Metric", ["gross_irr", "net_irr", "gross_moic", "exit_leverage", "min_interest_coverage"])
    exits = [round(exit_multiple + step, 2) for step in (-1.5, -0.75, 0.0, 0.75, 1.5)]
    entries = [round(entry + step, 2) for step in (-1.5, -0.75, 0.0, 0.75, 1.5)]
    grid = sensitivity_grid(deal, "exit_multiple", exits, "entry_multiple", entries, metric=metric)
    scale = 100 if metric.endswith("irr") else 1
    heatmap = go.Figure(
        go.Heatmap(
            z=scale * grid.to_numpy(),
            x=[f"{v:g}x" for v in grid.columns],
            y=[f"{v:g}x" for v in grid.index],
            colorscale="RdYlGn",
            texttemplate="%{z:.1f}",
        )
    )
    heatmap.update_layout(height=420, margin={"t": 20}, xaxis_title="exit multiple", yaxis_title="entry multiple")
    st.plotly_chart(heatmap, use_container_width=True)

with risk_tab:
    trials = st.slider("Trials", 200, 3000, 800, 200)
    if st.button("Run Monte Carlo", type="primary"):
        st.session_state["trials"] = simulate(deal, trials, 11)
    if "trials" in st.session_state:
        frame = st.session_state["trials"]
        irr = frame["gross_irr"].to_numpy()
        histogram = go.Figure()
        histogram.add_trace(go.Histogram(x=irr[frame["recession_year"] == 0], name="no downturn", marker_color=BLUE))
        histogram.add_trace(go.Histogram(x=irr[frame["recession_year"] > 0], name="downturn", marker_color=RED))
        histogram.update_layout(
            barmode="stack", height=420, xaxis_tickformat=".0%", xaxis_title="gross IRR", legend={"orientation": "h", "y": -0.2}
        )
        st.plotly_chart(histogram, use_container_width=True)
        cols = st.columns(4)
        cols[0].metric("Median IRR", f"{100 * np.nanmedian(irr):.1f}%")
        cols[1].metric("5th percentile", f"{100 * np.nanpercentile(irr, 5):.1f}%")
        cols[2].metric("P(loss)", f"{100 * np.nanmean(irr < 0):.1f}%")
        cols[3].metric("P(covenant breach)", f"{100 * frame['covenant_breach'].mean():.1f}%")

with screen_tab:
    st.write(
        "Solve for the highest premium to today's share price that still clears a target IRR, "
        "assuming an exit at the company's current trading multiple."
    )
    tickers = st.text_input("Tickers", "HRB, KMB, SJM").replace(" ", "").split(",")
    target = st.slider("Target gross IRR", 0.10, 0.35, 0.20, 0.01)
    if st.button("Run screen"):
        with st.spinner("Downloading fundamentals ..."):
            st.session_state["screen"] = screen_companies([t for t in tickers if t], ScreenAssumptions(), target)
    if "screen" in st.session_state:
        st.dataframe(st.session_state["screen"].round(3), hide_index=True, use_container_width=True)
