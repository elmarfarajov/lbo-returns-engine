# Methodology

How the model is built, why each choice was made, and where it stops being reliable.

- [1. Entry: sources, uses and purchase accounting](#1-entry-sources-uses-and-purchase-accounting)
- [2. The operating forecast](#2-the-operating-forecast)
- [3. The debt schedule and the circularity problem](#3-the-debt-schedule-and-the-circularity-problem)
- [4. Accounting identities as tests](#4-accounting-identities-as-tests)
- [5. Exit and the equity waterfall](#5-exit-and-the-equity-waterfall)
- [6. Return measures and the public market equivalent](#6-return-measures-and-the-public-market-equivalent)
- [7. Value creation attribution](#7-value-creation-attribution)
- [8. Sensitivity, break-evens and Monte Carlo](#8-sensitivity-break-evens-and-monte-carlo)
- [9. The public-to-private screen](#9-the-public-to-private-screen)
- [10. Limitations](#10-limitations)
- [References](#references)

---

## 1. Entry: sources, uses and purchase accounting

Enterprise value at entry is a multiple of last-twelve-month EBITDA,
$EV_0 = M_0 \times EBITDA_0$, and the funding must balance:

$$\underbrace{D + E_{\text{sponsor}} + E_{\text{rollover}}}_{\text{sources}} = \underbrace{EV_0 + F_{\text{transaction}} + F_{\text{financing}} + C_{\min}}_{\text{uses}}$$

Sponsor equity is the plug: whatever the debt, the rollover and the balance sheet cash
do not cover. If it turns negative the structure over-funds the purchase and the model
refuses to run rather than quietly reporting an infinite return.

Fees are treated the way they are treated in practice, and the difference matters:

- **Transaction fees** (advisory, legal) are expensed at close, so they reduce opening
  equity and never return.
- **Financing fees** are capitalised as a deferred asset and amortised into interest
  expense over the weighted-average term of the debt, so they shelter tax.

The opening balance sheet allocates the purchase price: identifiable net assets
(working capital and fixed assets) at book value, and the remainder to goodwill,
$GW = EV_0 - NWC_0 - PPE_0$. Substituting the funding identity shows the sheet balances
at close by construction, which is the first thing the model checks.

## 2. The operating forecast

Revenue compounds at the assumed growth path and EBITDA follows a margin path, so a
sponsor's two levers - grow the business and improve the margin - are separable in the
attribution later. Working capital is driven by days, not a flat percentage:

$$NWC = \underbrace{\frac{DSO}{365}\,\text{Revenue}}_{\text{receivables}} + \underbrace{\frac{DIO}{365}\,\text{COGS}}_{\text{inventory}} - \underbrace{\frac{DPO}{365}\,\text{COGS}}_{\text{payables}}, \qquad COGS = (1 - \text{gross margin}) \times \text{Revenue}$$

Growth therefore consumes cash, which is exactly the tension a leveraged business
lives with. Tax is paid on earnings after interest and depreciation, with losses
carried forward: the carryforward absorbs positive income before it is taxed and grows
by any loss in the year.

## 3. The debt schedule and the circularity problem

Each tranche carries either a floating rate (base curve plus spread) or a fixed coupon,
optional mandatory amortisation, optional PIK accrual, and a place in the cash sweep
queue. A revolver is drawn when cash flow cannot cover the year and repaid first.

Interest accrues on the **average** of the opening and closing balance, which creates a
genuine circular reference:

$$I = r \cdot \frac{B_{\text{open}} + B_{\text{close}}}{2}, \qquad B_{\text{close}} = B_{\text{open}} - \text{sweep}(FCF - I)$$

Spreadsheets resolve this with an iterative calculation setting; here each year is
solved by fixed-point iteration to a tolerance of $10^{-10}$, which converges in
roughly ten passes. For a single tranche with a full sweep the fixed point is available
in closed form,

$$I = \frac{r\,(2B_{\text{open}} - EBITDA)}{2 - r},$$

and the validation suite checks the solver against it, then recomputes interest
independently from the finished schedule.

The waterfall of cash within each year is the market convention: mandatory amortisation
first, then the revolver, then term debt by seniority, with any remaining cash swept at
the agreed percentage and the rest retained above a minimum balance.

Maintenance covenants are tested each year - net leverage against a stepping-down
schedule, interest coverage, and fixed-charge coverage - and breaches are reported with
the headroom. The model does not simulate a default; it flags the year the lender gets
a seat at the table.

## 4. Accounting identities as tests

Two identities must hold every year, and both are recomputed and reported:

$$\text{Cash} + NWC + PPE + GW + \text{deferred fees} = \text{Debt} + \text{Equity}$$

$$\Delta\text{Cash} = \underbrace{NI + D\&A + \text{fee amortisation} + \text{PIK} - \Delta NWC}_{\text{operating}} - \underbrace{\text{capex}}_{\text{investing}} + \underbrace{\text{draws} - \text{repayments} - \text{dividends}}_{\text{financing}}$$

Substituting the second into the first shows the balance sheet can only tie if the cash
flow statement does, so a single silent error in the debt schedule breaks both. In the
worked deals the residuals are below $10^{-12}$.

## 5. Exit and the equity waterfall

Exit enterprise value is the exit multiple times final-year EBITDA; equity proceeds are
that less net debt and exit fees. The proceeds are then split in the order real
documents specify:

1. **Management incentive plan**: a share of value creation above the entry equity
   value, so management is paid for the gain rather than the asset.
2. **Investors pro rata**: sponsor and rollover holders by ownership.
3. **The fund waterfall** on the sponsor's own cash flows: investors first receive their
   capital plus a preferred return, then the general partner catches up, then profits
   split at the carry rate.

With a full catch-up the carry is piecewise in total profit $P$ against the preferred
return's profit $H$:

$$\text{carry} = \begin{cases} 0 & P \le H \\ P - H & H < P \le H + \frac{cH}{1-c} \\ c\,P & \text{otherwise} \end{cases}$$

The third branch is the useful sanity check: past the catch-up the general partner
holds exactly its carry share of all profit, which the test suite verifies.

## 6. Return measures and the public market equivalent

IRR solves $\sum_t CF_t (1+r)^{-t} = 0$ by bracketed root finding, returning NaN rather
than a fabricated number when the flows never change sign; XIRR does the same on actual
dates. MOIC is gross distributions over gross contributions.

Neither says whether the return was worth the risk, so the model also reports the
**Kaplan-Schoar PME**, which discounts the same cash flows by a public index:

$$\text{PME} = \frac{\sum_t D_t / I_t}{\sum_t C_t / I_t}, \qquad \text{direct alpha} = \text{IRR}\left(CF_t \cdot \frac{I_T}{I_t}\right)$$

A PME above 1.0 means the deal beat the index on identical timing; direct alpha
annualises the gap. An index compounding at the deal's own IRR must give a PME of
exactly 1.00 and an alpha of zero - another identity the suite checks.

## 7. Value creation attribution

Sponsors report IRRs; investors want to know where the money came from. The bridge is
an exact identity, not an approximation:

$$\underbrace{E_1 + \text{dividends} - E_0}_{\text{value created}} = \underbrace{M_0 (R_1 - R_0) m_0}_{\text{revenue growth}} + \underbrace{M_0 R_1 (m_1 - m_0)}_{\text{margin expansion}} + \underbrace{(M_1 - M_0) EBITDA_1}_{\text{multiple}} + \underbrace{\textstyle\sum_t FCF_t}_{\text{cash generation}} - \underbrace{\textstyle\sum_t PIK_t}_{\text{accretion}} - \text{fees}$$

It follows from three facts: entry equity is $EV_0 - ND_0 + \text{fees}$, exit equity is
$EV_1 - ND_1 - \text{fees}$, and the movement in net debt is exactly cumulative levered
free cash flow less PIK accretion less dividends. The residual is reported with every
run; in the worked deals it is around $10^{-13}$.

Each component is also expressed as an IRR contribution by compounding the cumulative
value created back into an annualised return, which answers the question sponsors are
actually asked in fundraising: how many points of the IRR came from the market rather
than from the business?

## 8. Sensitivity, break-evens and Monte Carlo

Any assumption can be shifted on a common interface, so two-way grids, a lever ranking
and a break-even solver share one code path. The solver scans its bracket before root
finding because extreme inputs make a deal unfinanceable, and returns NaN when no value
in range reaches the target rather than extrapolating nonsense.

The simulation perturbs four things, and splits growth uncertainty in two because the
distinction matters under leverage:

- a **persistent** growth surprise (the business is structurally better or worse than
  underwritten) and **annual** noise on top,
- margin delivery, phased in along the same ramp as the base case,
- the exit multiple, floored to keep the deal saleable,
- the base-rate curve.

Normal noise alone never produces the outcomes leverage is judged on, so a downturn
also hits with a configurable probability: a revenue shock in a random year, a partial
rebound the year after, a margin hit that decays, and a lower exit multiple. This is
what turns a covenant breach probability of 3% at five turns into 53% at six and a half.

## 9. The public-to-private screen

For each listed company the screen builds a standard structure and solves for the
maximum premium to the current share price that still clears a target IRR, assuming an
exit at **today's** trading multiple - so the premium has to be earned back through
growth, margin and deleveraging rather than multiple expansion.

Credit capacity comes first. Leverage is reduced in half-turn steps until year-one
interest coverage passes the test, and a company that never passes is reported as
unfinanceable instead of being silently levered to an impossible structure. Fundamentals
come from Yahoo Finance, with capex and depreciation taken from the cash flow statement
and revenue growth from the historical revenue series, capped so that a single strong
year cannot be extrapolated for five.

## 10. Limitations

- **A model, not a diligence process.** One EBITDA, one margin path and one multiple
  stand in for a quality-of-earnings review, a market study and a management
  assessment.
- **No default machinery.** A covenant breach is flagged, not resolved: there is no
  amend-and-extend, equity cure, or restructuring of the capital structure.
- **Taxes are simplified.** A single rate with loss carryforwards, no interest
  deductibility caps (such as the 30% of EBITDA limit under section 163(j)), no
  cross-border structuring, and no cash-versus-book differences beyond the carryforward.
- **Purchase accounting is stylised.** No step-up of intangibles or the deferred tax
  that comes with it.
- **Exit timing is fixed.** Real sponsors sell when the window opens; hold length here
  is an input, not a decision. The lever table shows what a year of patience costs.
- **Screen inputs are noisy.** Yahoo Finance fundamentals are unaudited, occasionally
  stale, and EBITDA definitions vary; the screen is a shortlist generator, not a
  valuation.

## References

- Gompers, P., Kaplan, S. N. & Mukharlyamov, V. (2016). What do private equity firms say they do? *Journal of Financial Economics* 121(3).
- Kaplan, S. N. & Schoar, A. (2005). Private equity performance: returns, persistence and capital flows. *Journal of Finance* 60(4).
- Kaplan, S. N. & Stromberg, P. (2009). Leveraged buyouts and private equity. *Journal of Economic Perspectives* 23(1).
- Gredil, O., Griffiths, B. & Stucke, R. (2014). Benchmarking private equity: the direct alpha method. Working paper.
- Axelson, U., Jenkinson, T., Stromberg, P. & Weisbach, M. (2013). Borrow cheap, buy high? The determinants of leverage and pricing in buyouts. *Journal of Finance* 68(6).
- Metrick, A. & Yasuda, A. (2010). The economics of private equity funds. *Review of Financial Studies* 23(6).
- Rosenbaum, J. & Pearl, J. (2020). *Investment Banking: Valuation, LBOs, M&A and IPOs*. 3rd edition, Wiley.
- Phalippou, L. (2020). *Private Equity Laid Bare*. 3rd edition.
