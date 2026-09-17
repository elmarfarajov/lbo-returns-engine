"""Two worked deals: a conservatively financed carve-out and an aggressive roll-up.

The same assumptions live in ``deals/*.yaml`` for people who would rather read a config
file than Python; a test checks the two never drift apart.
"""

from __future__ import annotations

from .assumptions import (
    CovenantPackage,
    Deal,
    DebtTranche,
    DividendRecap,
    ExitAssumptions,
    FinancingAssumptions,
    OperatingAssumptions,
    WaterfallAssumptions,
)


def helios_carveout() -> Deal:
    """Corporate carve-out at 9.5x with 5.0x leverage: the base case of the report."""
    return Deal(
        name="Helios Industrials carve-out",
        notes=(
            "Corporate carve-out of a speciality components business. The sponsor underwrites modest "
            "organic growth and a 300 basis point margin improvement from standalone cost actions, "
            "funded with 5.0x total leverage and exited at the entry multiple after five years."
        ),
        operating=OperatingAssumptions(
            entry_revenue=420.0,
            ebitda_margin=(0.180, 0.185, 0.192, 0.200, 0.206, 0.210),
            revenue_growth=(0.080, 0.070, 0.060, 0.050, 0.045),
            gross_margin=0.42,
            days_sales_outstanding=52.0,
            days_inventory=65.0,
            days_payable=45.0,
            capex_pct_revenue=0.032,
            depreciation_pct_revenue=0.028,
            tax_rate=0.25,
            opening_ppe=120.0,
        ),
        financing=FinancingAssumptions(
            tranches=(
                DebtTranche(
                    name="Revolver",
                    turns=0.0,
                    commitment_turns=0.5,
                    spread=0.035,
                    is_revolver=True,
                    sweep_priority=1,
                    term_years=6,
                ),
                DebtTranche(name="Term Loan B", turns=4.0, spread=0.0425, amortisation_pct=0.01, sweep_priority=2, term_years=7),
                DebtTranche(name="Senior notes", turns=1.0, spread=0.085, floating=False, term_years=8),
            ),
            base_rate=(0.042, 0.040, 0.037, 0.035, 0.035),
            transaction_fee_pct=0.020,
            financing_fee_pct=0.020,
            exit_fee_pct=0.010,
            minimum_cash=15.0,
            cash_sweep_pct=0.75,
        ),
        exit=ExitAssumptions(hold_years=5, entry_multiple=9.5, exit_multiple=9.5),
        covenants=CovenantPackage(
            max_total_leverage=(5.75, 5.25, 4.75, 4.25, 4.00),
            min_interest_coverage=2.00,
            min_fixed_charge_coverage=1.10,
        ),
        waterfall=WaterfallAssumptions(
            rollover_equity=20.0, management_pool_pct=0.10, preferred_return=0.08, carry=0.20, catch_up=True
        ),
    )


def apex_rollup() -> Deal:
    """Aggressive structure: 6.5x leverage including PIK, a dividend recap in year three."""
    return Deal(
        name="Apex Services roll-up",
        notes=(
            "Buy-and-build in fragmented business services, financed at 6.5x including a PIK note and "
            "recapitalised in year three once leverage falls below four turns. The structure shows what "
            "happens to covenant headroom when returns are manufactured with debt rather than margin."
        ),
        operating=OperatingAssumptions(
            entry_revenue=260.0,
            ebitda_margin=(0.220, 0.226, 0.232, 0.238, 0.242, 0.245),
            revenue_growth=(0.120, 0.110, 0.095, 0.080, 0.070),
            gross_margin=0.55,
            days_sales_outstanding=48.0,
            days_inventory=15.0,
            days_payable=38.0,
            capex_pct_revenue=0.025,
            depreciation_pct_revenue=0.022,
            tax_rate=0.25,
            opening_ppe=60.0,
        ),
        financing=FinancingAssumptions(
            tranches=(
                DebtTranche(
                    name="Revolver",
                    turns=0.0,
                    commitment_turns=0.5,
                    spread=0.0375,
                    is_revolver=True,
                    sweep_priority=1,
                    term_years=6,
                ),
                DebtTranche(name="Unitranche", turns=5.0, spread=0.055, amortisation_pct=0.01, sweep_priority=2, term_years=7),
                DebtTranche(name="PIK note", turns=1.5, spread=0.115, floating=False, pik=True, term_years=8),
            ),
            base_rate=(0.042, 0.040, 0.037, 0.035, 0.035),
            transaction_fee_pct=0.025,
            financing_fee_pct=0.025,
            exit_fee_pct=0.010,
            minimum_cash=10.0,
            cash_sweep_pct=1.0,
        ),
        exit=ExitAssumptions(
            hold_years=5,
            entry_multiple=11.0,
            exit_multiple=10.5,
            dividend_recaps=(DividendRecap(year=3, incremental_turns=1.0, tranche="Unitranche"),),
        ),
        covenants=CovenantPackage(max_total_leverage=(7.0, 6.5, 6.0, 5.5, 5.0), min_interest_coverage=1.75),
        waterfall=WaterfallAssumptions(
            rollover_equity=10.0,
            management_pool_pct=0.12,
            preferred_return=0.08,
            carry=0.20,
            catch_up=True,
            management_fee_pct=0.015,
        ),
    )


EXAMPLES = {"helios": helios_carveout, "apex": apex_rollup}
