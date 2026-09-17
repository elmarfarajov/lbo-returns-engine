"""Deal assumptions: typed, validated and loadable from YAML.

A deal is a plain data structure, so a reader can inspect exactly what drives a
result, and a test can build a deal in three lines. Every rate is a decimal
(0.085 = 8.5%), every amount is in millions of the deal currency, and every
per-year sequence is indexed from the first forecast year.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OperatingAssumptions:
    """Revenue, margin and working-capital drivers of the operating forecast."""

    entry_revenue: float
    ebitda_margin: tuple[float, ...]
    """Margin path with one entry per year, index 0 being the last twelve months at entry."""
    revenue_growth: tuple[float, ...]
    gross_margin: float = 0.45
    days_sales_outstanding: float = 45.0
    days_inventory: float = 60.0
    days_payable: float = 40.0
    capex_pct_revenue: float = 0.03
    depreciation_pct_revenue: float = 0.028
    tax_rate: float = 0.25
    opening_ppe: float = 0.0

    def __post_init__(self) -> None:
        if self.entry_revenue <= 0:
            raise ValueError("entry_revenue must be positive")
        if len(self.ebitda_margin) != len(self.revenue_growth) + 1:
            raise ValueError("ebitda_margin needs one more entry than revenue_growth (year 0 plus each forecast year)")
        if not 0.0 < self.gross_margin < 1.0:
            raise ValueError("gross_margin must lie in (0, 1)")
        if not 0.0 <= self.tax_rate < 1.0:
            raise ValueError("tax_rate must lie in [0, 1)")

    @property
    def entry_ebitda(self) -> float:
        return self.entry_revenue * self.ebitda_margin[0]

    def revenue(self, year: int) -> float:
        revenue = self.entry_revenue
        for growth in self.revenue_growth[:year]:
            revenue *= 1.0 + growth
        return revenue

    def net_working_capital(self, revenue: float) -> float:
        cogs = revenue * (1.0 - self.gross_margin)
        receivables = revenue * self.days_sales_outstanding / 365.0
        inventory = cogs * self.days_inventory / 365.0
        payables = cogs * self.days_payable / 365.0
        return receivables + inventory - payables


@dataclass(frozen=True)
class DebtTranche:
    """One layer of the capital structure.

    Size is given either as an absolute amount or as turns of entry EBITDA, which is
    how leverage is actually quoted in the market ("5.0x total, 3.5x first lien").
    """

    name: str
    turns: float | None = None
    amount: float | None = None
    spread: float = 0.0
    """Margin over the base rate for floating debt, or the full coupon when fixed."""
    floating: bool = True
    amortisation_pct: float = 0.0
    """Mandatory repayment per year as a share of the original principal."""
    pik: bool = False
    sweep_priority: int | None = None
    """Lower numbers are repaid first by the cash sweep; None means no sweep."""
    is_revolver: bool = False
    commitment_turns: float | None = None
    undrawn_fee: float = 0.005
    term_years: int = 7

    def __post_init__(self) -> None:
        if (self.turns is None) == (self.amount is None):
            raise ValueError(f"{self.name}: give exactly one of turns or amount")
        if self.pik and self.sweep_priority is not None:
            raise ValueError(f"{self.name}: PIK debt does not participate in the cash sweep")
        if self.is_revolver and self.commitment_turns is None and self.turns is None:
            raise ValueError(f"{self.name}: a revolver needs a commitment")

    def opening_balance(self, entry_ebitda: float) -> float:
        return self.amount if self.amount is not None else float(self.turns) * entry_ebitda

    def commitment(self, entry_ebitda: float) -> float:
        if not self.is_revolver:
            return self.opening_balance(entry_ebitda)
        turns = self.commitment_turns if self.commitment_turns is not None else self.turns
        return float(turns) * entry_ebitda


@dataclass(frozen=True)
class FinancingAssumptions:
    tranches: tuple[DebtTranche, ...]
    base_rate: tuple[float, ...] = (0.04,)
    """Forward base-rate curve (e.g. SOFR). The last value is held flat if the deal runs longer."""
    transaction_fee_pct: float = 0.02
    """Advisory and other deal fees, as a share of enterprise value, expensed at close."""
    financing_fee_pct: float = 0.02
    """Underwriting fees on debt raised, capitalised and amortised over the term."""
    exit_fee_pct: float = 0.01
    minimum_cash: float = 15.0
    cash_sweep_pct: float = 1.0

    def rate_for(self, year: int) -> float:
        index = min(year - 1, len(self.base_rate) - 1)
        return self.base_rate[max(index, 0)]


@dataclass(frozen=True)
class CovenantPackage:
    """Maintenance covenants. Leverage tests usually step down over the life of the loan."""

    max_total_leverage: tuple[float, ...] | None = None
    min_interest_coverage: float | None = None
    min_fixed_charge_coverage: float | None = None

    def leverage_limit(self, year: int) -> float | None:
        if not self.max_total_leverage:
            return None
        index = min(year - 1, len(self.max_total_leverage) - 1)
        return self.max_total_leverage[max(index, 0)]


@dataclass(frozen=True)
class DividendRecap:
    """Mid-hold recapitalisation: raise incremental debt and distribute the proceeds."""

    year: int
    incremental_turns: float
    tranche: str


@dataclass(frozen=True)
class WaterfallAssumptions:
    """How exit proceeds are shared between management, the sponsor and its investors."""

    rollover_equity: float = 0.0
    management_pool_pct: float = 0.10
    """Share of equity value creation paid to the management incentive plan."""
    preferred_return: float = 0.08
    carry: float = 0.20
    catch_up: bool = True
    management_fee_pct: float = 0.0
    """Annual fee charged to investors on invested capital."""

    def __post_init__(self) -> None:
        for name in ("management_pool_pct", "carry"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")


@dataclass(frozen=True)
class ExitAssumptions:
    hold_years: int
    entry_multiple: float
    exit_multiple: float
    dividend_recaps: tuple[DividendRecap, ...] = ()

    def __post_init__(self) -> None:
        if self.hold_years < 1:
            raise ValueError("hold_years must be at least 1")
        if self.entry_multiple <= 0 or self.exit_multiple <= 0:
            raise ValueError("multiples must be positive")


@dataclass(frozen=True)
class Deal:
    name: str
    operating: OperatingAssumptions
    financing: FinancingAssumptions
    exit: ExitAssumptions
    waterfall: WaterfallAssumptions = field(default_factory=WaterfallAssumptions)
    covenants: CovenantPackage = field(default_factory=CovenantPackage)
    currency: str = "USD m"
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.operating.revenue_growth) < self.exit.hold_years:
            raise ValueError("revenue_growth must cover every year of the hold period")
        for recap in self.exit.dividend_recaps:
            if not 1 <= recap.year <= self.exit.hold_years:
                raise ValueError(f"dividend recap in year {recap.year} falls outside the hold period")
            if recap.tranche not in {t.name for t in self.financing.tranches}:
                raise ValueError(f"dividend recap references unknown tranche {recap.tranche!r}")

    @property
    def entry_ebitda(self) -> float:
        return self.operating.entry_ebitda

    @property
    def entry_enterprise_value(self) -> float:
        return self.exit.entry_multiple * self.entry_ebitda

    def total_debt_turns(self) -> float:
        drawn = sum(t.opening_balance(self.entry_ebitda) for t in self.financing.tranches if not t.is_revolver)
        return drawn / self.entry_ebitda

    def with_overrides(self, **overrides: Any) -> Deal:
        """Return a copy with nested fields replaced, e.g. ``with_overrides(exit={"exit_multiple": 9.0})``."""
        updated: dict[str, Any] = {}
        for key, value in overrides.items():
            current = getattr(self, key)
            updated[key] = replace(current, **value) if isinstance(value, dict) else value
        return replace(self, **updated)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Deal:
        data = dict(payload)
        operating = OperatingAssumptions(**_tuples(data.pop("operating")))
        financing_payload = _tuples(data.pop("financing"))
        tranches = tuple(DebtTranche(**t) for t in financing_payload.pop("tranches"))
        financing = FinancingAssumptions(tranches=tranches, **financing_payload)
        exit_payload = _tuples(data.pop("exit"))
        recaps = tuple(DividendRecap(**r) for r in exit_payload.pop("dividend_recaps", ()) or ())
        exit_assumptions = ExitAssumptions(dividend_recaps=recaps, **exit_payload)
        waterfall = WaterfallAssumptions(**data.pop("waterfall", {}) or {})
        covenants = CovenantPackage(**_tuples(data.pop("covenants", {}) or {}))
        return cls(
            operating=operating,
            financing=financing,
            exit=exit_assumptions,
            waterfall=waterfall,
            covenants=covenants,
            **data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Deal:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _tuples(payload: dict[str, Any]) -> dict[str, Any]:
    """YAML gives lists; the dataclasses are frozen, so sequences become tuples."""
    return {key: tuple(value) if isinstance(value, list) and key != "tranches" else value for key, value in payload.items()}
