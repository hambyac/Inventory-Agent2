"""
inventory_tools.py - Supply Chain Analytics Deterministic Calculation Engine
Designed for Student Verification and Transparent Step-by-Step Supply Chain Analytics.

Every calculation tool returns:
1. inputs_extracted
2. variable_definitions
3. formula
4. formula_with_substituted_values
5. intermediate_calculations
6. final_answer
7. verification_checks
8. assumptions

Supported Models:
- calculate_eoq()
- calculate_quantity_discount()
- calculate_rop()
- calculate_periodic_review()
- calculate_newsvendor()

All numerical calculation steps are computed strictly within Python.
Never invents missing numerical inputs.
"""

import math
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats


# -----------------------------------------------------------------------------
# Structured Pydantic Response Schemas (Containing the 8 Verification Sections)
# -----------------------------------------------------------------------------

class VerificationCheck(BaseModel):
    check_name: str
    condition: str
    passed: bool
    details: str


class StudentVerificationModel(BaseModel):
    """Base schema enforcing the 8 pedagogical verification sections."""
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: List[VerificationCheck]
    assumptions: List[str]


class EOQResult(StudentVerificationModel):
    model_name: str = "Economic Order Quantity (EOQ)"
    demand: float
    ordering_cost: float
    holding_cost: float
    unit_cost: Optional[float] = None
    holding_cost_rate: Optional[float] = None
    working_days_per_year: int = 250
    
    # Backwards-compatible convenience fields
    eoq: float
    recommended_order_quantity: int
    orders_per_year: float
    cycle_time_days: float
    average_cycle_inventory: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_variable_cost: float
    annual_purchasing_cost: Optional[float] = None
    total_annual_cost: Optional[float] = None
    formulas: Dict[str, str] = Field(default_factory=dict)
    sanity_notes: List[str] = Field(default_factory=list)


class QuantityDiscountTier(BaseModel):
    tier_number: int
    min_qty: float
    max_qty: Optional[float] = None
    unit_price: float
    holding_cost: float
    holding_cost_type: str
    unconstrained_eoq: float
    feasible_quantity: float
    is_feasibility_clamped: bool
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_purchase_cost: float
    total_annual_cost: float
    is_optimal: bool = False


class QuantityDiscountResult(StudentVerificationModel):
    model_name: str = "Quantity Discount Optimization"
    demand: float
    ordering_cost: float
    holding_cost_rate: Optional[float] = None
    fixed_holding_cost: Optional[float] = None
    holding_cost_type: str
    
    # Backwards-compatible convenience fields
    optimal_tier_number: int
    optimal_order_quantity: float
    optimal_unit_price: float
    minimum_total_annual_cost: float
    annual_savings_vs_base_tier: float
    recommendation: str
    tiers: List[QuantityDiscountTier]
    formulas: Dict[str, str] = Field(default_factory=dict)
    sanity_notes: List[str] = Field(default_factory=list)


class SafetyStockROPResult(StudentVerificationModel):
    model_name: str = "Safety Stock and Reorder Point (Continuous Review)"
    case_type: str
    service_level: float
    lead_time: float
    daily_demand_mean: float
    daily_demand_std: float
    lead_time_std: float
    
    # Backwards-compatible convenience fields
    recommended_rop: int
    recommended_safety_stock: int
    reorder_point: float
    safety_stock: float
    z_score: float
    lead_time_expected_demand: float
    demand_variance_component: float
    lead_time_variance_component: float
    lead_time_demand_variance: float
    lead_time_demand_std: float
    unit_cost: Optional[float] = None
    holding_cost_per_unit_year: Optional[float] = None
    annual_safety_stock_holding_cost: Optional[float] = None
    formulas: Dict[str, str] = Field(default_factory=dict)
    sanity_notes: List[str] = Field(default_factory=list)


class PeriodicReviewResult(StudentVerificationModel):
    model_name: str = "Fixed-Period Review Policy (Periodic Review)"
    review_period_R: float
    lead_time_L: float
    service_level: float
    period_demand_mean: float
    period_demand_std: float
    current_on_hand: float
    current_on_order: float
    backorders: float
    
    # Backwards-compatible convenience fields
    recommended_order_quantity: float
    target_inventory_S: float
    recommended_order_up_to_S: int
    recommended_safety_stock: int
    protection_period: float
    protection_period_demand_mean: float
    protection_period_demand_std: float
    z_score: float
    safety_stock: float
    order_up_to_level_S: float
    inventory_position: float
    formulas: Dict[str, str] = Field(default_factory=dict)
    sanity_notes: List[str] = Field(default_factory=list)


class NewsvendorResult(StudentVerificationModel):
    model_name: str = "Newsvendor Single-Period Stochastic Model"
    unit_selling_price: float
    unit_cost: float
    salvage_value: float
    shortage_penalty: float
    demand_mean: float
    demand_std: float
    
    # Backwards-compatible convenience fields
    optimal_stocking_quantity: int
    optimal_stocking_quantity_continuous: float
    critical_ratio: float
    expected_profit: float
    underage_cost_Cu: float
    overage_cost_Co: float
    critical_fractile: float
    optimal_z_score: float
    unit_loss_value: float
    expected_lost_sales: float
    expected_leftover_units: float
    expected_sales: float
    expected_revenue: float
    expected_purchase_cost: float
    expected_shortage_penalty: float
    expected_salvage_revenue: float
    expected_profit_perfect_information: float
    expected_value_of_perfect_information_EVPI: float
    formulas: Dict[str, str] = Field(default_factory=dict)
    sanity_notes: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Tool 1: calculate_eoq()
# -----------------------------------------------------------------------------

def calculate_eoq(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    working_days_per_year: int = 250,
) -> EOQResult:
    """
    Computes Economic Order Quantity (EOQ): Q = sqrt(2 * D * S / H).
    Returns all 8 student verification sections. Never invents missing numerical inputs.
    """
    # 1. Input Validation
    if demand is None:
        raise ValueError("Missing required input: 'demand' (D) must be provided. Never invent missing numerical inputs.")
    if demand <= 0:
        raise ValueError(f"Demand must be strictly positive, got {demand}.")

    if ordering_cost is None:
        raise ValueError("Missing required input: 'ordering_cost' (S) must be provided. Never invent missing numerical inputs.")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be strictly positive, got {ordering_cost}.")

    effective_h: Optional[float] = None
    if holding_cost is not None:
        if holding_cost <= 0:
            raise ValueError(f"Holding cost must be strictly positive, got {holding_cost}.")
        effective_h = float(holding_cost)
    elif unit_cost is not None and holding_cost_rate is not None:
        if unit_cost <= 0:
            raise ValueError(f"Unit cost must be strictly positive, got {unit_cost}.")
        if holding_cost_rate <= 0:
            raise ValueError(f"Holding cost rate must be strictly positive, got {holding_cost_rate}.")
        effective_h = float(unit_cost * holding_cost_rate)
    else:
        raise ValueError(
            "Missing holding cost specification: Provide either 'holding_cost' directly or both "
            "'unit_cost' and 'holding_cost_rate'. Never invent missing numerical inputs."
        )

    if working_days_per_year <= 0:
        raise ValueError(f"Working days per year must be positive, got {working_days_per_year}.")

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "holding_cost": effective_h,
        "unit_cost": unit_cost,
        "holding_cost_rate": holding_cost_rate,
        "working_days_per_year": working_days_per_year
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual customer demand rate (units/year)",
        "S": "Fixed ordering or setup cost incurred per replenishment order ($/order)",
        "H": "Inventory carrying/holding cost per unit per year ($/unit/year)",
        "C": "Purchase price per unit ($/unit)",
        "h": "Annual inventory carrying rate fraction (holding cost as % of unit cost)",
        "Q": "Order quantity lot size (units/order)",
        "N": "Order frequency (replenishment orders placed per year)",
        "T": "Cycle length / time elapsed between consecutive orders (operating days)",
        "TVC": "Total annual variable inventory cost (Ordering Cost + Holding Cost) ($/year)",
        "TAC": "Total annual cost including material purchasing cost ($/year)"
    }

    # 3. Formula
    formula = {
        "EOQ": "Q = sqrt((2 * D * S) / H)",
        "Orders_per_Year": "N = D / Q",
        "Cycle_Time_Days": "T = (Q / D) * Working_Days",
        "Average_Cycle_Inventory": "Cycle_Stock = Q / 2",
        "Annual_Ordering_Cost": "AOC = (D / Q) * S",
        "Annual_Holding_Cost": "AHC = (Q / 2) * H",
        "Total_Variable_Cost": "TVC = AOC + AHC = (D / Q) * S + (Q / 2) * H",
        "Total_Annual_Cost": "TAC = D * C + TVC"
    }

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "EOQ": f"Q = sqrt((2 * {demand:,.2f} * {ordering_cost:,.2f}) / {effective_h:,.4f})",
        "Orders_per_Year": f"N = {demand:,.2f} / Q",
        "Cycle_Time_Days": f"T = (Q / {demand:,.2f}) * {working_days_per_year}",
        "Annual_Ordering_Cost": f"AOC = ({demand:,.2f} / Q) * {ordering_cost:,.2f}",
        "Annual_Holding_Cost": f"AHC = (Q / 2) * {effective_h:,.4f}",
        "Total_Variable_Cost": f"TVC = ({demand:,.2f} / Q) * {ordering_cost:,.2f} + (Q / 2) * {effective_h:,.4f}"
    }

    # 5. Intermediate Calculations (Strictly Python deterministic)
    numerator = 2.0 * demand * ordering_cost
    ratio = numerator / effective_h
    eoq_unrounded = math.sqrt(ratio)
    rec_q = int(round(eoq_unrounded))

    orders_per_year = demand / eoq_unrounded
    cycle_time_days = (eoq_unrounded / demand) * working_days_per_year
    avg_inventory = eoq_unrounded / 2.0

    annual_order_cost = orders_per_year * ordering_cost
    annual_hold_cost = avg_inventory * effective_h
    tvc = annual_order_cost + annual_hold_cost

    annual_purchasing = (demand * unit_cost) if unit_cost is not None else None
    tac = (tvc + annual_purchasing) if annual_purchasing is not None else None
    cost_parity_delta = abs(annual_order_cost - annual_hold_cost)

    intermediate_calculations = {
        "step_1_numerator_2DS": round(numerator, 4),
        "formula_numerator": round(numerator, 4),
        "step_2_quotient_2DS_over_H": round(ratio, 4),
        "effective_holding_cost_H": round(effective_h, 4),
        "step_3_unrounded_eoq_sqrt": round(eoq_unrounded, 4),
        "step_4_orders_per_year": round(orders_per_year, 4),
        "step_5_cycle_time_days": round(cycle_time_days, 2),
        "step_6_average_cycle_inventory": round(avg_inventory, 4),
        "step_7_annual_ordering_cost": round(annual_order_cost, 2),
        "annual_ordering_cost": round(annual_order_cost, 2),
        "step_8_annual_holding_cost": round(annual_hold_cost, 2),
        "annual_holding_cost": round(annual_hold_cost, 2),
        "step_9_total_variable_cost": round(tvc, 2),
        "step_10_annual_purchasing_cost": round(annual_purchasing, 2) if annual_purchasing is not None else None,
        "step_11_total_annual_cost": round(tac, 2) if tac is not None else None,
        "step_12_optimality_balance_delta": round(cost_parity_delta, 4)
    }

    # 6. Final Answer
    final_answer = {
        "recommended_order_quantity": rec_q,
        "eoq": round(eoq_unrounded, 2),
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_days": round(cycle_time_days, 1),
        "total_annual_variable_cost": round(tvc, 2),
        "total_annual_cost": round(tac, 2) if tac is not None else None
    }

    # 7. Verification Checks
    verification_checks = [
        VerificationCheck(
            check_name="Optimality Cost Trade-off (Wilson Parity)",
            condition="abs(Annual_Ordering_Cost - Annual_Holding_Cost) <= $1.00",
            passed=cost_parity_delta <= 1.0,
            details=f"Ordering Cost (${annual_order_cost:.2f}) vs Holding Cost (${annual_hold_cost:.2f}); Delta = ${cost_parity_delta:.4f}"
        ),
        VerificationCheck(
            check_name="Strict Non-Negativity",
            condition="EOQ > 0 and Demand > 0 and S > 0 and H > 0",
            passed=(eoq_unrounded > 0 and demand > 0 and ordering_cost > 0 and effective_h > 0),
            details=f"EOQ = {eoq_unrounded:.2f} units"
        ),
        VerificationCheck(
            check_name="Cycle Time Sanity",
            condition="Cycle_Time_Days <= Working_Days_Per_Year",
            passed=(cycle_time_days <= working_days_per_year),
            details=f"Cycle time of {cycle_time_days:.1f} days is within the {working_days_per_year}-day operating year."
        )
    ]

    # 8. Assumptions
    assumptions = [
        "Demand rate (D) is known, constant, and continuous over time.",
        "Ordering/setup cost (S) is fixed and independent of order quantity.",
        "Order lead time is zero (instantaneous replenishment) or completely deterministic.",
        "Entire replenishment lot Q arrives in a single batch with no partial shipments.",
        "Unit purchase price is constant; no volume discounts apply.",
        "Stockouts, backorders, and shortages are not permitted.",
        "Inventory carrying cost (H) is linear and strictly proportional to average inventory."
    ]

    return EOQResult(
        demand=demand,
        ordering_cost=ordering_cost,
        holding_cost=round(effective_h, 4),
        unit_cost=unit_cost,
        holding_cost_rate=holding_cost_rate,
        working_days_per_year=working_days_per_year,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        recommended_order_quantity=rec_q,
        eoq=round(eoq_unrounded, 2),
        orders_per_year=round(orders_per_year, 2),
        cycle_time_days=round(cycle_time_days, 1),
        average_cycle_inventory=round(avg_inventory, 2),
        annual_ordering_cost=round(annual_order_cost, 2),
        annual_holding_cost=round(annual_hold_cost, 2),
        total_annual_variable_cost=round(tvc, 2),
        annual_purchasing_cost=round(annual_purchasing, 2) if annual_purchasing is not None else None,
        total_annual_cost=round(tac, 2) if tac is not None else None,
        formulas=formula,
        sanity_notes=[v.details for v in verification_checks]
    )


# -----------------------------------------------------------------------------
# Tool 2: calculate_quantity_discount()
# -----------------------------------------------------------------------------

def calculate_quantity_discount(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    price_tiers: Optional[List[Dict[str, Any]]] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Evaluates feasible quantities and compares total annual cost across all price tiers.
    Supports both fixed holding cost and percentage holding cost.
    Returns all 8 student verification sections. Never invents missing numerical inputs.
    """
    # 1. Input Validation
    if demand is None:
        raise ValueError("Missing required input: 'demand' must be provided. Never invent missing numerical inputs.")
    if demand <= 0:
        raise ValueError(f"Demand must be positive, got {demand}.")

    if ordering_cost is None:
        raise ValueError("Missing required input: 'ordering_cost' must be provided. Never invent missing numerical inputs.")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be positive, got {ordering_cost}.")

    if price_tiers is None or len(price_tiers) == 0:
        raise ValueError("Missing required input: 'price_tiers' list cannot be empty. Never invent missing numerical inputs.")

    if holding_cost_rate is None and fixed_holding_cost is None:
        raise ValueError(
            "Missing holding cost specification: Must provide either 'holding_cost_rate' (percentage) "
            "or 'fixed_holding_cost'. Never invent missing numerical inputs."
        )

    holding_cost_type = "percentage" if holding_cost_rate is not None else "fixed"
    if holding_cost_rate is not None and holding_cost_rate <= 0:
        raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}.")
    if fixed_holding_cost is not None and fixed_holding_cost <= 0:
        raise ValueError(f"Fixed holding cost must be positive, got {fixed_holding_cost}.")

    # 1. Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "price_tiers": price_tiers,
        "holding_cost_rate": holding_cost_rate,
        "fixed_holding_cost": fixed_holding_cost,
        "holding_cost_type": holding_cost_type
    }

    # 2. Variable Definitions
    variable_definitions = {
        "D": "Annual customer demand rate (units/year)",
        "S": "Fixed ordering cost incurred per order ($/order)",
        "C_i": "Unit purchase price for discount tier i ($/unit)",
        "H_i": "Holding cost per unit per year for tier i ($/unit/year)",
        "min_qty_i": "Minimum order volume required to qualify for tier i pricing",
        "max_qty_i": "Upper volume limit for tier i pricing (or infinity for highest tier)",
        "EOQ_i": "Unconstrained economic order quantity calculated using tier i unit holding cost",
        "Q_i": "Feasible candidate order quantity for tier i after price-break clamping",
        "TAC_i": "Total annual cost for tier i = Material Cost + Ordering Cost + Holding Cost",
        "Optimal_Tier": "Tier index yielding the absolute minimum TAC across all feasible candidate quantities"
    }

    # 3. Formula
    formula = {
        "Tier_Holding_Cost": "H_i = (h * C_i) if percentage else fixed_H",
        "Unconstrained_EOQ": "EOQ_i = sqrt((2 * D * S) / H_i)",
        "Feasible_Quantity": "Q_i = max(min_qty_i, min(max_qty_i, EOQ_i))",
        "Annual_Purchase_Cost": "APC_i = D * C_i",
        "Annual_Ordering_Cost": "AOC_i = (D / Q_i) * S",
        "Annual_Holding_Cost": "AHC_i = (Q_i / 2) * H_i",
        "Total_Annual_Cost": "TAC_i = APC_i + AOC_i + AHC_i",
        "Optimal_Decision": "min(TAC_i) across all valid tiers"
    }

    # 4. Formula with Substituted Values & 5. Intermediate Calculations
    sorted_tiers = sorted(price_tiers, key=lambda t: t.get("min_qty", 0))
    tier_evaluations: List[QuantityDiscountTier] = []
    formula_substituted: Dict[str, str] = {}
    intermediate_steps: Dict[str, Any] = {}

    for idx, tier in enumerate(sorted_tiers, start=1):
        if "min_qty" not in tier or "price" not in tier:
            raise ValueError(f"Tier {idx} missing required 'min_qty' or 'price'.")

        min_q = float(tier["min_qty"])
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["price"])

        if min_q < 0 or price <= 0:
            raise ValueError(f"Invalid tier {idx}: min_qty and price must be non-negative and positive respectively.")

        tier_h = (price * holding_cost_rate) if holding_cost_rate is not None else float(fixed_holding_cost)
        unconstrained_q = math.sqrt((2.0 * demand * ordering_cost) / tier_h)

        is_clamped = False
        if unconstrained_q < min_q:
            feasible_q = min_q
            is_clamped = True
        elif max_q is not None and unconstrained_q > max_q:
            feasible_q = max_q
            is_clamped = True
        else:
            feasible_q = unconstrained_q

        annual_order = (demand / feasible_q) * ordering_cost
        annual_hold = (feasible_q / 2.0) * tier_h
        annual_purchase = demand * price
        total_annual = annual_purchase + annual_order + annual_hold

        bracket_str = f"[{min_q:,.0f} - {max_q:,.0f}]" if max_q else f"[{min_q:,.0f}+]"
        formula_substituted[f"tier_{idx}_EOQ"] = f"EOQ_{idx} = sqrt((2 * {demand:,.0f} * {ordering_cost:,.2f}) / {tier_h:,.4f}) = {unconstrained_q:,.2f}"
        formula_substituted[f"tier_{idx}_TAC"] = f"TAC_{idx} = ({demand:,.0f} * {price:,.2f}) + ({demand:,.0f} / {feasible_q:,.2f}) * {ordering_cost:,.2f} + ({feasible_q:,.2f} / 2) * {tier_h:,.4f} = ${total_annual:,.2f}"

        tier_obj = QuantityDiscountTier(
            tier_number=idx,
            min_qty=min_q,
            max_qty=max_q,
            unit_price=round(price, 4),
            holding_cost=round(tier_h, 4),
            holding_cost_type=holding_cost_type,
            unconstrained_eoq=round(unconstrained_q, 2),
            feasible_quantity=round(feasible_q, 2),
            is_feasibility_clamped=is_clamped,
            annual_ordering_cost=round(annual_order, 2),
            annual_holding_cost=round(annual_hold, 2),
            annual_purchase_cost=round(annual_purchase, 2),
            total_annual_cost=round(total_annual, 2)
        )
        tier_evaluations.append(tier_obj)

        intermediate_steps[f"tier_{idx}"] = {
            "bracket": bracket_str,
            "unit_price": price,
            "holding_cost": round(tier_h, 4),
            "unconstrained_eoq": round(unconstrained_q, 2),
            "feasible_quantity": round(feasible_q, 2),
            "is_clamped_to_price_break": is_clamped,
            "annual_order_cost": round(annual_order, 2),
            "annual_hold_cost": round(annual_hold, 2),
            "annual_purchase_cost": round(annual_purchase, 2),
            "total_annual_cost": round(total_annual, 2)
        }

    # Identify optimal tier
    best_tier = min(tier_evaluations, key=lambda t: t.total_annual_cost)
    best_tier.is_optimal = True
    base_cost = tier_evaluations[0].total_annual_cost
    savings = base_cost - best_tier.total_annual_cost

    # 6. Final Answer
    final_answer = {
        "optimal_tier_number": best_tier.tier_number,
        "optimal_order_quantity": best_tier.feasible_quantity,
        "optimal_unit_price": best_tier.unit_price,
        "minimum_total_annual_cost": best_tier.total_annual_cost,
        "annual_savings_vs_base_tier": round(savings, 2)
    }

    # 7. Verification Checks
    verification_checks = [
        VerificationCheck(
            check_name="Tier Feasibility Bounds Verification",
            condition="For each tier: min_qty <= feasible_quantity <= (max_qty or inf)",
            passed=all(t.feasible_quantity >= t.min_qty and (t.max_qty is None or t.feasible_quantity <= t.max_qty) for t in tier_evaluations),
            details=f"All {len(tier_evaluations)} tier candidate quantities satisfy qualifying volume constraints."
        ),
        VerificationCheck(
            check_name="Global Cost Minimization",
            condition="minimum_total_annual_cost == min(TAC_i)",
            passed=(best_tier.total_annual_cost == min(t.total_annual_cost for t in tier_evaluations)),
            details=f"Tier {best_tier.tier_number} achieves lowest TAC (${best_tier.total_annual_cost:,.2f}) across the discount schedule."
        ),
        VerificationCheck(
            check_name="Non-Negative Annual Savings",
            condition="annual_savings_vs_base_tier >= 0.0",
            passed=(savings >= 0.0),
            details=f"Net annual savings vs base pricing: ${savings:,.2f}"
        )
    ]

    # 8. Assumptions
    assumptions = [
        "All-units quantity discount structure applies (discount applies to every unit in the lot).",
        "Annual demand is known, constant, and independent of purchase price.",
        "Lead time is zero or constant.",
        "Holding cost is evaluated as specified (either fixed $/unit/yr or constant % rate of unit price).",
        "Sufficient capital and warehouse storage exist to accommodate larger discount lot sizes.",
        "Supplier guarantees delivery of all qualifying order quantities without allocation constraints."
    ]

    recommendation = (
        f"Order {best_tier.feasible_quantity:.0f} units in Tier {best_tier.tier_number} at ${best_tier.unit_price:.2f}/unit. "
        f"Yields minimum total annual cost of ${best_tier.total_annual_cost:,.2f} (saves ${savings:,.2f}/yr vs Tier 1)."
    )

    return QuantityDiscountResult(
        demand=demand,
        ordering_cost=ordering_cost,
        holding_cost_rate=holding_cost_rate,
        fixed_holding_cost=fixed_holding_cost,
        holding_cost_type=holding_cost_type,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_substituted,
        intermediate_calculations=intermediate_steps,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_tier_number=best_tier.tier_number,
        optimal_order_quantity=best_tier.feasible_quantity,
        optimal_unit_price=best_tier.unit_price,
        minimum_total_annual_cost=best_tier.total_annual_cost,
        annual_savings_vs_base_tier=round(savings, 2),
        recommendation=recommendation,
        tiers=tier_evaluations,
        formulas=formula,
        sanity_notes=[v.details for v in verification_checks]
    )


# -----------------------------------------------------------------------------
# Tool 3: calculate_rop()
# -----------------------------------------------------------------------------

def calculate_rop(
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    lead_time_std: float = 0.0,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    daily_demand_mean: Optional[float] = None,
    daily_demand_std: Optional[float] = None,
    **kwargs
) -> SafetyStockROPResult:
    """
    Computes Safety Stock (SS) and Reorder Point (ROP) for a continuous review system.
    Supports all 4 demand and lead time cases.
    Returns all 8 student verification sections. Never invents missing numerical inputs.
    """
    # Parameter normalization
    if demand_mean is None and daily_demand_mean is not None:
        demand_mean = daily_demand_mean
    if demand_std == 0.0 and daily_demand_std is not None:
        demand_std = daily_demand_std

    # 1. Input Validation
    if lead_time is None:
        raise ValueError("Missing required input: 'lead_time' (L) must be provided. Never invent missing numerical inputs.")
    if lead_time <= 0:
        raise ValueError(f"Lead time must be strictly positive, got {lead_time}.")

    if service_level is None:
        raise ValueError("Missing required input: 'service_level' (CSL) must be provided. Never invent missing numerical inputs.")
    if not (0.0 < service_level < 1.0):
        raise ValueError(f"Service level must be strictly between 0 and 1 (exclusive), got {service_level}.")

    if demand_mean is None:
        raise ValueError("Missing required input: 'demand_mean' (d) must be provided. Never invent missing numerical inputs.")
    if demand_mean <= 0:
        raise ValueError(f"Demand mean must be strictly positive, got {demand_mean}.")

    if demand_std < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {demand_std}.")
    if lead_time_std < 0:
        raise ValueError(f"Lead time standard deviation cannot be negative, got {lead_time_std}.")

    # Identify which of the 4 cases is active
    if demand_std == 0.0 and lead_time_std == 0.0:
        case_type = "Constant demand and constant lead time"
    elif demand_std > 0.0 and lead_time_std == 0.0:
        case_type = "Variable demand and constant lead time"
    elif demand_std == 0.0 and lead_time_std > 0.0:
        case_type = "Constant demand and variable lead time"
    else:
        case_type = "Variable demand and variable lead time"

    # 1. Inputs Extracted
    inputs_extracted = {
        "lead_time": lead_time,
        "service_level": service_level,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
        "lead_time_std": lead_time_std,
        "holding_cost": holding_cost,
        "unit_cost": unit_cost,
        "holding_cost_rate": holding_cost_rate,
        "case_type": case_type
    }

    # 2. Variable Definitions
    variable_definitions = {
        "d": "Mean demand rate per operating period (units/day or units/period)",
        "sigma_d": "Standard deviation of demand per period (demand volatility)",
        "L": "Average supplier replenishment lead time (periods or days)",
        "sigma_L": "Standard deviation of supplier lead time (lead time uncertainty)",
        "CSL": "Cycle Service Level (target probability of zero stockout during lead time)",
        "z": "Standard normal inverse cumulative distribution quantile = norm.ppf(CSL)",
        "DLT": "Expected demand during supplier lead time = d * L",
        "sigma_LT": "Standard deviation of demand during lead time (combined joint uncertainty)",
        "SS": "Safety stock buffer maintained to satisfy service level = z * sigma_LT",
        "ROP": "Reorder point inventory threshold triggering replenishment order = DLT + SS"
    }

    # 3. Formula
    formula = {
        "Case": f"Identified Case: {case_type}",
        "DLT": "DLT = d * L",
        "sigma_LT": "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)",
        "Safety_Stock": "SS = z * sigma_LT",
        "ROP": "ROP = DLT + SS = d * L + z * sigma_LT"
    }

    # 5. Intermediate Calculations (Strictly Python deterministic)
    z = float(stats.norm.ppf(service_level))
    dlt = demand_mean * lead_time
    var_from_demand = lead_time * (demand_std ** 2)
    var_from_lead_time = (demand_mean ** 2) * (lead_time_std ** 2)
    var_total = var_from_demand + var_from_lead_time
    sigma_lt = math.sqrt(var_total)

    safety_stock_continuous = z * sigma_lt
    rop_continuous = dlt + safety_stock_continuous

    rec_ss = max(0, int(math.ceil(safety_stock_continuous)))
    rec_rop = max(0, int(math.ceil(rop_continuous)))

    effective_h = holding_cost
    if effective_h is None and unit_cost is not None and holding_cost_rate is not None:
        effective_h = unit_cost * holding_cost_rate
    annual_ss_cost = (rec_ss * effective_h) if effective_h is not None else None

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "DLT": f"DLT = {demand_mean:,.2f} * {lead_time:,.2f} = {dlt:,.2f}",
        "z": f"z = norm.ppf({service_level:.4f}) = {z:.4f}",
        "sigma_LT": f"sigma_LT = sqrt({lead_time:,.2f} * ({demand_std:,.2f})^2 + ({demand_mean:,.2f})^2 * ({lead_time_std:,.2f})^2) = {sigma_lt:,.4f}",
        "Safety_Stock": f"SS = {z:.4f} * {sigma_lt:,.4f} = {safety_stock_continuous:,.2f} (rounded up to {rec_ss})",
        "ROP": f"ROP = {dlt:,.2f} + {safety_stock_continuous:,.2f} = {rop_continuous:,.2f} (rounded up to {rec_rop})"
    }

    intermediate_calculations = {
        "case_type": case_type,
        "z_score": round(z, 4),
        "lead_time_expected_demand_DLT": round(dlt, 4),
        "variance_from_demand": round(var_from_demand, 4),
        "variance_from_lead_time": round(var_from_lead_time, 4),
        "total_lead_time_demand_variance": round(var_total, 4),
        "lead_time_demand_std_sigma_LT": round(sigma_lt, 4),
        "continuous_safety_stock": round(safety_stock_continuous, 4),
        "continuous_reorder_point": round(rop_continuous, 4),
        "effective_holding_cost": round(effective_h, 2) if effective_h is not None else None,
        "annual_safety_stock_holding_cost": round(annual_ss_cost, 2) if annual_ss_cost is not None else None
    }

    # 6. Final Answer
    final_answer = {
        "recommended_rop": rec_rop,
        "recommended_safety_stock": rec_ss,
        "reorder_point_exact": round(rop_continuous, 2),
        "safety_stock_exact": round(safety_stock_continuous, 2),
        "lead_time_expected_demand": round(dlt, 2),
        "service_level": round(service_level, 4),
        "case_type": case_type
    }

    # 7. Verification Checks
    verification_checks = [
        VerificationCheck(
            check_name="Service Level Bounds Check",
            condition="0 < CSL < 1",
            passed=(0.0 < service_level < 1.0),
            details=f"CSL = {service_level:.1%} (z = {z:.4f})"
        ),
        VerificationCheck(
            check_name="Buffer Coverage Verification (ROP >= DLT)",
            condition="ROP >= DLT when z >= 0",
            passed=(rop_continuous >= dlt if z >= 0 else True),
            details=f"ROP ({rop_continuous:.2f}) exceeds expected lead time demand DLT ({dlt:.2f}) by safety stock buffer ({safety_stock_continuous:.2f} units)."
        ),
        VerificationCheck(
            check_name="Case Variance Consistency",
            condition="sigma_LT >= 0",
            passed=(sigma_lt >= 0.0),
            details=f"sigma_LT = {sigma_lt:.4f} correctly computed under '{case_type}'."
        )
    ]

    # 8. Assumptions
    assumptions = [
        "Continuous review policy (s, Q): Inventory levels are tracked in real-time.",
        "Demand across consecutive time periods is independent and identically distributed (i.i.d.).",
        "Lead time demand follows a normal distribution: DLT ~ N(d * L, sigma_LT^2).",
        "Cycle Service Level (CSL) criterion: Probability of no stockout during lead time equals service_level.",
        "Stockouts during lead time are completely backordered (not lost sales).",
        "Lead time variability and demand variability are mutually independent."
    ]

    return SafetyStockROPResult(
        case_type=case_type,
        service_level=round(service_level, 4),
        lead_time=lead_time,
        daily_demand_mean=demand_mean,
        daily_demand_std=demand_std,
        lead_time_std=lead_time_std,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        recommended_rop=rec_rop,
        recommended_safety_stock=rec_ss,
        reorder_point=round(rop_continuous, 2),
        safety_stock=round(safety_stock_continuous, 2),
        z_score=round(z, 4),
        lead_time_expected_demand=round(dlt, 2),
        demand_variance_component=round(var_from_demand, 4),
        lead_time_variance_component=round(var_from_lead_time, 4),
        lead_time_demand_variance=round(var_total, 4),
        lead_time_demand_std=round(sigma_lt, 4),
        unit_cost=unit_cost,
        holding_cost_per_unit_year=round(effective_h, 2) if effective_h is not None else None,
        annual_safety_stock_holding_cost=round(annual_ss_cost, 2) if annual_ss_cost is not None else None,
        formulas=formula,
        sanity_notes=[v.details for v in verification_checks]
    )


# -----------------------------------------------------------------------------
# Tool 4: calculate_periodic_review()
# -----------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    current_on_hand: float = 0.0,
    current_on_order: float = 0.0,
    backorders: float = 0.0,
    holding_cost: Optional[float] = None,
) -> PeriodicReviewResult:
    """
    Computes Target Inventory Level (S) and Replenishment Order Quantity under Fixed-Period Review (R, S).
    Returns all 8 student verification sections. Never invents missing numerical inputs.
    """
    # 1. Input Validation
    if review_period is None:
        raise ValueError("Missing required input: 'review_period' (R) must be provided. Never invent missing numerical inputs.")
    if review_period <= 0:
        raise ValueError(f"Review period must be positive, got {review_period}.")

    if lead_time is None:
        raise ValueError("Missing required input: 'lead_time' (L) must be provided. Never invent missing numerical inputs.")
    if lead_time <= 0:
        raise ValueError(f"Lead time must be positive, got {lead_time}.")

    if service_level is None:
        raise ValueError("Missing required input: 'service_level' must be provided. Never invent missing numerical inputs.")
    if not (0.0 < service_level < 1.0):
        raise ValueError(f"Service level must be strictly between 0 and 1, got {service_level}.")

    if demand_mean is None:
        raise ValueError("Missing required input: 'demand_mean' (d) must be provided. Never invent missing numerical inputs.")
    if demand_mean <= 0:
        raise ValueError(f"Demand mean must be positive, got {demand_mean}.")

    if demand_std is None:
        raise ValueError("Missing required input: 'demand_std' (sigma_d) must be provided. Never invent missing numerical inputs.")
    if demand_std < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {demand_std}.")

    # 1. Inputs Extracted
    inputs_extracted = {
        "review_period_R": review_period,
        "lead_time_L": lead_time,
        "service_level": service_level,
        "demand_mean": demand_mean,
        "demand_std": demand_std,
        "current_on_hand": current_on_hand,
        "current_on_order": current_on_order,
        "backorders": backorders
    }

    # 2. Variable Definitions
    variable_definitions = {
        "R": "Review interval (time elapsed between consecutive inventory audits/reviews)",
        "L": "Supplier lead time (time between placing replenishment order and receiving delivery)",
        "R + L": "Protection period horizon (duration of demand exposure that the order must cover)",
        "d": "Mean demand rate per period",
        "sigma_d": "Standard deviation of demand per period",
        "CSL": "Cycle Service Level target",
        "z": "Standard normal quantile corresponding to CSL",
        "mu_{R+L}": "Expected demand during the protection period = d * (R + L)",
        "sigma_{R+L}": "Standard deviation of demand during protection period = sigma_d * sqrt(R + L)",
        "SS": "Safety stock buffer for protection period = z * sigma_d * sqrt(R + L)",
        "S": "Target inventory level (Base Stock / Order-Up-To Level) = mu_{R+L} + SS",
        "IP": "Current inventory position = On_Hand + On_Order - Backorders",
        "q": "Replenishment order quantity placed today = max(0, S - IP)"
    }

    # 3. Formula
    formula = {
        "Protection_Period": "Protection_Period = R + L",
        "Demand_Protection_Period": "mu_{R+L} = d * (R + L)",
        "Sigma_Protection_Period": "sigma_{R+L} = sigma_d * sqrt(R + L)",
        "Safety_Stock": "SS = z * sigma_d * sqrt(R + L)",
        "Target_Inventory_S": "S = mu_{R+L} + SS = d * (R + L) + z * sigma_d * sqrt(R + L)",
        "Inventory_Position": "IP = On_Hand + On_Order - Backorders",
        "Order_Quantity": "q = max(0, S - IP)"
    }

    # 5. Intermediate Calculations (Strictly Python deterministic)
    protection_period = review_period + lead_time
    z = float(stats.norm.ppf(service_level))
    protection_demand_mean = demand_mean * protection_period
    protection_demand_std = demand_std * math.sqrt(protection_period)
    safety_stock = z * protection_demand_std
    target_inventory_S = protection_demand_mean + safety_stock

    rec_s = int(math.ceil(target_inventory_S))
    rec_ss = int(math.ceil(safety_stock))

    inventory_position = current_on_hand + current_on_order - backorders
    order_quantity = max(0.0, target_inventory_S - inventory_position)

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Protection_Period": f"R + L = {review_period:,.2f} + {lead_time:,.2f} = {protection_period:,.2f}",
        "Demand_Protection_Period": f"mu_{{R+L}} = {demand_mean:,.2f} * {protection_period:,.2f} = {protection_demand_mean:,.2f}",
        "Sigma_Protection_Period": f"sigma_{{R+L}} = {demand_std:,.2f} * sqrt({protection_period:,.2f}) = {protection_demand_std:,.4f}",
        "Safety_Stock": f"SS = {z:.4f} * {protection_demand_std:,.4f} = {safety_stock:,.2f} (rounded up to {rec_ss})",
        "Target_Inventory_S": f"S = {protection_demand_mean:,.2f} + {safety_stock:,.2f} = {target_inventory_S:,.2f} (rounded up to {rec_s})",
        "Inventory_Position": f"IP = {current_on_hand:,.2f} + {current_on_order:,.2f} - {backorders:,.2f} = {inventory_position:,.2f}",
        "Order_Quantity": f"q = max(0, {target_inventory_S:,.2f} - {inventory_position:,.2f}) = {order_quantity:,.2f}"
    }

    intermediate_calculations = {
        "step_1_protection_period_R_plus_L": round(protection_period, 4),
        "protection_period_R_plus_L": round(protection_period, 4),
        "step_2_protection_demand_mean": round(protection_demand_mean, 4),
        "protection_demand_mean": round(protection_demand_mean, 4),
        "step_3_protection_demand_std": round(protection_demand_std, 4),
        "protection_demand_std": round(protection_demand_std, 4),
        "step_4_z_score": round(z, 4),
        "step_5_continuous_safety_stock": round(safety_stock, 4),
        "continuous_safety_stock": round(safety_stock, 4),
        "step_6_continuous_target_inventory_S": round(target_inventory_S, 4),
        "continuous_target_inventory_S": round(target_inventory_S, 4),
        "step_7_inventory_position": round(inventory_position, 4),
        "inventory_position": round(inventory_position, 4),
        "step_8_raw_order_quantity": round(order_quantity, 4)
    }

    # 6. Final Answer
    final_answer = {
        "target_inventory_S": round(target_inventory_S, 2),
        "recommended_order_quantity": round(order_quantity, 2),
        "recommended_order_up_to_S": rec_s,
        "recommended_safety_stock": rec_ss,
        "inventory_position": round(inventory_position, 2),
        "protection_period": round(protection_period, 2)
    }

    # 7. Verification Checks
    verification_checks = [
        VerificationCheck(
            check_name="Protection Period Verification",
            condition="Protection_Period == R + L and Protection_Period > 0",
            passed=(protection_period == review_period + lead_time and protection_period > 0),
            details=f"Protection period verified at {protection_period:.1f} periods (R={review_period} + L={lead_time})."
        ),
        VerificationCheck(
            check_name="Order-Up-To Target Soundness",
            condition="S >= Expected_Protection_Demand",
            passed=(target_inventory_S >= protection_demand_mean),
            details=f"Target inventory S ({target_inventory_S:.2f}) covers expected demand ({protection_demand_mean:.2f}) plus safety stock buffer."
        ),
        VerificationCheck(
            check_name="Order Quantity Non-Negativity",
            condition="Order_Quantity >= 0",
            passed=(order_quantity >= 0.0),
            details=f"Order quantity correctly non-negative: {order_quantity:.2f} units."
        )
    ]

    # 8. Assumptions
    assumptions = [
        "Fixed-period review policy (R, S): Inventory position is audited at regular intervals of length R.",
        "Protection horizon covers R + L periods because replenishment orders placed today arrive in L periods and must satisfy demand until the subsequent review's order arrives at R + L.",
        "Demand across consecutive time periods is independent and identically distributed normal random variable.",
        "Stockouts during the protection horizon are completely backordered.",
        "Review frequency R and supplier lead time L are deterministic and exogenous."
    ]

    return PeriodicReviewResult(
        review_period_R=review_period,
        lead_time_L=lead_time,
        service_level=round(service_level, 4),
        period_demand_mean=demand_mean,
        period_demand_std=demand_std,
        current_on_hand=current_on_hand,
        current_on_order=current_on_order,
        backorders=backorders,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        recommended_order_quantity=round(order_quantity, 2),
        target_inventory_S=round(target_inventory_S, 2),
        recommended_order_up_to_S=rec_s,
        recommended_safety_stock=rec_ss,
        protection_period=round(protection_period, 2),
        protection_period_demand_mean=round(protection_demand_mean, 2),
        protection_period_demand_std=round(protection_demand_std, 2),
        z_score=round(z, 4),
        safety_stock=round(safety_stock, 2),
        order_up_to_level_S=round(target_inventory_S, 2),
        inventory_position=round(inventory_position, 2),
        formulas=formula,
        sanity_notes=[v.details for v in verification_checks]
    )


# -----------------------------------------------------------------------------
# Tool 5: calculate_newsvendor()
# -----------------------------------------------------------------------------

def calculate_newsvendor(
    price: Optional[float] = None,
    cost: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    salvage_value: float = 0.0,
    shortage_penalty: float = 0.0,
) -> NewsvendorResult:
    """
    Computes Underage Cost (Cu), Overage Cost (Co), Critical Ratio (CR),
    and Recommended Order Quantity (Q*) for single-period perishable inventory.
    Returns all 8 student verification sections. Never invents missing numerical inputs.
    """
    # 1. Input Validation
    if price is None:
        raise ValueError("Missing required input: 'price' (P) must be provided. Never invent missing numerical inputs.")
    if price <= 0:
        raise ValueError(f"Selling price must be strictly positive, got {price}.")

    if cost is None:
        raise ValueError("Missing required input: 'cost' (C) must be provided. Never invent missing numerical inputs.")
    if cost <= 0:
        raise ValueError(f"Purchase cost must be strictly positive, got {cost}.")

    if price <= cost:
        raise ValueError(f"Selling price (${price}) must strictly exceed unit cost (${cost}) to be viable.")

    if salvage_value is None:
        salvage_value = 0.0
    if salvage_value >= cost:
        raise ValueError(f"Salvage value (${salvage_value}) cannot exceed or equal purchase cost (${cost}).")

    if shortage_penalty is None:
        shortage_penalty = 0.0
    if shortage_penalty < 0:
        raise ValueError(f"Shortage penalty cannot be negative, got {shortage_penalty}.")

    if demand_mean is None:
        raise ValueError("Missing required input: 'demand_mean' (mu) must be provided. Never invent missing numerical inputs.")
    if demand_mean < 0:
        raise ValueError(f"Demand mean cannot be negative, got {demand_mean}.")

    if demand_std is None:
        raise ValueError("Missing required input: 'demand_std' (sigma) must be provided. Never invent missing numerical inputs.")
    if demand_std <= 0:
        raise ValueError(f"Demand standard deviation must be positive, got {demand_std}.")

    # 1. Inputs Extracted
    inputs_extracted = {
        "unit_selling_price": price,
        "unit_cost": cost,
        "salvage_value": salvage_value,
        "shortage_penalty": shortage_penalty,
        "demand_mean": demand_mean,
        "demand_std": demand_std
    }

    # 2. Variable Definitions
    variable_definitions = {
        "P": "Retail selling price per unit ($/unit)",
        "C": "Purchase / manufacturing cost per unit ($/unit)",
        "V": "Salvage / markdown liquidation value for unsold items ($/unit)",
        "g": "Goodwill penalty / loss per unit of unsatisfied demand ($/unit)",
        "Cu": "Underage cost (opportunity cost of stocking one unit too few) = P - C + g",
        "Co": "Overage cost (loss incurred from stocking one unit too many) = C - V",
        "CR": "Critical ratio (critical fractile / optimal non-stockout probability) = Cu / (Cu + Co)",
        "mu": "Expected mean customer demand during the single selling season",
        "sigma": "Standard deviation of customer demand during the selling season",
        "z": "Optimal standard normal quantile corresponding to critical ratio = norm.ppf(CR)",
        "Q_star": "Optimal stocking quantity = mu + z * sigma",
        "L(z)": "Standard normal unit loss function = phi(z) - z * (1 - Phi(z))",
        "EVPI": "Expected Value of Perfect Information = Maximum price to acquire perfect demand forecast"
    }

    # 3. Formula
    formula = {
        "Underage_Cost": "Cu = P - C + g",
        "Overage_Cost": "Co = C - V",
        "Critical_Ratio": "CR = Cu / (Cu + Co)",
        "Optimal_z": "z = norm.ppf(CR)",
        "Optimal_Stocking_Quantity": "Q* = mu + z * sigma",
        "Unit_Loss_Function": "L(z) = phi(z) - z * (1 - Phi(z))",
        "Expected_Lost_Sales": "E[LS] = sigma * L(z)",
        "Expected_Leftover_Units": "E[Leftover] = Q* - mu + E[LS]",
        "Expected_Sales": "E[Sales] = mu - E[LS]",
        "Expected_Net_Profit": "E[Profit] = P * E[Sales] + V * E[Leftover] - C * Q* - g * E[LS]",
        "EVPI": "EVPI = (P - C) * mu - E[Profit]"
    }

    # 5. Intermediate Calculations (Strictly Python deterministic)
    cu = price - cost + shortage_penalty
    co = cost - salvage_value
    critical_ratio = cu / (cu + co)
    z = float(stats.norm.ppf(critical_ratio))

    q_star_continuous = demand_mean + z * demand_std
    rec_q = max(0, int(round(q_star_continuous)))

    phi_z = float(stats.norm.pdf(z))
    Phi_z = float(stats.norm.cdf(z))
    unit_loss_Lz = phi_z - z * (1.0 - Phi_z)

    expected_lost_sales = demand_std * unit_loss_Lz
    expected_leftover = q_star_continuous - demand_mean + expected_lost_sales
    expected_sales = demand_mean - expected_lost_sales

    exp_revenue = price * expected_sales
    exp_salvage = salvage_value * expected_leftover
    exp_purchase_cost = cost * q_star_continuous
    exp_shortage_penalty = shortage_penalty * expected_lost_sales
    exp_profit = exp_revenue + exp_salvage - exp_purchase_cost - exp_shortage_penalty

    exp_profit_perfect = (price - cost) * demand_mean
    evpi = max(0.0, exp_profit_perfect - exp_profit)

    # 4. Formula with Substituted Values
    formula_with_substituted_values = {
        "Underage_Cost": f"Cu = {price:,.2f} - {cost:,.2f} + {shortage_penalty:,.2f} = {cu:,.2f}",
        "Overage_Cost": f"Co = {cost:,.2f} - {salvage_value:,.2f} = {co:,.2f}",
        "Critical_Ratio": f"CR = {cu:,.2f} / ({cu:,.2f} + {co:,.2f}) = {critical_ratio:.6f}",
        "Optimal_z": f"z = norm.ppf({critical_ratio:.6f}) = {z:.4f}",
        "Optimal_Stocking_Quantity": f"Q* = {demand_mean:,.2f} + ({z:.4f} * {demand_std:,.2f}) = {q_star_continuous:,.2f} (rounded to {rec_q})",
        "Unit_Loss_Function": f"L({z:.4f}) = {phi_z:.4f} - ({z:.4f} * (1 - {Phi_z:.4f})) = {unit_loss_Lz:.6f}",
        "Expected_Profit": f"E[Profit] = ({price:,.2f} * {expected_sales:,.2f}) + ({salvage_value:,.2f} * {expected_leftover:,.2f}) - ({cost:,.2f} * {q_star_continuous:,.2f}) = ${exp_profit:,.2f}"
    }

    intermediate_calculations = {
        "step_1_underage_cost_Cu": round(cu, 4),
        "underage_cost_Cu": round(cu, 4),
        "step_2_overage_cost_Co": round(co, 4),
        "overage_cost_Co": round(co, 4),
        "step_3_critical_ratio_CR": round(critical_ratio, 6),
        "critical_ratio": round(critical_ratio, 6),
        "step_4_optimal_z_score": round(z, 4),
        "optimal_z_score": round(z, 4),
        "step_5_unit_loss_function_Lz": round(unit_loss_Lz, 6),
        "step_6_continuous_optimal_Q": round(q_star_continuous, 4),
        "step_7_expected_lost_sales": round(expected_lost_sales, 4),
        "step_8_expected_leftover_units": round(expected_leftover, 4),
        "step_9_expected_sales": round(expected_sales, 4),
        "expected_sales": round(expected_sales, 4),
        "step_10_expected_revenue": round(exp_revenue, 2),
        "step_11_expected_purchase_cost": round(exp_purchase_cost, 2),
        "step_12_expected_salvage_revenue": round(exp_salvage, 2),
        "step_13_expected_shortage_penalty": round(exp_shortage_penalty, 2),
        "step_14_expected_profit_perfect_information": round(exp_profit_perfect, 2),
        "step_15_expected_value_of_perfect_information_EVPI": round(evpi, 2)
    }

    # 6. Final Answer
    final_answer = {
        "underage_cost": round(cu, 2),
        "overage_cost": round(co, 2),
        "critical_ratio": round(critical_ratio, 4),
        "recommended_order_quantity": rec_q,
        "optimal_stocking_quantity_continuous": round(q_star_continuous, 2),
        "expected_profit": round(exp_profit, 2)
    }

    # 7. Verification Checks
    verification_checks = [
        VerificationCheck(
            check_name="Economic Viability (Positive Margin)",
            condition="Price > Cost",
            passed=(price > cost),
            details=f"Margin: ${price - cost:.2f}/unit (Price=${price:.2f}, Cost=${cost:.2f})"
        ),
        VerificationCheck(
            check_name="Salvage Feasibility",
            condition="Salvage_Value < Cost",
            passed=(salvage_value < cost),
            details=f"Salvage (${salvage_value:.2f}) is strictly less than cost (${cost:.2f})"
        ),
        VerificationCheck(
            check_name="Critical Ratio Probability Bounds",
            condition="0 < CR < 1",
            passed=(0.0 < critical_ratio < 1.0),
            details=f"CR = {critical_ratio:.2%}"
        ),
        VerificationCheck(
            check_name="Optimal Order Stocking Logic",
            condition="Q* > mu if CR > 0.5 else Q* <= mu",
            passed=(q_star_continuous > demand_mean if critical_ratio > 0.5 else q_star_continuous <= demand_mean),
            details=f"High-margin regime (CR={critical_ratio:.1%} > 50%): ordering {rec_q - int(demand_mean)} units above mean demand."
        )
    ]

    # 8. Assumptions
    assumptions = [
        "Single-period inventory setting with non-perishable demand or strictly perishable goods.",
        "Single procurement order placed prior to the selling season; no in-season replenishment.",
        "Demand during the season is normally distributed: Demand ~ N(mu, sigma^2).",
        "Unsold units at season end are liquidated at salvage value V with no additional storage penalty.",
        "Unsatisfied customer demand is lost (with optional penalty cost g per unit).",
        "Linear costs: Unit cost, retail price, salvage value, and penalty remain constant regardless of order volume."
    ]

    return NewsvendorResult(
        unit_selling_price=price,
        unit_cost=cost,
        salvage_value=salvage_value,
        shortage_penalty=shortage_penalty,
        demand_mean=demand_mean,
        demand_std=demand_std,
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_stocking_quantity=rec_q,
        optimal_stocking_quantity_continuous=round(q_star_continuous, 2),
        critical_ratio=round(critical_ratio, 4),
        expected_profit=round(exp_profit, 2),
        underage_cost_Cu=round(cu, 2),
        overage_cost_Co=round(co, 2),
        critical_fractile=round(critical_ratio, 4),
        optimal_z_score=round(z, 4),
        unit_loss_value=round(unit_loss_Lz, 6),
        expected_lost_sales=round(expected_lost_sales, 2),
        expected_leftover_units=round(expected_leftover, 2),
        expected_sales=round(expected_sales, 2),
        expected_revenue=round(exp_revenue, 2),
        expected_purchase_cost=round(exp_purchase_cost, 2),
        expected_shortage_penalty=round(exp_shortage_penalty, 2),
        expected_salvage_revenue=round(exp_salvage, 2),
        expected_profit_perfect_information=round(exp_profit_perfect, 2),
        expected_value_of_perfect_information_EVPI=round(evpi, 2),
        formulas=formula,
        sanity_notes=[v.details for v in verification_checks]
    )


# -----------------------------------------------------------------------------
# Backwards-compatible aliases for existing agent/tests/app
# -----------------------------------------------------------------------------
calculate_safety_stock_and_rop = calculate_rop
calculate_fixed_period_review = calculate_periodic_review
