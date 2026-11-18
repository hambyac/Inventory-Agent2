"""
inventory_agent.py - Supply Chain Analytics Inventory Decision Agent

Implements the multi-stage decision architecture:
  Customer Question
       ↓
  Identify Model
       ↓
  Extract Inputs
       ↓
  Select Python Tool
       ↓
  Calculate
       ↓
  Verify
       ↓
  Explain
"""

import re
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

import inventory_tools as tools


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount"
    SAFETY_STOCK_ROP = "Safety Stock and Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Policy"
    NEWSVENDOR = "Newsvendor Single-Period"
    UNKNOWN = "Unknown / Ambiguous"


class AgentResponse(BaseModel):
    query: str
    identified_model: InventoryModelType
    extracted_inputs: Dict[str, Any]
    selected_tool: str
    calculation_result: Optional[Any] = None
    verification_passed: bool
    verification_notes: List[str]
    explanation: str


class InventoryAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.
    Accepts natural-language questions and executes deterministic inventory calculations.
    """

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # Stage 1: Identify Model
    # -------------------------------------------------------------------------
    def identify_model(self, query: str) -> InventoryModelType:
        """
        Classifies the customer query into one of the 5 supported inventory models
        using semantic keyword patterns and domain markers.
        """
        text = query.lower()

        # Newsvendor indicators (perishable, single season, salvage, critical fractile, underage/overage)
        newsvendor_keywords = [
            "newsvendor", "newsboy", "single period", "single-period", "perishable",
            "salvage", "liquidat", "overage", "underage", "critical fractile", "critical ratio",
            "shortage penalty", "goodwill penalty", "unsold units", "event", "fashion", "markdown"
        ]
        if any(kw in text for kw in newsvendor_keywords) and ("lead time" not in text or "salvage" in text or "selling price" in text):
            return InventoryModelType.NEWSVENDOR

        # Fixed Period Review indicators (periodic review, review period, order-up-to, base stock, R and S)
        periodic_keywords = [
            "fixed period", "fixed-period", "periodic review", "order up to", "order-up-to",
            "base stock", "review period", "review interval", "review every", "reviewed every",
            "inventory position", "on hand and", "on hand of", "on order"
        ]
        if any(kw in text for kw in periodic_keywords):
            return InventoryModelType.FIXED_PERIOD_REVIEW

        # Quantity Discount indicators (tiers, discount, price break, bracket, bulk order pricing)
        discount_keywords = [
            "quantity discount", "discount", "price break", "price breaks", "tier", "tiers",
            "bracket", "all-units", "bulk price", "volume discount"
        ]
        if any(kw in text for kw in discount_keywords) and ("price" in text or "tier" in text or "break" in text or "$" in text):
            return InventoryModelType.QUANTITY_DISCOUNT

        # Safety Stock & Reorder Point indicators (continuous review, lead time, service level, std dev, ROP)
        ss_rop_keywords = [
            "safety stock", "reorder point", "rop", "service level", "cycle service level",
            "csl", "stockout probability", "lead time demand", "continuous review"
        ]
        if any(kw in text for kw in ss_rop_keywords):
            # If review period is also present, it could be periodic, but checked earlier
            return InventoryModelType.SAFETY_STOCK_ROP

        # Economic Order Quantity indicators (classic Wilson EOQ)
        eoq_keywords = [
            "eoq", "economic order quantity", "optimal order quantity", "annual demand",
            "ordering cost", "setup cost", "holding cost", "carrying cost", "how many units should we order",
            "order frequency", "orders per year"
        ]
        if any(kw in text for kw in eoq_keywords):
            return InventoryModelType.EOQ

        # Fallback heuristic
        if "lead time" in text and "demand" in text:
            return InventoryModelType.SAFETY_STOCK_ROP
        if "holding" in text and "order" in text and "demand" in text:
            return InventoryModelType.EOQ

        return InventoryModelType.UNKNOWN

    # -------------------------------------------------------------------------
    # Stage 2: Extract Inputs
    # -------------------------------------------------------------------------
    def extract_inputs(self, query: str, model_type: InventoryModelType) -> Dict[str, Any]:
        """
        Extracts numerical parameters, rates, time horizons, and tier schedules
        from natural language text for the designated model.
        """
        inputs: Dict[str, Any] = {}
        text = query

        # Helper regex extractors
        def extract_first_float(patterns: List[str], text_to_search: str) -> Optional[float]:
            for pat in patterns:
                m = re.search(pat, text_to_search, re.IGNORECASE)
                if m:
                    raw_val = m.group(1).replace(",", "").strip()
                    try:
                        return float(raw_val)
                    except ValueError:
                        continue
            return None

        # 1. Model: EOQ
        if model_type == InventoryModelType.EOQ:
            # Demand
            demand = extract_first_float([
                r"(?:annual\s+demand|demand\s+is|demand\s+of|faces\s+an\s+annual\s+demand\s+of|demand)\s*(?:is|of|averages|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+per\s+year|units\s+annually|units\s+annual\s+demand)",
                r"demand[:=]?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            if demand is not None:
                inputs["demand"] = demand

            # Ordering / Setup Cost
            ordering_cost = extract_first_float([
                r"(?:ordering\s+cost|order\s+cost|setup\s+cost|cost\s+per\s+order)\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\$([\d,]+(?:\.\d+)?)\s*(?:per\s+order|to\s+place\s+an\s+order)",
                r"orders?\s+costs?\s*\$?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            if ordering_cost is not None:
                inputs["ordering_cost"] = ordering_cost

            # Holding Cost Rate (%)
            holding_rate_match = re.search(
                r"(?:holding|carrying)\s+(?:cost\s+rate|rate|cost\s+is)\s*(?:is|of|[:=])?\s*([\d\.]+)%",
                text, re.IGNORECASE
            )
            if holding_rate_match:
                inputs["holding_cost_rate"] = float(holding_rate_match.group(1)) / 100.0
            else:
                holding_pct_match = re.search(r"([\d\.]+)%\s*(?:holding|carrying)", text, re.IGNORECASE)
                if holding_pct_match:
                    inputs["holding_cost_rate"] = float(holding_pct_match.group(1)) / 100.0

            # Direct Holding Cost ($/unit/year) - extract BEFORE general unit cost
            holding_cost = extract_first_float([
                r"(?:holding\s+cost|carrying\s+cost|holding\s+is|holding)\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+unit|\/unit|\$)?",
                r"\$([\d,]+(?:\.\d+)?)\s*(?:per\s+unit\s+per\s+year|\/unit\/year)"
            ], text)
            if holding_cost is not None and "holding_cost_rate" not in inputs:
                inputs["holding_cost"] = holding_cost

            # Unit Cost (only if explicitly referring to item / purchase / unit price, not holding cost)
            unit_cost = extract_first_float([
                r"(?:purchase\s+price|purchase\s+cost|unit\s+price|item\s+cost)\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"(?<!holding\s)(?<!carrying\s)(?:unit\s+cost|cost\s+per\s+unit)\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"(?<!holding\s)(?<!carrying\s)costs?\s*\$([\d,]+(?:\.\d+)?)\s*(?:each|per\s+item)"
            ], text)
            if unit_cost is not None:
                inputs["unit_cost"] = unit_cost

        # 2. Model: Quantity Discount
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            # Demand
            demand = extract_first_float([
                r"(?:annual\s+demand|demand\s+is|demand\s+of)\s*[:=]?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+per\s+year|units\s+annually)",
                r"demand[:=]?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            if demand is not None:
                inputs["demand"] = demand

            # Ordering Cost
            ordering_cost = extract_first_float([
                r"(?:ordering\s+cost|order\s+cost|setup\s+cost)\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\$([\d,]+(?:\.\d+)?)\s*(?:per\s+order)"
            ], text)
            if ordering_cost is not None:
                inputs["ordering_cost"] = ordering_cost

            # Holding rate or fixed holding cost
            holding_rate_match = re.search(r"([\d\.]+)%\s*(?:holding|carrying)", text, re.IGNORECASE)
            if not holding_rate_match:
                holding_rate_match = re.search(
                    r"(?:holding|carrying)\s+(?:cost\s+rate|cost|rate)\s*(?:is|of|[:=])?\s*([\d\.]+)%?",
                    text, re.IGNORECASE
                )
            
            if holding_rate_match:
                val = float(holding_rate_match.group(1))
                if val > 1.0:
                    val = val / 100.0
                inputs["holding_cost_rate"] = val

            # Price Tiers Extraction
            # Matches bracketed patterns like:
            # "1-999 units at $10.00", "1,000 to 1,999: $9.50"
            tiers = []
            tier_pattern = re.findall(
                r"([\d,]+)\s*(?:to|-)\s*([\d,]+)\s*(?:units)?\s*(?:at|:|\$)?\s*\$?([\d\.]+)",
                text, re.IGNORECASE
            )
            if tier_pattern:
                for match in tier_pattern:
                    low_s, high_s, price_s = match
                    low_val = float(low_s.replace(",", "").rstrip("."))
                    high_val = float(high_s.replace(",", "").rstrip(".")) if high_s.strip() else None
                    price_val = float(price_s.rstrip("."))
                    tiers.append({"min_qty": low_val, "max_qty": high_val, "price": price_val})

            # Check open-ended tiers like "2000+ units at $9.20" or "over 2000 units at $9.00"
            plus_matches = re.findall(
                r"(?:(?:over|above|>=|at least)\s*([\d,]+)|([\d,]+)\s*\+|([\d,]+)\s*(?:and\s+above|or\s+more))\s*units?\s*(?:at|:|\$)?\s*\$?([\d\.]+)",
                text, re.IGNORECASE
            )
            for m1, m2, m3, price_s in plus_matches:
                raw_low = m1 or m2 or m3
                if raw_low:
                    low_val = float(raw_low.replace(",", "").rstrip("."))
                    if not any(t["min_qty"] == low_val for t in tiers):
                        tiers.append({"min_qty": low_val, "max_qty": None, "price": float(price_s.rstrip("."))})

            if tiers:
                inputs["price_tiers"] = tiers

        # 3. Model: Safety Stock and Reorder Point
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            # Lead Time
            lead_time = extract_first_float([
                r"(?:lead\s+time)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|periods)?",
                r"([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)\s*lead\s+time"
            ], text)
            if lead_time is not None:
                inputs["lead_time"] = lead_time

            # Service Level (e.g., 95% or 0.95)
            csl_match = re.search(r"([\d\.]+)%\s*(?:cycle\s+)?service\s+level", text, re.IGNORECASE)
            if not csl_match:
                csl_match = re.search(r"(?:target\s+)?(?:cycle\s+)?(?:service\s+level|csl)\s*(?:is|of|at|target\s+is|[:=])?\s*([\d\.]+)%?", text, re.IGNORECASE)
            if csl_match:
                csl_val = float(csl_match.group(1))
                if csl_val > 1.0:
                    csl_val = csl_val / 100.0
                inputs["service_level"] = csl_val

            # Daily Demand Mean
            demand_mean = extract_first_float([
                r"(?:daily\s+demand|average\s+daily\s+demand|mean\s+daily\s+demand|demand\s+averages?|daily\s+demand)\s*(?:is|of|averages|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+per\s+day|units\/day)",
                r"mean\s*(?:demand)?\s*[:=]?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            if demand_mean is not None:
                inputs["daily_demand_mean"] = demand_mean

            # Daily Demand Std Dev
            demand_std = extract_first_float([
                r"(?:standard\s+deviation|std\s+dev|std\s+deviation|sigma_d)\s*(?:of\s+demand|in\s+demand)?\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"sigma_d\s*[:=]?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            inputs["daily_demand_std"] = demand_std if demand_std is not None else 0.0

            # Lead time std dev (if variable lead time)
            lt_std = extract_first_float([
                r"(?:lead\s+time\s+std\s+dev|standard\s+deviation\s+of\s+lead\s+time|lead\s+time\s+standard\s+deviation)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            inputs["lead_time_std"] = lt_std if lt_std is not None else 0.0

            # Holding cost / unit cost
            h_cost = extract_first_float([r"holding\s+cost\s*(?:is|of|[:=])?\s*\$?\s*([\d\.]+)"], text)
            if h_cost is not None:
                inputs["holding_cost"] = h_cost

        # 4. Model: Fixed-Period Review Policy
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            # Review Period R
            review_period = extract_first_float([
                r"(?:review\s+period|review\s+interval|reviewed\s+every|review\s+every)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)?",
                r"every\s+([\d,]+(?:\.\d+)?)\s*(?:days|weeks|months)"
            ], text)
            if review_period is not None:
                inputs["review_period"] = review_period

            # Lead Time L
            lead_time = extract_first_float([
                r"(?:lead\s+time)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)\s*(?:days|weeks|periods)?",
                r"([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)\s*lead\s+time"
            ], text)
            if lead_time is not None:
                inputs["lead_time"] = lead_time

            # Service Level
            csl_match = re.search(r"([\d\.]+)%\s*(?:cycle\s+)?service\s+level", text, re.IGNORECASE)
            if not csl_match:
                csl_match = re.search(r"(?:target\s+)?(?:service\s+level|csl)\s*(?:is|of|at|target\s+is|[:=])?\s*([\d\.]+)%?", text, re.IGNORECASE)
            if csl_match:
                csl_val = float(csl_match.group(1))
                if csl_val > 1.0:
                    csl_val = csl_val / 100.0
                inputs["service_level"] = csl_val

            # Demand Mean
            demand_mean = extract_first_float([
                r"(?:daily\s+demand|average\s+daily\s+demand|mean\s+demand|demand\s+averages?|demand)\s*(?:is|of|averages|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+per\s+day|units\/day)"
            ], text)
            if demand_mean is not None:
                inputs["demand_mean"] = demand_mean

            # Demand Std
            demand_std = extract_first_float([
                r"(?:standard\s+deviation|std\s+dev|std\s+deviation)\s*(?:of\s+demand)?\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            inputs["demand_std"] = demand_std if demand_std is not None else 0.0

            # Current On-Hand
            on_hand = extract_first_float([
                r"(?:current\s+inventory|on[- ]hand\s+inventory|on[- ]hand\s+is|currently\s+have|on[- ]hand)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+on\s+hand)"
            ], text)
            inputs["current_on_hand"] = on_hand if on_hand is not None else 0.0

            # Current On-Order (Pipeline)
            on_order = extract_first_float([
                r"(?:on[- ]order\s+inventory|on[- ]order\s+is|pipeline\s+inventory|on[- ]order)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*(?:units\s+on\s+order)"
            ], text)
            inputs["current_on_order"] = on_order if on_order is not None else 0.0

            # Backorders
            backorders = extract_first_float([
                r"(?:backorders?|backlogged)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            inputs["backorders"] = backorders if backorders is not None else 0.0

        # 5. Model: Newsvendor
        elif model_type == InventoryModelType.NEWSVENDOR:
            # Selling Price
            price = extract_first_float([
                r"(?:retail\s+price|selling\s+price|sell\s+for|sells\s+for|price)\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\$([\d,]+(?:\.\d+)?)\s*(?:retail|selling\s+price|each)"
            ], text)
            if price is not None:
                inputs["price"] = price

            # Purchase Cost
            cost = extract_first_float([
                r"(?:purchase\s+cost|unit\s+cost|wholesale\s+price)\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"costs?\s+(?:us\s+)?\$?([\d,]+(?:\.\d+)?)",
                r"\$([\d,]+(?:\.\d+)?)\s*(?:to\s+buy|to\s+purchase|cost\s+per\s+unit|to\s+print|to\s+produce|to\s+make)",
                r"cost\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            if cost is not None:
                inputs["cost"] = cost

            # Salvage Value
            salvage = extract_first_float([
                r"(?:salvage\s+value|salvage|liquidat\w+\s+at|liquidat\w+\s+for|markdown\s+price|markdown\s+to|scrap\s+value)\s*(?:is|of|at|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)",
                r"\$([\d,]+(?:\.\d+)?)\s*(?:salvage|liquidation)"
            ], text)
            inputs["salvage_value"] = salvage if salvage is not None else 0.0

            # Shortage Penalty (Goodwill loss)
            penalty = extract_first_float([
                r"(?:shortage\s+penalty|penalty\s+cost|goodwill\s+loss|penalty)\s*(?:is|of|[:=])?\s*\$?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            inputs["shortage_penalty"] = penalty if penalty is not None else 0.0

            # Demand Mean
            mean_demand = extract_first_float([
                r"(?:mean\s+demand|demand\s+mean|average\s+demand|mean)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"mean\s*(?:of)?\s*([\d,]+(?:\.\d+)?)"
            ], text)
            if mean_demand is not None:
                inputs["demand_mean"] = mean_demand

            # Demand Std Dev
            std_demand = extract_first_float([
                r"(?:standard\s+deviation|std\s+dev|std\s+deviation|sigma)\s*(?:is|of|[:=])?\s*([\d,]+(?:\.\d+)?)",
                r"std\s+dev\s+is\s*([\d,]+(?:\.\d+)?)"
            ], text)
            if std_demand is not None:
                inputs["demand_std"] = std_demand

        return inputs

    # -------------------------------------------------------------------------
    # Stage 3: Select Python Tool
    # -------------------------------------------------------------------------
    def select_tool(self, model_type: InventoryModelType) -> str:
        """
        Maps identified model to the appropriate numerical Python tool name.
        """
        tool_mapping = {
            InventoryModelType.EOQ: "inventory_tools.calculate_eoq",
            InventoryModelType.QUANTITY_DISCOUNT: "inventory_tools.calculate_quantity_discount",
            InventoryModelType.SAFETY_STOCK_ROP: "inventory_tools.calculate_safety_stock_and_rop",
            InventoryModelType.FIXED_PERIOD_REVIEW: "inventory_tools.calculate_fixed_period_review",
            InventoryModelType.NEWSVENDOR: "inventory_tools.calculate_newsvendor"
        }
        return tool_mapping.get(model_type, "none")

    # -------------------------------------------------------------------------
    # Stage 4: Calculate
    # -------------------------------------------------------------------------
    def calculate(self, model_type: InventoryModelType, inputs: Dict[str, Any]) -> Any:
        """
        Executes deterministic numerical calculation using Python tools.
        """
        if model_type == InventoryModelType.EOQ:
            return tools.calculate_eoq(
                demand=inputs["demand"],
                ordering_cost=inputs["ordering_cost"],
                holding_cost=inputs.get("holding_cost"),
                unit_cost=inputs.get("unit_cost"),
                holding_cost_rate=inputs.get("holding_cost_rate"),
                working_days_per_year=inputs.get("working_days_per_year", 250)
            )

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            return tools.calculate_quantity_discount(
                demand=inputs["demand"],
                ordering_cost=inputs["ordering_cost"],
                price_tiers=inputs["price_tiers"],
                holding_cost_rate=inputs.get("holding_cost_rate"),
                fixed_holding_cost=inputs.get("fixed_holding_cost")
            )

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            return tools.calculate_safety_stock_and_rop(
                lead_time=inputs["lead_time"],
                service_level=inputs["service_level"],
                daily_demand_mean=inputs["daily_demand_mean"],
                daily_demand_std=inputs.get("daily_demand_std", 0.0),
                lead_time_std=inputs.get("lead_time_std", 0.0),
                holding_cost=inputs.get("holding_cost"),
                unit_cost=inputs.get("unit_cost"),
                holding_cost_rate=inputs.get("holding_cost_rate")
            )

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            return tools.calculate_fixed_period_review(
                review_period=inputs["review_period"],
                lead_time=inputs["lead_time"],
                service_level=inputs["service_level"],
                demand_mean=inputs["demand_mean"],
                demand_std=inputs.get("demand_std", 0.0),
                current_on_hand=inputs.get("current_on_hand", 0.0),
                current_on_order=inputs.get("current_on_order", 0.0),
                backorders=inputs.get("backorders", 0.0),
                holding_cost=inputs.get("holding_cost")
            )

        elif model_type == InventoryModelType.NEWSVENDOR:
            return tools.calculate_newsvendor(
                price=inputs["price"],
                cost=inputs["cost"],
                salvage_value=inputs.get("salvage_value", 0.0),
                shortage_penalty=inputs.get("shortage_penalty", 0.0),
                demand_mean=inputs["demand_mean"],
                demand_std=inputs["demand_std"]
            )

        raise ValueError(f"Unsupported model: {model_type}")

    # -------------------------------------------------------------------------
    # Stage 5: Verify
    # -------------------------------------------------------------------------
    def verify(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        result: Any
    ) -> Tuple[bool, List[str]]:
        """
        Validates mathematical integrity, boundary constraints, and domain realism.
        """
        notes = []
        is_valid = True

        if model_type == InventoryModelType.EOQ:
            eoq_res: tools.EOQResult = result
            if eoq_res.eoq <= 0:
                notes.append("Failure: Calculated EOQ is non-positive.")
                is_valid = False
            # Check cost balance
            diff = abs(eoq_res.annual_ordering_cost - eoq_res.annual_holding_cost)
            if diff > 1.0:
                notes.append(f"Notice: Annual ordering cost (${eoq_res.annual_ordering_cost}) and holding cost (${eoq_res.annual_holding_cost}) delta is ${diff:.2f}.")
            else:
                notes.append("Optimal cost trade-off verified: Ordering cost equals holding cost at EOQ.")
            if eoq_res.cycle_time_days > 250:
                notes.append("Warning: Cycle time exceeds 1 full operating year; check demand and holding parameters.")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            qd_res: tools.QuantityDiscountResult = result
            if qd_res.optimal_order_quantity <= 0:
                notes.append("Failure: Optimal quantity is non-positive.")
                is_valid = False
            notes.append(f"Verified all-units discount schedule across {len(qd_res.tiers)} tiers.")
            notes.append(f"Global minimum total annual cost verified at ${qd_res.minimum_total_annual_cost:,.2f}.")

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            rop_res: tools.SafetyStockROPResult = result
            if rop_res.reorder_point < 0:
                notes.append("Failure: Reorder point cannot be negative.")
                is_valid = False
            if rop_res.safety_stock < 0:
                notes.append("Notice: Service level below 50% results in negative safety stock.")
            else:
                notes.append(f"Positive safety stock buffer verified: {rop_res.recommended_safety_stock} units for {rop_res.service_level:.1%} CSL (z={rop_res.z_score:.2f}).")
            if rop_res.reorder_point < rop_res.lead_time_expected_demand:
                notes.append("Notice: Reorder point is lower than expected lead time demand.")

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            pr_res: tools.FixedPeriodReviewResult = result
            if pr_res.order_up_to_level_S <= 0:
                notes.append("Failure: Order-up-to level S must be positive.")
                is_valid = False
            notes.append(f"Protection interval of {pr_res.protection_period:.1f} periods (R={pr_res.review_period_R} + L={pr_res.lead_time_L}) verified.")
            notes.append(f"Target base stock level S set to {pr_res.recommended_order_up_to_S} units.")

        elif model_type == InventoryModelType.NEWSVENDOR:
            nv_res: tools.NewsvendorResult = result
            if nv_res.unit_selling_price <= nv_res.unit_cost:
                notes.append("Failure: Selling price must exceed cost for product to be profitable.")
                is_valid = False
            if nv_res.salvage_value >= nv_res.unit_cost:
                notes.append("Failure: Salvage value cannot exceed cost.")
                is_valid = False
            if not (0.0 < nv_res.critical_fractile < 1.0):
                notes.append("Failure: Critical fractile must lie in (0, 1).")
                is_valid = False
            notes.append(f"Critical ratio Cu/(Cu+Co) verified at {nv_res.critical_fractile:.2%}.")
            notes.append(f"Expected profit at optimal order quantity: ${nv_res.expected_profit:,.2f}.")

        return is_valid, notes

    # -------------------------------------------------------------------------
    # Stage 6: Explain (Structured for Student Verification across 8 Sections)
    # -------------------------------------------------------------------------
    def explain(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        result: Any,
        verification_notes: List[str]
    ) -> str:
        """
        Synthesizes student verification report across the 8 mandatory pedagogical sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations (from deterministic tool)
        6. Final Answer
        7. Verification Checks
        8. Assumptions
        """
        lines = []
        model_title = getattr(result, "model_name", model_type.value)
        lines.append(f"### Supply Chain Decision Analysis: {model_title}\n")

        # 1. Inputs Extracted
        lines.append("#### 1. Inputs Extracted")
        if hasattr(result, "inputs_extracted") and result.inputs_extracted:
            for k, v in result.inputs_extracted.items():
                if k != "price_tiers":
                    lines.append(f"- **{k}:** `{v}`")
                else:
                    lines.append(f"- **price_tiers:** `{len(v)} tiers evaluated`")
        else:
            for k, v in inputs.items():
                lines.append(f"- **{k}:** `{v}`")
        lines.append("")

        # 2. Variable Definitions
        lines.append("#### 2. Variable Definitions")
        if hasattr(result, "variable_definitions") and result.variable_definitions:
            for var, desc in result.variable_definitions.items():
                lines.append(f"- **`{var}`**: {desc}")
        lines.append("")

        # 3. Formula
        lines.append("#### 3. Formula")
        if hasattr(result, "formula") and result.formula:
            for f_name, f_expr in result.formula.items():
                lines.append(f"- **{f_name}:** `{f_expr}`")
        lines.append("")

        # 4. Formula with Substituted Values
        lines.append("#### 4. Formula with Substituted Values")
        if hasattr(result, "formula_with_substituted_values") and result.formula_with_substituted_values:
            for s_name, s_expr in result.formula_with_substituted_values.items():
                lines.append(f"- **{s_name}:** `{s_expr}`")
        lines.append("")

        # 5. Intermediate Calculations (Strictly from deterministic Python engine)
        lines.append("#### 5. Intermediate Calculations")
        if hasattr(result, "intermediate_calculations") and result.intermediate_calculations:
            for step_key, step_val in result.intermediate_calculations.items():
                if isinstance(step_val, dict):
                    formatted_dict = ", ".join(f"{sub_k}: {sub_v}" for sub_k, sub_v in step_val.items())
                    lines.append(f"- **{step_key}:** `{formatted_dict}`")
                else:
                    lines.append(f"- **{step_key}:** `{step_val}`")
        lines.append("")

        # 6. Final Answer
        lines.append("#### 6. Final Answer")
        if hasattr(result, "final_answer") and result.final_answer:
            for ans_key, ans_val in result.final_answer.items():
                lines.append(f"- **{ans_key.replace('_', ' ').title()}:** **`{ans_val}`**")
        lines.append("")

        # 7. Verification Checks
        lines.append("#### 7. Verification Checks")
        if hasattr(result, "verification_checks") and result.verification_checks:
            for check in result.verification_checks:
                status = "PASSED" if check.passed else "FAILED"
                lines.append(f"- [{status}] **{check.check_name}**: `{check.condition}` → *{check.details}*")
        elif verification_notes:
            for note in verification_notes:
                lines.append(f"- {note}")
        lines.append("")

        # 8. Assumptions
        lines.append("#### 8. Theoretical & Operational Assumptions")
        if hasattr(result, "assumptions") and result.assumptions:
            for idx, asm in enumerate(result.assumptions, 1):
                lines.append(f"{idx}. {asm}")
        lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # End-to-End Orchestrator Pipeline
    # -------------------------------------------------------------------------
    def process_query(self, query: str, input_overrides: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Executes the full Supply Chain Analytics Decision Architecture:
        Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        # Step 1: Identify Model
        model_type = self.identify_model(query)
        if model_type == InventoryModelType.UNKNOWN:
            return AgentResponse(
                query=query,
                identified_model=model_type,
                extracted_inputs={},
                selected_tool="none",
                calculation_result=None,
                verification_passed=False,
                verification_notes=["Could not identify an inventory decision model from the question."],
                explanation=(
                    "Unable to classify your question into one of the 5 supported inventory models:\n"
                    "1. Economic Order Quantity (EOQ)\n"
                    "2. Quantity Discount\n"
                    "3. Safety Stock and Reorder Point\n"
                    "4. Fixed-Period Review\n"
                    "5. Newsvendor\n\n"
                    "Please include parameters like demand, ordering costs, holding costs, lead time, or price brackets."
                )
            )

        # Step 2: Extract Inputs
        extracted = self.extract_inputs(query, model_type)
        if input_overrides:
            extracted.update(input_overrides)

        # Check for mandatory inputs
        missing = []
        if model_type == InventoryModelType.EOQ:
            if "demand" not in extracted: missing.append("annual demand")
            if "ordering_cost" not in extracted: missing.append("ordering cost")
            if "holding_cost" not in extracted and not ("unit_cost" in extracted and "holding_cost_rate" in extracted):
                missing.append("holding cost (or unit cost and holding cost rate)")
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            if "demand" not in extracted: missing.append("annual demand")
            if "ordering_cost" not in extracted: missing.append("ordering cost")
            if "price_tiers" not in extracted: missing.append("price discount tiers")
            if "holding_cost_rate" not in extracted and "fixed_holding_cost" not in extracted:
                missing.append("holding cost rate or fixed holding cost")
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            if "lead_time" not in extracted: missing.append("lead time")
            if "service_level" not in extracted: missing.append("cycle service level")
            if "daily_demand_mean" not in extracted: missing.append("mean daily demand")
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            if "review_period" not in extracted: missing.append("review period")
            if "lead_time" not in extracted: missing.append("lead time")
            if "service_level" not in extracted: missing.append("service level")
            if "demand_mean" not in extracted: missing.append("period demand mean")
        elif model_type == InventoryModelType.NEWSVENDOR:
            if "price" not in extracted: missing.append("selling price")
            if "cost" not in extracted: missing.append("purchase cost")
            if "demand_mean" not in extracted: missing.append("demand mean")
            if "demand_std" not in extracted: missing.append("demand standard deviation")

        if missing:
            return AgentResponse(
                query=query,
                identified_model=model_type,
                extracted_inputs=extracted,
                selected_tool=self.select_tool(model_type),
                calculation_result=None,
                verification_passed=False,
                verification_notes=[f"Missing required parameter(s): {', '.join(missing)}"],
                explanation=(
                    f"**Identified Model:** `{model_type.value}`\n\n"
                    f"The following required parameters could not be detected from your question: **{', '.join(missing)}**.\n"
                    f"Extracted parameters so far: `{extracted}`.\n\n"
                    "Please supply the missing values to proceed with calculation."
                )
            )

        # Step 3: Select Tool
        tool_name = self.select_tool(model_type)

        # Step 4: Calculate
        try:
            result = self.calculate(model_type, extracted)
        except Exception as e:
            return AgentResponse(
                query=query,
                identified_model=model_type,
                extracted_inputs=extracted,
                selected_tool=tool_name,
                calculation_result=None,
                verification_passed=False,
                verification_notes=[f"Calculation exception: {str(e)}"],
                explanation=f"Error executing `{tool_name}`: {str(e)}"
            )

        # Step 5: Verify
        is_valid, verification_notes = self.verify(model_type, extracted, result)

        # Step 6: Explain
        explanation = self.explain(model_type, extracted, result, verification_notes)

        return AgentResponse(
            query=query,
            identified_model=model_type,
            extracted_inputs=extracted,
            selected_tool=tool_name,
            calculation_result=result,
            verification_passed=is_valid,
            verification_notes=verification_notes,
            explanation=explanation
        )
