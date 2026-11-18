"""
app.py - Interactive CLI Application for Supply Chain Analytics Inventory Decision Agent

Executes the decision architecture:
  Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
"""

import sys
import argparse
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from inventory_agent import InventoryAgent, AgentResponse, InventoryModelType

console = Console()

# Preloaded Benchmark Inventory Questions for Quick Testing
BENCHMARK_SCENARIOS = [
    {
        "name": "1. Economic Order Quantity (EOQ)",
        "query": (
            "A distribution center has an annual demand of 12,000 units. "
            "Ordering cost is $60 per order, and annual holding cost is $2.50 per unit. "
            "What is our optimal order quantity, how many orders per year should we place, "
            "and what is the total annual variable cost?"
        )
    },
    {
        "name": "2. Quantity Discount Optimization",
        "query": (
            "Annual demand is 10,000 units. Ordering cost is $40 per order. "
            "Carrying cost is 20%. The supplier offers price breaks: "
            "1 to 999 units at $10.00, 1000 to 1999 units at $9.60, 2000+ units at $9.20. "
            "Which price tier minimizes total cost and what is the optimal order quantity?"
        )
    },
    {
        "name": "3. Safety Stock and Reorder Point (Continuous Review)",
        "query": (
            "Our supplier lead time is 6 days. Average daily demand is 80 units "
            "with a standard deviation of 15 units. We target a 95% cycle service level. "
            "What safety stock should we carry, and at what inventory level should we place a reorder?"
        )
    },
    {
        "name": "4. Fixed-Period Review Policy (Order-Up-To Base Stock)",
        "query": (
            "We review store inventory every 14 days, and lead time is 7 days. "
            "Daily demand averages 35 units with standard deviation of 9 units. "
            "Target cycle service level is 98%. "
            "Currently, on-hand inventory is 180 units and on-order inventory is 120 units. "
            "What is our order-up-to level S, and how many units should we order today?"
        )
    },
    {
        "name": "5. Newsvendor Single-Period Perishable Inventory",
        "query": (
            "We are ordering commemorative sports jackets for a championship weekend. "
            "Selling price is $120 each, and wholesale purchase cost is $50 each. "
            "Unsold jackets will be liquidated at $20 salvage value. "
            "Demand is normally distributed with a mean of 1,200 jackets and standard deviation of 250 jackets. "
            "How many jackets should we order to maximize expected profit?"
        )
    }
]


def render_response(response: AgentResponse):
    """Renders the 7-stage agent lifecycle cleanly using rich panels and tables."""
    console.print("\n" + "=" * 80, style="dim")
    
    # Header & Question
    console.print(Panel(
        f"[bold white]{response.query}[/bold white]",
        title="[bold cyan]1. Customer Question[/bold cyan]",
        border_style="cyan"
    ))

    # Architecture Pipeline Summary Table
    pipe_table = Table(title="Agent Execution Trace (Architecture Pipeline)", show_header=True, header_style="bold magenta")
    pipe_table.add_column("Pipeline Stage", style="bold yellow", width=22)
    pipe_table.add_column("Agent Action / Output", style="white")

    pipe_table.add_row(
        "2. Identify Model",
        f"[bold green]{response.identified_model.value}[/bold green]"
    )
    
    # Formatted extracted inputs
    formatted_inputs = ", ".join(f"{k}: [cyan]{v}[/cyan]" for k, v in response.extracted_inputs.items())
    pipe_table.add_row("3. Extract Inputs", formatted_inputs or "[red]None[/red]")

    pipe_table.add_row("4. Select Python Tool", f"`[bold blue]{response.selected_tool}[/bold blue]`")

    calc_status = "[bold green]Successfully Computed[/bold green]" if response.calculation_result else "[bold red]Calculation Failed[/bold red]"
    pipe_table.add_row("5. Calculate", calc_status)

    verify_status = "[bold green]PASSED[/bold green]" if response.verification_passed else "[bold red]FAILED[/bold red]"
    verify_summary = f"{verify_status} - " + "; ".join(response.verification_notes)
    pipe_table.add_row("6. Verify", verify_summary)

    console.print(pipe_table)

    # Stage 7: Explanation & Executive Report
    console.print("\n")
    console.print(Panel(
        Markdown(response.explanation),
        title="[bold green]7. Explain & Decision Report[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))
    console.print("=" * 80 + "\n", style="dim")


def interactive_repl():
    """Interactive loop allowing the user to select benchmark questions or enter custom queries."""
    agent = InventoryAgent()

    console.print(Panel.fit(
        "[bold cyan]Supply Chain Analytics - Inventory Decision Agent[/bold cyan]\n"
        "[italic white]Natural-Language Inventory Optimization Engine[/italic white]\n\n"
        "[dim]Supported Models:[/dim]\n"
        "  1. Economic Order Quantity (EOQ)\n"
        "  2. Quantity Discount Optimization (All-Units)\n"
        "  3. Safety Stock & Reorder Point (Continuous Review)\n"
        "  4. Fixed-Period Review (Periodic Review / Order-Up-To Base Stock)\n"
        "  5. Newsvendor (Single-Period Stochastic Demand)\n\n"
        "[bold green]Architecture:[/bold green] Question → Identify Model → Extract Inputs → Select Tool → Calculate → Verify → Explain",
        border_style="bright_blue"
    ))

    while True:
        console.print("\n[bold yellow]Menu Options:[/bold yellow]")
        for idx, scenario in enumerate(BENCHMARK_SCENARIOS, start=1):
            console.print(f"  [{idx}] {scenario['name']}")
        console.print("  [C] Custom Question (Enter your own inventory problem)")
        console.print("  [Q] Quit")

        choice = Prompt.ask("\nSelect an option", default="1").strip()

        if choice.lower() in ("q", "quit", "exit"):
            console.print("[green]Exiting Inventory Decision Agent. Goodbye![/green]")
            break

        query_text = ""
        if choice.isdigit() and 1 <= int(choice) <= len(BENCHMARK_SCENARIOS):
            scenario = BENCHMARK_SCENARIOS[int(choice) - 1]
            query_text = scenario["query"]
            console.print(f"\n[bold]Selected Scenario:[/bold] {scenario['name']}")
        elif choice.lower() in ("c", "custom"):
            query_text = Prompt.ask("\n[bold cyan]Enter your inventory question[/bold cyan]").strip()
            if not query_text:
                console.print("[red]Empty input, returning to menu.[/red]")
                continue
        else:
            console.print("[red]Invalid selection. Please choose 1-5, C, or Q.[/red]")
            continue

        with console.status("[bold green]Agent thinking: Parsing → Selecting Tool → Computing...[/bold green]"):
            response = agent.process_query(query_text)

        render_response(response)


def main():
    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent CLI")
    parser.add_argument("query", nargs="?", help="Natural language inventory question to solve")
    parser.add_argument("--demo", action="store_true", help="Run all 5 benchmark scenarios sequentially")
    args = parser.parse_args()

    agent = InventoryAgent()

    if args.demo:
        console.print(Panel("[bold yellow]Running All 5 Benchmark Supply Chain Scenarios[/bold yellow]"))
        for s in BENCHMARK_SCENARIOS:
            console.print(f"\n[bold cyan]Executing Scenario: {s['name']}[/bold cyan]")
            resp = agent.process_query(s["query"])
            render_response(resp)
        return

    if args.query:
        resp = agent.process_query(args.query)
        render_response(resp)
        return

    interactive_repl()


if __name__ == "__main__":
    main()
