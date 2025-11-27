"""
Knowledge graph construction utilities.

For now we use an in-memory NetworkX MultiDiGraph.
Later, you can:
- serialize it
- or swap it out with Neo4j
"""

from typing import Dict, Any
import networkx as nx

# Global in-memory graph
_graph = nx.MultiDiGraph()


def get_graph() -> nx.MultiDiGraph:
    """Return the global knowledge graph instance."""
    return _graph


def add_company_node(ticker: str, name: str | None = None) -> None:
    """
    Add a company node like: company:AAPL
    """
    g = get_graph()
    node_id = f"company:{ticker}"
    g.add_node(
        node_id,
        type="company",
        ticker=ticker,
        name=name or ticker,
    )


def add_quarter_node(company_ticker: str, quarter: str) -> None:
    """
    Add a quarter node (e.g., 2025Q4) and connect it to the company.
    """
    g = get_graph()
    company_node = f"company:{company_ticker}"
    quarter_node = f"quarter:{company_ticker}:{quarter}"

    # Ensure company exists
    if company_node not in g:
        add_company_node(company_ticker)

    g.add_node(
        quarter_node,
        type="quarter",
        company=company_ticker,
        quarter=quarter,
    )

    g.add_edge(company_node, quarter_node, type="HAS_QUARTER")


def add_metric(
    company_ticker: str,
    quarter: str,
    metric_name: str,
    metric_value: float | None,
    unit: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Add a metric node for a given company + quarter.

    Example:
        add_metric("AAPL", "2025Q4", "Revenue growth", 0.12, unit="YoY")

    This will create:
    - Node: metric:AAPL:2025Q4:Revenue growth
    - Edge: quarter:AAPL:2025Q4 -> metric:... (REPORTS_METRIC)
    """
    g = get_graph()

    # Ensure the quarter node exists
    add_quarter_node(company_ticker, quarter)

    quarter_node = f"quarter:{company_ticker}:{quarter}"
    metric_node = f"metric:{company_ticker}:{quarter}:{metric_name}"

    g.add_node(
        metric_node,
        type="metric",
        name=metric_name,
        value=metric_value,
        unit=unit,
        **(extra or {}),
    )
    g.add_edge(quarter_node, metric_node, type="REPORTS_METRIC")
