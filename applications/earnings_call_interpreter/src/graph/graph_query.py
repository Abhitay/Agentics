"""
Query functions for the knowledge graph.
Extracts structured data (metrics, relationships)
and returns readable text snippets for the RAG pipeline.
"""

from typing import List, Dict, Any
from src.graph.build_graph import get_graph


def query_graph(
    company: str,
    quarter: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve KG-backed context for a company.

    Returns:
        [
            {
                "text": "...",
                "metadata": {...}
            },
            ...
        ]

    Example outputs:
    - "AAPL 2024Q3 Revenue = 89500 USD millions"
    - "AAPL 2024Q3 Operating Margin = 0.42"
    """

    g = get_graph()
    snippets: List[Dict[str, Any]] = []

    # if quarter is provided, focus on that specific reporting period
    if quarter:
        q_node = f"quarter:{company}:{quarter}"

        if q_node not in g:
            return snippets

        # Get all metrics connected to the quarter node
        for _, metric_node, edge_data in g.out_edges(q_node, data=True):
            if edge_data.get("type") == "REPORTS_METRIC":
                metric = g.nodes[metric_node]
                metric_name = metric.get("name")
                metric_value = metric.get("value")
                unit = metric.get("unit", "")

                text = f"{company} {quarter} {metric_name} = {metric_value} {unit}".strip()

                snippets.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": "graph",
                            "company": company,
                            "quarter": quarter,
                            "metric": metric_name,
                            "value": metric_value,
                            "unit": unit,
                        },
                    }
                )

        return snippets

    # If NO quarter given → return all known data about the company
    for node_id, data in g.nodes(data=True):
        if data.get("type") == "metric" and data.get("name"):
            if node_id.startswith(f"metric:{company}:"):
                quarter = data.get("quarter")
                metric_name = data.get("name")
                metric_value = data.get("value")
                unit = data.get("unit", "")

                text = f"{company} {quarter} {metric_name} = {metric_value} {unit}".strip()

                snippets.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": "graph",
                            "company": company,
                            "quarter": quarter,
                            "metric": metric_name,
                            "value": metric_value,
                            "unit": unit,
                        },
                    }
                )

    return snippets
