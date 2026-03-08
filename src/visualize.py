import json
import networkx as nx
from pyvis.network import Network

INPUT_GRAPH_FILE = "stage3_memory_graph.json"
OUTPUT_HTML_FILE = "memory_graph_visualization.html"


# -------------------------------
# Load graph safely
# -------------------------------
def load_graph():

    with open(INPUT_GRAPH_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.DiGraph()

    # Detect where edges are stored
    if isinstance(data, list):
        edges = data

    elif isinstance(data, dict):

        if "edges" in data:
            edges = data["edges"]

        elif "links" in data:
            edges = data["links"]

        elif "claims" in data:
            edges = data["claims"]

        else:
            edges = []

    else:
        edges = []

    for edge in edges:

        # Skip strings (unstructured claims)
        if not isinstance(edge, dict):
            continue

        subject = str(edge.get("subject", edge.get("source", ""))).lower().strip()
        obj = str(edge.get("object", edge.get("target", ""))).lower().strip()
        relation = edge.get("relation", edge.get("predicate", ""))

        if not subject or not obj:
            continue

        graph.add_node(subject)
        graph.add_node(obj)

        graph.add_edge(
            subject,
            obj,
            relation=relation,
            confidence=edge.get("confidence", ""),
            evidence=edge.get("evidence", "")
        )

    return graph


# -------------------------------
# Build visualization
# -------------------------------
def build_visualization(graph):

    net = Network(height="750px", width="100%", directed=True)

    net.barnes_hut()

    for node in graph.nodes():

        net.add_node(
            node,
            label=node,
            title=f"Entity: {node}",
            size=20
        )

    for u, v, data in graph.edges(data=True):

        relation = data.get("relation", "")
        confidence = data.get("confidence", "")
        evidence = data.get("evidence", "")

        edge_title = f"""
Relation: {relation}
Confidence: {confidence}

Evidence:
{evidence}
"""

        net.add_edge(
            u,
            v,
            label=relation,
            title=edge_title
        )

    net.toggle_physics(True)

    return net


# -------------------------------
# Run visualization
# -------------------------------
def run():

    print("Loading memory graph...")

    graph = load_graph()

    print("Nodes:", len(graph.nodes()))
    print("Edges:", len(graph.edges()))

    net = build_visualization(graph)

    print("Generating visualization...")

    net.show(OUTPUT_HTML_FILE)

    print("Saved to:", OUTPUT_HTML_FILE)


if __name__ == "__main__":
    run()