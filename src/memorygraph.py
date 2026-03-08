import json
import hashlib
import re
from collections import defaultdict

INPUT_FILE = "stage2_memory_deduplicated.json"
OUTPUT_FILE = "stage3_memory_graph.json"


# -----------------------------
# ENTITY NORMALIZATION
# -----------------------------

def normalize_entity(name):

    name = name.lower()

    name = name.replace("@enron.com", "")
    name = name.replace(".", " ")
    name = name.replace("/", " ")
    name = name.replace("_", " ")

    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# -----------------------------
# CLAIM ID GENERATION
# -----------------------------

def claim_id(source, relation, target):

    key = f"{source}|{relation}|{target}"
    return hashlib.md5(key.encode()).hexdigest()


# -----------------------------
# BUILD MEMORY GRAPH
# -----------------------------

def build_graph(data):

    graph = {
        "meta": {
            "schema_version": 1,
            "source": "stage2_memory_deduplicated"
        },
        "entities": {},
        "claims": {},
        "artifacts": {},
        "entity_index": defaultdict(list)
    }

    # -------------------
    # ENTITIES
    # -------------------

    for e in data.get("entities", []):

        name = e["name"]

        if name not in graph["entities"]:

            graph["entities"][name] = {
                "name": name,
                "type": e.get("type", "unknown"),
                "claims_out": [],
                "claims_in": []
            }

        norm = normalize_entity(name)

        graph["entity_index"][norm].append(name)

    # -------------------
    # CLAIMS
    # -------------------

    for claim in data.get("claims", []):

        source = claim.get("source")
        relation = claim.get("type")
        target = claim.get("target")

        if not source or not relation or not target:
            continue

        cid = claim_id(source, relation, target)

        claim["claim_id"] = cid
        claim["relation"] = relation

        # prevent duplicate claims
        if cid not in graph["claims"]:
            graph["claims"][cid] = claim

        src = source
        tgt = target

        # -------------------
        # LINK CLAIM TO ENTITIES
        # -------------------

        if src in graph["entities"]:

            if cid not in graph["entities"][src]["claims_out"]:
                graph["entities"][src]["claims_out"].append(cid)

        if tgt in graph["entities"]:

            if cid not in graph["entities"][tgt]["claims_in"]:
                graph["entities"][tgt]["claims_in"].append(cid)

        # -------------------
        # ARTIFACTS / EVIDENCE
        # -------------------

        for ev in claim.get("evidence_set", []):

            aid = (
                ev.get("artifact_id")
                or ev.get("source_id")
                or ev.get("artifact")
                or "unknown_artifact"
            )

            if aid not in graph["artifacts"]:

                text = ev.get("text") or ev.get("excerpt", "")

                graph["artifacts"][aid] = {
                    "artifact_id": aid,
                    "text": text[:2000],  # trim large artifacts
                    "timestamp": ev.get("timestamp"),
                    "claims": []
                }

            if cid not in graph["artifacts"][aid]["claims"]:
                graph["artifacts"][aid]["claims"].append(cid)

    return graph


# -----------------------------
# GRAPH STATISTICS
# -----------------------------

def graph_stats(graph):

    entity_count = len(graph["entities"])
    claim_count = len(graph["claims"])
    artifact_count = len(graph["artifacts"])

    total_edges = sum(
        len(e["claims_out"]) for e in graph["entities"].values()
    )

    avg_claims = total_edges / entity_count if entity_count else 0

    print("\nGraph statistics")
    print("----------------")
    print("Entities:", entity_count)
    print("Claims:", claim_count)
    print("Artifacts:", artifact_count)
    print("Average claims per entity:", round(avg_claims, 2))


# -----------------------------
# MAIN RUNNER
# -----------------------------

def run():

    with open(INPUT_FILE) as f:
        data = json.load(f)

    graph = build_graph(data)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(graph, f, indent=2)

    print("Memory graph built successfully")

    graph_stats(graph)


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":
    run()