import json
import re
from rapidfuzz import fuzz
from collections import defaultdict

GRAPH_FILE = "stage3_memory_graph.json"

MAX_ENTITIES = 5
MAX_CLAIMS = 20
MAX_EVIDENCE = 3


# -----------------------------
# LOAD GRAPH
# -----------------------------

def load_graph():

    with open(GRAPH_FILE) as f:
        graph = json.load(f)

    for cid, claim in graph["claims"].items():

        if "relation" not in claim:
            claim["relation"] = claim.get("type", "related_to")

    return graph


# -----------------------------
# ENTITY NORMALIZATION
# -----------------------------

def normalize_entity(name):

    name = name.lower()

    name = re.sub(r'@.*', '', name)
    name = name.replace(".", " ")
    name = name.replace("/", " ")

    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# -----------------------------
# INTENT DETECTION
# -----------------------------

def detect_intent(question):

    q = question.lower()

    if "email" in q or "emailed" in q or "sent" in q:
        return "email"

    if "mention" in q or "mentioned" in q:
        return "mentions_person"

    if "discuss" in q or "topic" in q:
        return "discusses_media"

    return None


# -----------------------------
# ENTITY RETRIEVAL
# -----------------------------

def find_candidate_entities(question, graph):

    q = normalize_entity(question)

    results = []

    for name in graph["entities"]:

        norm = normalize_entity(name)

        score = max(
            fuzz.partial_ratio(q, name.lower()),
            fuzz.partial_ratio(q, norm)
        )

        if score > 70:
            results.append((name, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in results[:MAX_ENTITIES]]


# -----------------------------
# CLAIM EXPANSION
# -----------------------------

def collect_claims(entities, graph):

    claim_ids = set()

    for e in entities:

        if e not in graph["entities"]:
            continue

        claim_ids.update(graph["entities"][e].get("claims_out", []))
        claim_ids.update(graph["entities"][e].get("claims_in", []))

    return list(claim_ids)


# -----------------------------
# CLAIM SCORING
# -----------------------------

def score_claim(claim):

    confidence = claim.get("confidence", 0)

    evidence_count = len(claim.get("evidence_set", []))

    recency_score = 0

    for ev in claim.get("evidence_set", []):

        ts = ev.get("timestamp")

        if ts:
            try:
                year = int(str(ts)[:4])
                recency_score = max(recency_score, year)
            except:
                pass

    score = (
        0.5 * confidence +
        0.3 * evidence_count +
        0.2 * (recency_score / 2025 if recency_score else 0)
    )

    return score


# -----------------------------
# RANK CLAIMS
# -----------------------------

def rank_claims(claim_ids, graph, intent=None, entities=None, question=None):

    scored = []

    for cid in claim_ids:

        claim = graph["claims"][cid]

        # intent filtering
        if intent and claim["relation"] != intent:
            continue

        # direction filtering for email questions
        if intent == "email" and entities:
            if claim["target"] not in entities:
                continue

        score = score_claim(claim)

        if entities:
            if claim["source"] in entities:
                score += 1

            if claim["target"] in entities:
                score += 1

        scored.append((cid, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    entity_counter = defaultdict(int)
    seen_pairs = set()

    final = []

    for cid, score in scored:

        claim = graph["claims"][cid]

        pair = tuple(sorted([claim["source"], claim["target"]]))

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)

        src = claim["source"]

        entity_counter[src] += 1

        if entity_counter[src] <= 5:
            final.append(cid)

        if len(final) >= MAX_CLAIMS:
            break

    return final


# -----------------------------
# CONFLICT DETECTION
# -----------------------------

def detect_conflicts(claim_ids, graph):

    conflict_relations = {"works_for", "ceo_of", "born_in"}

    relation_map = defaultdict(list)

    for cid in claim_ids:

        claim = graph["claims"][cid]

        if claim["relation"] not in conflict_relations:
            continue

        key = (claim["source"], claim["relation"])

        relation_map[key].append(claim)

    conflicts = []

    for key, claims in relation_map.items():

        targets = set(c["target"] for c in claims)

        if len(targets) > 1:

            conflicts.append({
                "entity": key[0],
                "relation": key[1],
                "targets": list(targets)
            })

    return conflicts


# -----------------------------
# CONTEXT PACK
# -----------------------------

def build_context_pack(claim_ids, graph):

    context = []

    for cid in claim_ids:

        claim = graph["claims"][cid]

        entry = {
            "claim": f"{claim['source']} --{claim['relation']}--> {claim['target']}",
            "confidence": claim.get("confidence", 0),
            "evidence": []
        }

        for ev in claim.get("evidence_set", [])[:MAX_EVIDENCE]:

            snippet = ev.get("text") or ev.get("excerpt", "")

            entry["evidence"].append({
                "artifact_id": ev.get("artifact_id") or ev.get("source_id"),
                "timestamp": ev.get("timestamp"),
                "snippet": snippet[:200]
            })

        context.append(entry)

    return context


# -----------------------------
# MAIN RETRIEVAL PIPELINE
# -----------------------------

def retrieve(question, graph):

    intent = detect_intent(question)

    entities = find_candidate_entities(question, graph)

    if not entities:

        for cid, claim in graph["claims"].items():

            text = f"{claim['source']} {claim['relation']} {claim['target']}".lower()

            if any(word in text for word in question.lower().split()):
                entities.append(claim["source"])
                break

    claim_ids = collect_claims(entities, graph)

    ranked_claims = rank_claims(claim_ids, graph, intent, entities, question)

    conflicts = detect_conflicts(ranked_claims, graph)

    context = build_context_pack(ranked_claims, graph)

    return {
        "entities_found": entities,
        "intent": intent,
        "context_pack": context,
        "conflicts": conflicts
    }


# -----------------------------
# CLI DEMO
# -----------------------------

if __name__ == "__main__":

    graph = load_graph()

    print("\nMemory Retrieval Ready\n")

    while True:

        q = input("\nAsk a question (or 'exit'): ")

        if q.lower() == "exit":
            break

        result = retrieve(q, graph)

        print("\nEntities detected:")

        for e in result["entities_found"]:
            print(" -", e)

        if result["intent"]:
            print("\nDetected intent:", result["intent"])

        print("\nContext Pack:\n")

        for item in result["context_pack"]:

            print("CLAIM:", item["claim"])
            print("Confidence:", item["confidence"])

            for ev in item["evidence"]:

                print(f"  Evidence [{ev['artifact_id']} | {ev['timestamp']}]")
                print("   ", ev["snippet"])

            print()

        if result["conflicts"]:

            print("\n⚠ Conflicts detected:")

            for c in result["conflicts"]:

                print(f"{c['entity']} has multiple {c['relation']} targets:")

                for t in c["targets"]:
                    print(" -", t)