import json
import re
import hashlib
from rapidfuzz import fuzz
from collections import defaultdict

INPUT_FILE = "stage1_extraction_output.json"
OUTPUT_FILE = "stage2_memory_deduplicated.json"
AUDIT_FILE = "merge_audit_log.json"


# -----------------------------
# GLOBAL STORES
# -----------------------------

artifact_store = {}
artifact_texts = {}

entity_store = {}
entity_alias_map = {}
entity_normalized_map = {}

claim_store = {}

audit_log = []

NOISE_ENTITIES = {
    "content-type",
    "mime-version",
    "content-transfer-encoding",
    "message-id",
    "date",
}


# -----------------------------
# TEXT NORMALIZATION
# -----------------------------

def normalize_text(text):

    text = text.lower()
    text = re.sub(r">+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# ENTITY NORMALIZATION
# -----------------------------

def normalize_entity_name(name):

    name = name.lower()

    # remove email domain
    name = re.sub(r'@.*', '', name)

    # replace separators
    name = name.replace(".", " ")
    name = name.replace("/", " ")
    name = name.replace("_", " ")

    # remove punctuation
    name = re.sub(r'[^a-z0-9 ]', '', name)

    name = name.replace(" jr", "")
    name = name.replace(" sr", "")
    name = name.replace(" xcc", "")

    name = re.sub(r"\s+", " ", name).strip()

    return name


def is_noise_entity(name):

    name = name.lower()

    if name in NOISE_ENTITIES:
        return True

    if len(name) < 2:
        return True

    return False


# -----------------------------
# ARTIFACT HASH
# -----------------------------

def artifact_hash(text):

    cleaned = normalize_text(text)

    return hashlib.sha256(cleaned.encode()).hexdigest()


# -----------------------------
# ARTIFACT DEDUP
# -----------------------------

def deduplicate_artifact(source_id, text):

    cleaned = normalize_text(text)

    h = artifact_hash(cleaned)

    if h in artifact_store:

        audit_log.append({
            "type": "artifact_merge",
            "duplicate": source_id,
            "canonical": artifact_store[h],
            "reason": "exact_hash"
        })

        return artifact_store[h]

    # near duplicate detection

    for existing_hash, existing_text in artifact_texts.items():

        if abs(len(cleaned) - len(existing_text)) > 200:
            continue

        score = fuzz.token_set_ratio(cleaned, existing_text)

        if score > 97:

            audit_log.append({
                "type": "artifact_merge",
                "duplicate": source_id,
                "canonical": artifact_store[existing_hash],
                "reason": "near_duplicate",
                "score": score
            })

            return artifact_store[existing_hash]

    artifact_store[h] = source_id
    artifact_texts[h] = cleaned

    return source_id


# -----------------------------
# ENTITY CANONICALIZATION
# -----------------------------

def canonicalize_entity(entity):

    name = entity["name"]
    etype = entity.get("type", "unknown")

    if is_noise_entity(name):
        return None

    normalized = normalize_entity_name(name)

    # exact normalized match

    if normalized in entity_normalized_map:

        canonical = entity_normalized_map[normalized]

        entity_alias_map[name] = canonical

        audit_log.append({
            "type": "entity_merge",
            "alias": name,
            "canonical": canonical,
            "reason": "normalized_match"
        })

        return canonical

    # fuzzy match

    for existing_name, existing in entity_store.items():

        if existing["type"] != etype:
            continue

        score = fuzz.token_set_ratio(
            normalized,
            normalize_entity_name(existing_name)
        )

        if score > 85:

            entity_alias_map[name] = existing_name

            audit_log.append({
                "type": "entity_merge",
                "alias": name,
                "canonical": existing_name,
                "score": score,
                "reason": "fuzzy_match"
            })

            return existing_name

    # new entity

    entity_store[name] = entity
    entity_normalized_map[normalized] = name

    return name


# -----------------------------
# RELATIONSHIP NORMALIZATION
# -----------------------------

RELATION_NORMALIZATION = {
    "receives_email": "email",
    "sends_email": "email",
}


def canonicalize_relationship(source, relation, target):

    if relation in RELATION_NORMALIZATION:
        relation = RELATION_NORMALIZATION[relation]

    return source, relation, target


# -----------------------------
# CLAIM KEY
# -----------------------------

def claim_key(source, relation, target):

    return f"{source}|{relation}|{target}"


# -----------------------------
# EVIDENCE DEDUP
# -----------------------------

def evidence_exists(evidence_set, new_evidence):

    snippet = (
        new_evidence.get("snippet")
        or new_evidence.get("excerpt")
    )

    for ev in evidence_set:

        existing = (
            ev.get("snippet")
            or ev.get("excerpt")
        )

        if existing == snippet:
            return True

    return False


# -----------------------------
# CLAIM DEDUP
# -----------------------------

def deduplicate_claim(claim):

    source = claim["source"]
    relation = claim["type"]
    target = claim["target"]

    source, relation, target = canonicalize_relationship(
        source, relation, target
    )

    key = claim_key(source, relation, target)

    evidence = claim["evidence"]

    timestamp = evidence.get("timestamp")

    confidence = claim.get("confidence", 0)

    if key not in claim_store:

        claim_store[key] = {
            "source": source,
            "type": relation,
            "target": target,
            "evidence_set": [evidence],
            "confidence": confidence,
            "support_count": 1,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "valid_from": timestamp,
            "valid_to": None
        }

        return

    if not evidence_exists(claim_store[key]["evidence_set"], evidence):

        claim_store[key]["evidence_set"].append(evidence)

    claim_store[key]["support_count"] += 1

    claim_store[key]["confidence"] = max(
        claim_store[key]["confidence"],
        confidence
    )

    claim_store[key]["confidence"] = min(
        1.0,
        claim_store[key]["confidence"] +
        0.05 * claim_store[key]["support_count"]
    )

    if timestamp:

        claim_store[key]["last_seen"] = timestamp

    audit_log.append({
        "type": "claim_merge",
        "claim": key,
        "timestamp": timestamp
    })


# -----------------------------
# CONFLICT DETECTION
# -----------------------------

def detect_conflicts():

    ownership_map = defaultdict(list)

    for key, claim in claim_store.items():

        if claim["type"] in {"works_for", "owned_by", "managed_by"}:

            ownership_map[claim["source"]].append(claim)

    for entity, claims in ownership_map.items():

        if len(claims) <= 1:
            continue

        claims.sort(key=lambda x: x["valid_from"] or "")

        for i in range(len(claims) - 1):

            old = claims[i]
            new = claims[i + 1]

            old["valid_to"] = new["valid_from"]

            audit_log.append({
                "type": "claim_revision",
                "entity": entity,
                "old_target": old["target"],
                "new_target": new["target"],
                "timestamp": new["valid_from"]
            })


# -----------------------------
# PIPELINE
# -----------------------------

def run_dedup_pipeline():

    with open(INPUT_FILE) as f:
        data = json.load(f)

    for doc in data:

        source_id = doc.get("source_id", "unknown")

        text = doc.get("text", "")

        deduplicate_artifact(source_id, text)

        for entity in doc.get("entities", []):

            canonicalize_entity(entity)

        for rel in doc.get("relationships", []):

            rel["source"] = entity_alias_map.get(
                rel["source"],
                rel["source"]
            )

            rel["target"] = entity_alias_map.get(
                rel["target"],
                rel["target"]
            )

            deduplicate_claim(rel)

    detect_conflicts()

    output = {
        "entities": list(entity_store.values()),
        "claims": list(claim_store.values()),
        "aliases": entity_alias_map
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    with open(AUDIT_FILE, "w") as f:
        json.dump(audit_log, f, indent=4)

    print("Deduplication complete.")
    print("Entities:", len(entity_store))
    print("Claims:", len(claim_store))
    print("Aliases:", len(entity_alias_map))
    print("Audit entries:", len(audit_log))


# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":

    run_dedup_pipeline()