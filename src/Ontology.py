import spacy
import json
import re
import pandas as pd
import google.generativeai as genai
from datetime import datetime


# -----------------------------
# Load NLP model
# -----------------------------
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key="AIzaSyDtGnjjKo7kWjs7GTlCtsrLoV-J_WW-Geo")
model = genai.GenerativeModel("gemini-2.5-flash-lite")


# -----------------------------
# Extraction Versioning
# -----------------------------
EXTRACTION_METADATA = {
    "model": "gemini-2.5-flash-lite",
    "prompt_version": "v1.0",
    "schema_version": "v1.0",
    "ontology_version": "v1.0"
}


# -----------------------------
# Ontology
# -----------------------------
ontology = {
    "entities": [
        "Person",
        "EmailAddress",
        "Organization",
        "EmailMessage",
        "Event",
        "Media",
        "Activity"
    ],
    "relationships": [
        "sends_email",
        "receives_email",
        "has_email",
        "works_for",
        "mentions_person",
        "discusses_event",
        "discusses_media",
        "participates_in"
    ]
}


# -----------------------------
# Deterministic Normalization
# -----------------------------
def normalize_entity_name(name):

    name = name.strip()
    name = name.replace("\n", " ")
    name = re.sub(r"\s+", " ", name)

    return name.lower()


# -----------------------------
# Entity Extraction
# -----------------------------
def extract_entities(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.label_ == "PERSON":
            entity_type = "Person"

        elif ent.label_ == "ORG":
            entity_type = "Organization"

        elif ent.label_ == "EVENT":
            entity_type = "Event"

        else:
            continue

        entities.append({
            "name": normalize_entity_name(ent.text),
            "type": entity_type
        })

    return entities


# -----------------------------
# Email Address Extraction
# -----------------------------
def extract_email_addresses(text):

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    emails = re.findall(email_pattern, text)

    results = []

    for email in emails:

        results.append({
            "name": normalize_entity_name(email),
            "type": "EmailAddress"
        })

    return results


# -----------------------------
# Entity Deduplication
# -----------------------------
def deduplicate_entities(entities):

    seen = set()
    unique = []

    for e in entities:

        key = (e["name"], e["type"])

        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# -----------------------------
# Offset Computation
# -----------------------------
def compute_offsets(text, excerpt):

    start = text.find(excerpt)

    if start == -1:
        return None, None

    end = start + len(excerpt)

    return start, end


# -----------------------------
# Safe JSON Parsing
# -----------------------------
def safe_json_load(result):

    try:
        return json.loads(result)

    except json.JSONDecodeError:

        print("Invalid JSON from model. Attempting repair.")

        start = result.find("{")
        end = result.rfind("}")

        if start != -1 and end != -1:

            try:
                return json.loads(result[start:end+1])
            except:
                return {"relationships": []}

        return {"relationships": []}


# -----------------------------
# Schema Validation
# -----------------------------
def validate_relationship_schema(data):

    valid_relationships = []

    for r in data.get("relationships", []):

        if not isinstance(r, dict):
            continue

        if "type" not in r:
            continue

        if "source" not in r or "target" not in r:
            continue

        if r["type"] not in ontology["relationships"]:
            continue

        if "evidence" not in r:
            continue

        valid_relationships.append(r)

    return {"relationships": valid_relationships}


# -----------------------------
# Quality Gate
# -----------------------------
def quality_gate(relationships):

    filtered = []

    for r in relationships:

        confidence = r.get("confidence", 0)

        if confidence >= 0.6:
            filtered.append(r)

        elif 0.5 < confidence < 0.6:
            r["requires_review"] = True
            filtered.append(r)

    return filtered


# -----------------------------
# Relationship Extraction
# -----------------------------
def extract_relationships(text, entities, source_id, timestamp):

    prompt = f"""
You are an information extraction system.

TEXT:
{text}

ENTITIES:
{json.dumps(entities, indent=2)}

ALLOWED RELATIONSHIPS:
{ontology["relationships"]}

Extract relationships ONLY from the allowed list.

Each relationship MUST include grounding evidence.

Return JSON:

{{
 "relationships":[
  {{
   "type":"",
   "source":"",
   "target":"",
   "confidence":0.0,
   "evidence":{{
        "source_id":"",
        "excerpt":"",
        "start_offset":0,
        "end_offset":0,
        "timestamp":""
   }}
  }}
 ]
}}

Rules:
excerpt must be exact text from TEXT
return ONLY valid JSON
"""

    max_retries = 3
    attempt = 0

    while attempt < max_retries:

        response = model.generate_content(prompt)

        result = response.text
        result = result.replace("```json", "").replace("```", "").strip()

        data = safe_json_load(result)
        data = validate_relationship_schema(data)

        if "relationships" in data:
            break

        attempt += 1

    for r in data.get("relationships", []):

        r["source"] = normalize_entity_name(r["source"])
        r["target"] = normalize_entity_name(r["target"])

        r["evidence"]["source_id"] = source_id
        r["evidence"]["timestamp"] = timestamp

        excerpt = r["evidence"].get("excerpt", "")

        start, end = compute_offsets(text, excerpt)

        r["evidence"]["start_offset"] = start
        r["evidence"]["end_offset"] = end

    return data


# -----------------------------
# Cross Evidence Support
# -----------------------------
def update_support_counts(all_results):

    support = {}

    for result in all_results:

        for r in result["relationships"]:

            key = (r["source"], r["type"], r["target"])

            support[key] = support.get(key, 0) + 1

    for result in all_results:

        for r in result["relationships"]:

            key = (r["source"], r["type"], r["target"])

            r["support_count"] = support[key]


# -----------------------------
# Full Pipeline
# -----------------------------
def run_pipeline(text, source_id, timestamp):

    entities = extract_entities(text)

    email_entities = extract_email_addresses(text)

    entities.extend(email_entities)

    entities = deduplicate_entities(entities)

    relationships_data = extract_relationships(
        text,
        entities,
        source_id,
        timestamp
    )

    filtered_relationships = quality_gate(
        relationships_data["relationships"]
    )

    return {
        "source_id": source_id,
        "text": text,
        "timestamp": timestamp,
        "extraction_metadata": EXTRACTION_METADATA,
        "entities": entities,
        "relationships": filtered_relationships
    }


# -----------------------------
# CSV Processing
# -----------------------------
def process_email_csv(file_path, n=None):

    df = pd.read_csv(file_path)

    if n is not None:
        df = df.head(n)

    all_results = []

    for i, row in df.iterrows():

        print(f"\nProcessing {i+1}/{len(df)}")

        text = str(row["message"])
        source_id = f"email_{i}"
        timestamp = row["date"] if "date" in df.columns else str(datetime.now())

        result = run_pipeline(text, source_id, timestamp)

        print("\nEntities Found:")

        if len(result["entities"]) == 0:
            print(" - None")

        for e in result["entities"]:
            print(" -", e["name"], "|", e["type"])

        print("\nRelationships Found:")

        if len(result["relationships"]) == 0:
            print(" - None")

        for r in result["relationships"]:
            print(" -", r["source"], "--", r["type"], "-->", r["target"])
            ev = r.get("evidence", {})
            print("    Evidence:", ev.get("excerpt", "")[:120])
            print("    Offset:", ev.get("start_offset"), "-", ev.get("end_offset"))
            print("    Timestamp:", ev.get("timestamp"))
            print("    Confidence:", r.get("confidence"))
            print("------------------------------------------")

        all_results.append(result)

    update_support_counts(all_results)

    return all_results


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    results = process_email_csv("subset3000.csv", n=5)

    with open("stage1_extraction_output.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nExtraction complete. Results saved to stage1_extraction_output.json")