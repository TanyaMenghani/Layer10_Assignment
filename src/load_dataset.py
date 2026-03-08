import json
import re
import pandas as pd
import spacy
import os
from dotenv import load_dotenv
from openai import OpenAI


# -------------------------
# Load API key
# -------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------
# Load spaCy
# -------------------------

nlp = spacy.load("en_core_web_sm")


# -------------------------
# Ontology
# -------------------------

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


# -------------------------
# Entity Extraction
# -------------------------

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

        if entity_type in ontology["entities"]:

            entities.append({
                "name": ent.text,
                "type": entity_type
            })

    return entities


# -------------------------
# Email Address Extraction
# -------------------------

def extract_email_addresses(text):

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    emails = re.findall(email_pattern, text)

    results = []

    for email in emails:

        results.append({
            "name": email,
            "type": "EmailAddress"
        })

    return results


# -------------------------
# Relationship Extraction (LLM)
# -------------------------

def extract_relationships(text, entities):

    prompt = f"""
You are an information extraction system.

TEXT:
{text}

ENTITIES:
{json.dumps(entities, indent=2)}

ALLOWED RELATIONSHIPS:
{ontology["relationships"]}

Extract relationships only from the allowed list.

Return JSON only in this format:

{{
 "relationships":[
  {{
   "type":"",
   "source":"",
   "target":""
  }}
 ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    output = response.choices[0].message.content

    return json.loads(output)


# -------------------------
# Pipeline
# -------------------------

def run_pipeline(email_text):

    entities = extract_entities(email_text)

    email_entities = extract_email_addresses(email_text)

    entities.extend(email_entities)

    relationships = extract_relationships(email_text, entities)

    return {
        "entities": entities,
        "relationships": relationships["relationships"]
    }


# -------------------------
# Load CSV
# -------------------------

def process_csv(file_path):

    df = pd.read_csv(file_path)

    all_results = []

    # Change column name if needed
    column_name = "body"

    if column_name not in df.columns:
        column_name = df.columns[0]

    for index, row in df.iterrows():

        text = str(row[column_name])

        result = run_pipeline(text)

        all_results.append(result)

    return all_results


# -------------------------
# Run
# -------------------------

if __name__ == "__main__":

    results = process_csv("subset3000.csv")

    print(json.dumps(results[:3], indent=4))

    print("\nProcessed emails:", len(results))