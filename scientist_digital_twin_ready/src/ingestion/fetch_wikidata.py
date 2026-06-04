"""
Phase 1.1 - Wikidata Structured Facts Fetcher.
Fetches entity JSON for scientist QID and resolves labels.
No LLM.
"""
import json
import time
import yaml
import requests
from pathlib import Path

BASE_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
CONFIG_PATH = Path("config/scientist.yaml")
OUTPUT_PATH = Path("data/processed/scientist_wikidata.json")

PROPERTY_MAP = {
    "P31": "instance_of",
    "P21": "gender",
    "P569": "date_of_birth",
    "P570": "date_of_death",
    "P27": "country_of_citizenship",
    "P19": "place_of_birth",
    "P69": "educated_at",
    "P108": "employer",
    "P184": "doctoral_advisor",
    "P185": "doctoral_students",
    "P101": "fields_of_work",
    "P737": "influenced_by",
    "P738": "influenced",
    "P166": "awards_received",
    "P800": "notable_works",
    "P1344": "participant_of",
    "P106": "occupation",
}

session = requests.Session()
session.headers.update({"User-Agent": "ScientistTwin/1.0 (research project)"})
label_cache: dict[str, str] = {}


def fetch_entity(qid: str) -> dict:
    url = BASE_URL.format(qid=qid)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Retry {attempt+1}/3 for {qid}: {e} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {qid} after 3 retries")


def resolve_label(qid: str) -> str:
    if qid in label_cache:
        return label_cache[qid]
    try:
        data = fetch_entity(qid)
        entities = data.get("entities", {})
        entity = entities.get(qid, {})
        label = (
            entity.get("labels", {}).get("en", {}).get("value")
            or entity.get("descriptions", {}).get("en", {}).get("value")
            or qid
        )
        label_cache[qid] = label
        time.sleep(0.2)
        return label
    except Exception:
        label_cache[qid] = qid
        return qid


def extract_time_value(snak: dict) -> str | None:
    dv = snak.get("datavalue", {})
    if dv.get("type") == "time":
        return dv["value"].get("time", "").lstrip("+")
    return None


def extract_snak_value(snak: dict) -> str | None:
    dv = snak.get("datavalue", {})
    dtype = dv.get("type")
    if dtype == "wikibase-entityid":
        qid = dv["value"].get("id")
        return resolve_label(qid) if qid else None
    if dtype == "string":
        return dv.get("value")
    if dtype == "monolingualtext":
        return dv["value"].get("text")
    if dtype == "time":
        return dv["value"].get("time", "").lstrip("+")
    return None


def parse_claims(claims: dict, entity_qid: str) -> dict:
    result: dict[str, any] = {}
    for prop, label in PROPERTY_MAP.items():
        if prop not in claims:
            continue
        values = []
        for statement in claims[prop]:
            mainsnak = statement.get("mainsnak", {})
            val = extract_snak_value(mainsnak)
            if not val:
                continue
            entry: dict = {"value": val}
            # Extract qualifiers (start/end dates, year)
            qualifiers = statement.get("qualifiers", {})
            if "P580" in qualifiers:  # start time
                t = qualifiers["P580"][0].get("datavalue", {}).get("value", {})
                entry["start"] = t.get("time", "").lstrip("+")[:10] if isinstance(t, dict) else ""
            if "P582" in qualifiers:  # end time
                t = qualifiers["P582"][0].get("datavalue", {}).get("value", {})
                entry["end"] = t.get("time", "").lstrip("+")[:10] if isinstance(t, dict) else ""
            if "P585" in qualifiers:  # point in time
                t = qualifiers["P585"][0].get("datavalue", {}).get("value", {})
                entry["year"] = t.get("time", "").lstrip("+")[:4] if isinstance(t, dict) else ""
            values.append(entry)
        if values:
            # Simplify single-value scalar fields
            scalars = {"date_of_birth", "date_of_death", "gender", "country_of_citizenship", "place_of_birth"}
            if label in scalars:
                result[label] = values[0]["value"]
            else:
                result[label] = values
    return result


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    scientist = config["scientist"]
    qid = scientist["wikidata_qid"]
    name = scientist["name"]

    print(f"Fetching Wikidata entity for {name} ({qid})...")
    data = fetch_entity(qid)
    entities = data.get("entities", {})
    entity = entities.get(qid, {})

    en_label = entity.get("labels", {}).get("en", {}).get("value", name)
    en_description = entity.get("descriptions", {}).get("en", {}).get("value", "")
    claims = entity.get("claims", {})

    print("Parsing claims and resolving QID labels...")
    parsed = parse_claims(claims, qid)
    parsed["name"] = en_label
    parsed["qid"] = qid
    parsed["description"] = en_description

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {OUTPUT_PATH}")

    print("\n--- Summary ---")
    for k, v in parsed.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} entries")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
