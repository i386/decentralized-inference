#!/usr/bin/env python3
import argparse
import csv
import json
import urllib.request
from datetime import date
from pathlib import Path


DEFAULT_MATCH_CONFIDENCE = {
    "exact": 1.0,
    "family": 0.7,
    "fallback": 0.4,
}


def parse_percent(value: str):
    value = (value or "").strip()
    if not value or value == "N/A":
        return None
    if value.endswith("%"):
        value = value[:-1]
    return round(float(value), 2)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def fetch_csv(url: str):
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def build_bfcl_index(rows, score_field):
    out = {}
    for row in rows:
        score = parse_percent(row.get(score_field, ""))
        if score is None:
            continue
        out[row["Model"]] = score
    return out


def mapping_confidence(entry):
    override = entry.get("mapping_confidence")
    if override is not None:
        return round(float(override), 2)
    return DEFAULT_MATCH_CONFIDENCE.get(entry["match_quality"], 0.5)


def source_metadata(swe_score, overall_score, agentic_score):
    has_swe = swe_score is not None
    has_bfcl = overall_score is not None or agentic_score is not None
    source_count = int(has_swe) + int(has_bfcl)

    if has_swe and agentic_score is not None:
        source_confidence = 1.0
    elif has_swe and overall_score is not None:
        source_confidence = 0.9
    elif agentic_score is not None:
        source_confidence = 0.8
    elif has_swe:
        source_confidence = 0.7
    elif overall_score is not None:
        source_confidence = 0.55
    else:
        source_confidence = 0.0

    return source_count, round(source_confidence, 2)


def main():
    parser = argparse.ArgumentParser(description="Generate mesh-llm benchmark snapshot")
    parser.add_argument(
        "--aliases",
        default="mesh-llm/benchmarks/model_aliases.json",
        help="Alias and source config JSON",
    )
    parser.add_argument(
        "--swe-snapshot",
        default="mesh-llm/benchmarks/swe_rebench_snapshot.json",
        help="Manual SWE-rebench snapshot JSON",
    )
    parser.add_argument(
        "--output",
        default="mesh-llm/benchmarks/agentic_scores.json",
        help="Generated benchmark snapshot JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    aliases_path = repo_root / args.aliases
    swe_path = repo_root / args.swe_snapshot
    output_path = repo_root / args.output

    aliases = load_json(aliases_path)
    swe_snapshot = load_json(swe_path)

    bfcl_overall_rows = fetch_csv(aliases["bfcl"]["overall_url"])
    bfcl_agentic_rows = fetch_csv(aliases["bfcl"]["agentic_url"])

    bfcl_overall = build_bfcl_index(bfcl_overall_rows, "Overall Acc")
    bfcl_agentic = build_bfcl_index(bfcl_agentic_rows, "Agentic Overall Acc")
    swe_scores = {
        row["name"]: row["resolved_rate"]
        for row in swe_snapshot["scores"]
        if "resolved_rate" in row
    }

    models = []
    for entry in sorted(aliases["models"], key=lambda item: item["local_names"][0]):
        out = {
            "local_names": entry["local_names"],
            "benchmark_name": entry["benchmark_name"],
            "match_quality": entry["match_quality"],
            "mapping_confidence": mapping_confidence(entry),
        }
        notes = entry.get("notes")
        if notes:
            out["notes"] = notes

        swe_lookup = entry.get("swe_lookup", entry["benchmark_name"])
        swe_score = swe_scores.get(swe_lookup)
        if swe_score is not None:
            out["swe_rebench_resolved_rate"] = swe_score

        bfcl_lookup = entry.get("bfcl_lookup", entry["benchmark_name"])
        overall_score = bfcl_overall.get(bfcl_lookup)
        if overall_score is not None:
            out["bfcl_overall"] = overall_score

        bfcl_agentic_lookup = entry.get("bfcl_agentic_lookup", bfcl_lookup)
        agentic_score = bfcl_agentic.get(bfcl_agentic_lookup)
        if agentic_score is not None:
            out["bfcl_agentic"] = agentic_score

        source_count, source_confidence = source_metadata(
            swe_score, overall_score, agentic_score
        )
        out["source_count"] = source_count
        out["source_confidence"] = source_confidence

        models.append(out)

    output = {
        "version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/update_agentic_scores.py",
        "sources": {
            "swe_rebench": swe_snapshot["source"],
            "bfcl": {
                "url": aliases["bfcl"]["overall_url"],
                "agentic_url": aliases["bfcl"]["agentic_url"],
                "notes": aliases["bfcl"]["notes"],
            },
        },
        "models": models,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2, sort_keys=False)
        f.write("\n")


if __name__ == "__main__":
    main()
