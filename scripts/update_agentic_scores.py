#!/usr/bin/env python3
import argparse
import csv
import json
import re
import urllib.request
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path


DEFAULT_MATCH_CONFIDENCE = {
    "exact": 1.0,
    "family": 0.7,
    "fallback": 0.4,
}

FAMILY_ALIASES = {
    "qwen25": "qwen",
    "qwen35": "qwen",
    "qwen3": "qwen",
    "qwen2": "qwen",
    "llama3": "llama",
    "llama4": "llama",
    "glm4": "glm",
    "glm46": "glm",
    "glm47": "glm",
    "gemma3": "gemma",
}

STOPWORDS = {
    "gguf",
    "instruct",
    "it",
    "chat",
    "prompt",
    "fc",
    "thinking",
    "exp",
}

SPECIAL_TOKENS = {"coder", "flash", "scout", "small", "large", "distill", "mini"}


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


def strip_split_suffix(name: str) -> str:
    return re.sub(r"-\d{5}-of-\d{5}$", "", name)


def strip_local_quantization(name: str) -> str:
    name = strip_split_suffix(name)
    return re.sub(r"-Q\d+(?:_K(?:_M)?)?$", "", name, flags=re.IGNORECASE)


def strip_bfcl_variant(name: str) -> str:
    return re.sub(r"\s+\([^)]*\)$", "", name).strip()


def normalize_family_token(token: str) -> str:
    token = FAMILY_ALIASES.get(token, token)
    match = re.match(r"([a-z]+)\d+$", token)
    if match and match.group(1) in {
        "qwen",
        "llama",
        "glm",
        "gemma",
        "mistral",
        "mixtral",
        "devstral",
        "deepseek",
        "minimax",
        "hermes",
    }:
        return match.group(1)
    return token


def normalized_tokens(name: str):
    name = strip_local_quantization(name)
    name = strip_bfcl_variant(name)
    name = re.sub(r"(?<=\d)\.(?=\d)", "", name)
    name = name.lower().replace("/", " ")
    name = re.sub(r"[^a-z0-9]+", " ", name)

    tokens = []
    for raw in name.split():
        if raw in STOPWORDS:
            continue
        if raw.isdigit() and len(raw) >= 4:
            continue
        token = normalize_family_token(raw)
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def literal_tokens(name: str):
    name = strip_local_quantization(name)
    name = strip_bfcl_variant(name)
    name = re.sub(r"(?<=\d)\.(?=\d)", "", name)
    name = name.lower().replace("/", " ")
    name = re.sub(r"[^a-z0-9]+", " ", name)

    tokens = []
    for raw in name.split():
        if raw in STOPWORDS:
            continue
        if raw.isdigit() and len(raw) >= 4:
            continue
        if raw:
            tokens.append(raw)
    return tokens


def family_key(tokens):
    for token in tokens:
        if token in {
            "qwen",
            "llama",
            "glm",
            "gemma",
            "mistral",
            "mixtral",
            "devstral",
            "deepseek",
            "minimax",
            "hermes",
        }:
            return token
    return tokens[0] if tokens else ""


def size_tokens(tokens):
    out = set()
    for token in tokens:
        if re.fullmatch(r"\d+b", token):
            out.add(token)
        elif re.fullmatch(r"\d+x\d+b", token):
            out.add(token)
        elif re.fullmatch(r"a\d+b", token):
            out.add(token)
    return out


def match_score(local_name: str, benchmark_name: str) -> int:
    local_tokens = normalized_tokens(local_name)
    benchmark_tokens = normalized_tokens(benchmark_name)

    local_family = family_key(local_tokens)
    benchmark_family = family_key(benchmark_tokens)

    score = 0
    if local_family == benchmark_family:
        score += 60
    elif {local_family, benchmark_family} <= {"mixtral", "mistral"}:
        score += 15
    elif local_family == "hermes" and benchmark_family == "mistral":
        score += 10
    else:
        score -= 80

    local_sizes = size_tokens(local_tokens)
    benchmark_sizes = size_tokens(benchmark_tokens)
    shared_sizes = local_sizes & benchmark_sizes
    if shared_sizes:
        score += 25 + 5 * len(shared_sizes)
    elif local_sizes and benchmark_sizes:
        score -= 10

    local_set = set(local_tokens)
    benchmark_set = set(benchmark_tokens)
    score += 4 * len(local_set & benchmark_set)
    score += 8 * len((local_set & benchmark_set) & SPECIAL_TOKENS)
    score += int(20 * SequenceMatcher(None, " ".join(local_tokens), " ".join(benchmark_tokens)).ratio())

    if local_tokens == benchmark_tokens:
        score += 40
    return score


def candidate_quality(local_name: str, benchmark_name: str, score: int):
    literal_local = literal_tokens(local_name)
    literal_benchmark = literal_tokens(benchmark_name)
    local_tokens = normalized_tokens(local_name)
    benchmark_tokens = normalized_tokens(benchmark_name)
    local_sizes = size_tokens(local_tokens)
    benchmark_sizes = size_tokens(benchmark_tokens)

    if literal_local == literal_benchmark:
        return "exact", 1.0
    if family_key(local_tokens) == family_key(benchmark_tokens):
        if (local_sizes & benchmark_sizes) or score >= 95:
            return "family", 0.8
        return "family", 0.65
    return "fallback", 0.4


def preferred_bfcl_lookup(rows):
    def priority(name: str):
        if "(FC" in name:
            return (0, name)
        if "(Prompt" in name:
            return (1, name)
        return (2, name)

    return sorted(rows, key=priority)[0]


def build_benchmark_catalog(swe_snapshot, bfcl_overall_rows, bfcl_agentic_rows):
    catalog = {}

    for row in swe_snapshot["scores"]:
        benchmark_name = row["name"]
        catalog.setdefault(benchmark_name, {"benchmark_name": benchmark_name})
        catalog[benchmark_name]["swe_lookup"] = benchmark_name

    grouped_overall = {}
    for row in bfcl_overall_rows:
        grouped_overall.setdefault(strip_bfcl_variant(row["Model"]), []).append(row["Model"])

    grouped_agentic = {}
    for row in bfcl_agentic_rows:
        grouped_agentic.setdefault(strip_bfcl_variant(row["Model"]), []).append(row["Model"])

    for benchmark_name, rows in grouped_overall.items():
        entry = catalog.setdefault(benchmark_name, {"benchmark_name": benchmark_name})
        entry["bfcl_lookup"] = preferred_bfcl_lookup(rows)

    for benchmark_name, rows in grouped_agentic.items():
        entry = catalog.setdefault(benchmark_name, {"benchmark_name": benchmark_name})
        entry["bfcl_agentic_lookup"] = preferred_bfcl_lookup(rows)

    return catalog


def extract_local_models(router_path: Path):
    text = router_path.read_text()
    section = text.split("pub static MODEL_PROFILES: &[ModelProfile] = &[", 1)[1].split("];", 1)[0]
    blocks = re.findall(r"ModelProfile\s*\{(.*?)\n\s*\},", section, re.S)

    models = []
    for block in blocks:
        name_match = re.search(r'name:\s*"([^"]+)"', block)
        tools_match = re.search(r"tools:\s*(true|false)", block)
        if not name_match:
            continue
        models.append(
            {
                "name": name_match.group(1),
                "tools": tools_match and tools_match.group(1) == "true",
            }
        )
    return models


def matching_rule(local_name: str, rules):
    name = local_name.lower()
    for rule in rules:
        if rule["match_substring"] in name:
            return rule
    return None


def auto_alias(local_model, catalog, rules):
    local_name = local_model["name"]
    rule = matching_rule(local_name, rules)

    if rule is not None:
        benchmark_name = rule["benchmark_name"]
        candidate = dict(catalog.get(benchmark_name, {"benchmark_name": benchmark_name}))
        candidate.update(rule)
        match_quality = candidate.get("match_quality", "family")
        mapping_conf = candidate.get(
            "mapping_confidence",
            DEFAULT_MATCH_CONFIDENCE.get(match_quality, 0.5),
        )
        return {
            "local_names": [local_name],
            "benchmark_name": benchmark_name,
            "match_quality": match_quality,
            "mapping_confidence": round(float(mapping_conf), 2),
            **{
                key: candidate[key]
                for key in ("swe_lookup", "bfcl_lookup", "bfcl_agentic_lookup", "notes")
                if key in candidate
            },
        }

    candidates = sorted(
        catalog.values(),
        key=lambda item: match_score(local_name, item["benchmark_name"]),
        reverse=True,
    )
    best = candidates[0]
    score = match_score(local_name, best["benchmark_name"])
    match_quality, mapping_conf = candidate_quality(local_name, best["benchmark_name"], score)

    alias = {
        "local_names": [local_name],
        "benchmark_name": best["benchmark_name"],
        "match_quality": match_quality,
        "mapping_confidence": round(mapping_conf, 2),
    }

    if "swe_lookup" in best and best["swe_lookup"] != best["benchmark_name"]:
        alias["swe_lookup"] = best["swe_lookup"]
    if "bfcl_lookup" in best:
        alias["bfcl_lookup"] = best["bfcl_lookup"]
    if "bfcl_agentic_lookup" in best and best["bfcl_agentic_lookup"] != best.get("bfcl_lookup"):
        alias["bfcl_agentic_lookup"] = best["bfcl_agentic_lookup"]
    if match_quality == "fallback":
        alias["notes"] = "Auto-generated low-confidence fallback from benchmark name similarity."
    return alias


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


def build_agentic_snapshot(aliases, swe_snapshot, bfcl_config, bfcl_overall_rows, bfcl_agentic_rows):
    bfcl_overall = build_bfcl_index(bfcl_overall_rows, "Overall Acc")
    bfcl_agentic = build_bfcl_index(bfcl_agentic_rows, "Agentic Overall Acc")
    swe_scores = {
        row["name"]: row["resolved_rate"]
        for row in swe_snapshot["scores"]
        if "resolved_rate" in row
    }

    models = []
    for entry in aliases["models"]:
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

    return {
        "version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/update_agentic_scores.py",
        "sources": {
            "swe_rebench": swe_snapshot["source"],
            "bfcl": bfcl_config,
        },
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate mesh-llm benchmark snapshots")
    parser.add_argument(
        "--rules",
        default="mesh-llm/benchmarks/model_alias_rules.json",
        help="Rules and source config JSON",
    )
    parser.add_argument(
        "--swe-snapshot",
        default="mesh-llm/benchmarks/swe_rebench_snapshot.json",
        help="Manual SWE-rebench snapshot JSON",
    )
    parser.add_argument(
        "--aliases-output",
        default="mesh-llm/benchmarks/model_aliases.json",
        help="Generated alias snapshot JSON",
    )
    parser.add_argument(
        "--output",
        default="mesh-llm/benchmarks/agentic_scores.json",
        help="Generated benchmark snapshot JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rules_path = repo_root / args.rules
    swe_path = repo_root / args.swe_snapshot
    aliases_output_path = repo_root / args.aliases_output
    output_path = repo_root / args.output
    router_path = repo_root / "mesh-llm/src/router.rs"

    rules = load_json(rules_path)
    swe_snapshot = load_json(swe_path)

    bfcl_overall_rows = fetch_csv(rules["bfcl"]["overall_url"])
    bfcl_agentic_rows = fetch_csv(rules["bfcl"]["agentic_url"])
    catalog = build_benchmark_catalog(swe_snapshot, bfcl_overall_rows, bfcl_agentic_rows)
    local_models = extract_local_models(router_path)

    aliases = {
        "version": 1,
        "generated_at": date.today().isoformat(),
        "generated_by": "scripts/update_agentic_scores.py",
        "sources": {
            "router": str(router_path.relative_to(repo_root)),
            "swe_rebench": swe_snapshot["source"],
            "bfcl": rules["bfcl"],
            "rules": str(rules_path.relative_to(repo_root)),
        },
        "models": sorted(
            [auto_alias(model, catalog, rules["rules"]) for model in local_models],
            key=lambda item: item["local_names"][0],
        ),
    }

    agentic_snapshot = build_agentic_snapshot(
        aliases,
        swe_snapshot,
        rules["bfcl"],
        bfcl_overall_rows,
        bfcl_agentic_rows,
    )

    aliases_output_path.parent.mkdir(parents=True, exist_ok=True)
    aliases_output_path.write_text(json.dumps(aliases, indent=2) + "\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(agentic_snapshot, indent=2) + "\n")


if __name__ == "__main__":
    main()
