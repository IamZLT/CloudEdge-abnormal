#!/usr/bin/env python3
"""Describe when a completed detection Agent corrected or harmed PatchCore.

This is a retrospective diagnostic.  It never selects an inference threshold and
must not be presented as validation of a rule tuned from the reported labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def collect(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_root.glob("mvtec/*/agent_result.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["rows"]:
            agent = row["agent"]
            expert = agent["expert"]
            review = agent.get("review")
            expert_prediction = int(str(expert["decision"]).upper() == "NG")
            final_prediction = int(row["prediction"])
            if expert_prediction == final_prediction:
                continue
            label = int(row["label"])
            expert_correct = expert_prediction == label
            final_correct = final_prediction == label
            outcome = "corrected" if (not expert_correct and final_correct) else "harmed"
            if expert_correct == final_correct:
                outcome = "neutral"
            probability = float(expert["probability"])
            rows.append({
                "category": report["category"],
                "path": row["path"],
                "label": label,
                "direction": f"{expert['decision']}->{agent['decision']}",
                "outcome": outcome,
                "expert_probability": probability,
                "expert_margin": abs(probability - 0.5),
                "expert_score": float(expert["score"]),
                "concentration": float(expert.get("concentration", 0.0)),
                "reference_similarity": _safe_float(expert.get("reference_similarity")),
                "review_decision": "" if review is None else review.get("decision", ""),
                "review_confidence": None if review is None else _safe_float(review.get("confidence")),
                "review_region_agreement": False if review is None else bool(review.get("region_agreement")),
                "review_reason": "" if review is None else review.get("reason", ""),
            })
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction = Counter(row["direction"] for row in rows)
    by_outcome = Counter(row["outcome"] for row in rows)
    category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category[row["category"]][row["outcome"]] += 1
    groups = {}
    for outcome in ("corrected", "harmed", "neutral"):
        selected = [row for row in rows if row["outcome"] == outcome]
        groups[outcome] = {
            "n": len(selected),
            "mean_expert_margin": _mean(selected, "expert_margin"),
            "mean_concentration": _mean(selected, "concentration"),
            "mean_reference_similarity": _mean(selected, "reference_similarity"),
            "mean_review_confidence": _mean(selected, "review_confidence"),
        }
    corrected = by_outcome["corrected"]
    harmed = by_outcome["harmed"]
    return {
        "changed_decisions": len(rows),
        "by_direction": dict(by_direction),
        "by_outcome": dict(by_outcome),
        "override_precision": corrected / max(1, corrected + harmed),
        "group_means": groups,
        "by_category": {key: dict(value) for key, value in sorted(category.items())},
        "note": "Retrospective test-label diagnostic only; do not use as an unbiased validation result.",
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "override_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (out / "override_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Detection Agent override diagnostic", "",
        "> Retrospective test-label diagnostic only; not an unbiased validation result.", "",
        f"- Changed decisions: {summary['changed_decisions']}",
        f"- Corrected: {summary['by_outcome'].get('corrected', 0)}",
        f"- Harmed: {summary['by_outcome'].get('harmed', 0)}",
        f"- Override precision: {summary['override_precision']:.2%}", "",
        "| Category | Corrected | Harmed |", "|---|---:|---:|",
    ]
    for category, counts in summary["by_category"].items():
        lines.append(f"| {category} | {counts.get('corrected', 0)} | {counts.get('harmed', 0)} |")
    (out / "override_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.result_root)
    summary = summarize(rows)
    write_outputs(rows, summary, args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
